# rag_fastapi_service.py

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.protocol import CompletionRequest, TokenizeRequest, DetokenizeRequest
from vllm.entrypoints.openai.serving_models import OpenAIServingModels, BaseModelPath
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_tokenization import OpenAIServingTokenization
import uvicorn
import logging
# 从 log_service 导入日志记录器和辅助函数
from log_service import get_logger, log_rag_request_start, log_rag_request_end, log_first_llm_call, log_neo4j_query, log_second_llm_call, log_entity_relationship_group_processing
import sys
import os
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
import json
from typing import AsyncGenerator, Dict, Any, List, Optional, Tuple
from neo4j_service import Neo4jService # 导入Neo4j服务类
import pydantic
from pydantic import BaseModel, Field
from transformers import AutoTokenizer # 用于计算token长度
import datetime

# --- 1. 配置与日志 ---
# 日志已在 log_service.py 中配置
logger = get_logger(__name__) # 获取特定于当前模块的记录器

# --- 2. 配置类 (集中管理所有配置) ---
class Config:
    # vLLM 模型配置
    MODEL_PATH = "/home/project/MedicalQA/base_model"
    SERVED_MODEL_NAME = "Qwen3-4B-Instruct-2507"
    # vLLM 引擎参数 (复用您提供的配置)
    ENGINE_ARGS = AsyncEngineArgs(
        model=MODEL_PATH,
        enable_lora=False,
        trust_remote_code=True,
        tokenizer_mode="auto",
        dtype="auto",
        quantization="bitsandbytes", # 注意：您在pip list中安装了bitsandbytes
        kv_cache_dtype="auto",
        enable_prefix_caching=False,
        gpu_memory_utilization=0.40,
        max_model_len=4096, # 这是关键限制
        max_num_batched_tokens=4096,
        max_num_seqs=16,
        enforce_eager=False,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        swap_space=8
    )
    # Neo4j 配置
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://74f0576d.databases.neo4j.io") # 请替换为实际的云端Neo4j地址
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")           # 请替换为实际用户名
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Jvabu2hncwBYexP_vWoQpRQd3tIp0pul7QK4sK6xh_s") # 请替换为实际密码
    # 映射文件路径
    ENTITY_TABLE_FILE = "mapping/entity.json" # 与脚本同目录
    ENTITY_CLASS_FILE = "mapping/entity_class.json" # 与脚本同目录，用于中文到英文类型的映射
    RELATIONSHIP_TABLE_FILE = "mapping/relationship.json" # 与脚本同目录
    INTENT_MAPPING_FILE = "mapping/intent_interface_mapping.json" # 与脚本同目录
    # 为安全起见，为输出预留一些token
    SAFETY_MARGIN = 512
    MAX_PROMPT_TOKENS = ENGINE_ARGS.max_model_len - SAFETY_MARGIN
    # LLM 调用参数
    LLM_TEMPERATURE_PARSE = 0.0 # 解析阶段通常用较低温度
    LLM_TEMPERATURE_GENERATE = 0.1 # 生成阶段可稍高
    LLM_MAX_TOKENS_PARSE = 1024 # 预估解析结果的token数
    LLM_MAX_TOKENS_GENERATE = 1024 # 生成回答的token数

# --- 3. 数据模型 (Pydantic) ---
class RAGRequest(BaseModel):
    question: str = Field(..., description="用户提出的问题")
    session_id: Optional[str] = Field(None, description="会话ID，用于上下文管理（当前RAG流程未使用，但可预留）")

class Entity(BaseModel):
    name: str
    type: str # 这里将是英文类型

class EntityRelationshipGroup(BaseModel):
    entities: List[Entity]
    relationships: List[str]
    intent: str

class RAGParseResult(BaseModel):
    entity_relationship_groups: List[EntityRelationshipGroup]
    standalone_entities: List[Entity]
    overall_intent: str

# --- 4. 加载映射表 ---
def load_mapping_tables():
    """加载实体表、实体分类表、关系表和意图映射表"""
    # 获取当前脚本文件的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建配置文件的完整路径
    entity_file_path = os.path.join(script_dir, Config.ENTITY_TABLE_FILE)
    entity_class_file_path = os.path.join(script_dir, Config.ENTITY_CLASS_FILE)
    relationship_file_path = os.path.join(script_dir, Config.RELATIONSHIP_TABLE_FILE)
    intent_mapping_file_path = os.path.join(script_dir, Config.INTENT_MAPPING_FILE)

    # 加载实体表 (包含中文分类名)
    try:
        with open(entity_file_path, 'r', encoding='utf-8') as f:
            entity_data = json.load(f)
        logger.info(f"成功加载实体表文件: {entity_file_path}")
    except FileNotFoundError:
        logger.error(f"实体表文件未找到: {entity_file_path}")
        entity_data = {}
    except json.JSONDecodeError as e:
        logger.error(f"实体表文件JSON格式错误: {e}")
        entity_data = {}

    # 加载实体分类表 (中文 -> 英文类型映射)
    try:
        with open(entity_class_file_path, 'r', encoding='utf-8') as f:
            entity_class_data = json.load(f)
        # 将列表转换为字典，方便查找
        entity_type_mapping = {k: v for k, v in entity_class_data.items()}
        logger.info(f"成功加载实体分类映射文件: {entity_class_file_path}")
    except FileNotFoundError:
        logger.error(f"实体分类映射文件未找到: {entity_class_file_path}")
        entity_type_mapping = {}
    except json.JSONDecodeError as e:
        logger.error(f"实体分类映射文件JSON格式错误: {e}")
        entity_type_mapping = {}

    # 加载关系表
    try:
        with open(relationship_file_path, 'r', encoding='utf-8') as f:
            raw_rel_data = json.load(f)

        # 根据您提供的relationship.json格式，提取中文关系名作为列表
        # 格式: {"relationships": [{"并发症": "acompany_with", ...}, {"其他关系": "other_mapping"}, ...]}
        # 需要遍历列表中的每个字典，提取所有键
        relationship_data = []
        rel_mapping = {}
        for item in raw_rel_data.get("relationships", []):
             if isinstance(item, dict):
                 # 从当前字典中提取所有中文关系名
                 relationship_data.extend(item.keys())
                 # 将当前字典的映射关系更新到总映射表中
                 rel_mapping.update(item)
             else:
                 logger.warning(f"Skipping non-dict item in relationships list: {item}")

        # 去重关系列表（以防万一有重复）
        relationship_data = list(set(relationship_data))

        logger.info(f"成功加载关系表文件: {relationship_file_path}")

    except FileNotFoundError:
        logger.error(f"关系表文件未找到: {relationship_file_path}")
        relationship_data = []
        rel_mapping = {}
    except json.JSONDecodeError as e:
        logger.error(f"关系表文件JSON格式错误: {e}")
        relationship_data = []
        rel_mapping = {}

    # 加载意图映射表
    try:
        with open(intent_mapping_file_path, 'r', encoding='utf-8') as f:
            intent_mapping_data = json.load(f)
        logger.info(f"成功加载意图映射表文件: {intent_mapping_file_path}")
    except FileNotFoundError:
        logger.error(f"意图映射表文件未找到: {intent_mapping_file_path}")
        intent_mapping_data = {}
    except json.JSONDecodeError as e:
        logger.error(f"意图映射表文件JSON格式错误: {e}")
        intent_mapping_data = {}

    return entity_data, entity_type_mapping, relationship_data, intent_mapping_data, rel_mapping

# --- 5. vLLM 服务集成类 ---
class VLLMService:
    def __init__(self, engine: AsyncLLMEngine, model_config, base_model_paths: List[BaseModelPath]):
        self.engine = engine
        self.model_config = model_config
        self.openai_serving_models = OpenAIServingModels(
            engine_client=self.engine,
            model_config=model_config,
            base_model_paths=base_model_paths
        )
        self.openai_serving_chat = OpenAIServingChat(
            self.engine,
            self.model_config,
            self.openai_serving_models,
            response_role="assistant",
            chat_template=None,
            request_logger=None,
            chat_template_content_format="auto"
        )

    async def chat_completion(self, messages: List[Dict[str, str]], stream: bool = False, temperature: float = 0.1, max_tokens: int = 16) -> Any:
        # 构造 ChatCompletionRequest
        request = ChatCompletionRequest(
            model=Config.SERVED_MODEL_NAME,
            messages=messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens
        )
        if stream:
            generator = await self.openai_serving_chat.create_chat_completion(request)
            return generator
        else:
            response = await self.openai_serving_chat.create_chat_completion(request)
            return response

# --- 6. 主 FastAPI 应用 ---
async def create_app():
    """创建并配置 FastAPI 应用"""
    logger.info("正在初始化模型引擎...")
    engine = AsyncLLMEngine.from_engine_args(Config.ENGINE_ARGS)

    # 获取模型配置
    async def get_model_config():
        return await engine.get_model_config()
    model_config = await get_model_config()

    # 创建 BaseModelPath
    base_model_paths = [BaseModelPath(name=Config.SERVED_MODEL_NAME, model_path=Config.MODEL_PATH)]

    # 创建 VLLM 服务实例 (传递已创建的 engine 实例)
    vllm_service = VLLMService(engine, model_config, base_model_paths)

    # 加载映射表
    logger.info("正在加载映射表...")
    entity_data, entity_type_mapping, relationship_data, intent_mapping_data, rel_mapping = load_mapping_tables()

    # 创建 Neo4j 服务实例
    logger.info(f"正在连接Neo4j数据库: {Config.NEO4J_URI}")
    neo4j_service = Neo4jService(Config.NEO4J_URI, Config.NEO4J_USER, Config.NEO4J_PASSWORD)

    # 加载分词器用于计算token
    logger.info(f"正在加载分词器: {Config.MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)

    # 创建 FastAPI 应用
    app = FastAPI(
        title="Medical RAG API Server",
        description="API server for medical RAG using vLLM and Neo4j",
        version="0.1.0"
    )

    # --- 7. 自定义 RAG 端点 (核心) ---
    @app.post("/v1/medical_rag_stream", response_class=StreamingResponse)
    async def medical_rag_stream(request: RAGRequest):
        question = request.question
        session_id = request.session_id # 当前流程未使用，但可预留

        logger.info(f"Received RAG request for session {session_id}, question: {question}")
        log_rag_request_start(question, session_id)

        try:
            # --- Step 1: Entity/Intent Parsing (LLM) ---
            # 构建解析Prompt，包含实体表、关系表、意图列表
            entities_str = json.dumps(entity_data, ensure_ascii=False, indent=2)
            relationships_str = json.dumps(relationship_data, ensure_ascii=False, indent=2)
            intents_str = json.dumps(list(intent_mapping_data.keys()), ensure_ascii=False, indent=2) # 提取意图名称列表

            # 优化后的Prompt，将标准列表和输出格式放入 system 部分
            parse_prompt = f"""<|system|>
你是一个专业的医疗助手。你的知识库包含以下标准实体、关系和意图：

标准实体列表:
{entities_str}

标准关系列表:
{relationships_str}

标准意图列表:
{intents_str}

你的任务是：
1.  分析用户提出的问题。
2.  从问题中识别出属于标准实体列表中的实体，或生成专业医学术语。
3.  从标准关系列表中选择最匹配的关系。
4.  从标准意图列表中选择最匹配的意图。（用户一句话可能有多个意图，如“我可能有点感冒怎么办？”，这里用户的意图是“查找症状”和“查找治疗方法”）
5.  将识别出的实体、关系与意图进行逻辑分组。
6.  识别整体意图和独立实体。

请严格按照以下JSON格式输出，不要输出其他内容：

输出格式为:
{{
  "entity_relationship_groups": [
    {{
      "entities": [{{"name": "标准实体名或生成的专业术语", "type": "疾病/症状/药物/部门/食物/检查/厂商" # 使用entity.json中的中文类型名}}, ...],
      "relationships": ["标准关系名"], # 必须从关系表中选择，可以为空列表 []
      "intent": "标准意图名" # 必须从意图列表中选择，不可为空
    }}
  ],
  "standalone_entities": [{{"name": "实体名", "type": "中文类型名"}}],
  "overall_intent": "整体意图"
}}
</|system|>
<|user|>
请分析以下用户问题：

用户问题: {question}
</|user|>
<|assistant|>
"""
            parse_messages = [{"role": "user", "content": parse_prompt}]

            # 记录第一次调用的完整输入
            # log_to_file(f"First LLM Call - Input Prompt:\n{parse_prompt}") # 移除旧的日志调用
            logger.debug(f"First LLM Call - Input Prompt (truncated): {parse_prompt[:200]}...") # 控制台记录简略信息

            parse_response = await vllm_service.chat_completion(
                parse_messages,
                stream=False,
                temperature=Config.LLM_TEMPERATURE_PARSE,
                max_tokens=Config.LLM_MAX_TOKENS_PARSE
            )
            logger.debug(f"Parse response: {parse_response}")

            # 记录第一次调用的完整输出
            first_llm_output = parse_response.choices[0].message.content
            log_first_llm_call(parse_prompt, first_llm_output) # 使用辅助函数记录到文件

            # 假设模型返回的是JSON格式的字符串
            try:
                parse_data = json.loads(first_llm_output)
                parse_result = RAGParseResult(
                    entity_relationship_groups=parse_data.get("entity_relationship_groups", []),
                    standalone_entities=parse_data.get("standalone_entities", []),
                    overall_intent=parse_data.get("overall_intent", "")
                )
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                # log_to_file(f"Error: Failed to parse first LLM output as JSON: {e}") # 移除旧的日志调用
                raise HTTPException(status_code=500, detail="解析模型响应失败")

            logger.info(f"Parsed result: {parse_result.dict()}")
            # log_to_file(f"Parsed Result:\n{json.dumps(parse_result.dict(), ensure_ascii=False, indent=2)}") # 移除旧的日志调用

            # --- Step 2: Map Chinese Entity Types to English & Map Relationships & Query Neo4j (Service-side) ---
            # 遍历 entity_relationship_groups 并调用相应接口
            neo4j_results = []
            for i, group in enumerate(parse_result.entity_relationship_groups):
                logger.info(f"Processing group {i}: {group.dict()}")
                # log_to_file(f"Processing group {i} (before mapping): {group.dict()}") # 移除旧的日志调用
                log_entity_relationship_group_processing(i, group.dict(), None, None) # 记录处理开始

                # 将中文实体类型映射为英文类型
                mapped_entities = []
                for entity in group.entities:
                    english_type = entity_type_mapping.get(entity.type, entity.type) # 如果找不到映射，保留原名
                    mapped_entity = Entity(name=entity.name, type=english_type)
                    mapped_entities.append(mapped_entity)
                # 更新group中的实体列表
                group.entities = mapped_entities

                # 将中文关系名映射为英文关系类型
                english_relationships = [rel_mapping.get(rel, rel) for rel in group.relationships] # 如果找不到映射，保留原名

                # 记录映射结果到日志文件
                # logger.info(f"Group {i} - Mapped Entities: {[e.dict() for e in group.entities]}") # 控制台记录
                # logger.info(f"Group {i} - Mapped Relationships: {english_relationships}") # 控制台记录
                # log_to_file(f"Group {i} - Mapped Entities: {[e.dict() for e in group.entities]}") # 移除旧的日志调用
                # log_to_file(f"Group {i} - Mapped Relationships: {english_relationships}") # 移除旧的日志调用
                log_entity_relationship_group_processing(i, group.dict(), [e.dict() for e in group.entities], english_relationships) # 使用辅助函数记录到文件

                # 检查关系和意图是否为空
                if not group.relationships:
                    logger.warning(f"Group {i} has no relationships: {group.dict()}")
                    # log_to_file(f"Warning: Group {i} has no relationships.") # 移除旧的日志调用
                    # 可以选择跳过此组，或根据意图使用默认关系
                    # 这里选择跳过
                    continue
                if not group.intent:
                    logger.warning(f"Group {i} has no intent: {group.dict()}")
                    # log_to_file(f"Warning: Group {i} has no intent.") # 移除旧的日志调用
                    # 可以选择跳过此组，或尝试推断意图
                    # 这里选择跳过
                    continue

                # 根据意图查找映射的接口
                intent = group.intent
                if intent not in intent_mapping_data:
                    logger.warning(f"未知意图: {intent}")
                    # log_to_file(f"Warning: Unknown intent '{intent}' in group {i}.") # 移除旧的日志调用
                    continue

                interface_info = intent_mapping_data[intent]
                interface_name = interface_info["neo4j_interface"]
                required_params = interface_info.get("requires", []) # 获取接口需要的参数

                # 检查接口所需的参数是否提供
                missing_params = []
                if "entities" in required_params and not group.entities:
                    missing_params.append("entities")
                if "relationships" in required_params and not english_relationships:
                    missing_params.append("relationships")

                if missing_params:
                    logger.warning(f"接口 {interface_name} 需要参数 {missing_params}，但在组 {i} 中未提供。跳过。")
                    # log_to_file(f"Warning: Interface '{interface_name}' requires {missing_params} but they are missing in group {i}. Skipping.") # 移除旧的日志调用
                    continue

                # 调用Neo4j服务 - 使用 if/elif 映射
                entities_for_query = [{"name": e.name, "type": e.type} for e in group.entities]

                if interface_name == "find_connections_between_entities":
                    # 需要至少两个实体
                    if len(entities_for_query) >= 2:
                        result = neo4j_service.find_connections_between_entities(
                            entities=entities_for_query,
                            relationships=english_relationships
                        )
                        neo4j_results.extend(result)
                        logger.info(f"Neo4j query for intent '{intent}' (interface '{interface_name}') returned {len(result)} records.")
                        # log_to_file(f"Neo4j Query - Intent: '{intent}', Interface: '{interface_name}', Entities: {entities_for_query}, Relationships: {english_relationships}, Results Count: {len(result)}") # 移除旧的日志调用
                        log_neo4j_query(intent, interface_name, entities_for_query, english_relationships, len(result)) # 使用辅助函数记录到文件
                    else:
                        logger.warning(f"接口 {interface_name} 需要至少两个实体，但只有 {len(entities_for_query)} 个。")
                        # log_to_file(f"Warning: Interface '{interface_name}' requires at least 2 entities, but only {len(entities_for_query)} provided in group {i}.") # 移除旧的日志调用
                elif interface_name == "find_properties_of_entity":
                    result = neo4j_service.find_properties_of_entity(
                        entities=entities_for_query
                    )
                    neo4j_results.extend(result)
                    logger.info(f"Neo4j query for intent '{intent}' (interface '{interface_name}') returned {len(result)} records.")
                    # log_to_file(f"Neo4j Query - Intent: '{intent}', Interface: '{interface_name}', Entities: {entities_for_query}, Results Count: {len(result)}") # 移除旧的日志调用
                    log_neo4j_query(intent, interface_name, entities_for_query, [], len(result)) # 使用辅助函数记录到文件
                elif interface_name == "find_related_entities_by_relationship":
                    result = neo4j_service.find_related_entities_by_relationship(
                        entities=entities_for_query,
                        relationships=english_relationships
                    )
                    neo4j_results.extend(result)
                    logger.info(f"Neo4j query for intent '{intent}' (interface '{interface_name}') returned {len(result)} records.")
                    # log_to_file(f"Neo4j Query - Intent: '{intent}', Interface: '{interface_name}', Entities: {entities_for_query}, Relationships: {english_relationships}, Results Count: {len(result)}") # 移除旧的日志调用
                    log_neo4j_query(intent, interface_name, entities_for_query, english_relationships, len(result)) # 使用辅助函数记录到文件
                elif interface_name == "find_common_connections":
                    # 需要至少两个实体
                    if len(entities_for_query) >= 2:
                         result = neo4j_service.find_common_connections(
                             entities=entities_for_query,
                             relationships=english_relationships
                         )
                         neo4j_results.extend(result)
                         logger.info(f"Neo4j query for intent '{intent}' (interface '{interface_name}') returned {len(result)} records.")
                         # log_to_file(f"Neo4j Query - Intent: '{intent}', Interface: '{interface_name}', Entities: {entities_for_query}, Relationships: {english_relationships}, Results Count: {len(result)}") # 移除旧的日志调用
                         log_neo4j_query(intent, interface_name, entities_for_query, english_relationships, len(result)) # 使用辅助函数记录到文件
                    else:
                        logger.warning(f"接口 {interface_name} 需要至少两个实体，但只有 {len(entities_for_query)} 个。")
                        # log_to_file(f"Warning: Interface '{interface_name}' requires at least 2 entities, but only {len(entities_for_query)} provided in group {i}.") # 移除旧的日志调用
                elif interface_name == "query_entity_relationships":
                    # 这个接口只需要实体
                    for entity in entities_for_query:
                         result = neo4j_service.query_entity_relationships(
                             entities=[entity]
                         )
                         neo4j_results.extend(result)
                         logger.info(f"Neo4j query for intent '{intent}' (interface '{interface_name}') on entity '{entity['name']}' returned {len(result)} records.")
                         # log_to_file(f"Neo4j Query - Intent: '{intent}', Interface: '{interface_name}', Entity: {entity}, Results Count: {len(result)}") # 移除旧的日志调用
                         log_neo4j_query(intent, interface_name, [entity], [], len(result)) # 使用辅助函数记录到文件
                elif interface_name == "query_entity_connections":
                    for entity in entities_for_query:
                         result = neo4j_service.query_entity_connections(
                             entities=[entity]
                         )
                         neo4j_results.extend(result)
                         logger.info(f"Neo4j query for intent '{intent}' (interface '{interface_name}') on entity '{entity['name']}' returned {len(result)} records.")
                         # log_to_file(f"Neo4j Query - Intent: '{intent}', Interface: '{interface_name}', Entity: {entity}, Results Count: {len(result)}") # 移除旧的日志调用
                         log_neo4j_query(intent, interface_name, [entity], [], len(result)) # 使用辅助函数记录到文件
                # find_entities_by_property 被排除
                elif interface_name == "query_relationship_properties":
                    # 需要至少两个实体来定位关系
                    if len(entities_for_query) >= 2:
                        result = neo4j_service.query_relationship_properties(
                            entities=entities_for_query,
                            relationships=english_relationships
                        )
                        neo4j_results.extend(result)
                        logger.info(f"Neo4j query for intent '{intent}' (interface '{interface_name}') returned {len(result)} records.")
                        # log_to_file(f"Neo4j Query - Intent: '{intent}', Interface: '{interface_name}', Entities: {entities_for_query}, Relationships: {english_relationships}, Results Count: {len(result)}") # 移除旧的日志调用
                        log_neo4j_query(intent, interface_name, entities_for_query, english_relationships, len(result)) # 使用辅助函数记录到文件
                    else:
                        logger.warning(f"接口 {interface_name} 需要至少两个实体，但只有 {len(entities_for_query)} 个。")
                        # log_to_file(f"Warning: Interface '{interface_name}' requires at least 2 entities, but only {len(entities_for_query)} provided in group {i}.") # 移除旧的日志调用
                else:
                    logger.error(f"Neo4j服务中未找到接口: {interface_name}")
                    # log_to_file(f"Error: Interface '{interface_name}' not found in Neo4j service.") # 移除旧的日志调用

            # 处理 standalone_entities (如果需要)
            standalone_results = []
            if parse_result.standalone_entities:
                # 将standalone_entities的中文类型也映射为英文
                mapped_standalone_entities = []
                for entity in parse_result.standalone_entities:
                    english_type = entity_type_mapping.get(entity.type, entity.type)
                    mapped_standalone_entities.append(Entity(name=entity.name, type=english_type))
                parse_result.standalone_entities = mapped_standalone_entities # 更新

                standalone_entities_for_query = [{"name": e.name, "type": e.type} for e in parse_result.standalone_entities]
                # 示例：查询 standalone_entities 的属性
                for entity in standalone_entities_for_query:
                     rel_result = neo4j_service.find_properties_of_entity(entities=[entity])
                     standalone_results.extend(rel_result)
                logger.info(f"Neo4j query for standalone_entities returned {len(standalone_results)} records.")
                # log_to_file(f"Neo4j Query - Standalone Entities: {standalone_entities_for_query}, Results Count: {len(standalone_results)}") # 移除旧的日志调用
                log_neo4j_query("standalone_properties", "find_properties_of_entity", standalone_entities_for_query, [], len(standalone_results)) # 使用辅助函数记录到文件

            # 合并所有Neo4j查询结果
            all_context_data = neo4j_results + standalone_results
            # log_to_file(f"All Neo4j Results:\n{json.dumps(all_context_data, ensure_ascii=False, indent=2)}") # 移除旧的日志调用
            logger.debug(f"All Neo4j Results count: {len(all_context_data)}") # 控制台记录简略信息

            # --- Step 3: Generate Final Answer (LLM, Streamed) ---
            # 构造生成Prompt，并管理长度
            # --- 修改后的 Prompt 结构 ---
            context_str = json.dumps(all_context_data, ensure_ascii=False, indent=2)
            
            # 新的 Prompt 模板：将知识库信息放入 system prompt
            initial_system_prompt_part = f"""<|system|>
你是一个专业的医疗知识助手，基于提供的可能用上的知识库信息，能够详细地分析出用户的需求并回答用户的问题，态度认真亲切，十分关心用户健康，并且欢迎用户询问健康知识。
知识库信息:
{context_str}
</|system|>
"""
            initial_user_prompt_part = f"""<|user|>
请根据以上信息回答用户问题: {question}
</|user|>
<|assistant|>
"""
            
            # 分别计算 system 和 user 部分的 token 数
            initial_system_tokens = tokenizer.encode(initial_system_prompt_part, add_special_tokens=False)
            initial_user_tokens = tokenizer.encode(initial_user_prompt_part, add_special_tokens=False)
            initial_system_token_count = len(initial_system_tokens)
            initial_user_token_count = len(initial_user_tokens)
            initial_total_token_count = initial_system_token_count + initial_user_token_count

            # 检查总长度是否超过限制
            if initial_total_token_count > Config.MAX_PROMPT_TOKENS:
                logger.warning(f"Initial prompt tokens (System: {initial_system_token_count}, User: {initial_user_token_count}, Total: {initial_total_token_count}) exceed max allowed ({Config.MAX_PROMPT_TOKENS}). Truncating context.")
                
                # --- 修改后的截断逻辑 ---
                # 我们需要保留完整的 initial_user_prompt_part 和 <|assistant|> 标记
                # 因此，计算可用于 system prompt 中 context 部分的 token 数
                
                # 1. 计算固定的 system prompt 模板头和尾的 token 数
                fixed_system_header = """<|system|>
你是一个专业的医疗知识助手，基于提供的可能用上的知识库信息，能够详细地分析出用户的需求并回答用户的问题，态度认真亲切，十分关心用户健康，并且欢迎用户询问健康知识。
知识库信息:
"""
                fixed_system_footer = """
</|system|>
"""
                header_tokens = tokenizer.encode(fixed_system_header, add_special_tokens=False)
                footer_tokens = tokenizer.encode(fixed_system_footer, add_special_tokens=False)
                
                # 2. 计算 user prompt 和 assistant 标记的 token 数
                user_and_assistant_tokens = tokenizer.encode(f"""<|user|>
请根据以上信息回答用户问题: {question}
</|user|>
<|assistant|>
""", add_special_tokens=False) # 这包含了 user prompt 和 <|assistant|>
                
                # 3. 计算可用于 context_str 的 token 数
                available_context_tokens = Config.MAX_PROMPT_TOKENS - len(header_tokens) - len(footer_tokens) - len(user_and_assistant_tokens)

                if available_context_tokens <= 0:
                    logger.error("Available tokens for context is 0 or negative, cannot proceed after accounting for fixed parts.")
                    raise HTTPException(status_code=500, detail="上下文过长，无法处理，即使截断也无法容纳固定提示词。")

                # 4. 简单截断 context_str (可以优化为按句子或段落截断)
                # 这里使用 json 字符串进行截断，可能不够精确，更好的方式是先处理 all_context_data 列表
                context_lines = context_str.split('\n')
                current_context_tokens = 0
                final_context = []
                for line in context_lines:
                    line_tokens = tokenizer.encode(line, add_special_tokens=False)
                    if current_context_tokens + len(line_tokens) <= available_context_tokens:
                        final_context.append(line)
                        current_context_tokens += len(line_tokens)
                    else:
                        # 添加截断标记或直接停止 (这里选择停止)
                        break

                final_context_str = '\n'.join(final_context)
                
                # 5. 构造最终的截断后的 Prompt
                final_system_prompt_part = f"""<|system|>
你是一个专业的医疗知识助手，基于提供的可能用上的知识库信息，能够详细地分析出用户的需求并回答用户的问题，态度认真亲切，十分关心用户健康，并且欢迎用户询问健康知识。知识库作为你知识的补充，不能让用户知道知识库的存在。
知识库信息:
{final_context_str}
</|system|>
"""
                final_user_prompt_part = f"""<|user|>
请根据以上信息回答用户问题: {question}
</|user|>
<|assistant|>
"""
                final_prompt = final_system_prompt_part + final_user_prompt_part
                
                # 记录截断后的Prompt（用于日志）
                logger.debug(f"Second LLM Call - Prompt truncated. System part length: ~{len(tokenizer.encode(final_system_prompt_part, add_special_tokens=False))}, User part length: ~{len(tokenizer.encode(final_user_prompt_part, add_special_tokens=False))}")
                # 使用 log_service 记录到文件
                log_second_llm_call(final_prompt, "[Awaiting Output...]") # 在流开始前记录输入

            else:
                # 不需要截断
                final_prompt = initial_system_prompt_part + initial_user_prompt_part
                # 使用 log_service 记录到文件
                log_second_llm_call(final_prompt, "[Awaiting Output...]") # 在流开始前记录输入
                logger.debug(f"Second LLM Call - Prompt not truncated. Total length: ~{initial_total_token_count}")

            generate_messages = [{"role": "user", "content": final_prompt}]

            # 获取流式生成器
            stream_generator = await vllm_service.chat_completion(
                generate_messages,
                stream=True,
                temperature=Config.LLM_TEMPERATURE_GENERATE,
                max_tokens=Config.LLM_MAX_TOKENS_GENERATE
            )

            # 定义一个异步生成器来处理流式输出
            async def generate_stream():
                second_llm_output = "" # 用于累积并最终记录第二次调用的完整输出
                async for chunk in stream_generator: # chunk 可能是 str 或 ServerSentEvent 对象
                    # 检查 chunk 的类型
                    chunk_data_str = None # 初始化，用于存储待处理的 JSON 字符串
                    if hasattr(chunk, 'data'):
                        # 假设 chunk 是 ServerSentEvent 对象 (旧版 vLLM 可能)
                        chunk_data_str = chunk.data
                    elif isinstance(chunk, str):
                        # 假设 chunk 是字符串 (SSE 格式, 新版 vLLM 默认)
                        chunk_str = chunk
                        # --- 修改点 1: 正确识别 SSE data 前缀 ---
                        # SSE 消息通常以 "data: " 开头 (注意冒号后有空格)
                        DATA_PREFIX = "data: "
                        if chunk_str.startswith(DATA_PREFIX):
                            # --- 修改点 2: 正确提取 JSON 数据 ---
                            # 去掉 "data: " 前缀并去除首尾可能的空白字符
                            chunk_data_str = chunk_str[len(DATA_PREFIX):].strip()
                        else:
                            # 如果不是以 "data: " 开头，可能是其他SSE字段（如 event:, id:, retry:）或格式错误
                            # 这些是非数据块，记录为 DEBUG 级别更合适，避免刷屏 WARNING
                            logger.debug(f"Ignoring non-data SSE field or malformed chunk: {chunk_str[:100]}...")
                            # 跳过非数据块的处理
                            continue
                    else:
                        logger.error(f"Unexpected chunk type received from vLLM stream: {type(chunk)}")
                        # 对于未知类型，也选择跳过
                        continue

                    # --- 修改点 3: 确保只有在 chunk_data_str 被成功提取后才进行处理 ---
                    # 如果 chunk_data_str 仍未被赋值，则跳过
                    if not chunk_data_str:
                         logger.warning("chunk_data_str is empty or None after processing. Skipping.")
                         continue

                    # --- 修改点 4: 处理 "[DONE]" 和 JSON 数据 ---
                    if chunk_data_str.strip() == "[DONE]":
                        # [DONE] 标志，结束循环
                        logger.debug("Received [DONE] signal from vLLM stream.")
                        break
                    else:
                        try:
                            # 尝试解析 chunk_data_str 为 JSON
                            chunk_data = json.loads(chunk_data_str)
                            # 提取 delta content
                            choices = chunk_data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content: # 只发送有内容的部分
                                    second_llm_output += content # 累积输出内容
                                    logger.debug(f"Yielding content chunk: {content}")
                                    yield content.encode("utf-8") # FastAPI StreamingResponse 需要 bytes
                            # else:
                            #     # 可能是 keep-alive 或其他没有 content 的块，可以忽略或记录为 debug
                            #     logger.debug(f"Received SSE chunk with no 'content': {chunk_data}")
                        except json.JSONDecodeError as e:
                            # 如果不是JSON，可能是格式错误或其他问题，记录警告
                            logger.warning(f"Failed to parse chunk data as JSON: {chunk_data_str[:100]}... Error: {e}")
                            # 可以选择继续处理下一个块或中断
                            continue # 选择继续
                        except Exception as e:
                            # 捕获其他可能的意外错误
                            logger.error(f"Unexpected error while processing SSE chunk: {chunk_data_str[:100]}... Error: {e}", exc_info=True)
                            continue # 选择继续

                # 记录第二次调用的完整输出（在流结束时）
                # 使用 log_service 记录到文件
                # 注意：这里只记录输出，输入已在上面记录
                log_second_llm_call("[Input was logged previously]", second_llm_output)
                logger.info("Finished streaming response from second LLM call.")

            logger.info(f"Starting to stream response for question: {question}")
            return StreamingResponse(generate_stream(), media_type="text/plain")

        except Exception as e:
            logger.error(f"Error in RAG process: {str(e)}", exc_info=True)
            # 使用 log_service 记录错误到文件
            # 注意：这里记录的是整个 RAG 过程的错误，可能与第二次调用的具体输入输出不同
            # 如果需要更精确的错误日志，可以在更具体的 catch 块中记录
            from log_service import log_error_to_file
            log_error_to_file(f"RAG process error for question '{question}': {str(e)}", "RAG_Request")
            raise HTTPException(status_code=500, detail=f"RAG处理过程出错: {str(e)}")
        finally:
            # 确保在请求结束时记录日志
            log_rag_request_end() # 使用辅助函数记录到文件

    # --- 8. (可选) 原有的 OpenAI 兼容端点 ---
    openai_serving_completion = OpenAIServingCompletion(
        engine,
        model_config,
        vllm_service.openai_serving_models,
        request_logger=None
    )
    openai_serving_tokenization = OpenAIServingTokenization(
        engine,
        model_config,
        vllm_service.openai_serving_models,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto"
    )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        if request.stream:
            generator = await vllm_service.openai_serving_chat.create_chat_completion(request)
            return StreamingResponse(generator, media_type="text/event-stream")
        else:
            return await vllm_service.openai_serving_chat.create_chat_completion(request)

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        return await openai_serving_completion.create_completion(request)

    @app.post("/v1/tokenize")
    async def tokenize(request: TokenizeRequest):
        return await openai_serving_tokenization.create_tokenize(request)

    @app.post("/v1/detokenize")
    async def detokenize(request: DetokenizeRequest):
        return await openai_serving_tokenization.create_detokenize(request)

    @app.get("/v1/models")
    async def show_available_models():
        models = await vllm_service.openai_serving_chat.show_available_models()
        return models

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app, vllm_service, neo4j_service

# --- 9. 启动服务 ---
# 注意：启动逻辑将移到 start_MedicalQA.py
# if __name__ == "__main__":
#     try:
#         logger.info("正在启动 Medical RAG API 服务...")
#
#         # 创建应用和依赖实例
#         app, vllm_service, neo4j_service = asyncio.run(create_app())
#
#         # 定义关闭时的清理函数
#         async def shutdown_event():
#             logger.info("Shutting down services...")
#             neo4j_service.close()
#             logger.info("Neo4j connection closed.")
#
#         app.add_event_handler("shutdown", shutdown_event)
#
#         logger.info("服务启动成功!")
#         logger.info("RAG流式API地址: http://0.0.0.0:8001/v1/medical_rag_stream")
#         logger.info("原有OpenAI API地址: http://0.0.0.0:8001/v1/chat/completions")
#         logger.info("文档地址: http://0.0.0.0:8001/docs")
#
#         # 启动 Uvicorn 服务器
#         uvicorn.run(
#             app,
#             host="0.0.0.0",
#             port=8001, # 修改端口以区分
#             log_level="info", # 保持uvicorn日志级别以便调试
#             timeout_keep_alive=60
#         )
#
#     except Exception as e:
#         logger.error(f"启动服务时出错: {str(e)}", exc_info=True)
#         sys.exit(1)
