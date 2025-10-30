# rag_fastapi_service.py

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStreamResponse,
    DeltaMessage, ChatMessage, UsageInfo
)
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
import uuid # 用于生成默认的 request_id
from datetime import datetime
import time

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
    # 修改后的 medical_rag_stream，符合 OpenAI API 格式
    @app.post("/v1/medical_rag_stream", response_model=ChatCompletionResponse) # 使用标准响应模型
    async def medical_rag_stream(
        request: Request, # 用于读取 headers
        chat_request: ChatCompletionRequest # 使用标准的 ChatCompletionRequest
    ):
        """
        符合 OpenAI API 格式的医疗 RAG 流式聊天补全端点。
        从 HTTP Headers 读取会话ID和时间戳。
        """
        # --- 从 Headers 读取信息 ---
        # 使用 .get() 方法并提供默认值，以防 Header 不存在
        session_id = request.headers.get("X-Session-ID")
        timestamp_str = request.headers.get("X-Timestamp") # 通常为 Unix 时间戳字符串

        # --- 处理时间戳 ---
        request_timestamp = None
        if timestamp_str:
            try:
                # 假设前端发送的是 Unix 时间戳（秒）
                request_timestamp = float(timestamp_str)
            except ValueError:
                logger.warning(f"Invalid timestamp format in header X-Timestamp: {timestamp_str}")
                # 可以选择忽略无效时间戳或使用当前时间
                request_timestamp = time.time()
        else:
            # 如果没有提供时间戳，使用当前时间
            request_timestamp = time.time()

        # --- 从 ChatCompletionRequest 提取信息 ---
        # 检查是否有消息
        if not chat_request.messages:
            logger.error("No messages provided in the request.")
            raise HTTPException(status_code=400, detail="Messages are required.")

        # 获取最后一个用户消息作为当前问题
        # 注意：标准 OpenAI API 允许 roles 为 "system", "user", "assistant"
        # 我们通常处理最后一个 "user" 消息
        latest_user_message = None
        for message in reversed(chat_request.messages):
            if message["role"] == "user":
                latest_user_message = message
                break

        if not latest_user_message:
            logger.error("No 'user' message found in the request.")
            raise HTTPException(status_code=400, detail="No user message found.")

        question = latest_user_message["content"]
        # model_name = chat_request.model # 可以用来验证请求的模型是否匹配

        # --- 生成或使用请求ID ---
        # OpenAI API 响应中通常包含一个 id 字段
        # 如果请求中没有提供 (vLLM 的 ChatCompletionRequest 可能没有 id 字段，或者我们想用自己的),
        # 我们可以生成一个
        request_id = f"chatcmpl-{uuid.uuid4().hex}" # 例如: chatcmpl-a1b2c3d4e5f6

        logger.info(f"[Session: {session_id}] [TS: {request_timestamp}] Received RAG request (ID: {request_id}) for model '{chat_request.model}', question: {question}")
        # 使用 log_service 记录到文件 (如果需要区分来源，可以添加标记)
        log_rag_request_start(question, session_id) # 假设 log_service 支持 session_id

        try:
            # --- Step 1: Entity/Intent Parsing (LLM) ---
            # 这里复用您原有的 RAG 逻辑，将 `question` 传递进去
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
1.  分析用户的问题，提取出关键的实体。对于每个识别出的实体，请在提供的标准实体列表中寻找最匹配的名称。如果列表中没有合适的名称，生成一个准确的专业医学术语。
2.  从提供的标准关系列表中，选择最能描述问题中实体间联系的关系名称。
3.  从提供的标准意图列表中，选择最能概括用户查询目的的意图名称。
4.  将提取出的实体、关系与意图进行关联，形成一个或多个意图-实体-关系组。允许一个实体被多个意图组使用。
5.  识别整体查询意图。
6.  将未被明确分组到任何意图中的实体放入独立实体列表。

请严格按照以下JSON格式输出，不输出其他内容：

输出格式为:
{{
  "entity_relationship_groups": [
    {{
      "entities": [{{"name": "标准实体名或生成的专业术语", "type": "疾病/症状/药物/部门/食物/检查/厂商" # 使用entity.json中的中文类型名}}, ...],
      "relationships": ["标准关系名"], # 只能从关系表中选择，可以为空列表 []
      "intent": "标准意图名" # 只能从意图列表中选择，不可为空
    }}
  ],
  "standalone_entities": [{{"name": "实体名", "type": "中文类型名"}}],
  "overall_intent": "整体意图" # 只能从意图列表中选择，可以为空
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
            logger.debug(f"First LLM Call - Input Prompt:\n{parse_prompt}")

            parse_response = await vllm_service.chat_completion(
                parse_messages,
                stream=False,
                temperature=Config.LLM_TEMPERATURE_PARSE,
                max_tokens=Config.LLM_MAX_TOKENS_PARSE
            )
            logger.debug(f"Parse response: {parse_response}")

            # 记录第一次调用的完整输出
            first_llm_output = parse_response.choices[0].message.content
            logger.debug(f"First LLM Call - Output:\n{first_llm_output}")

            # 假设模型返回的是JSON格式的字符串
            parse_text = parse_response.choices[0].message.content
            try:
                parse_data = json.loads(parse_text)
                parse_result = RAGParseResult(
                    entity_relationship_groups=parse_data.get("entity_relationship_groups", []),
                    standalone_entities=parse_data.get("standalone_entities", []),
                    overall_intent=parse_data.get("overall_intent", "")
                )
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.debug(f"Error: Failed to parse first LLM output as JSON: {e}")
                raise HTTPException(status_code=500, detail="解析模型响应失败")

            logger.info(f"Parsed result: {parse_result.dict()}")
            logger.debug(f"Parsed Result:\n{json.dumps(parse_result.dict(), ensure_ascii=False, indent=2)}")

            # --- Step 2: Map Chinese Entity Types to English & Map Relationships & Query Neo4j (Service-side) ---
            # 遍历 entity_relationship_groups 并调用相应接口
            neo4j_results = []
            for i, group in enumerate(parse_result.entity_relationship_groups):
                # 将中文实体类型映射为英文类型
                mapped_entities = []
                for entity in group.entities:
                    english_type = entity_type_mapping.get(entity.type, entity.type) # 如果找不到映射，保留原名
                    mapped_entities.append(Entity(name=entity.name, type=english_type))
                group.entities = mapped_entities # 更新group中的实体列表

                # 检查关系和意图是否为空
                if not group.relationships:
                    logger.warning(f"Group {i} has no relationships: {group.dict()}")
                    logger.debug(f"Warning: Group {i} has no relationships.")
                    # 可以选择跳过此组，或根据意图使用默认关系
                    # 这里选择跳过
                    continue
                if not group.intent:
                    logger.warning(f"Group {i} has no intent: {group.dict()}")
                    logger.debug(f"Warning: Group {i} has no intent.")
                    # 可以选择跳过此组，或尝试推断意图
                    # 这里选择跳过
                    continue

                # 将中文关系名映射为英文关系类型
                english_relationships = [rel_mapping.get(rel, rel) for rel in group.relationships] # 如果找不到映射，保留原名

                # 根据意图查找映射的接口
                intent = group.intent
                if intent not in intent_mapping_data:
                    logger.warning(f"未知意图: {intent}")
                    logger.debug(f"Warning: Unknown intent '{intent}' in group {i}.")
                    continue

                interface_info = intent_mapping_data[intent]
                interface_name = interface_info["neo4j_interface"]

                # 调用Neo4j服务 - 使用 if/elif 映射
                # 注意：这里需要根据接口定义的参数来传递
                # 通常需要 entities 和 relationships
                # 某些接口可能还需要 direction 等其他参数
                # 这里假设所有接口都接受 entities 和 relationships (映射后)
                entities_for_query = [{"name": e.name, "type": e.type} for e in group.entities]

                if interface_name == "find_connections_between_entities":
                    # 需要至少两个实体
                    if len(entities_for_query) >= 2:
                        result = neo4j_service.find_connections_between_entities(
                            entities=entities_for_query,
                            relationships=english_relationships
                        )
                        neo4j_results.extend(result)
                    else:
                        logger.warning(f"接口 {interface_name} 需要至少两个实体，但只有 {len(entities_for_query)} 个。")
                elif interface_name == "find_properties_of_entity":
                    result = neo4j_service.find_properties_of_entity(
                        entities=entities_for_query
                    )
                    neo4j_results.extend(result)
                elif interface_name == "find_related_entities_by_relationship":
                    result = neo4j_service.find_related_entities_by_relationship(
                        entities=entities_for_query,
                        relationships=english_relationships
                    )
                    neo4j_results.extend(result)
                elif interface_name == "find_common_connections":
                    # 需要至少两个实体
                    if len(entities_for_query) >= 2:
                         result = neo4j_service.find_common_connections(
                             entities=entities_for_query,
                             relationships=english_relationships
                         )
                         neo4j_results.extend(result)
                    else:
                        logger.warning(f"接口 {interface_name} 需要至少两个实体，但只有 {len(entities_for_query)} 个。")
                elif interface_name == "query_entity_relationships":
                    # 这个接口只需要实体
                    for entity in entities_for_query:
                         result = neo4j_service.query_entity_relationships(
                             entities=[entity]
                         )
                         neo4j_results.extend(result)
                elif interface_name == "query_entity_connections":
                    for entity in entities_for_query:
                         result = neo4j_service.query_entity_connections(
                             entities=[entity]
                         )
                         neo4j_results.extend(result)
                # find_entities_by_property 被排除
                # elif interface_name == "find_entities_by_property":
                #     # 需要 property_name 和 property_value，无法从当前解析结果直接获得
                #     logger.warning(f"接口 {interface_name} 需要特殊参数处理，暂时跳过。")
                #     continue
                elif interface_name == "query_relationship_properties":
                    # 需要至少两个实体来定位关系
                    if len(entities_for_query) >= 2:
                        result = neo4j_service.query_relationship_properties(
                            entities=entities_for_query,
                            relationships=english_relationships
                        )
                        neo4j_results.extend(result)
                    else:
                        logger.warning(f"接口 {interface_name} 需要至少两个实体，但只有 {len(entities_for_query)} 个。")
                else:
                    logger.error(f"Neo4j服务中未找到接口: {interface_name}")
                    logger.debug(f"Error: Interface '{interface_name}' not found in Neo4j service.")
                    continue

                logger.info(f"Neo4j query for intent '{intent}' (interface '{interface_name}') returned {len(result)} records.")
                logger.debug(f"Neo4j Query - Intent: '{intent}', Interface: '{interface_name}', Entities: {entities_for_query}, Relationships: {english_relationships}, Results Count: {len(result)}")

            # 处理 standalone_entities (如果需要)
            standalone_results = []
            if parse_result.standalone_entities:
                standalone_entities_for_query = [{"name": e.name, "type": e.type} for e in parse_result.standalone_entities]
                # 示例：查询 standalone_entities 的属性
                for entity in standalone_entities_for_query:
                     rel_result = neo4j_service.find_properties_of_entity(entities=[entity])
                     standalone_results.extend(rel_result)
                logger.info(f"Neo4j query for standalone_entities returned {len(standalone_results)} records.")
                logger.debug(f"Neo4j Query - Standalone Entities: {standalone_entities_for_query}, Results Count: {len(standalone_results)}")

            # 合并所有Neo4j查询结果
            all_context_data = neo4j_results + standalone_results
            logger.debug(f"All Neo4j Results:\n{json.dumps(all_context_data, ensure_ascii=False, indent=2)}")

            # --- Step 3: Generate Final Answer (LLM, Streamed) ---
            # 构造生成Prompt，并管理长度
            context_str = json.dumps(all_context_data, ensure_ascii=False, indent=2)
            initial_prompt_part = f"""<|system|>
你是一个专业的医疗知识助手，基于提供的可能用上的知识库信息，能够详细地分析出用户的需求并回答用户的问题，态度认真亲切，十分关心用户健康，并且欢迎用户询问健康知识。
知识库信息:
{context_str}
</|system|>
<|user|>
根据以上信息回答用户问题: {question}
</|user|>
<|assistant|>
"""
            initial_tokens = tokenizer.encode(initial_prompt_part, add_special_tokens=False)
            initial_token_count = len(initial_tokens)

            # 检查是否超过限制
            if initial_token_count > Config.MAX_PROMPT_TOKENS:
                logger.warning(f"Initial prompt tokens ({initial_token_count}) exceed max allowed ({Config.MAX_PROMPT_TOKENS}). Truncating context.")
                logger.debug(f"Warning: Initial prompt tokens ({initial_token_count}) exceed limit ({Config.MAX_PROMPT_TOKENS}). Truncating context.")
                # 需要截断context部分
                prompt_template_tokens = tokenizer.encode(f"""<|system|>
你是一个专业的医疗知识助手，基于提供的可能用上的知识库信息，能够详细地分析出用户的需求并回答用户的问题，态度认真亲切，十分关心用户健康，并且欢迎用户询问健康知识。
知识库信息:
""", add_special_tokens=False)
                question_tokens = tokenizer.encode(f"</|system|>\n<|user|>\n根据以上信息回答用户问题: {question}\n</|user|>\n<|assistant|>\n", add_special_tokens=False)
                available_context_tokens = Config.MAX_PROMPT_TOKENS - len(prompt_template_tokens) - len(question_tokens)

                if available_context_tokens <= 0:
                    logger.error("Available tokens for context is 0 or negative, cannot proceed.")
                    logger.debug(f"Error: Available tokens for context is 0 or negative.")
                    raise HTTPException(status_code=500, detail="上下文过长，无法处理")

                # 简单截断context (可以优化为按句子或段落截断)
                context_lines = context_str.split('\n')
                current_context_tokens = 0
                final_context = []
                for line in context_lines:
                    line_tokens = tokenizer.encode(line, add_special_tokens=False)
                    if current_context_tokens + len(line_tokens) <= available_context_tokens:
                        final_context.append(line)
                        current_context_tokens += len(line_tokens)
                    else:
                        # 添加截断标记或直接停止
                        break

                final_context_str = '\n'.join(final_context)
                final_prompt = f"""<|system|>
你是一个专业的医疗知识助手，基于提供的可能用上的知识库信息，能够详细地分析出用户的需求并回答用户的问题，态度认真亲切，十分关心用户健康，并且欢迎用户询问健康知识。
知识库信息:
{final_context_str}
</|system|>
<|user|>
根据以上信息回答用户问题: {question}
</|user|>
<|assistant|>
"""
            else:
                # 不需要截断
                final_prompt = initial_prompt_part

            # --- 构造传递给 vLLM 的消息 ---
            # 使用最终的 Prompt 替换原始的 messages
            # 注意：这里我们用最终构造的 Prompt 作为唯一的 user 消息传递给 vLLM
            # 如果需要保留原始对话历史用于上下文（非 RAG 检索到的信息），逻辑会更复杂
            # 这里简化为只使用 RAG 检索到的信息和当前问题
            generate_messages = [{"role": "user", "content": final_prompt}]

            logger.debug("第二次调用模型的输入："+json.dumps(generate_messages))

            # --- 调用 vLLM 生成 ---
            # 使用 chat_request 中的参数（如 temperature, max_tokens, stream）
            stream_generator = await vllm_service.chat_completion(
                generate_messages, # 使用我们构造的消息
                stream=chat_request.stream, # 使用请求中的 stream 参数
                temperature=chat_request.temperature if chat_request.temperature is not None else 0.1,
                max_tokens=chat_request.max_tokens if chat_request.max_tokens is not None else Config.LLM_MAX_TOKENS_GENERATE
            )

            # --- 构造 OpenAI 格式的响应 ---
            if chat_request.stream:
                # --- 流式响应 ---
                logger.info(f"[Session: {session_id}] Starting to stream response for question (ID: {request_id}): {question}")
                
                # --- 修改点：定义一个新的内部函数来处理流式逻辑 ---
                async def generate_openai_stream():
                    """生成符合 OpenAI SSE 格式的流式响应"""
                    created_time = int(time.time())
                    choice_index = 0
    
                    try:
                        # --- 修改点：健壮地处理来自 vLLM 的流式响应块 (chunk) ---
                        async for chunk in stream_generator: # chunk 来自 await vllm_service.chat_completion(..., stream=True)
                            
                            # 1. 首先，明确检查 chunk 是否为字符串 (这是 vLLM 0.10.1+ 常见的流式输出格式)
                            if isinstance(chunk, str):
                                # 2. 检查字符串是否以 "data: " 开头 (标准 SSE 格式)
                                if chunk.startswith("data: "):
                                    # 3. 提取 JSON 数据部分 (移除 "data: " 前缀和末尾的 \n\n)
                                    json_data_str = chunk[6:].strip() # chunk[6:] removes "data: "
                        
                                    # 4. 检查是否为 [DONE] 信号
                                    if json_data_str == "[DONE]":
                                        
                                        logger.debug(f"[Session: {session_id}] [Req ID: {request_id}] Sending [DONE] signal.")
                                        yield " [DONE]\n\n"
                                        break # 退出循环
                                    
                                    # 5. 处理包含实际内容的 JSON 数据块
                                    else:
                                        try:
                                            # 6. 解析 JSON 字符串
                                            chunk_data = json.loads(json_data_str)
                    
                                            # 7. 提取 delta 内容和 finish_reason (与之前逻辑一致)
                                            choices = chunk_data.get("choices", [])
                                            if choices:
                                                delta = choices[0].get("delta", {})
                                                content = delta.get("content", "")
                                                finish_reason = choices[0].get("finish_reason", None)
                    
                                                # 8. 构造并发送符合 OpenAI 格式的 SSE 响应块
                                                if content or finish_reason: # 只有有内容或结束原因时才发送
                                                    delta_message_dict = {
                                                        "content": content,
                                                        "role": "assistant"
                                                    }
                                                    choice_dict = {
                                                        "index": 0, # 通常流式响应 index 为 0
                                                        "delta": delta_message_dict,
                                                        "finish_reason": finish_reason # 通常为 None 直到最后
                                                    }
                                                    # 使用请求中传入的 model 名称 或 vLLM 返回的
                                                    stream_response_dict = {
                                                        "id": chunk_data.get("id", request_id), # 优先使用 vLLM 的 ID，备选 request_id
                                                        "object": "chat.completion.chunk",
                                                        "created": chunk_data.get("created", created_time), # 优先使用 vLLM 的 created 时间
                                                        "model": chunk_data.get("model", chat_request.model), # 优先使用 vLLM 的 model 名
                                                        "choices": [choice_dict],
                                                        # "usage": None # 流式响应通常不包含 usage
                                                    }
                    
                                                    # 9. 通过 yield 发送格式化的 SSE 响应
                                                    yield f" {json.dumps(stream_response_dict, separators=(',', ':'))}\n\n"
                                                # else:
                                                #     # 可选：处理空内容块（例如 keep-alive）
                                                #     pass
                    
                                        except json.JSONDecodeError as e:
                                            # 10. 如果 JSON 解析失败，记录警告并跳过该块
                                            logger.warning(f"[Session: {session_id}] [Req ID: {request_id}] Failed to decode JSON chunk: {e}. Raw chunk: {chunk[:200]}...")
                                            continue # 跳过有问题的块
                                        
                                else:
                                    # 11. 如果字符串不以 "data: " 开头，可能是其他 SSE 字段 (如 event:, id:, retry:) 或格式错误
                                    # 记录为 DEBUG 级别，因为这很常见且通常可以安全忽略
                                    logger.debug(f"[Session: {session_id}] [Req ID: {request_id}] Ignoring non-data SSE line or malformed chunk: {chunk[:100]}...")
                                    continue # 跳过非 data 行
                                
                            else:
                                # 12. 如果 chunk 不是字符串 (理论上 stream_generator 应该只产生 str，但这作为兜底检查)
                                logger.warning(f"[Session: {session_id}] [Req ID: {request_id}] Received unexpected chunk type (not str): {type(chunk)}. Value preview: {repr(chunk)[:100]}...")
                                continue # 跳过未知类型
                            
                    except Exception as e: # 捕获生成器内部的任何未处理异常
                        logger.error(f"[Session: {session_id}] [Req ID: {request_id}] Error inside stream generator: {e}", exc_info=True)
                        # 可以选择在这里 yield 一个错误信息给客户端，但这比较复杂且非标准
                        # 通常让异常传播出去，由 FastAPI 的错误处理机制处理
                        # 注意：一旦开始发送流，再抛出 HTTPException 可能导致客户端收到不完整的响应
                        raise # 重新抛出异常，让外层捕获
                    
                    finally:
                        # --- 在这里记录流式响应结束的日志 ---
                        # 这是确保即使在循环中 break 或发生异常也能执行的地方
                        logger.debug(f"[Session: {session_id}] [Req ID: {request_id}] Streaming response ended.")
                        log_second_llm_call("[Input was logged previously]", "[Streamed output - see client logs]")
                        log_rag_request_end() # 记录请求结束 via log_service
    
                # --- 修改点：在 if block 末尾调用 generate_openai_stream 并返回 StreamingResponse ---
                # 注意：这里调用函数名 `generate_openai_stream`，它返回一个异步生成器对象
                # StreamingResponse 会消费这个生成器
                return StreamingResponse(generate_openai_stream(), media_type="text/event-stream")
                # --- 修改点结束 ---

            else:
                # --- 非流式响应 ---
                # 注意：当前 RAG 流程设计为流式，非流式可能需要调整
                # 这里假设 stream_generator 在非流式下返回完整响应（虽然通常不是）
                # 更常见的是，vLLM 的非流式调用直接返回 ChatCompletionResponse
                # 但我们为了复用现有逻辑，假设它返回一个包含完整内容的 "chunk"
                logger.warning("Non-streaming mode requested for RAG endpoint, which is primarily designed for streaming. Attempting to adapt...")
                full_response_text = ""
                created_time = int(time.time())
                finish_reason = "stop" # Assume normal stop
                choice_index = 0


                # --- 修改点：正确收集非流式响应内容 ---
                collected_content = []
                async for chunk in stream_generator: # Iterate through the stream even if not streaming to client
                     if hasattr(chunk, 'data') and chunk.data != "[DONE]":
                         try:
                             chunk_data_str = chunk.data
                             chunk_data = json.loads(chunk_data_str)
                             choices = chunk_data.get("choices", [])
                             if choices:
                                 delta = choices[0].get("delta", {})
                                 content = delta.get("content", "")
                                 if content:
                                     collected_content.append(content)
                                 # Check for finish reason in non-streaming context (less common here, but possible)
                                 fr = choices[0].get("finish_reason")
                                 if fr:
                                     finish_reason = fr
                         except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
                             logger.error(f"[Session: {session_id}] [Req ID: {request_id}] Error processing non-stream chunk for collection: {e}")
                             continue
                     elif hasattr(chunk, 'data') and chunk.data == "[DONE]":
                         # End of internal stream
                         break
                     else:
                         logger.warning(f"[Session: {session_id}] [Req ID: {request_id}] Unexpected chunk in non-stream collection: {type(chunk)}")

                full_response_text = "".join(collected_content)


                # --- 修改点：使用 vLLM 协议中的 ChatCompletionResponse ---
                # 构造标准的 ChatCompletionResponse
                # message = ChatMessage(role="assistant", content=full_response_text) # Use dict instead
                message_dict = {
                     "role": "assistant",
                     "content": full_response_text
                     # "tool_calls": None, "function_call": None etc. if applicable
                }
                # choice = CompletionChoice(index=choice_index, message=message, finish_reason=finish_reason) # Use dict
                choice_dict = {
                    "index": choice_index,
                    "message": message_dict,
                    "finish_reason": finish_reason
                }

                # Note: Accurate token usage requires vLLM to provide it. vLLM's non-streaming API usually includes this.
                # We attempt to extract it if present in the last chunk or assume it's handled internally by vLLM
                # For now, we leave usage as None or attempt basic estimation (not recommended)
                # A better approach is to let vLLM calculate it if it can and include it if present in stream chunks

                # Create the response dictionary manually to match ChatCompletionResponse structure
                response_dict = {
                    "id": request_id, # Use our generated ID
                    "object": "chat.completion",
                    "created": created_time,
                    "model": chat_request.model, # Use model from request
                    "choices": [choice_dict],
                    # "usage": usage_dict_or_none # Add if you can reliably get it
                    "usage": None # Placeholder, vLLM might populate this differently or require specific config
                }

                logger.info(f"[Session: {session_id}] Generated non-streaming response (ID: {request_id}) for question: {question[:50]}...")
                # Log full output via log_service
                log_second_llm_call("[Input was logged previously]", full_response_text)
                log_rag_request_end() # 记录请求结束 via log_service
                # Return the constructed dictionary. FastAPI will serialize it based on the response_model (ChatCompletionResponse)
                # Returning a dict is generally fine if it matches the Pydantic model structure
                return response_dict # Let FastAPI handle serialization

        except Exception as e:
            logger.error(f"[Session: {session_id}] Error in RAG process (ID: {request_id}): {str(e)}", exc_info=True)
            logger.debug(f"Error in RAG process (ID: {request_id}): {str(e)}")
            log_rag_request_end() # 确保即使出错也记录结束
            raise HTTPException(status_code=500, detail=f"RAG处理过程出错: {str(e)}")

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
