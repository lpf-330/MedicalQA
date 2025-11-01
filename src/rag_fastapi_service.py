# rag_fastapi_service.py

import os
import sys
import json
import logging
import asyncio
import uuid
from datetime import datetime
from typing import AsyncGenerator, Dict, Any, List, Optional, Tuple
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import pydantic
from pydantic import BaseModel, Field
from transformers import AutoTokenizer # 用于计算token长度

# --- 1. 导入 vLLM OpenAI 兼容服务 ---
# 注意：根据 vLLM 0.10.1 的结构，需要正确导入
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    DeltaMessage,
    ChatCompletionResponseStreamChoice,
    ChatMessage,
    UsageInfo,
    ErrorResponse
)
from vllm.entrypoints.openai.serving_models import OpenAIServingModels, BaseModelPath
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_tokenization import OpenAIServingTokenization

# --- 2. 导入自定义模块 ---
from neo4j_service import Neo4jService # 导入Neo4j服务类
from log_service import (
    get_logger,
    log_rag_request_start,
    log_rag_request_end,
    log_first_llm_call,
    log_neo4j_query,
    log_second_llm_call,
    log_error_to_file
) # 导入日志服务

import time

# --- 3. 配置类 (集中管理所有配置) ---
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
        gpu_memory_utilization=0.4,
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

# --- 4. 加载映射表 ---
def load_mapping_tables():
    """加载实体表、实体分类表、关系表和意图映射表"""
    logger = get_logger(__name__)
    script_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前脚本文件的绝对路径

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

# --- 5. 主 FastAPI 应用 ---
# 使用 lifespan 事件处理器来管理应用生命周期
@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用启动和关闭时的资源"""
    logger = get_logger(__name__)
    logger.info("正在初始化模型引擎...")
    engine = AsyncLLMEngine.from_engine_args(Config.ENGINE_ARGS)

    # 获取模型配置
    async def get_model_config():
        return await engine.get_model_config()
    model_config = await get_model_config()

    # 创建 BaseModelPath
    base_model_paths = [BaseModelPath(name=Config.SERVED_MODEL_NAME, model_path=Config.MODEL_PATH)]

    # 创建 VLLM 服务实例
    global vllm_service # 声明为全局变量以便在路由中访问
    vllm_service = OpenAIServingChat(
        engine,
        model_config,
        OpenAIServingModels(
            engine_client=engine,
            model_config=model_config,
            base_model_paths=base_model_paths
        ),
        response_role="assistant",
        chat_template=None,
        request_logger=None,
        chat_template_content_format="auto"
    )

    # 加载映射表
    logger.info("正在加载映射表...")
    global entity_data, entity_type_mapping, relationship_data, intent_mapping_data, rel_mapping
    entity_data, entity_type_mapping, relationship_data, intent_mapping_data, rel_mapping = load_mapping_tables()

    # 创建 Neo4j 服务实例
    logger.info(f"正在连接Neo4j数据库: {Config.NEO4J_URI}")
    global neo4j_service # 声明为全局变量
    neo4j_service = Neo4jService(Config.NEO4J_URI, Config.NEO4J_USER, Config.NEO4J_PASSWORD)

    # 加载分词器用于计算token
    logger.info(f"正在加载分词器: {Config.MODEL_PATH}")
    global tokenizer # 声明为全局变量
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)

    logger.info("服务启动成功!")
    logger.info("RAG流式API地址: http://0.0.0.0:8001/v1/medical_rag_stream")
    logger.info("原有OpenAI API地址: http://0.0.0.0:8001/v1/chat/completions")
    logger.info("文档地址: http://0.0.0.0:8001/docs")

    # --- 应用启动完成 ---
    yield # 应用在此处运行

    # --- 应用关闭时的清理 ---
    logger.info("正在关闭服务...")
    neo4j_service.close()
    logger.info("Neo4j连接已关闭。")


# 创建 FastAPI 应用实例，传入 lifespan 处理器
app = FastAPI(
    title="Medical RAG API Server",
    description="API server for medical RAG using vLLM and Neo4j, compatible with OpenAI API",
    version="0.1.0",
    lifespan=lifespan # 注册 lifespan 处理器
)

# --- 6. 工具函数 ---

def truncate_messages_for_parse(messages: List[Dict[str, str]], tokenizer, max_tokens: int) -> List[Dict[str, str]]:
    """
    为第一次解析LLM调用截断消息历史，优先保留最新的消息。
    """
    if not messages:
        return messages

    # 计算所有消息的总token数
    total_tokens = 0
    message_tokens = []
    for msg in messages:
        tokens = tokenizer.encode(msg["content"], add_special_tokens=False)
        message_tokens.append(len(tokens))
        total_tokens += len(tokens)

    # 如果总token数未超限，直接返回
    if total_tokens <= max_tokens:
        return messages

    # 如果超限，从最早的非system消息开始裁剪
    # （假设system消息很重要，尽量保留）
    truncated_messages = []
    current_tokens = 0
    system_messages = [msg for msg in messages if msg["role"] == "system"]
    other_messages = [msg for msg in messages if msg["role"] != "system"]

    # 先加上所有system消息的token数
    for msg in system_messages:
        tokens = tokenizer.encode(msg["content"], add_special_tokens=False)
        current_tokens += len(tokens)
    truncated_messages.extend(system_messages)

    # 从最新的other消息开始添加，直到达到token限制
    for msg in reversed(other_messages):
        tokens = tokenizer.encode(msg["content"], add_special_tokens=False)
        if current_tokens + len(tokens) <= max_tokens:
            # 在列表开头插入，以保持时间顺序（但相对于原始顺序是倒序的）
            # 更好的做法是维护一个临时列表，最后反转或按正确顺序插入
            # 这里简单处理：在 truncated_messages 末尾追加，最后反转非system部分
            truncated_messages.append(msg)
            current_tokens += len(tokens)
        else:
            # 如果加上这条消息就超了，就不再添加了
            break
    
    # 重新排列非system消息的顺序，使其符合时间顺序
    final_other_messages = [msg for msg in truncated_messages if msg["role"] != "system"]
    final_other_messages.reverse() # 因为我们是从后往前加的
    
    # 重新组合：system消息 + 按时间顺序排列的其他消息
    final_messages = system_messages + [msg for msg in messages if msg["role"] != "system" and msg in final_other_messages]
    
    # 再次检查总长度，如果还是超了（例如system消息本身很长），则需要进一步裁剪system消息或第一条非system消息
    # 这里做一个简化的最终检查和调整
    final_total_tokens = sum(len(tokenizer.encode(msg["content"], add_special_tokens=False)) for msg in final_messages)
    if final_total_tokens > max_tokens:
        # 简单地从第一条非system消息开始裁剪内容
        if len(final_messages) > len(system_messages):
            first_non_system_idx = len(system_messages)
            content_to_truncate = final_messages[first_non_system_idx]["content"]
            # 简单按字符截断，更精确的方法是按token截断
            available_tokens_for_first_msg = max_tokens - sum(len(tokenizer.encode(m["content"], add_special_tokens=False)) for m in final_messages[:first_non_system_idx]) - sum(len(tokenizer.encode(m["content"], add_special_tokens=False)) for m in final_messages[first_non_system_idx+1:])
            if available_tokens_for_first_msg > 0:
                # 粗略估计需要保留的字符数 (假设平均token长度)
                avg_token_len = len(content_to_truncate) / max(1, len(tokenizer.encode(content_to_truncate, add_special_tokens=False)))
                chars_to_keep = int(available_tokens_for_first_msg * avg_token_len * 0.8) # 留点余量
                if chars_to_keep < len(content_to_truncate):
                    final_messages[first_non_system_idx]["content"] = content_to_truncate[:chars_to_keep] + "...[内容已截断]"
            else:
                 # 如果连第一条非system消息都放不下，就只保留system消息
                 final_messages = system_messages

    logger = get_logger(__name__)
    logger.info(f"为解析截断消息历史，原始token数: {total_tokens}, 截断后token数: {sum(len(tokenizer.encode(m['content'], add_special_tokens=False)) for m in final_messages)}")
    return final_messages


async def parse_user_query_with_context(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    第一次LLM调用：结合上下文解析用户问题，提取实体、关系、意图。
    """
    logger = get_logger(__name__)
    
    # --- Step 1: 构建解析Prompt (使用标准 messages 数组) ---
    # 构建解析Prompt，包含实体表、关系表、意图列表
    entities_str = json.dumps(entity_data, ensure_ascii=False, indent=2)
    relationships_str = json.dumps(relationship_data, ensure_ascii=False, indent=2)
    intents_str = json.dumps(list(intent_mapping_data.keys()), ensure_ascii=False, indent=2) # 提取意图名称列表

    # 截断 messages 以适应解析模型的 token 限制
    # 为解析留出一些空间
    parse_max_tokens = Config.MAX_PROMPT_TOKENS - 1024 # 预留1024 tokens给Prompt模板和输出
    truncated_messages = truncate_messages_for_parse(messages, tokenizer, parse_max_tokens)
    
    # 将截断后的对话历史序列化为字符串，作为解析的一部分上下文
    # history_context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in truncated_messages])

    # 使用标准的 messages 格式构建解析请求
    parse_system_prompt = f"""你是一个专业的医疗助手。你的任务是分析用户提供的一段对话历史，并执行以下步骤：

1.理解上下文：仔细阅读对话历史，理解用户当前问题的背景和之前的交流内容。
2.提取实体：识别用户当前问题（对话历史的最后一部分）中属于以下类型的实体（如疾病、症状、药物等）。对于每个识别出的实体，请在提供的标准实体列表中寻找最匹配的名称。如果列表中没有完全匹配的名称，请生成一个准确的专业医学术语。
3.提取关系：从提供的标准关系列表中，选择最能描述问题中实体间联系的关系名称。
4.提取意图：从提供的标准意图列表中，选择最能概括用户查询目的的意图名称。
5.分组：将提取出的实体、关系与意图进行关联，形成一个或多个意图-实体-关系组。允许一个实体被多个意图组使用。如果某个意图不需要特定关系（例如，仅查询实体属性），则其关系列表可为空。
6.识别整体意图：概括整个对话历史的主要查询意图。
7.识别独立实体：将未被明确分组到任何意图中的实体放入独立实体列表。

请严格按照以下JSON格式输出，不要输出其他内容：

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
  "overall_intent": "整体意图"
}}
---
标准实体列表:
{entities_str}

标准关系列表:
{relationships_str}

标准意图列表:
{intents_str}
"""

    # 构造传递给解析模型的 messages
    parse_messages = [
        {"role": "system", "content": parse_system_prompt},
        # 可以添加一个空的 user 消息来触发模型生成，或者直接在 system prompt 中包含所有信息
        # {"role": "user", "content": ""} 
    ]+truncated_messages

    # --- Step 2: 调用LLM进行解析 ---
    # 记录第一次调用的完整输入
    log_first_llm_call(json.dumps(parse_messages,ensure_ascii=False,indent=2), "[第一次调用的完整massage输入...]") # 记录到文件

    try:
        # 注意：这里我们直接调用 vLLM 的底层服务方法
        # 需要构造一个 ChatCompletionRequest 对象
        parse_request = ChatCompletionRequest(
            model=Config.SERVED_MODEL_NAME,
            messages=parse_messages,
            stream=False,
            temperature=Config.LLM_TEMPERATURE_PARSE,
            max_tokens=Config.LLM_MAX_TOKENS_PARSE
        )
        
        # 调用 vLLM 服务
        parse_response = await vllm_service.create_chat_completion(parse_request)
        logger.debug(f"Parse response: {parse_response}")

        # 记录第一次调用的完整输出
        first_llm_output = parse_response.choices[0].message.content
        log_first_llm_call(parse_system_prompt, first_llm_output) # 记录到文件

        # 假设模型返回的是JSON格式的字符串
        parse_data = json.loads(first_llm_output)
        return parse_data

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        log_error_to_file(f"解析模型响应失败: {e}", "RAG_Request")
        raise HTTPException(status_code=500, detail="解析模型响应失败")
    except Exception as e:
        logger.error(f"Error during parsing LLM call: {e}")
        log_error_to_file(f"第一次LLM调用错误: {e}", "RAG_Request")
        raise HTTPException(status_code=500, detail="第一次LLM调用失败")


async def query_knowledge_graph(parse_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    根据解析结果查询Neo4j知识图谱。
    """
    logger = get_logger(__name__)
    neo4j_results = []
    
    # 遍历 entity_relationship_groups 并调用相应接口
    for i, group in enumerate(parse_result.get("entity_relationship_groups", [])):
        logger.info(f"Processing group {i}: {group}")
        logger.debug(f"Processing group {i} (before mapping): {group}") # 记录到文件

        # 将中文实体类型映射为英文类型
        mapped_entities = []
        for entity in group.get("entities", []):
            english_type = entity_type_mapping.get(entity["type"], entity["type"]) # 如果找不到映射，保留原名
            mapped_entities.append({"name": entity["name"], "type": english_type})
        group["entities"] = mapped_entities # 更新group中的实体列表

        # 将中文关系名映射为英文关系类型
        english_relationships = [rel_mapping.get(rel, rel) for rel in group.get("relationships", [])] # 如果找不到映射，保留原名

        # 记录映射结果到日志文件
        logger.info(f"Group {i} - Mapped Entities: {mapped_entities}")
        logger.info(f"Group {i} - Mapped Relationships: {english_relationships}")
        # 记录到文件
        logger.debug(f"Group {i} - Mapped Entities: {mapped_entities}")
        logger.debug(f"Group {i} - Mapped Relationships: {english_relationships}")

        # 检查关系和意图是否为空
        if not group.get("relationships"):
            logger.warning(f"Group {i} has no relationships: {group}")
            logger.debug(f"Warning: Group {i} has no relationships.")
            # 可以选择跳过此组，或根据意图使用默认关系
            # 这里选择跳过
            continue
        if not group.get("intent"):
            logger.warning(f"Group {i} has no intent: {group}")
            logger.debug(f"Warning: Group {i} has no intent.")
            # 可以选择跳过此组，或尝试推断意图
            # 这里选择跳过
            continue

        # 根据意图查找映射的接口
        intent = group["intent"]
        if intent not in intent_mapping_data:
            logger.warning(f"未知意图: {intent}")
            logger.debug(f"Warning: Unknown intent '{intent}' in group {i}.")
            continue

        interface_info = intent_mapping_data[intent]
        interface_name = interface_info["neo4j_interface"]
        required_params = interface_info.get("requires", []) # 获取接口需要的参数

        # 检查接口所需的参数是否提供
        missing_params = []
        if "entities" in required_params and not group.get("entities"):
            missing_params.append("entities")
        if "relationships" in required_params and not english_relationships:
            missing_params.append("relationships")

        if missing_params:
            logger.warning(f"接口 {interface_name} 需要参数 {missing_params}，但在组 {i} 中未提供。跳过。")
            logger.debug(f"Warning: Interface '{interface_name}' requires {missing_params} but they are missing in group {i}. Skipping.")
            continue

        # 调用Neo4j服务 - 使用 if/elif 映射
        entities_for_query = group["entities"]

        if interface_name == "find_connections_between_entities":
            # 需要至少两个实体
            if len(entities_for_query) >= 2:
                result = neo4j_service.find_connections_between_entities(
                    entities=entities_for_query,
                    relationships=english_relationships
                )
                neo4j_results.extend(result)
                logger.info(f"Neo4j query for intent '{intent}' (interface '{interface_name}') returned {len(result)} records.")
                log_neo4j_query(intent, interface_name, entities_for_query, english_relationships, len(result)) # 记录到文件
            else:
                logger.warning(f"接口 {interface_name} 需要至少两个实体，但只有 {len(entities_for_query)} 个。")
                logger.debug(f"Warning: Interface '{interface_name}' requires at least 2 entities, but only {len(entities_for_query)} provided in group {i}.")
        elif interface_name == "find_properties_of_entity":
            result = neo4j_service.find_properties_of_entity(
                entities=entities_for_query
            )
            neo4j_results.extend(result)
            logger.info(f"Neo4j query for intent '{intent}' (interface '{interface_name}') returned {len(result)} records.")
            log_neo4j_query(intent, interface_name, entities_for_query, [], len(result)) # 记录到文件
        elif interface_name == "find_related_entities_by_relationship":
            result = neo4j_service.find_related_entities_by_relationship(
                entities=entities_for_query,
                relationships=english_relationships
            )
            neo4j_results.extend(result)
            logger.info(f"Neo4j query for intent '{intent}' (interface '{interface_name}') returned {len(result)} records.")
            log_neo4j_query(intent, interface_name, entities_for_query, english_relationships, len(result)) # 记录到文件
        elif interface_name == "find_common_connections":
            # 需要至少两个实体
            if len(entities_for_query) >= 2:
                 result = neo4j_service.find_common_connections(
                     entities=entities_for_query,
                     relationships=english_relationships
                 )
                 neo4j_results.extend(result)
                 logger.info(f"Neo4j query for intent '{intent}' (interface '{interface_name}') returned {len(result)} records.")
                 log_neo4j_query(intent, interface_name, entities_for_query, english_relationships, len(result)) # 记录到文件
            else:
                logger.warning(f"接口 {interface_name} 需要至少两个实体，但只有 {len(entities_for_query)} 个。")
                logger.debug(f"Warning: Interface '{interface_name}' requires at least 2 entities, but only {len(entities_for_query)} provided in group {i}.")
        elif interface_name == "query_entity_relationships":
            # 这个接口只需要实体
            for entity in entities_for_query:
                 result = neo4j_service.query_entity_relationships(
                     entities=[entity]
                 )
                 neo4j_results.extend(result)
                 logger.info(f"Neo4j query for intent '{intent}' (interface '{interface_name}') on entity '{entity['name']}' returned {len(result)} records.")
                 log_neo4j_query(intent, interface_name, [entity], [], len(result)) # 记录到文件
        elif interface_name == "query_entity_connections":
            for entity in entities_for_query:
                 result = neo4j_service.query_entity_connections(
                     entities=[entity]
                 )
                 neo4j_results.extend(result)
                 logger.info(f"Neo4j query for intent '{intent}' (interface '{interface_name}') on entity '{entity['name']}' returned {len(result)} records.")
                 log_neo4j_query(intent, interface_name, [entity], [], len(result)) # 记录到文件
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
                log_neo4j_query(intent, interface_name, entities_for_query, english_relationships, len(result)) # 记录到文件
            else:
                logger.warning(f"接口 {interface_name} 需要至少两个实体，但只有 {len(entities_for_query)} 个。")
                logger.debug(f"Warning: Interface '{interface_name}' requires at least 2 entities, but only {len(entities_for_query)} provided in group {i}.")
        else:
            logger.error(f"Neo4j服务中未找到接口: {interface_name}")
            logger.debug(f"Error: Interface '{interface_name}' not found in Neo4j service.")
            continue # 跳过未知接口

    # 处理 standalone_entities (如果需要)
    standalone_results = []
    standalone_entities = parse_result.get("standalone_entities", [])
    if standalone_entities:
        # 将standalone_entities的中文类型也映射为英文
        mapped_standalone_entities = []
        for entity in standalone_entities:
            english_type = entity_type_mapping.get(entity["type"], entity["type"])
            mapped_standalone_entities.append({"name": entity["name"], "type": english_type})
        parse_result["standalone_entities"] = mapped_standalone_entities # 更新

        standalone_entities_for_query = [{"name": e["name"], "type": e["type"]} for e in parse_result["standalone_entities"]]
        # 示例：查询 standalone_entities 的属性
        for entity in standalone_entities_for_query:
             rel_result = neo4j_service.find_properties_of_entity(entities=[entity])
             standalone_results.extend(rel_result)
        logger.info(f"Neo4j query for standalone_entities returned {len(standalone_results)} records.")
        log_neo4j_query("standalone_properties", "find_properties_of_entity", standalone_entities_for_query, [], len(standalone_results)) # 记录到文件

    # 合并所有Neo4j查询结果
    all_context_data = neo4j_results + standalone_results
    logger.debug(f"All Neo4j Results:\n{json.dumps(all_context_data, ensure_ascii=False, indent=2)}") # 记录到文件
    return all_context_data


async def generate_final_response(
    messages: List[Dict[str, str]], # 完整的原始对话历史
    neo4j_context: List[Dict[str, Any]], # 从Neo4j检索到的上下文信息
    request_id: str,
    model_name: str,
    stream: bool,
    temperature: float,
    max_tokens: int,
) -> Any:
    """
    第二次LLM调用：结合检索到的上下文和完整对话历史生成最终回答。
    使用标准 OpenAI messages 格式：
    - 用户对话历史 (user/assistant) 在前。
    - 数据库检索到的上下文 (system) 插入在用户历史之后，问题之前。
    """
    logger = get_logger(__name__)
    
    # --- Step 1: 准备上下文信息 ---
    context_str = json.dumps(neo4j_context, ensure_ascii=False, indent=2) if neo4j_context else "无相关信息。"

    # --- Step 2: 构造符合 OpenAI 标准的 messages 数组 ---
    # 策略：
    # 1. 复制原始对话历史 (user/assistant messages)。
    # 2. 在历史末尾插入一条 system message，包含数据库上下文。
    # 3. (可选) 如果需要，可以在最后添加一条 user message 来明确指令。
    #    但通常最后一条 user message 已经是当前问题了。

    final_messages = []
    if messages:
        # 复制所有原始消息
        final_messages.extend(messages)
        
        # 在原始消息列表的末尾（即在最后一条 user 消息之前）插入系统上下文
        # 这样可以确保上下文靠近用户问题，同时不破坏原始对话顺序
        # 注意：需要在倒数第一个 user 消息之前插入，以确保它在问题前被看到
        # 但简单地 append 然后 insert(-1, ...) 可能不准确
        # 更稳健的方法是找到最后一个 user 消息的位置并插入
        
        # 查找最后一个 user 消息的索引
        last_user_index = -1
        for i, msg in enumerate(reversed(final_messages)):
            if msg["role"] == "user":
                last_user_index = len(final_messages) - 1 - i
                break
        
        system_context_message = {
            "role": "system", 
            "content": f"背景知识:\n{context_str}"
        }
        
        if last_user_index != -1:
            # 在最后一个 user 消息之前插入上下文
            final_messages.insert(last_user_index, system_context_message)
        else:
            # 如果没有找到 user 消息（理论上不太可能），则在末尾添加
            final_messages.append(system_context_message)
            
    else:
        # 如果没有原始消息（不太可能），则创建一个包含上下文和问题的对话
        final_messages = [
            {"role": "system", "content": f"背景知识:\n{context_str}"},
            {"role": "user", "content": "请根据以上背景知识回答问题。"}
        ]

    # --- Step 3: 处理 token 截断 ---
    # 计算最终消息的总 token 数
    # 注意：需要计算整个 final_messages 列表的 token 总和
    total_tokens = 0
    for msg in final_messages:
        # 使用 tokenizer 计算每个消息内容的 token 数
        tokens = tokenizer.encode(msg["content"], add_special_tokens=False)
        total_tokens += len(tokens)
    
    if total_tokens > Config.MAX_PROMPT_TOKENS:
        logger.warning(f"Final prompt tokens ({total_tokens}) exceed max allowed ({Config.MAX_PROMPT_TOKENS}). Truncating context.")
        logger.debug(f"Warning: Final prompt tokens ({total_tokens}) exceed limit ({Config.MAX_PROMPT_TOKENS}). Truncating context.") # 记录到文件
        
        # 策略：优先保留 system 消息的开头和结尾，以及 user/assistant 消息
        # 最简单的策略是截断 system message 的内容
        # 找到 system message
        system_msg_indices = [i for i, msg in enumerate(final_messages) if msg["role"] == "system"]
        if system_msg_indices:
            # 假设只有一个 system message 用于上下文
            system_msg_index = system_msg_indices[0]
            original_system_content = final_messages[system_msg_index]["content"]
            
            # 计算除 system message 外其他消息的 token 数
            other_messages_tokens = total_tokens - len(tokenizer.encode(original_system_content, add_special_tokens=False))
            
            # 计算可用于 system message 的 token 数
            available_system_tokens = Config.MAX_PROMPT_TOKENS - other_messages_tokens - Config.SAFETY_MARGIN # 预留安全边际
            
            if available_system_tokens > 0:
                # 简单截断 system message 内容
                # 可以优化为按句子或段落截断
                system_lines = original_system_content.split('\n')
                current_system_tokens = 0
                truncated_system_lines = []
                for line in system_lines:
                    line_tokens = tokenizer.encode(line, add_special_tokens=False)
                    if current_system_tokens + len(line_tokens) <= available_system_tokens:
                        truncated_system_lines.append(line)
                        current_system_tokens += len(line_tokens)
                    else:
                        # 添加截断标记
                        truncated_system_lines.append("...[上下文已截断]")
                        break
                
                truncated_system_content = '\n'.join(truncated_system_lines)
                final_messages[system_msg_index]["content"] = truncated_system_content
                logger.info(f"System message truncated to {current_system_tokens} tokens.")
                logger.debug(f"System message truncated to {current_system_tokens} tokens.") # 记录到文件
            else:
                # 可用 token 非常少，移除 system message
                logger.warning("Available tokens for system message is 0 or negative, removing system message.")
                logger.debug("Warning: Available tokens for system message is 0 or negative, removing system message.") # 记录到文件
                del final_messages[system_msg_index]
        else:
            logger.warning("No system message found for truncation.")
            logger.debug("Warning: No system message found for truncation.") # 记录到文件

    # --- Step 4: 调用LLM生成 ---
    generate_messages = final_messages
    log_second_llm_call(f"Final Messages with Context:\n{json.dumps(generate_messages, ensure_ascii=False, indent=2)}", "[Awaiting Output...]") # 记录到文件

    try:
        # 构造 ChatCompletionRequest
        generate_request = ChatCompletionRequest(
            model=model_name,
            messages=generate_messages, # 使用构造好的 final_messages
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens
            # 可以添加其他参数，如 top_p, frequency_penalty 等
        )

        if stream:
            # --- 流式响应 ---
            logger.info("[开始处理流式输出]")
            stream_generator = await vllm_service.create_chat_completion(generate_request)
            
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
                                        
                                        logger.debug(f"[Req ID: {request_id}] Sending [DONE] signal.")
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
                                                        "model": chunk_data.get("model", model_name), # 优先使用 vLLM 的 model 名
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
                                            logger.warning(f"[Req ID: {request_id}] Failed to decode JSON chunk: {e}. Raw chunk: {chunk[:200]}...")
                                            continue # 跳过有问题的块
                                        
                                else:
                                    # 11. 如果字符串不以 "data: " 开头，可能是其他 SSE 字段 (如 event:, id:, retry:) 或格式错误
                                    # 记录为 DEBUG 级别，因为这很常见且通常可以安全忽略
                                    logger.debug(f"[Req ID: {request_id}] Ignoring non-data SSE line or malformed chunk: {chunk[:100]}...")
                                    continue # 跳过非 data 行
                                
                            else:
                                # 12. 如果 chunk 不是字符串 (理论上 stream_generator 应该只产生 str，但这作为兜底检查)
                                logger.warning(f"[Req ID: {request_id}] Received unexpected chunk type (not str): {type(chunk)}. Value preview: {repr(chunk)[:100]}...")
                                continue # 跳过未知类型
                            
                    except Exception as e: # 捕获生成器内部的任何未处理异常
                        logger.error(f"[Req ID: {request_id}] Error inside stream generator: {e}", exc_info=True)
                        # 可以选择在这里 yield 一个错误信息给客户端，但这比较复杂且非标准
                        # 通常让异常传播出去，由 FastAPI 的错误处理机制处理
                        # 注意：一旦开始发送流，再抛出 HTTPException 可能导致客户端收到不完整的响应
                        raise # 重新抛出异常，让外层捕获
                    
                    finally:
                        # --- 在这里记录流式响应结束的日志 ---
                        # 这是确保即使在循环中 break 或发生异常也能执行的地方
                        logger.debug(f"[Req ID: {request_id}] Streaming response ended.")
                        log_second_llm_call("[Input was logged previously]", "[Streamed output - see client logs]")
                        log_rag_request_end() # 记录请求结束 via log_service
            

            # return StreamingResponse(generate_openai_stream(), media_type="text/event-stream")
            return generate_openai_stream

        else:
            # # --- 非流式响应 ---
            # non_stream_response = await vllm_service.create_chat_completion(generate_request) # 假设 vllm_service.chat_completion 支持非流式
            
            # # 记录第二次调用的完整输出
            # full_response_text = non_stream_response.choices[0].message.content
            # log_second_llm_call(f"Final Messages with Context:\n{json.dumps(generate_messages, ensure_ascii=False, indent=2)}", full_response_text) # 记录到文件
            # log_rag_request_end() # 记录请求结束

            # # 返回标准的 ChatCompletionResponse
            # # 注意：vLLM 的非流式响应通常是 ChatCompletionResponse 格式，可以直接返回
            # # 但为了确保 ID 和 model 等字段是我们期望的，可以稍微调整
            # response_dict = non_stream_response.model_dump()
            # response_dict["id"] = request_id
            # # response_dict["model"] = model_name # vLLM 通常会正确设置
            # # response_dict["created"] = int(datetime.now().timestamp()) # 可以更新时间戳
            
            # return JSONResponse(content=response_dict)


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
                         logger.error(f"[Req ID: {request_id}] Error processing non-stream chunk for collection: {e}")
                         continue
                 elif hasattr(chunk, 'data') and chunk.data == "[DONE]":
                     # End of internal stream
                     break
                 else:
                     logger.warning(f"[Req ID: {request_id}] Unexpected chunk in non-stream collection: {type(chunk)}")
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
                "model": model_name, # Use model from request
                "choices": [choice_dict],
                # "usage": usage_dict_or_none # Add if you can reliably get it
                "usage": None # Placeholder, vLLM might populate this differently or require specific config
            }
            
            logger.info(f"Generated non-streaming response (ID: {request_id})")
            
            # Log full output via log_service
            log_second_llm_call("[Input was logged previously]", full_response_text)
            log_rag_request_end() # 记录请求结束 via log_service
            
            # Return the constructed dictionary. FastAPI will serialize it based on the response_model (ChatCompletionResponse)
            # Returning a dict is generally fine if it matches the Pydantic model structure
            return response_dict # Let FastAPI handle serialization

    except Exception as e:
        logger.error(f"Error in RAG process: {str(e)}", exc_info=True)
        log_error_to_file(f"RAG process error: {str(e)}", "RAG_Request") # 记录到文件
        log_rag_request_end() # 确保即使出错也记录结束
        raise HTTPException(status_code=500, detail=f"RAG处理过程出错: {str(e)}")


# --- 7. 自定义 RAG 端点 (核心) ---
@app.post("/v1/medical_rag_stream", response_model=None) # response_model=None 允许动态返回
async def medical_rag_stream(request: Request, chat_request: ChatCompletionRequest):
    """
    严格的 OpenAI API 兼容 RAG 流式/非流式聊天补全端点。
    """
    # --- 从 Headers 读取信息 ---
    session_id = request.headers.get("X-Session-ID", "unknown_session")
    timestamp_str = request.headers.get("X-Timestamp", "unknown_timestamp")
    request_id = f"chatcmpl-{uuid.uuid4().hex}"

    # --- 从 ChatCompletionRequest 提取信息 ---
    if not chat_request.messages:
        logger = get_logger(__name__)
        logger.error("No messages provided in the request.")
        raise HTTPException(status_code=400, detail="Messages are required.")

    # 获取最后一个用户消息作为当前问题（用于日志）
    latest_user_message_content = ""
    for msg in reversed(chat_request.messages):
        if msg["role"] == "user":
            latest_user_message_content = msg["content"]
            break

    logger = get_logger(__name__)
    logger.info(f"[Session: {session_id}] Received RAG request (ID: {request_id}) for model '{chat_request.model}', question: {latest_user_message_content[:50]}...")
    log_rag_request_start(latest_user_message_content, session_id) # 记录到文件

    try:
        messages_dicts = []
        for msg in chat_request.messages:
            if isinstance(msg, dict):
                # 如果 msg 已经是 dict，则直接添加
                messages_dicts.append(msg)
            elif hasattr(msg, 'dict'):
                # 如果 msg 是 Pydantic 模型实例（如 ChatMessage），则调用 .dict()
                # Pydantic v2 推荐使用 .model_dump()
                messages_dicts.append(msg.model_dump() if hasattr(msg, 'model_dump') else msg.dict())
            else:
                # 如果既不是 dict 也没有 .dict() 方法，则记录警告并跳过
                logger.warning(f"[Session: {session_id}] [Req ID: {request_id}] Skipping message of unexpected type: {type(msg)}")
                continue

        # --- Step 1: Entity/Intent Parsing with Context (LLM) ---
        parse_result = await parse_user_query_with_context(messages_dicts)

        # --- Step 2: Map Chinese Entity Types to English & Map Relationships & Query Neo4j (Service-side) ---
        neo4j_context_data = await query_knowledge_graph(parse_result)

        # --- Step 3: Generate Final Answer (LLM, Streamed or Non-Streamed) ---
        response = await generate_final_response(
            messages=messages_dicts, # 传递完整的原始消息历史
            neo4j_context=neo4j_context_data,
            request_id=request_id,
            model_name=chat_request.model,
            stream=chat_request.stream,
            temperature=chat_request.temperature if chat_request.temperature is not None else Config.LLM_TEMPERATURE_GENERATE,
            max_tokens=chat_request.max_tokens if chat_request.max_tokens is not None else Config.LLM_MAX_TOKENS_GENERATE,
        )

        if chat_request.stream:
            logger.info(f"[Session: {session_id}] Starting to stream response (ID: {request_id}) for question: {latest_user_message_content[:50]}...")
            return StreamingResponse(response(), media_type="text/event-stream")
        else:
            logger.info(f"[Session: {session_id}] Generated non-streaming response (ID: {request_id}) for question: {latest_user_message_content[:50]}...")
            # vLLM 返回的已经是 ChatCompletionResponse，直接返回即可
            # 如果需要修改 ID 等字段，可以在这里处理
            response_dict = response.model_dump()
            response_dict["id"] = request_id
            return JSONResponse(content=response_dict)

    except Exception as e:
        logger.error(f"[Session: {session_id}] Error in RAG process (ID: {request_id}): {str(e)}", exc_info=True)
        log_error_to_file(f"RAG process error for question '{latest_user_message_content[:50]}...': {str(e)}", "RAG_Request") # 记录到文件
        log_rag_request_end() # 确保即使出错也记录结束
        raise HTTPException(status_code=500, detail=f"RAG处理过程出错: {str(e)}")


# --- 8. (可选) 原有的 OpenAI 兼容端点 ---
# 注意：这些端点需要正确地挂载 vLLM 的 OpenAI 服务
# 由于 vLLM 0.10.1 的 API 结构，直接挂载 router 可能不行，需要手动添加路由
# 这里提供一个示例，但可能需要根据 vLLM 的具体实现调整

@app.post("/v1/chat/completions")
async def chat_completions(chat_request: ChatCompletionRequest):
    # 直接调用 vLLM 的服务
    # 注意：需要确保 vllm_service 在此作用域内可用
    # 在 lifespan 中我们将其设为全局变量
    try:
        if chat_request.stream:
            generator = await vllm_service.create_chat_completion(chat_request)
            return StreamingResponse(generator, media_type="text/event-stream")
        else:
            response = await vllm_service.create_chat_completion(chat_request)
            return response
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error in OpenAI compatible chat completion: {e}")
        raise HTTPException(status_code=500, detail="OpenAI兼容接口调用失败")

@app.get("/v1/models")
async def show_available_models():
    # 注意：需要确保 vllm_service 在此作用域内可用
    try:
        # models_service = OpenAIServingModels(...) # 需要正确初始化
        # models = await models_service.show_available_models()
        # 临时方案：直接调用 chat 服务的模型信息（如果可行）
        # 或者返回静态信息
        # 这里简化处理
        return {"object": "list", "data": [{"id": Config.SERVED_MODEL_NAME, "object": "model", "created": int(datetime.now().timestamp()), "owned_by": "vllm"}]}
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail="获取模型列表失败")

@app.get("/health")
async def health():
    return {"status": "ok"}

# --- 9. 错误处理 ---
# FastAPI 会自动处理 HTTPException
# 对于未捕获的异常，可以添加全局异常处理器
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request, exc):
    logger = get_logger(__name__)
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    log_error_to_file(f"HTTP {exc.status_code}: {exc.detail}", "General")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger = get_logger(__name__)
    logger.error(f"Validation Error: {exc}")
    log_error_to_file(f"Validation Error: {exc}", "General")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger = get_logger(__name__)
    logger.error(f"Unhandled Exception: {exc}", exc_info=True)
    log_error_to_file(f"Unhandled Exception: {exc}", "General")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
