# log_service.py

import logging
import sys
import os
from datetime import datetime
from typing import Optional

# --- 全局配置 ---
LOG_DIR = "logs"
LOG_FILE_NAME = f"medical_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

# --- 日志格式定义 ---
CONSOLE_FORMAT = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
FILE_FORMAT = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d]: %(message)s')

def setup_global_logger() -> logging.Logger:
    """
    设置全局日志记录器，同时输出到控制台和文件。
    - 控制台输出：INFO及以上级别，格式简洁。
    - 文件输出：DEBUG及以上级别，格式详细，包含文件名和行号。
    """
    # 确保日志目录存在
    os.makedirs(LOG_DIR, exist_ok=True)

    # 创建根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # 设置根记录器级别为最低

    # 防止重复添加处理器
    if root_logger.handlers:
        root_logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO) # 控制台只记录INFO及以上
    console_handler.setFormatter(CONSOLE_FORMAT)

    # 文件处理器
    file_handler = logging.FileHandler(LOG_FILE_PATH, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG) # 文件记录DEBUG及以上
    file_handler.setFormatter(FILE_FORMAT)

    # 将处理器添加到根记录器
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # 返回一个特定的记录器实例用于业务代码调用
    logger = logging.getLogger(__name__)
    logger.info(f"全局日志服务已初始化，日志文件: {LOG_FILE_PATH}")
    return logger

# --- 全局日志记录器实例 ---
# 在模块加载时初始化
global_logger = setup_global_logger()

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取一个日志记录器实例。
    如果不提供name，则返回全局记录器。
    如果提供name，则返回一个子记录器，通常用于区分不同模块的日志。
    """
    if name:
        return logging.getLogger(name)
    else:
        return global_logger

# --- 便捷的日志记录函数 ---
def log_info_to_file(message: str, module_name: str = "General"):
    """将信息日志写入文件（通过子记录器）"""
    logger = get_logger(module_name)
    logger.info(message)

def log_error_to_file(message: str, module_name: str = "General"):
    """将错误日志写入文件（通过子记录器）"""
    logger = get_logger(module_name)
    logger.error(message)

def log_warning_to_file(message: str, module_name: str = "General"):
    """将警告日志写入文件（通过子记录器）"""
    logger = get_logger(module_name)
    logger.warning(message)

def log_debug_to_file(message: str, module_name: str = "General"):
    """将调试日志写入文件（通过子记录器）"""
    logger = get_logger(module_name)
    logger.debug(message)

# --- RAG 请求日志辅助函数 ---
def log_rag_request_start(question: str, session_id: Optional[str] = None):
    """记录RAG请求开始"""
    logger = get_logger("RAG_Request")
    logger.info(f"--- RAG Request Start ---")
    logger.info(f"Question: {question}")
    logger.info(f"Session ID: {session_id}")

def log_rag_request_end():
    """记录RAG请求结束"""
    logger = get_logger("RAG_Request")
    logger.info(f"--- RAG Request End ---")

def log_first_llm_call(input_prompt: str, output: str):
    """记录第一次LLM调用的输入和输出"""
    logger = get_logger("RAG_Request")
    logger.debug(f"First LLM Call - Input Prompt:\n{input_prompt}")
    logger.debug(f"First LLM Call - Output:\n{output}")

def log_neo4j_query(intent: str, interface: str, entities: list, relationships: list, results_count: int):
    """记录Neo4j查询详情"""
    logger = get_logger("RAG_Request")
    logger.info(f"Neo4j Query - Intent: '{intent}', Interface: '{interface}', Entities: {entities}, Relationships: {relationships}, Results Count: {results_count}")

def log_second_llm_call(input_prompt: str, output: str):
    """记录第二次LLM调用的输入和输出"""
    logger = get_logger("RAG_Request")
    logger.info(f"Second LLM Call - Input Prompt:\n{input_prompt}")
    logger.info(f"Second LLM Call - Output:\n{output}")

def log_entity_relationship_group_processing(index: int, original_group: dict, mapped_entities: list, mapped_relationships: list):
    """记录实体关系组的处理和映射过程"""
    logger = get_logger("RAG_Request")
    logger.info(f"Processing group {index} (before mapping): {original_group}")
    logger.info(f"Group {index} - Mapped Entities: {mapped_entities}")
    logger.info(f"Group {index} - Mapped Relationships: {mapped_relationships}")
