# -*- coding: utf-8 -*-
"""
资源类型与配置ID枚举

集中管理资源类型字符串和配置ID，消除代码中的硬编码字符串引用。
"""

from enum import Enum


class ResourceType(str, Enum):
    """资源类型枚举"""
    NEO4J_CONNECTION = "neo4j_connection"
    REASONING_MODEL = "reasoning_model"
    HEALTH_ASSESSMENT_MODEL = "health_assessment_model"
    MILVUS_CONNECTION = "milvus_connection"
    VECTOR_MODEL = "vector_model"
    INTENT_MODEL = "intent_model"
    NER_MODEL = "ner_model"


class ConfigId(str, Enum):
    """配置ID枚举"""
    NEO4J_CONFIG = "neo4j_connection_config"
    REASONING_CONFIG = "reasoning_model_config"
    HEALTH_ASSESSMENT_CONFIG = "health_assessment_model_config"
    MILVUS_CONFIG = "milvus_connection_config"
    VECTOR_MODEL_CONFIG = "vector_model_config"
    INTENT_MODEL_CONFIG = "intent_model_config"
    NER_MODEL_CONFIG = "ner_model_config"
