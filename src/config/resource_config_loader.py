# -*- coding: utf-8 -*-
"""
资源配置加载器

从配置文件加载全局资源配置。
支持从统一配置管理器加载配置。
"""

from .global_config import GlobalConfig
from .pool_config import PoolConfig
from .database_config import DatabaseConfig, ModelConfig
from .resource_config_manager import get_config_manager, load_global_config_from_manager


def load_global_config() -> GlobalConfig:
    """
    从配置文件加载全局资源配置
    
    使用统一配置管理器加载所有配置，并转换为GlobalConfig实例。
    
    Returns:
        GlobalConfig: 全局资源配置实例
    """
    return load_global_config_from_manager()


def load_global_config_legacy() -> GlobalConfig:
    """
    从配置文件加载全局资源配置（旧版兼容方法）
    
    直接从各配置类加载配置，不使用统一配置管理器。
    
    Returns:
        GlobalConfig: 全局资源配置实例
    """
    config = GlobalConfig()
    
    db_config = DatabaseConfig()
    model_config = ModelConfig()
    
    from src.resource_manager.neo4j_connection import Neo4jConnectionConfig
    
    neo4j_resource_config = Neo4jConnectionConfig(
        uri=db_config.neo4j_uri,
        user=db_config.neo4j_user,
        password=db_config.neo4j_password,
        database=db_config.neo4j_database
    )
    config.add_resource_config("neo4j_connection", neo4j_resource_config)
    
    neo4j_pool_config = PoolConfig(
        max_size=10,
        min_idle=2,
        idle_timeout=300000,
        max_wait_time=5000
    )
    config.add_pool_config("neo4j_connection", neo4j_pool_config)
    
    from src.resource_manager.vllm_model import VLLMModelConfig

    vllm_resource_config = VLLMModelConfig(
        model_path=model_config.model_path,
        model_name=model_config.model_name,
        tensor_parallel_size=model_config.tensor_parallel_size,
        max_model_len=model_config.max_model_len,
        gpu_memory_utilization=model_config.gpu_memory_utilization
    )
    config.add_resource_config("vllm_model", vllm_resource_config)

    vllm_pool_config = PoolConfig(
        max_size=1,
        min_idle=1,
        idle_timeout=600000,
        max_wait_time=30000
    )
    config.add_pool_config("vllm_model", vllm_pool_config)

    from src.resource_manager.milvus_connection import MilvusConnectionConfig

    milvus_resource_config = MilvusConnectionConfig(
        uri=db_config.milvus_uri if hasattr(db_config, 'milvus_uri') else "http://localhost:19530",
        user=db_config.milvus_user if hasattr(db_config, 'milvus_user') else "root",
        password=db_config.milvus_password if hasattr(db_config, 'milvus_password') else "Milvus",
        token=""
    )
    config.add_resource_config("milvus_connection", milvus_resource_config)

    milvus_pool_config = PoolConfig(
        max_size=10,
        min_idle=2,
        idle_timeout=300000,
        max_wait_time=5000
    )
    config.add_pool_config("milvus_connection", milvus_pool_config)

    from src.resource_manager.intent_model import IntentModelConfig

    intent_model_resource_config = IntentModelConfig(
        model_path=model_config.intent_model_path if hasattr(model_config, 'intent_model_path') else "",
        model_name=model_config.intent_model_name if hasattr(model_config, 'intent_model_name') else "intent-classification",
        device=model_config.intent_device if hasattr(model_config, 'intent_device') else "cpu",
        max_length=model_config.intent_max_length if hasattr(model_config, 'intent_max_length') else 128
    )
    config.add_resource_config("intent_model", intent_model_resource_config)

    intent_model_pool_config = PoolConfig(
        max_size=1,
        min_idle=1,
        idle_timeout=600000,
        max_wait_time=30000
    )
    config.add_pool_config("intent_model", intent_model_pool_config)

    from src.resource_manager.vector_model import VectorModelConfig

    vector_model_resource_config = VectorModelConfig(
        model_path=model_config.vector_model_path if hasattr(model_config, 'vector_model_path') else "",
        model_name=model_config.vector_model_name if hasattr(model_config, 'vector_model_name') else "vector-embedding",
        device=model_config.vector_device if hasattr(model_config, 'vector_device') else "cpu",
        dimension=model_config.vector_dimension if hasattr(model_config, 'vector_dimension') else 1024,
        batch_size=model_config.vector_batch_size if hasattr(model_config, 'vector_batch_size') else 32
    )
    config.add_resource_config("vector_model", vector_model_resource_config)

    vector_model_pool_config = PoolConfig(
        max_size=1,
        min_idle=1,
        idle_timeout=600000,
        max_wait_time=30000
    )
    config.add_pool_config("vector_model", vector_model_pool_config)

    return config
