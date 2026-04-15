# -*- coding: utf-8 -*-
"""
向量数据库部署项目配置模板
使用方法：复制此文件为config.py，填入实际的连接信息
"""

# Neo4j图数据库配置
NEO4J_CONFIG = {
    "uri": "neo4j+s://your-instance.databases.neo4j.io",  # Neo4j Aura连接地址
    "user": "your_username",  # 用户名
    "password": "your_password"  # 密码
}

# Zilliz Cloud/Milvus向量数据库配置
ZILLIZ_CONFIG = {
    "user": "your_username",  # Zilliz Cloud用户名
    "password": "your_password",  # Zilliz Cloud密码
    "uri": "https://your-instance.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn",  # Zilliz Cloud连接地址
    "token": "your_api_token"  # API Token
}

# 本地向量模型配置
LOCAL_MODEL_CONFIG = {
    "model_name": "BAAI/bge-large-zh-v1.5",  # 向量模型名称
    "device": "cuda",  # 运行设备：cuda或cpu
    "max_memory": "2GB",  # 模型最大内存
    "batch_size": 512,  # 批处理大小
    "max_batch_size": 1024,  # 最大批处理大小
    "dimension": 1024,  # 向量维度
    "max_length": 512,  # 最大序列长度
    "normalize": True,  # 是否归一化
    "cache_dir": "/home/project/MedicalQA/base_models/"  # 模型缓存目录
}

# GPU配置
GPU_CONFIG = {
    "device_id": 0,  # GPU设备ID
    "max_memory": "22GB",  # GPU最大内存
    "model_memory": "2GB",  # 模型内存
    "memory_fraction": 0.95  # 内存使用比例
}

# 向量配置
VECTOR_CONFIG = {
    "dimension": 1024,  # 向量维度
    "batch_size": 512,  # 批处理大小
    "max_batch_size": 1024  # 最大批处理大小
}

# Milvus集合配置
MILVUS_CONFIG = {
    "collection_name": "medical_entity",  # 集合名称
    "dimension": 1024,  # 向量维度
    "index_type": "IVF_FLAT",  # 索引类型
    "metric_type": "COSINE",  # 相似度度量
    "nlist": 1024  # 聚类中心数量
}

# 集合名称配置
COLLECTION_NAMES = {
    "entity": "medical_entity",  # 实体向量集合
    "attribute": "entity_attributes",  # 属性向量集合
    "relation": "entity_relations"  # 关系向量集合
}

# 数据文件路径配置
DATA_PATHS = {
    "entities": "data/entities.json",  # 实体数据文件
    "attributes": "data/disease_attributes.json",  # 属性数据文件
    "relations": "data/relations.json",  # 关系数据文件
    "vectors": "data/vectors.json"  # 向量数据文件
}

# 日志配置
LOG_CONFIG = {
    "log_dir": "logs",  # 日志目录
    "log_level": "INFO",  # 日志级别
    "max_file_size": 10 * 1024 * 1024,  # 最大文件大小（10MB）
    "backup_count": 5  # 备份文件数量
}

# 部署配置
DEPLOYMENT_CONFIG = {
    "timeout": 7200,  # 部署超时时间（秒）
    "retry_count": 3,  # 重试次数
    "retry_interval": 5  # 重试间隔（秒）
}
