# -*- coding: utf-8 -*-
"""
适配层

适配层负责对重要的外部框架或依赖进行适配对接，为内部其他层级提供统一的内部接口。

包结构：
- adapters/neo4j/ - Neo4j图数据库适配
- adapters/milvus/ - Milvus向量数据库适配
- adapters/sglang/ - SGLang推理引擎适配
- adapters/langchain/ - Langchain框架适配
- adapters/transformers/ - Transformers模型框架适配

每个适配包包含：
- {dependency}_adapter.py - 适配器接口
- {dependency}_adapter_impl.py - 适配器实现类
"""

from .neo4j import Neo4jAdapter, Neo4jAdapterImpl
from .sglang import SGLangAdapter, SGLangAdapterImpl
from .milvus import MilvusAdapter, MilvusAdapterImpl
from .transformers import TransformersAdapter, TransformersAdapterImpl
from .langchain import (
    LangchainAdapter,
    LangchainAdapterImpl,
    InternalChain,
    InternalChainImpl,
    InternalTool,
    InternalToolImpl,
    InternalMemory,
    InternalMemoryImpl,
    InternalAgent,
    InternalAgentImpl
)

__all__ = [
    'Neo4jAdapter',
    'Neo4jAdapterImpl',
    'SGLangAdapter',
    'SGLangAdapterImpl',
    'MilvusAdapter',
    'MilvusAdapterImpl',
    'TransformersAdapter',
    'TransformersAdapterImpl',
    'LangchainAdapter',
    'LangchainAdapterImpl',
    'InternalChain',
    'InternalChainImpl',
    'InternalTool',
    'InternalToolImpl',
    'InternalMemory',
    'InternalMemoryImpl',
    'InternalAgent',
    'InternalAgentImpl'
]
