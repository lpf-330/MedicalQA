# -*- coding: utf-8 -*-
"""
Neo4j适配器

为项目各层级、各类提供统一的Neo4j数据库操作接口。
"""

from .neo4j_adapter import Neo4jAdapter
from .neo4j_adapter_impl import Neo4jAdapterImpl

__all__ = ['Neo4jAdapter', 'Neo4jAdapterImpl']
