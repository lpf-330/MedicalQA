# AI辅助生成：GLM-5，2026-04-15
# -*- coding: utf-8 -*-
"""
Neo4j适配器接口

为项目各层级、各类提供统一的Neo4j数据库操作接口。
"""

from abc import abstractmethod
from typing import Any, Dict, List

from src.adapters.base_adapter import BaseAdapter


class Neo4jAdapter(BaseAdapter):
    """
    Neo4j适配器接口
    
    定义Neo4j数据库操作的标准接口，为项目各层级提供统一的访问方式。
    
    使用示例：
        adapter = Neo4jAdapterImpl(uri, user, password)
        adapter.connect()
        results = adapter.execute_query("MATCH (d:Disease) RETURN d LIMIT 10")
        adapter.disconnect()
    """
    
    @abstractmethod
    def is_initialized(self) -> bool:
        """
        检查适配器是否已初始化
        
        Returns:
            bool: 是否已初始化（已连接）
        """
        pass
    
    @abstractmethod
    def connect(self) -> None:
        """
        连接Neo4j数据库
        
        建立与Neo4j数据库的连接，初始化会话。
        
        Raises:
            ConnectionError: 当连接失败时抛出
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """
        断开与Neo4j数据库的连接
        
        关闭数据库会话，释放连接资源。
        """
        pass
    
    @abstractmethod
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        执行Cypher查询
        
        Args:
            query: Cypher查询语句
            
        Returns:
            查询结果列表，每个元素是一个字典
            
        Raises:
            QueryError: 当查询执行失败时抛出
        """
        pass
    
    @abstractmethod
    def execute_query_with_params(
        self, 
        query: str, 
        params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        执行带参数的Cypher查询（防止注入）
        
        Args:
            query: Cypher查询语句（使用$param占位符）
            params: 参数字典
            
        Returns:
            查询结果列表
            
        Raises:
            QueryError: 当查询执行失败时抛出
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        检查是否已连接
        
        Returns:
            bool: 是否已连接到数据库
        """
        pass
