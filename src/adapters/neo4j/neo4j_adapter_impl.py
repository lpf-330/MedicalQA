# -*- coding: utf-8 -*-
"""
Neo4j适配器实现类

转接适配Neo4j驱动，为项目各层级提供统一的Neo4j数据库操作接口。
"""

import logging
import time
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, Driver, Session

from . import Neo4jAdapter

logger = logging.getLogger(__name__)


class Neo4jAdapterImpl(Neo4jAdapter):
    """
    Neo4j适配器实现类
    
    封装neo4j-python-driver，为项目提供统一的Neo4j数据库操作接口。
    
    属性：
        _uri: 数据库连接URI
        _user: 用户名
        _password: 密码
        _driver: Neo4j驱动实例
        _session: 当前会话实例
    """
    
    def __init__(self, uri: str, user: str, password: str):
        """
        初始化Neo4j适配器
        
        Args:
            uri: Neo4j数据库连接URI
            user: 用户名
            password: 密码
        """
        self._uri = uri
        self._user = user
        self._password = password
        self._driver: Optional[Driver] = None
        self._session: Optional[Session] = None
        logger.debug(f"[Neo4jAdapter] 初始化Neo4j适配器: uri={uri}, user={user}")
    
    def connect(self) -> None:
        """连接Neo4j数据库"""
        if self._driver is not None:
            logger.debug("[Neo4jAdapter] 已连接，跳过")
            return
        
        logger.info(f"[Neo4jAdapter] 开始连接数据库: uri={self._uri}")
        start_time = time.time()
        
        self._driver = GraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password)
        )
        self._session = self._driver.session()
        
        elapsed = time.time() - start_time
        logger.info(f"[Neo4jAdapter] 数据库连接成功: elapsed={elapsed:.2f}s")
    
    def disconnect(self) -> None:
        """断开与Neo4j数据库的连接"""
        logger.info("[Neo4jAdapter] 开始断开数据库连接")
        
        if self._session is not None:
            self._session.close()
            self._session = None
            logger.debug("[Neo4jAdapter] Session已关闭")
        
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.debug("[Neo4jAdapter] Driver已关闭")
        
        logger.info("[Neo4jAdapter] 数据库连接已断开")
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        执行Cypher查询
        
        Args:
            query: Cypher查询语句
            
        Returns:
            查询结果列表
        """
        if self._session is None:
            logger.error("[Neo4jAdapter] 执行查询失败，未连接数据库")
            raise RuntimeError("Not connected to Neo4j database")
        
        logger.debug(f"[Neo4jAdapter] 执行查询: query={query[:100]}...")
        start_time = time.time()
        
        result = self._session.run(query)
        records = [record.data() for record in result]
        
        elapsed = time.time() - start_time
        logger.info(f"[Neo4jAdapter] 查询完成: result_count={len(records)}, elapsed={elapsed:.3f}s")
        return records
    
    def execute_query_with_params(
        self, 
        query: str, 
        params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        执行带参数的Cypher查询
        
        Args:
            query: Cypher查询语句
            params: 参数字典
            
        Returns:
            查询结果列表
        """
        if self._session is None:
            logger.error("[Neo4jAdapter] 执行参数查询失败，未连接数据库")
            raise RuntimeError("Not connected to Neo4j database")
        
        logger.debug(f"[Neo4jAdapter] 执行参数查询: query={query[:100]}..., params={params}")
        start_time = time.time()
        
        result = self._session.run(query, params)
        records = [record.data() for record in result]
        
        elapsed = time.time() - start_time
        logger.info(f"[Neo4jAdapter] 参数查询完成: result_count={len(records)}, elapsed={elapsed:.3f}s")
        return records
    
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._driver is not None and self._session is not None
    
    def __enter__(self) -> 'Neo4jAdapterImpl':
        """上下文管理器入口"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器退出"""
        self.disconnect()
