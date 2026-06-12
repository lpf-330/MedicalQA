# -*- coding: utf-8 -*-
"""
Neo4j适配器实现类

转接适配Neo4j驱动，为项目各层级提供统一的Neo4j数据库操作接口。
"""

import logging
import time
import threading
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, Driver

from . import Neo4jAdapter
from src.utils.logger import log_arch_event, truncate_for_log

logger = logging.getLogger(__name__)


class Neo4jAdapterImpl(Neo4jAdapter):
    """
    Neo4j适配器实现类
    
    封装neo4j-python-driver，为项目提供统一的Neo4j数据库操作接口。
    
    使用线程安全的session管理：每次查询创建新的session，查询完成后关闭。
    
    属性：
        _uri: 数据库连接URI
        _user: 用户名
        _password: 密码
        _driver: Neo4j驱动实例（线程安全）
        _lock: 线程锁
    """
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """
        初始化Neo4j适配器

        Args:
            uri: Neo4j数据库连接URI
            user: 用户名
            password: 密码
            database: 数据库名称
        """
        super().__init__()
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._driver: Optional[Driver] = None
        self._lock = threading.Lock()
        logger.debug(f"[Neo4jAdapter] 初始化Neo4j适配器: uri_present={bool(uri)}, user_present={bool(user)}, database_present={bool(database)}")
    
    def connect(self) -> None:
        """连接Neo4j数据库"""
        with self._lock:
            if self._driver is not None:
                logger.debug("[Neo4jAdapter] 已连接，跳过")
                return
            
            logger.info(f"[Neo4jAdapter] 开始连接数据库: uri_present={bool(self._uri)}, database_present={bool(self._database)}")
            start_time = time.time()
            
            driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
                connection_timeout=90.0,
                connection_acquisition_timeout=90.0
            )
            try:
                driver.verify_connectivity()
                session = driver.session(database=self._database)
                try:
                    session.run("RETURN 1 AS ok")
                finally:
                    session.close()
            except Exception as e:
                logger.debug(f"[Neo4jAdapter] 连接验证失败: {e}")
                driver.close()
                raise
            self._driver = driver
            self._set_initialized(True)

            elapsed = time.time() - start_time
            log_arch_event(logger, component="Neo4jAdapter", stage="ADAPTER", event="connect", status="success", design_id="ARCH-7.3", elapsed=f"{elapsed:.2f}s")
            logger.info(f"[Neo4jAdapter] 数据库连接成功: elapsed={elapsed:.2f}s")
    
    def disconnect(self) -> None:
        """断开与Neo4j数据库的连接"""
        with self._lock:
            logger.info("[Neo4jAdapter] 开始断开数据库连接")

            if self._driver is not None:
                self._driver.close()
                self._driver = None
                logger.debug("[Neo4jAdapter] Driver已关闭")

            self._set_initialized(False)
            log_arch_event(logger, component="Neo4jAdapter", stage="ADAPTER", event="disconnect", status="success", design_id="ARCH-7.3")
            logger.info("[Neo4jAdapter] 数据库连接已断开")
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        执行Cypher查询（线程安全）
        
        每次查询创建新的session，查询完成后关闭，确保线程安全。
        
        Args:
            query: Cypher查询语句
            
        Returns:
            查询结果列表
        """
        if self._driver is None:
            logger.error("[Neo4jAdapter] 执行查询失败，未连接数据库")
            raise RuntimeError("Not connected to Neo4j database")
        
        logger.debug(f"[Neo4jAdapter] 执行查询: query_len={len(query)}")
        logger.debug(f"[Neo4jAdapter] request: {truncate_for_log(repr(query), 500)}")
        start_time = time.time()

        session = None
        try:
            session = self._driver.session(database=self._database)
            result = session.run(query)
            records = [record.data() for record in result]

            elapsed = time.time() - start_time
            logger.debug(f"[Neo4jAdapter] response: {truncate_for_log(repr(records), 500)}")
            log_arch_event(logger, component="Neo4jAdapter", stage="ADAPTER", event="execute_query", status="success", design_id="ARCH-7.3", result_count=len(records), elapsed=f"{elapsed:.3f}s")
            logger.info(f"[Neo4jAdapter] 查询完成: result_count={len(records)}, elapsed={elapsed:.3f}s")
            return records
        finally:
            if session is not None:
                session.close()

    def execute_query_with_params(
        self, 
        query: str, 
        params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        执行带参数的Cypher查询（线程安全）
        
        每次查询创建新的session，查询完成后关闭，确保线程安全。
        
        Args:
            query: Cypher查询语句
            params: 参数字典
            
        Returns:
            查询结果列表
        """
        if self._driver is None:
            logger.error("[Neo4jAdapter] 执行参数查询失败，未连接数据库")
            raise RuntimeError("Not connected to Neo4j database")
        
        logger.debug(f"[Neo4jAdapter] 执行参数查询: query_len={len(query)}, params_keys={list(params.keys())}")
        logger.debug(f"[Neo4jAdapter] request: query={truncate_for_log(repr(query), 250)}, params={truncate_for_log(repr(params), 250)}")
        start_time = time.time()

        session = None
        try:
            session = self._driver.session(database=self._database)
            result = session.run(query, params)
            records = [record.data() for record in result]

            elapsed = time.time() - start_time
            logger.debug(f"[Neo4jAdapter] response: {truncate_for_log(repr(records), 500)}")
            log_arch_event(logger, component="Neo4jAdapter", stage="ADAPTER", event="execute_query_with_params", status="success", design_id="ARCH-7.3", result_count=len(records), elapsed=f"{elapsed:.3f}s")
            logger.info(f"[Neo4jAdapter] 参数查询完成: result_count={len(records)}, elapsed={elapsed:.3f}s")
            return records
        finally:
            if session is not None:
                session.close()
    
    def is_initialized(self) -> bool:
        return self._driver is not None
    
    def is_connected(self) -> bool:
        return self._driver is not None
    
    def __enter__(self) -> 'Neo4jAdapterImpl':
        """上下文管理器入口"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器退出"""
        self.disconnect()
