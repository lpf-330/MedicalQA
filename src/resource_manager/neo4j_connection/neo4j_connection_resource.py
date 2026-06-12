# -*- coding: utf-8 -*-
"""
Neo4j连接资源封装

封装Neo4j数据库连接资源，实现Resource接口。
"""

import time
from typing import TYPE_CHECKING, Optional

from src.resource_manager.resource import Resource
from src.adapters.neo4j.neo4j_adapter import Neo4jAdapter
from src.adapters.neo4j.neo4j_adapter_impl import Neo4jAdapterImpl

if TYPE_CHECKING:
    from src.resource_manager.neo4j_connection.neo4j_connection_config import Neo4jConnectionConfig


class Neo4jConnectionResource(Resource):
    """
    Neo4j连接资源类

    封装Neo4j数据库连接，实现Resource接口。

    属性：
        _config: Neo4j连接配置
        _adapter: Neo4j适配器实例
        _last_used_time: 最后使用时间戳
        _is_active: 资源活跃状态
    """

    def __init__(self, config: 'Neo4jConnectionConfig'):
        """
        初始化Neo4j连接资源

        Args:
            config: Neo4j连接配置
        """
        self._config = config
        self._adapter: Optional[Neo4jAdapter] = None
        self._last_used_time = int(time.time() * 1000)
        self._is_active = False

    def get_type(self) -> str:
        """获取资源类型标识"""
        return "neo4j_connection"

    def get_last_used_time(self) -> int:
        """获取最后使用时间戳"""
        return self._last_used_time

    def is_activate(self) -> bool:
        """校验资源活跃状态"""
        return self._is_active

    def activate(self) -> None:
        """激活资源"""
        if self._is_active:
            return

        if self._adapter is None:
            config_protocol = self._config.config_protocol
            self._adapter = Neo4jAdapterImpl(
                uri=config_protocol["uri"],
                user=config_protocol["user"],
                password=config_protocol["password"],
                database=config_protocol.get("database", "neo4j")
            )
            self._adapter.connect()
        self._is_active = True
        self._last_used_time = int(time.time() * 1000)

    def deactivate(self) -> None:
        """
        停用资源（释放回池，保持连接）

        语义：资源从活跃状态变为空闲状态，归还到资源池
        行为：仅标记状态，不断开连接
        场景：资源使用完毕，释放回资源池复用
        """
        if not self._is_active:
            return

        self._is_active = False

    def destroy(self) -> None:
        """
        销毁资源（彻底释放）

        语义：资源彻底销毁，从资源池移除
        行为：断开连接，释放所有资源
        场景：资源池关闭、资源过期、资源异常需销毁
        """
        if self._adapter is not None:
            self._adapter.disconnect()
        self._adapter = None
        self._is_active = False

    def get_adapter(self) -> Optional[Neo4jAdapter]:
        """
        获取Neo4j适配器实例

        Returns:
            Neo4jAdapter: Neo4j适配器实例
        """
        return self._adapter
