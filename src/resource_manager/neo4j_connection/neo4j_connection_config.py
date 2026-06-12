# -*- coding: utf-8 -*-
"""
Neo4j连接配置封装

封装Neo4j数据库连接配置，实现ResourceConfig接口。
"""

from typing import Dict

from src.resource_manager.resource_config import ResourceConfig


class Neo4jConnectionConfig(ResourceConfig[Dict[str, str]]):
    """
    Neo4j连接配置类

    实现ResourceConfig接口，存储Neo4j连接配置。

    属性：
        _resource_type: 资源类型标识
        _resource_name: 资源业务名称
        _config_protocol: 个性化配置协议
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        resource_name: str = "Neo4j图数据库"
    ):
        """
        初始化Neo4j连接配置

        Args:
            uri: Neo4j连接URI
            user: 用户名
            password: 密码
            database: 数据库名
            resource_name: 资源业务名称
        """
        self._resource_type = "neo4j_connection"
        self._resource_name = resource_name
        self._config_protocol: Dict[str, str] = {
            "uri": uri,
            "user": user,
            "password": password,
            "database": database
        }

    @property
    def resource_type(self) -> str:
        """获取资源类型标识"""
        return self._resource_type

    @property
    def resource_name(self) -> str:
        """获取资源业务名称"""
        return self._resource_name

    @property
    def config_protocol(self) -> Dict[str, str]:
        """获取个性化配置协议"""
        return self._config_protocol

    def to_dict(self) -> dict:
        """转换为字典"""
        config_protocol = {
            "uri_present": bool(self._config_protocol.get("uri")),
            "user_present": bool(self._config_protocol.get("user")),
            "password_present": bool(self._config_protocol.get("password")),
            "database_present": bool(self._config_protocol.get("database")),
        }
        return {
            "resource_type": self._resource_type,
            "resource_name": self._resource_name,
            "config_protocol": config_protocol
        }

    def validate(self) -> bool:
        """验证配置有效性"""
        for field_name in ("uri", "user", "password", "database"):
            field_value = self._config_protocol.get(field_name)
            if not isinstance(field_value, str) or not field_value.strip():
                return False
        return True
