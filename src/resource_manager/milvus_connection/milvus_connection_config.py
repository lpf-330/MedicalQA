# -*- coding: utf-8 -*-

"""Milvus连接资源配置类。"""

from typing import Dict

from src.resource_manager.resource_config import ResourceConfig


class MilvusConnectionConfig(ResourceConfig[Dict[str, str]]):

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        token: str = "",
        resource_name: str = "Milvus连接"
    ):
        self._resource_type = "milvus_connection"
        self._resource_name = resource_name
        self._config_protocol: Dict[str, str] = {
            "uri": uri,
            "user": user,
            "password": password,
            "token": token
        }

    @property
    def resource_type(self) -> str:
        return self._resource_type

    @property
    def resource_name(self) -> str:
        return self._resource_name

    @property
    def config_protocol(self) -> Dict[str, str]:
        return self._config_protocol

    def to_dict(self) -> dict:
        config_protocol = {
            "uri_present": bool(self._config_protocol.get("uri")),
            "user_present": bool(self._config_protocol.get("user")),
            "password_present": bool(self._config_protocol.get("password")),
            "token_present": bool(self._config_protocol.get("token")),
        }
        return {
            "resource_type": self._resource_type,
            "resource_name": self._resource_name,
            "config_protocol": config_protocol
        }

    def validate(self) -> bool:
        uri = self._config_protocol.get("uri")
        if not isinstance(uri, str) or not uri.strip():
            return False

        token = self._config_protocol.get("token")
        has_token = isinstance(token, str) and bool(token.strip())
        user = self._config_protocol.get("user")
        password = self._config_protocol.get("password")
        has_user_password = (
            isinstance(user, str) and bool(user.strip())
            and isinstance(password, str) and bool(password.strip())
        )
        return has_token or has_user_password
