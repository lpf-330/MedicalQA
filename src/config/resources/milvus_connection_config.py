# AI辅助生成：GLM-5，2026-04-15
# -*- coding: utf-8 -*-
"""
Milvus资源配置文件

定义Milvus向量数据库的连接参数和资源池配置。
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict

from src.config.base_config import BaseResourceConfig
from src.config.pool_config import PoolConfig


logger = logging.getLogger(__name__)


@dataclass(repr=False)
class MilvusConnectionConfig(BaseResourceConfig):
    """
    Milvus资源配置类

    属性：
        config_id: 配置ID（文件名作为唯一标识）
        resource_type: 资源类型
        uri: Milvus/Zilliz Cloud连接地址
        user: 用户名
        password: 密码
        token: 认证令牌
    """

    config_id: str = "milvus_connection_config"
    resource_type: str = "milvus_connection"
    uri: str = ""
    user: str = ""
    password: str = ""
    token: str = ""

    def validate(self) -> bool:
        """
        验证配置有效性

        Returns:
            bool: 配置是否有效
        """
        if not super().validate():
            return False

        if not isinstance(self.uri, str) or not self.uri.strip():
            logger.warning("[MilvusConnectionConfig.validate] Milvus URI 不能为空")
            return False
        has_token = isinstance(self.token, str) and bool(self.token.strip())
        has_user_password = (
            isinstance(self.user, str) and bool(self.user.strip())
            and isinstance(self.password, str) and bool(self.password.strip())
        )
        if not has_token and not has_user_password:
            logger.warning("[MilvusConnectionConfig.validate] Milvus 认证信息不能为空")
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        导出配置为字典

        Returns:
            Dict[str, Any]: 配置字典
        """
        base_dict = super().to_dict()
        base_dict.update({
            "uri_present": bool(self.uri),
            "user_present": bool(self.user),
            "password_present": bool(self.password),
            "token_present": bool(self.token),
        })
        return base_dict

    def __repr__(self) -> str:
        return (
            "MilvusConnectionConfig("
            f"config_id={self.config_id!r}, "
            f"resource_type={self.resource_type!r}, "
            f"uri_present={bool(self.uri)}, "
            f"user_present={bool(self.user)}, "
            f"password_present={bool(self.password)}, "
            f"token_present={bool(self.token)}"
            ")"
        )


resource_config = MilvusConnectionConfig()

resource_type = "milvus_connection"

pool_config = PoolConfig()
