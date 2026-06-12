# AI辅助生成：GLM-5，2026-04-15
# -*- coding: utf-8 -*-
"""
Neo4j资源配置文件

定义Neo4j数据库的连接参数和资源池配置。
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict

from src.config.base_config import BaseResourceConfig
from src.config.pool_config import PoolConfig


logger = logging.getLogger(__name__)


@dataclass(repr=False)
class Neo4jConnectionConfig(BaseResourceConfig):
    """
    Neo4j资源配置类
    
    属性：
        config_id: 配置ID（文件名作为唯一标识）
        resource_type: 资源类型
        uri: Neo4j连接URI
        user: 用户名
        password: 密码
        database: 数据库名称
    """
    
    config_id: str = "neo4j_connection_config"
    resource_type: str = "neo4j_connection"
    uri: str = ""
    user: str = ""
    password: str = ""
    database: str = "neo4j"
    
    def validate(self) -> bool:
        """
        验证配置有效性
        
        Returns:
            bool: 配置是否有效
        """
        if not super().validate():
            return False
        
        if not isinstance(self.uri, str) or not self.uri.strip():
            logger.warning("[Neo4jConnectionConfig.validate] Neo4j URI 不能为空")
            return False
        if not isinstance(self.user, str) or not self.user.strip():
            logger.warning("[Neo4jConnectionConfig.validate] Neo4j 用户名不能为空")
            return False
        if not isinstance(self.password, str) or not self.password.strip():
            logger.warning("[Neo4jConnectionConfig.validate] Neo4j 密码不能为空")
            return False
        if not isinstance(self.database, str) or not self.database.strip():
            logger.warning("[Neo4jConnectionConfig.validate] Neo4j 数据库名不能为空")
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
            "database_present": bool(self.database),
        })
        return base_dict

    def __repr__(self) -> str:
        return (
            "Neo4jConnectionConfig("
            f"config_id={self.config_id!r}, "
            f"resource_type={self.resource_type!r}, "
            f"uri_present={bool(self.uri)}, "
            f"user_present={bool(self.user)}, "
            f"password_present={bool(self.password)}, "
            f"database_present={bool(self.database)}"
            ")"
        )


resource_config = Neo4jConnectionConfig()

resource_type = "neo4j_connection"

pool_config = PoolConfig()
