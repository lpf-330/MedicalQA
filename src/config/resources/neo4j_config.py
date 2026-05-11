# AI辅助生成：GLM-5，2026-04-15
# -*- coding: utf-8 -*-
"""
Neo4j资源配置文件

定义Neo4j数据库的连接参数和资源池配置。
"""

from dataclasses import dataclass, field
from typing import Any, Dict

from src.config.base_config import BaseResourceConfig
from src.config.pool_config import PoolConfig


@dataclass
class Neo4jResourceConfig(BaseResourceConfig):
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
    
    config_id: str = "neo4j_config"
    resource_type: str = "neo4j_connection"
    uri: str = "neo4j+s://627658bb.databases.neo4j.io"
    user: str = "627658bb"
    password: str = "35No69NaLaoasxQqW-JhcjbxgQjeY_WzUVGHYtKWeNo"
    database: str = "neo4j"
    
    def validate(self) -> bool:
        """
        验证配置有效性
        
        Returns:
            bool: 配置是否有效
        """
        if not super().validate():
            return False
        
        if not self.uri:
            print("警告: Neo4j URI 不能为空")
            return False
        if not self.user:
            print("警告: Neo4j 用户名不能为空")
            return False
        if not self.password:
            print("警告: Neo4j 密码不能为空")
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
            "uri": self.uri,
            "user": self.user,
            "password": "***",
            "database": self.database,
        })
        return base_dict


resource_config = Neo4jResourceConfig()

resource_type = "neo4j_connection"

pool_config = PoolConfig(
    max_size=10,
    min_idle=2,
    idle_timeout=300000,
    max_wait_time=5000
)
