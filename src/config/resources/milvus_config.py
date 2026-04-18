# -*- coding: utf-8 -*-
"""
Milvus资源配置文件

定义Milvus向量数据库的连接参数和资源池配置。
"""

from dataclasses import dataclass, field
from typing import Any, Dict

from src.config.base_config import BaseResourceConfig
from src.config.pool_config import PoolConfig


@dataclass
class MilvusResourceConfig(BaseResourceConfig):
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

    config_id: str = "milvus_config"
    resource_type: str = "milvus_connection"
    uri: str = "https://in03-1c39e13a65460bf.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn"
    user: str = "db_1c39e13a65460bf"
    password: str = "Jk1*Xv*gJCv0}7Gg"
    token: str = "321a3d34b440e76d0e7d6bc5c4c40524aab8fee95cbd016f818b8e8285b3eb1258805be86fb100ad38c7b9fdcb2e33cf58e931e0"

    def validate(self) -> bool:
        """
        验证配置有效性

        Returns:
            bool: 配置是否有效
        """
        if not super().validate():
            return False

        if not self.uri:
            print("警告: Milvus URI 不能为空")
            return False
        if not self.user:
            print("警告: Milvus 用户名不能为空")
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
            "token": "***" if self.token else "",
        })
        return base_dict


resource_config = MilvusResourceConfig()

resource_type = "milvus_connection"

pool_config = PoolConfig(
    max_size=10,
    min_idle=2,
    idle_timeout=300000,
    max_wait_time=5000
)
