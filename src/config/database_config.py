# -*- coding: utf-8 -*-
"""
数据库配置类

提供Neo4j和Milvus数据库的连接配置。
"""

from typing import Any, Dict, Optional
from pathlib import Path

from .base_config import BaseConfig


class DatabaseConfig(BaseConfig):
    """
    数据库配置类
    
    管理Neo4j图数据库和Milvus向量数据库的连接配置。
    
    属性：
        neo4j_uri: Neo4j数据库连接URI
        neo4j_user: Neo4j用户名
        neo4j_password: Neo4j密码
        neo4j_database: Neo4j数据库名
        milvus_uri: Milvus数据库连接URI
        milvus_user: Milvus用户名
        milvus_password: Milvus密码
        milvus_token: Milvus API Token
    """
    
    def __init__(
        self,
        neo4j_uri: str = "neo4j+s://627658bb.databases.neo4j.io",
        neo4j_user: str = "627658bb",
        neo4j_password: str = "35No69NaLaoasxQqW-JhcjbxgQjeY_WzUVGHYtKWeNo",
        neo4j_database: str = "neo4j",
        milvus_uri: str = "https://in03-1c39e13a65460bf.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn",
        milvus_user: str = "db_1c39e13a65460bf",
        milvus_password: str = "Jk1*Xv*gJCv0}7Gg",
        milvus_token: str = "321a3d34b440e76d0e7d6bc5c4c40524aab8fee95cbd016f818b8e8285b3eb1258805be86fb100ad38c7b9fdcb2e33cf58e931e0",
        **kwargs
    ):
        """
        初始化数据库配置
        
        Args:
            neo4j_uri: Neo4j数据库连接URI
            neo4j_user: Neo4j用户名
            neo4j_password: Neo4j密码
            neo4j_database: Neo4j数据库名
            milvus_uri: Milvus数据库连接URI
            milvus_user: Milvus用户名
            milvus_password: Milvus密码
            milvus_token: Milvus API Token
            **kwargs: 其他基础配置参数
        """
        super().__init__(**kwargs)
        
        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password
        self._neo4j_database = neo4j_database
        
        self._milvus_uri = milvus_uri
        self._milvus_user = milvus_user
        self._milvus_password = milvus_password
        self._milvus_token = milvus_token
    
    @property
    def neo4j_uri(self) -> str:
        """获取Neo4j连接URI"""
        return self._neo4j_uri
    
    @property
    def neo4j_user(self) -> str:
        """获取Neo4j用户名"""
        return self._neo4j_user
    
    @property
    def neo4j_password(self) -> str:
        """获取Neo4j密码"""
        return self._neo4j_password
    
    @property
    def neo4j_database(self) -> str:
        """获取Neo4j数据库名"""
        return self._neo4j_database
    
    @property
    def neo4j_config(self) -> Dict[str, str]:
        """获取Neo4j完整配置"""
        return {
            "uri": self._neo4j_uri,
            "user": self._neo4j_user,
            "password": self._neo4j_password,
            "database": self._neo4j_database
        }
    
    @property
    def milvus_uri(self) -> str:
        """获取Milvus连接URI"""
        return self._milvus_uri
    
    @property
    def milvus_user(self) -> str:
        """获取Milvus用户名"""
        return self._milvus_user
    
    @property
    def milvus_password(self) -> str:
        """获取Milvus密码"""
        return self._milvus_password
    
    @property
    def milvus_token(self) -> str:
        """获取Milvus API Token"""
        return self._milvus_token
    
    @property
    def milvus_config(self) -> Dict[str, str]:
        """获取Milvus完整配置"""
        return {
            "uri": self._milvus_uri,
            "user": self._milvus_user,
            "password": self._milvus_password,
            "token": self._milvus_token
        }
    
    def validate(self) -> bool:
        """
        验证配置有效性
        
        Returns:
            bool: 配置是否有效
        """
        if not self._neo4j_uri:
            return False
        if not self._neo4j_user:
            return False
        if not self._neo4j_password:
            return False
        if not self._milvus_uri:
            return False
        if not self._milvus_user:
            return False
        if not self._milvus_password:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        导出配置为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        return {
            "project_name": self._project_name,
            "project_version": self._project_version,
            "environment": self._environment,
            "debug": self._debug,
            "neo4j": self.neo4j_config,
            "milvus": self.milvus_config
        }


class ModelConfig(BaseConfig):
    """
    模型配置类
    
    管理模型路径和相关配置。
    
    属性：
        model_path: 模型路径
        model_name: 模型名称
        tensor_parallel_size: 张量并行大小
        max_model_len: 最大模型长度
        gpu_memory_utilization: GPU内存利用率
    """
    
    def __init__(
        self,
        model_path: str = "",
        model_name: str = "Qwen3-4B-Instruct-2507",
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.9,
        **kwargs
    ):
        """
        初始化模型配置
        
        Args:
            model_path: 模型路径
            model_name: 模型名称
            tensor_parallel_size: 张量并行大小
            max_model_len: 最大模型长度
            gpu_memory_utilization: GPU内存利用率
            **kwargs: 其他基础配置参数
        """
        super().__init__(**kwargs)
        
        self._model_name = model_name
        self._tensor_parallel_size = tensor_parallel_size
        self._max_model_len = max_model_len
        self._gpu_memory_utilization = gpu_memory_utilization
        
        if model_path:
            self._model_path = model_path
        else:
            self._model_path = str(self._get_default_model_path())
    
    def _get_default_model_path(self) -> Path:
        """获取默认模型路径"""
        return self._project_root / "base_models" / self._model_name
    
    @property
    def model_path(self) -> str:
        """获取模型路径"""
        return self._model_path
    
    @property
    def model_name(self) -> str:
        """获取模型名称"""
        return self._model_name
    
    @property
    def tensor_parallel_size(self) -> int:
        """获取张量并行大小"""
        return self._tensor_parallel_size
    
    @property
    def max_model_len(self) -> int:
        """获取最大模型长度"""
        return self._max_model_len
    
    @property
    def gpu_memory_utilization(self) -> float:
        """获取GPU内存利用率"""
        return self._gpu_memory_utilization
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """获取模型完整配置"""
        return {
            "model_path": self._model_path,
            "model_name": self._model_name,
            "tensor_parallel_size": self._tensor_parallel_size,
            "max_model_len": self._max_model_len,
            "gpu_memory_utilization": self._gpu_memory_utilization
        }
    
    def validate(self) -> bool:
        """
        验证配置有效性
        
        Returns:
            bool: 配置是否有效
        """
        if not self._model_path:
            return False
        if not Path(self._model_path).exists():
            return False
        if self._tensor_parallel_size < 1:
            return False
        if self._gpu_memory_utilization <= 0 or self._gpu_memory_utilization > 1:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        导出配置为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        return {
            "project_name": self._project_name,
            "project_version": self._project_version,
            "environment": self._environment,
            "debug": self._debug,
            "model": self.model_config
        }


# 全局配置实例
_database_config: Optional[DatabaseConfig] = None
_model_config: Optional[ModelConfig] = None


def get_database_config() -> DatabaseConfig:
    """
    获取数据库配置实例（单例模式）
    
    Returns:
        DatabaseConfig: 数据库配置实例
    """
    global _database_config
    if _database_config is None:
        _database_config = DatabaseConfig()
    return _database_config


def get_model_config() -> ModelConfig:
    """
    获取模型配置实例（单例模式）
    
    Returns:
        ModelConfig: 模型配置实例
    """
    global _model_config
    if _model_config is None:
        _model_config = ModelConfig()
    return _model_config
