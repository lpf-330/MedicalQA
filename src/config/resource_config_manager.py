# -*- coding: utf-8 -*-
"""
资源配置管理器

统一管理所有资源配置，包括数据库、模型、资源池等配置。
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import logging

from .base_config import BaseConfig
from .database_config import DatabaseConfig, ModelConfig
from .pool_config import PoolConfig
from .logging_config import LoggingConfig
from .global_config import GlobalConfig

logger = logging.getLogger(__name__)


class ResourceConfigManager(BaseConfig):
    """
    资源配置管理器
    
    统一管理所有资源配置，提供配置加载、验证、导出等功能。
    
    属性：
        database_config: 数据库配置
        model_config: 模型配置
        logging_config: 日志配置
        pool_configs: 资源池配置字典
    """
    
    _instance: Optional['ResourceConfigManager'] = None
    
    def __new__(cls, *args, **kwargs) -> 'ResourceConfigManager':
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        project_name: str = "MedicalQA",
        project_version: str = "1.0.0",
        environment: str = "development",
        debug: bool = False,
        config_path: Optional[str] = None
    ):
        """
        初始化资源配置管理器
        
        Args:
            project_name: 项目名称
            project_version: 项目版本
            environment: 运行环境
            debug: 是否开启调试模式
            config_path: 配置文件路径
        """
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        super().__init__(
            project_name=project_name,
            project_version=project_version,
            environment=environment,
            debug=debug,
            config_path=config_path
        )
        
        self._database_config: Optional[DatabaseConfig] = None
        self._model_config: Optional[ModelConfig] = None
        self._logging_config: Optional[LoggingConfig] = None
        self._pool_configs: Dict[str, PoolConfig] = {}
        
        self._initialized = True
    
    @property
    def database_config(self) -> DatabaseConfig:
        """获取数据库配置"""
        if self._database_config is None:
            self._database_config = DatabaseConfig()
        return self._database_config
    
    @property
    def model_config(self) -> ModelConfig:
        """获取模型配置"""
        if self._model_config is None:
            self._model_config = ModelConfig()
        return self._model_config
    
    @property
    def logging_config(self) -> LoggingConfig:
        """获取日志配置"""
        if self._logging_config is None:
            self._logging_config = LoggingConfig()
        return self._logging_config
    
    @property
    def pool_configs(self) -> Dict[str, PoolConfig]:
        """获取所有资源池配置"""
        return self._pool_configs
    
    def get_pool_config(self, resource_type: str) -> Optional[PoolConfig]:
        """
        获取指定资源类型的资源池配置
        
        Args:
            resource_type: 资源类型
            
        Returns:
            PoolConfig: 资源池配置，如果不存在返回None
        """
        return self._pool_configs.get(resource_type)
    
    def set_database_config(self, config: DatabaseConfig) -> None:
        """
        设置数据库配置
        
        Args:
            config: 数据库配置实例
        """
        self._database_config = config
        logger.info(f"[ResourceConfigManager] 数据库配置已更新")
    
    def set_model_config(self, config: ModelConfig) -> None:
        """
        设置模型配置
        
        Args:
            config: 模型配置实例
        """
        self._model_config = config
        logger.info(f"[ResourceConfigManager] 模型配置已更新")
    
    def set_logging_config(self, config: LoggingConfig) -> None:
        """
        设置日志配置
        
        Args:
            config: 日志配置实例
        """
        self._logging_config = config
        logger.info(f"[ResourceConfigManager] 日志配置已更新")
    
    def add_pool_config(self, resource_type: str, config: PoolConfig) -> None:
        """
        添加资源池配置
        
        Args:
            resource_type: 资源类型
            config: 资源池配置实例
        """
        self._pool_configs[resource_type] = config
        logger.info(f"[ResourceConfigManager] 资源池配置已添加: {resource_type}")
    
    def load_all_configs(self) -> None:
        """
        加载所有配置
        
        按照配置加载顺序依次加载：
        1. 日志配置
        2. 数据库配置
        3. 模型配置
        4. 资源池配置
        """
        logger.info("[ResourceConfigManager] 开始加载所有配置...")
        
        self._logging_config = LoggingConfig()
        logger.info("[ResourceConfigManager] 日志配置加载完成")
        
        self._database_config = DatabaseConfig()
        logger.info("[ResourceConfigManager] 数据库配置加载完成")
        
        self._model_config = ModelConfig()
        logger.info("[ResourceConfigManager] 模型配置加载完成")
        
        self._load_default_pool_configs()
        logger.info("[ResourceConfigManager] 资源池配置加载完成")
        
        logger.info("[ResourceConfigManager] 所有配置加载完成")
    
    def _load_default_pool_configs(self) -> None:
        """加载默认资源池配置"""
        self._pool_configs["neo4j_connection"] = PoolConfig(
            max_size=10,
            min_idle=2,
            idle_timeout=300000,
            max_wait_time=5000
        )
        
        self._pool_configs["vllm_model"] = PoolConfig(
            max_size=1,
            min_idle=1,
            idle_timeout=600000,
            max_wait_time=30000
        )
    
    def validate(self) -> bool:
        """
        验证所有配置有效性
        
        Returns:
            bool: 所有配置是否有效
        """
        errors = []
        
        if self._logging_config and not self._logging_config.validate():
            errors.append("日志配置验证失败")
        
        if self._database_config and not self._database_config.validate():
            errors.append("数据库配置验证失败")
        
        if self._model_config and not self._model_config.validate():
            errors.append("模型配置验证失败")
        
        for resource_type, pool_config in self._pool_configs.items():
            if pool_config.max_size < 1:
                errors.append(f"资源池 {resource_type} max_size 必须 >= 1")
            if pool_config.min_idle < 0:
                errors.append(f"资源池 {resource_type} min_idle 不能为负数")
        
        if errors:
            for error in errors:
                logger.error(f"[ResourceConfigManager] 配置验证失败: {error}")
            return False
        
        logger.info("[ResourceConfigManager] 所有配置验证通过")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        导出所有配置为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        config_dict = {
            "project_name": self._project_name,
            "project_version": self._project_version,
            "environment": self._environment,
            "debug": self._debug,
            "config_path": self._config_path,
        }
        
        if self._logging_config:
            config_dict["logging"] = self._logging_config.to_dict()
        
        if self._database_config:
            config_dict["database"] = self._database_config.to_dict()
        
        if self._model_config:
            config_dict["model"] = self._model_config.to_dict()
        
        if self._pool_configs:
            config_dict["pools"] = {
                k: {"max_size": v.max_size, "min_idle": v.min_idle, 
                    "idle_timeout": v.idle_timeout, "max_wait_time": v.max_wait_time}
                for k, v in self._pool_configs.items()
            }
        
        return config_dict
    
    def to_global_config(self) -> GlobalConfig:
        """
        转换为GlobalConfig实例
        
        Returns:
            GlobalConfig: 全局资源配置实例
        """
        global_config = GlobalConfig()
        
        if self._database_config:
            from src.resource_manager.neo4j_connection import Neo4jConnectionConfig
            
            neo4j_config = Neo4jConnectionConfig(
                uri=self._database_config.neo4j_uri,
                user=self._database_config.neo4j_user,
                password=self._database_config.neo4j_password,
                database=self._database_config.neo4j_database
            )
            global_config.add_resource_config("neo4j_config", neo4j_config)
            global_config.add_pool_config("neo4j_config", self._pool_configs.get("neo4j_connection", PoolConfig()))
        
        if self._model_config:
            from src.resource_manager.vllm_model import VLLMModelConfig
            
            vllm_config = VLLMModelConfig(
                model_path=self._model_config.model_path,
                model_name=self._model_config.model_name,
                tensor_parallel_size=self._model_config.tensor_parallel_size,
                max_model_len=self._model_config.max_model_len,
                gpu_memory_utilization=self._model_config.gpu_memory_utilization
            )
            global_config.add_resource_config("vllm_config", vllm_config)
            global_config.add_pool_config("vllm_config", self._pool_configs.get("vllm_model", PoolConfig()))
        
        return global_config
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"project_name='{self._project_name}', "
            f"environment='{self._environment}', "
            f"database_config={'已配置' if self._database_config else '未配置'}, "
            f"model_config={'已配置' if self._model_config else '未配置'}, "
            f"pool_configs={len(self._pool_configs)})"
        )


_config_manager: Optional[ResourceConfigManager] = None


def get_config_manager() -> ResourceConfigManager:
    """
    获取资源配置管理器实例（单例模式）
    
    Returns:
        ResourceConfigManager: 资源配置管理器实例
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ResourceConfigManager()
        _config_manager.load_all_configs()
    return _config_manager


def load_global_config_from_manager() -> GlobalConfig:
    """
    从资源配置管理器加载全局配置
    
    Returns:
        GlobalConfig: 全局资源配置实例
    """
    manager = get_config_manager()
    return manager.to_global_config()
