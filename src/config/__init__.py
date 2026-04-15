"""
配置管理模块

提供项目配置的统一管理功能，包括：
- BaseConfig: 基础配置类
- LoggingConfig: 日志配置类
- ConfigLoader: 配置加载工具类
- DatabaseConfig: 数据库配置类
- ModelConfig: 模型配置类
- PoolConfig: 资源池配置类
- GlobalConfig: 全局资源配置类
- ResourceConfigLoader: 资源配置加载器
- ResourceConfigManager: 资源配置管理器（统一配置管理）
"""

from .base_config import BaseConfig
from .logging_config import LoggingConfig, LogLevel
from .config_loader import ConfigLoader
from .database_config import DatabaseConfig, ModelConfig, get_database_config, get_model_config
from .pool_config import PoolConfig
from .global_config import GlobalConfig
from .resource_config_loader import load_global_config, load_global_config_legacy
from .resource_config_manager import ResourceConfigManager, get_config_manager, load_global_config_from_manager

__all__ = [
    'BaseConfig',
    'LoggingConfig',
    'LogLevel',
    'ConfigLoader',
    'DatabaseConfig',
    'ModelConfig',
    'get_database_config',
    'get_model_config',
    'PoolConfig',
    'GlobalConfig',
    'load_global_config',
    'load_global_config_legacy',
    'ResourceConfigManager',
    'get_config_manager',
    'load_global_config_from_manager',
]
