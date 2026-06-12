"""
配置管理模块

提供项目配置的统一管理功能，包括：
- BaseConfig: 基础配置类
- LoggingConfig: 日志配置类
- ConfigLoader: 配置加载工具类
- PoolConfig: 资源池配置类
- GlobalConfig: 全局资源配置类
- ConfigManager: 统一配置管理器（唯一入口）
"""

from .base_config import BaseConfig
from .logging_config import LoggingConfig, LogLevel
from .config_loader import ConfigLoader
from .pool_config import PoolConfig
from .global_config import GlobalConfig
from .resource_config_loader import load_global_config
from .config_manager import ConfigManager, get_config_manager

__all__ = [
    'BaseConfig',
    'LoggingConfig',
    'LogLevel',
    'ConfigLoader',
    'PoolConfig',
    'GlobalConfig',
    'load_global_config',
    'ConfigManager',
    'get_config_manager',
]
