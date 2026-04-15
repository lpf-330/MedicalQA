# -*- coding: utf-8 -*-
"""
资源配置模块

存放各资源的配置文件，每个资源配置文件包含：
- 资源配置类：定义资源的连接参数
- 资源池配置：定义资源池的大小、超时等参数

文件名作为资源配置的唯一标识。
"""

from typing import Dict, Any, Type
import importlib
import os
from pathlib import Path

from src.config.pool_config import PoolConfig


def load_resource_config(config_name: str) -> Dict[str, Any]:
    """
    加载指定的资源配置
    
    Args:
        config_name: 资源配置文件名（不含.py后缀）
        
    Returns:
        Dict[str, Any]: 包含resource_config和pool_config的字典
    """
    module = importlib.import_module(f"src.config.resources.{config_name}")
    
    return {
        "resource_config": getattr(module, "resource_config", None),
        "pool_config": getattr(module, "pool_config", None),
        "resource_type": getattr(module, "resource_type", config_name),
    }


def get_all_resource_configs() -> Dict[str, str]:
    """
    获取所有资源配置文件名
    
    Returns:
        Dict[str, str]: 资源配置文件名列表
    """
    resources_dir = Path(__file__).parent
    configs = {}
    
    for file in resources_dir.glob("*_config.py"):
        config_name = file.stem
        configs[config_name] = str(file)
    
    return configs
