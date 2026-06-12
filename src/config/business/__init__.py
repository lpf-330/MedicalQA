# -*- coding: utf-8 -*-
"""
业务配置模块

存放各业务的配置文件，每个业务配置文件包含：
- 业务参数：定义业务的参数
- 资源配置引用：指定所需的资源配置文件名列表

文件名作为业务配置的唯一标识。
"""

from typing import Dict, Any, List
import importlib
from pathlib import Path


def load_business_config(config_name: str) -> Any:
    """
    加载指定的业务配置
    
    Args:
        config_name: 业务配置文件名（不含.py后缀）
        
    Returns:
        Any: 业务配置实例
    """
    module = importlib.import_module(f"src.config.business.{config_name}")
    
    from src.config.base_config import BusinessConfig
    
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, BusinessConfig) and attr is not BusinessConfig:
            return attr()
    
    raise ValueError(f"未找到业务配置类: {config_name}")


def get_all_business_configs() -> Dict[str, str]:
    """
    获取所有业务配置文件名
    
    Returns:
        Dict[str, str]: 业务配置文件名列表
    """
    business_dir = Path(__file__).parent
    configs = {}
    
    for file in business_dir.glob("*_config.py"):
        config_name = file.stem
        configs[config_name] = str(file)
    
    return configs


def get_required_resource_configs() -> List[str]:
    """
    获取所有业务配置所需的资源配置文件名（去重）
    
    Returns:
        List[str]: 资源配置文件名列表
    """
    required_resources = set()
    
    for config_name in get_all_business_configs():
        try:
            business_config = load_business_config(config_name)
            if hasattr(business_config, "resource_configs"):
                required_resources.update(business_config.resource_configs)
        except Exception as e:
            print(f"警告: 加载业务配置 {config_name} 失败: {e}")
    
    return list(required_resources)
