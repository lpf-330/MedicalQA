# -*- coding: utf-8 -*-
"""
全局资源配置类

包含所有资源的配置信息和资源池配置信息。
"""

from typing import Dict, Any

from .pool_config import PoolConfig


class GlobalConfig:
    """
    全局资源配置类
    
    包含所有资源的配置信息和资源池配置信息。
    
    属性：
        resource_configs: 资源配置总容器，key为资源类型唯一标识，value为ResourceConfig
        pool_configs: 资源池配置总容器，key为资源类型唯一标识，value为PoolConfig
    """
    
    def __init__(self):
        """初始化全局配置"""
        self._resource_configs: Dict[str, Any] = {}
        self._pool_configs: Dict[str, PoolConfig] = {}
    
    @property
    def resource_configs(self) -> Dict[str, Any]:
        """获取资源配置字典"""
        return self._resource_configs
    
    @property
    def pool_configs(self) -> Dict[str, PoolConfig]:
        """获取资源池配置字典"""
        return self._pool_configs
    
    def add_resource_config(self, resource_type: str, config: Any) -> None:
        """
        添加资源配置
        
        Args:
            resource_type: 资源类型唯一标识
            config: 资源配置对象
        """
        self._resource_configs[resource_type] = config
    
    def get_resource_config(self, resource_type: str) -> Any:
        """
        获取资源配置
        
        Args:
            resource_type: 资源类型唯一标识
            
        Returns:
            资源配置对象
        """
        return self._resource_configs.get(resource_type)
    
    def add_pool_config(self, resource_type: str, config: PoolConfig) -> None:
        """
        添加资源池配置
        
        Args:
            resource_type: 资源类型唯一标识
            config: 资源池配置对象
        """
        self._pool_configs[resource_type] = config
    
    def get_pool_config(self, resource_type: str) -> PoolConfig:
        """
        获取资源池配置
        
        Args:
            resource_type: 资源类型唯一标识
            
        Returns:
            PoolConfig: 资源池配置对象
        """
        return self._pool_configs.get(resource_type)
    
    def validate(self) -> bool:
        """
        验证配置有效性
        
        Returns:
            bool: 配置是否有效
        """
        for config in self._resource_configs.values():
            if hasattr(config, 'validate') and not config.validate():
                return False
        
        for config in self._pool_configs.values():
            if not config.validate():
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "resource_configs": {
                k: v.to_dict() if hasattr(v, 'to_dict') else str(v)
                for k, v in self._resource_configs.items()
            },
            "pool_configs": {
                k: v.to_dict() for k, v in self._pool_configs.items()
            }
        }
