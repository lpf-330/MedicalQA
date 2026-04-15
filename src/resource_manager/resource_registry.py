# -*- coding: utf-8 -*-
"""
资源工厂注册表

管理所有资源工厂的注册和获取。
"""

from typing import Dict, Optional

from .resource_factory import ResourceFactory


class ResourceRegistry:
    """
    资源工厂注册表
    
    管理所有资源工厂的注册和获取。
    
    属性：
        _factories: 工厂字典，key为资源类型，value为工厂实例
    """
    
    def __init__(self):
        """初始化资源工厂注册表"""
        self._factories: Dict[str, ResourceFactory] = {}
    
    def register_factory(self, resource_type: str, factory: ResourceFactory) -> None:
        """
        注册资源工厂
        
        Args:
            resource_type: 资源类型唯一标识
            factory: 资源工厂实例
        """
        self._factories[resource_type] = factory
    
    def get_factory(self, resource_type: str) -> Optional[ResourceFactory]:
        """
        获取资源工厂
        
        Args:
            resource_type: 资源类型唯一标识
            
        Returns:
            ResourceFactory: 资源工厂实例，如果不存在返回None
        """
        return self._factories.get(resource_type)
    
    def has_factory(self, resource_type: str) -> bool:
        """
        检查是否已注册工厂
        
        Args:
            resource_type: 资源类型唯一标识
            
        Returns:
            bool: 是否已注册
        """
        return resource_type in self._factories
    
    def unregister_factory(self, resource_type: str) -> None:
        """
        注销资源工厂
        
        Args:
            resource_type: 资源类型唯一标识
        """
        if resource_type in self._factories:
            del self._factories[resource_type]
    
    def get_all_resource_types(self) -> list:
        """获取所有已注册的资源类型"""
        return list(self._factories.keys())
