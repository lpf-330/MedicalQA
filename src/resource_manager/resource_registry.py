# AI辅助生成：GLM-5，2026-04-15
# -*- coding: utf-8 -*-
"""
资源工厂注册表

管理所有资源工厂的注册和获取。
"""

import logging
from typing import Dict, Optional

from .resource_factory import ResourceFactory

logger = logging.getLogger(__name__)


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
        logger.info("[ResourceRegistry.__init__] 资源工厂注册表初始化完成")
    
    def register_factory(self, resource_type: str, factory: ResourceFactory) -> None:
        """
        注册资源工厂
        
        Args:
            resource_type: 资源类型唯一标识
            factory: 资源工厂实例
        """
        self._factories[resource_type] = factory
        logger.info(f"[ResourceRegistry.register_factory] 工厂已注册: resource_type={resource_type}, factory_class={type(factory).__name__}")
    
    def get_factory(self, resource_type: str) -> Optional[ResourceFactory]:
        """
        获取资源工厂
        
        Args:
            resource_type: 资源类型唯一标识
            
        Returns:
            ResourceFactory: 资源工厂实例，如果不存在返回None
        """
        factory = self._factories.get(resource_type)
        if factory is None:
            logger.warning(f"[ResourceRegistry.get_factory] 工厂未注册: resource_type={resource_type}")
        else:
            logger.debug(f"[ResourceRegistry.get_factory] 获取工厂: resource_type={resource_type}, factory_class={type(factory).__name__}")
        return factory
    
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
            logger.info(f"[ResourceRegistry.unregister_factory] 工厂已注销: resource_type={resource_type}")
    
    def get_all_resource_types(self) -> list:
        """获取所有已注册的资源类型"""
        types = list(self._factories.keys())
        logger.debug(f"[ResourceRegistry.get_all_resource_types] 已注册资源类型: {types}")
        return types
