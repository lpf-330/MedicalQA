# AI辅助生成：GLM-5，2026-04-15
# -*- coding: utf-8 -*-
"""
资源句柄类

封装资源实例，提供统一的资源访问接口。
"""

import logging
from typing import Optional, TYPE_CHECKING, Any

from .resource import Resource

if TYPE_CHECKING:
    from .resource_pool import ResourcePool

logger = logging.getLogger(__name__)


class SimpleResourceClient:
    """
    简单资源客户端
    
    提供基本的资源访问功能。
    """
    
    def __init__(self, resource: Resource):
        """
        初始化简单资源客户端
        
        Args:
            resource: 资源实例
        """
        self._resource = resource
    
    def get_resource_type(self) -> str:
        """获取资源类型"""
        return self._resource.get_type()
    
    def get_raw_resource(self) -> Resource:
        """获取原始资源实例"""
        return self._resource
    
    @property
    def resource(self) -> Resource:
        """获取资源实例"""
        return self._resource


class ResourceHandle:
    """
    资源句柄类
    
    封装资源实例，提供统一的资源访问接口。
    支持上下文管理器协议，可自动释放资源。
    
    属性：
        _resource_id: 资源ID
        _resource: 资源实例
        _pool: 所属资源池
        _released: 是否已释放
    """
    
    def __init__(self, resource_id: str, resource: Resource, pool: 'ResourcePool'):
        """
        初始化资源句柄
        
        Args:
            resource_id: 资源ID
            resource: 资源实例
            pool: 所属资源池
        """
        self._resource_id = resource_id
        self._resource = resource
        self._pool = pool
        self._released = False
        logger.debug(f"[ResourceHandle] 创建资源句柄: resource_id={resource_id[:8]}..., type={resource.get_type()}")
    
    @property
    def resource_id(self) -> str:
        """获取资源ID"""
        return self._resource_id
    
    @property
    def resource(self) -> Resource:
        """获取原始资源实例"""
        return self._resource
    
    @property
    def is_released(self) -> bool:
        """检查是否已释放"""
        return self._released
    
    def get_client(self) -> SimpleResourceClient:
        """
        获取资源客户端
        
        Returns:
            SimpleResourceClient: 资源客户端实例
        """
        if self._released:
            logger.error(f"[ResourceHandle] 获取客户端失败，资源已释放: resource_id={self._resource_id[:8]}...")
            raise RuntimeError("Resource has been released")
        logger.debug(f"[ResourceHandle] 获取资源客户端: resource_id={self._resource_id[:8]}...")
        return SimpleResourceClient(self._resource)
    
    def release(self) -> None:
        """释放资源"""
        if self._released:
            logger.debug(f"[ResourceHandle] 资源已释放，跳过: resource_id={self._resource_id[:8]}...")
            return
        
        logger.info(f"[ResourceHandle] 释放资源句柄: resource_id={self._resource_id[:8]}..., type={self._resource.get_type()}")
        self._pool.release(self)
        self._released = True
    
    def __enter__(self) -> 'ResourceHandle':
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器退出"""
        self.release()
