# AI辅助生成：GLM-5，2026-04-15
# -*- coding: utf-8 -*-
"""
资源句柄类

封装资源实例，提供统一的资源访问接口。
"""

import logging
from typing import TYPE_CHECKING, Optional

from .resource import Resource
from .resource_client import ResourceClient, ModelResourceClient

if TYPE_CHECKING:
    from .global_resource_manager import GlobalResourceManager
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
    
    # 资源类型到客户端类的注册表
    _client_registry: dict = {}
    
    def __init__(self, resource_id: str, resource: Resource, pool: 'ResourcePool', manager_ref: Optional['GlobalResourceManager'] = None):
        """
        初始化资源句柄

        Args:
            resource_id: 资源ID
            resource: 资源实例
            pool: 所属资源池
            manager_ref: GlobalResourceManager引用，用于资源生命周期管理
        """
        self._resource_id = resource_id
        self._resource = resource
        self._pool = pool
        self._manager_ref = manager_ref
        self._released = False
        logger.debug(f"[ResourceHandle] 创建资源句柄: resource_id={resource_id[:8]}..., type={resource.get_type()}")
    
    @classmethod
    def register_client(cls, resource_type: str, client_class: type) -> None:
        """
        注册资源类型对应的客户端类
        
        Args:
            resource_type: 资源类型标识
            client_class: 客户端类（必须实现ResourceClient接口）
        """
        cls._client_registry[resource_type] = client_class
        logger.debug(f"[ResourceHandle] 注册资源客户端: resource_type={resource_type}, client_class={client_class.__name__}")
    
    @property
    def resource_id(self) -> str:
        """获取资源ID"""
        return self._resource_id
    
    @property
    def resource_type(self) -> str:
        """获取资源类型"""
        return self._resource.get_type()
    
    @property
    def resource(self) -> Resource:
        """获取原始资源实例"""
        return self._resource
    
    @property
    def client(self) -> ResourceClient:
        """
        获取资源客户端

        根据资源类型自动返回对应的客户端实例，
        与get_client()行为一致，提供属性访问方式。
        """
        return self.get_client()
    
    @property
    def manager_ref(self) -> Optional['GlobalResourceManager']:
        """
        获取GlobalResourceManager引用

        与架构设计文档中的manager_ref属性对应，
        用于关联资源生命周期管理，支撑资源释放的回调与调度。
        """
        return self._manager_ref
    
    @property
    def is_released(self) -> bool:
        """检查是否已释放"""
        return self._released
    
    def get_client(self) -> ResourceClient:
        """
        获取资源客户端
        
        根据资源类型自动创建对应的客户端实例。
        通过客户端注册表查找对应的客户端类，实现业务层与具体客户端类的解耦。
        
        Returns:
            ResourceClient: 资源客户端实例（具体类型取决于资源类型）
            
        Raises:
            RuntimeError: 资源已释放或未注册对应客户端类时抛出
        """
        if self._released:
            logger.error(f"[ResourceHandle] 获取客户端失败，资源已释放: resource_id={self._resource_id[:8]}...")
            raise RuntimeError("Resource has been released")
        
        resource_type = self._resource.get_type()
        client_class = self._client_registry.get(resource_type)
        
        if client_class is not None:
            logger.debug(f"[ResourceHandle] 获取资源客户端: resource_id={self._resource_id[:8]}..., type={resource_type}, client={client_class.__name__}")
            return client_class(self._resource)
        
        # 未注册客户端类时，返回SimpleResourceClient作为默认客户端
        logger.debug(f"[ResourceHandle] 获取资源客户端(默认): resource_id={self._resource_id[:8]}..., type={resource_type}")
        return SimpleResourceClient(self._resource)
    
    def get_model_client(self) -> ModelResourceClient:
        """
        获取模型资源客户端
        
        便捷方法，返回ModelResourceClient接口类型的客户端实例。
        业务层通过此方法获取模型客户端，无需直接依赖具体实现类。
        
        Returns:
            ModelResourceClient: 模型资源客户端实例
            
        Raises:
            RuntimeError: 资源已释放或客户端不是ModelResourceClient类型时抛出
        """
        client = self.get_client()
        if not isinstance(client, ModelResourceClient):
            raise RuntimeError(
                f"Resource type '{self._resource.get_type()}' does not provide a ModelResourceClient, "
                f"got {type(client).__name__}"
            )
        return client
    
    def release(self) -> None:
        """释放资源"""
        if self._released:
            logger.debug(f"[ResourceHandle] 资源已释放，跳过: resource_id={self._resource_id[:8]}...")
            return
        
        logger.info(f"[ResourceHandle] 释放资源句柄: resource_id={self._resource_id[:8]}..., type={self._resource.get_type()}")
        self._pool.release_to_pool(self)
        self._released = True
    
    def __enter__(self) -> 'ResourceHandle':
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器退出"""
        self.release()
