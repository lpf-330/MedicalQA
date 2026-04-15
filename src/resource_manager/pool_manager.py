# -*- coding: utf-8 -*-
"""
资源池管理器

管理所有资源池的创建、获取、销毁。
"""

from typing import Dict, Optional, Any

from .resource_pool import ResourcePool
from .resource_factory import ResourceFactory
from .resource_handle import ResourceHandle
from src.config.pool_config import PoolConfig


class PoolManager:
    """
    资源池管理器
    
    管理所有资源池的创建、获取、销毁。
    
    属性：
        _pools: 资源池字典，key为资源类型，value为资源池实例
        _factories: 工厂字典，key为资源类型，value为工厂实例
    """
    
    def __init__(self):
        """初始化资源池管理器"""
        self._pools: Dict[str, ResourcePool] = {}
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
            ResourceFactory: 资源工厂实例
        """
        return self._factories.get(resource_type)
    
    def create_pool(
        self,
        resource_type: str,
        pool_config: PoolConfig,
        resource_config: Any
    ) -> ResourcePool:
        """
        创建资源池
        
        Args:
            resource_type: 资源类型
            pool_config: 资源池配置
            resource_config: 资源配置
            
        Returns:
            ResourcePool: 资源池实例
        """
        factory = self._factories.get(resource_type)
        if factory is None:
            raise ValueError(f"Factory not registered for resource type: {resource_type}")
        
        pool = ResourcePool(resource_type, pool_config, factory, resource_config)
        self._pools[resource_type] = pool
        return pool
    
    def get_pool(self, resource_type: str) -> Optional[ResourcePool]:
        """
        获取资源池
        
        Args:
            resource_type: 资源类型
            
        Returns:
            ResourcePool: 资源池实例
        """
        return self._pools.get(resource_type)
    
    def acquire(self, resource_type: str, wait_ms: int = None) -> Optional[ResourceHandle]:
        """
        从资源池获取资源
        
        Args:
            resource_type: 资源类型
            wait_ms: 等待时间（毫秒）
            
        Returns:
            ResourceHandle: 资源句柄
        """
        pool = self._pools.get(resource_type)
        if pool is None:
            raise ValueError(f"Pool not found for resource type: {resource_type}")
        
        return pool.acquire(wait_ms)
    
    def release(self, handle: ResourceHandle) -> None:
        """
        释放资源到资源池
        
        Args:
            handle: 资源句柄
        """
        handle.release()
    
    def destroy_all(self) -> None:
        """销毁所有资源池"""
        for pool in self._pools.values():
            pool.destroy_all()
        self._pools.clear()
    
    def get_pool_stats(self) -> Dict[str, Dict]:
        """
        获取所有资源池的统计信息
        
        Returns:
            Dict: 统计信息字典
        """
        stats = {}
        for resource_type, pool in self._pools.items():
            stats[resource_type] = {
                "idle_count": pool.idle_count,
                "active_count": pool.active_count,
                "total_count": pool.total_count
            }
        return stats
