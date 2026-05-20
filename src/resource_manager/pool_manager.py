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


class ResourcePoolManager:
    """
    资源池管理器
    
    管理所有资源池的创建、获取、销毁。
    
    属性：
        _pools: 资源池字典，key为"资源类型:配置ID"，value为资源池实例
        _factories: 工厂字典，key为资源类型，value为工厂实例
    """
    
    # 默认配置ID
    DEFAULT_CONFIG_ID = "default"
    
    def __init__(self):
        """初始化资源池管理器"""
        self._pools: Dict[str, ResourcePool] = {}
        self._factories: Dict[str, ResourceFactory] = {}
    
    def _get_pool_key(self, resource_type: str, config_id: Optional[str] = None) -> str:
        """
        生成资源池key
        
        Args:
            resource_type: 资源类型
            config_id: 配置ID，如果为None则使用默认值
            
        Returns:
            str: 资源池key，格式为"资源类型:配置ID"
        """
        actual_config_id = config_id if config_id is not None else self.DEFAULT_CONFIG_ID
        return f"{resource_type}:{actual_config_id}"
    
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
        resource_config: Any,
        config_id: Optional[str] = None
    ) -> ResourcePool:
        """
        创建资源池
        
        Args:
            resource_type: 资源类型
            pool_config: 资源池配置
            resource_config: 资源配置
            config_id: 配置ID，如果为None则使用默认值
            
        Returns:
            ResourcePool: 资源池实例
        """
        pool_key = self._get_pool_key(resource_type, config_id)
        
        # 如果pool已存在，直接返回（共享pool）
        if pool_key in self._pools:
            return self._pools[pool_key]
        
        factory = self._factories.get(resource_type)
        if factory is None:
            raise ValueError(f"Factory not registered for resource type: {resource_type}")
        
        pool = ResourcePool(resource_type, pool_config, factory, resource_config)
        self._pools[pool_key] = pool
        return pool
    
    def get_pool(self, resource_type: str, config_id: Optional[str] = None) -> Optional[ResourcePool]:
        """
        获取资源池
        
        Args:
            resource_type: 资源类型
            config_id: 配置ID，如果为None则使用默认值
            
        Returns:
            ResourcePool: 资源池实例
        """
        pool_key = self._get_pool_key(resource_type, config_id)
        return self._pools.get(pool_key)
    
    def has_pool(self, resource_type: str, config_id: Optional[str] = None) -> bool:
        """
        检查资源池是否存在
        
        Args:
            resource_type: 资源类型
            config_id: 配置ID，如果为None则使用默认值
            
        Returns:
            bool: 资源池是否存在
        """
        pool_key = self._get_pool_key(resource_type, config_id)
        return pool_key in self._pools
    
    def acquire(self, resource_type: str, config_id: Optional[str] = None, wait_ms: int = None) -> Optional[ResourceHandle]:
        """
        从资源池获取资源
        
        Args:
            resource_type: 资源类型
            config_id: 配置ID，如果为None则使用默认值
            wait_ms: 等待时间（毫秒）
            
        Returns:
            ResourceHandle: 资源句柄
        """
        pool_key = self._get_pool_key(resource_type, config_id)
        pool = self._pools.get(pool_key)
        if pool is None:
            raise ValueError(f"Pool not found for {pool_key}")
        
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
            Dict: 统计信息字典，key为"资源类型:配置ID"
        """
        stats = {}
        for pool_key, pool in self._pools.items():
            stats[pool_key] = {
                "idle_count": pool.idle_count,
                "active_count": pool.active_count,
                "total_count": pool.total_count
            }
        return stats
