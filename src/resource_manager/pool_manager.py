# -*- coding: utf-8 -*-
"""
资源池管理器

管理所有资源池的创建、获取、销毁。
"""

import logging
from typing import Dict, Optional, Any

from .resource_pool import ResourcePool
from .resource_factory import ResourceFactory
from .resource_handle import ResourceHandle
from src.config.pool_config import PoolConfig
from src.utils.logger import log_arch_event

logger = logging.getLogger(__name__)


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
        logger.info("[ResourcePoolManager.__init__] 资源池管理器初始化完成")
    
    def _get_pool_key(self, resource_type: str, config_id: Optional[str] = None) -> str:
        """
        生成资源池key

        Args:
            resource_type: 资源类型（字符串或str枚举）
            config_id: 配置ID，如果为None则使用默认值（字符串或str枚举）

        Returns:
            str: 资源池key，格式为"资源类型:配置ID"
        """
        from enum import Enum
        rt = resource_type.value if isinstance(resource_type, Enum) else resource_type
        actual_config_id = config_id if config_id is not None else self.DEFAULT_CONFIG_ID
        cid = actual_config_id.value if isinstance(actual_config_id, Enum) else actual_config_id
        return f"{rt}:{cid}"
    
    def register_factory(self, resource_type: str, factory: ResourceFactory) -> None:
        """
        注册资源工厂
        
        Args:
            resource_type: 资源类型唯一标识
            factory: 资源工厂实例
        """
        self._factories[resource_type] = factory
        logger.info(f"[ResourcePoolManager.register_factory] 工厂已注册: resource_type={resource_type}, factory_class={type(factory).__name__}")
    
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
        
        if pool_key in self._pools:
            logger.info(f"[ResourcePoolManager.create_pool] 资源池已存在，共享: pool_key={pool_key}")
            logger.info(f"[CONFIG_SHARE] pool_key={pool_key}, is_shared=True, resource_type={resource_type}")
            return self._pools[pool_key]
        
        factory = self._factories.get(resource_type)
        if factory is None:
            logger.error(f"[ResourcePoolManager.create_pool] 工厂未注册: resource_type={resource_type}")
            raise ValueError(f"Factory not registered for resource type: {resource_type}")
        
        pool = ResourcePool(resource_type, pool_config, factory, resource_config)
        self._pools[pool_key] = pool
        log_arch_event(logger, component="ResourcePoolManager", stage="RESOURCE_POOL", event="pool_created", status="success", design_id="ARCH-6.2", pool_key=pool_key, resource_type=resource_type, max_size=pool_config.max_size)
        logger.info(f"[ResourcePoolManager.create_pool] 资源池已创建: pool_key={pool_key}, max_size={pool_config.max_size}, min_idle={pool_config.min_idle}")
        logger.info(f"[CONFIG_SHARE] pool_key={pool_key}, is_shared=False, resource_type={resource_type}, max_size={pool_config.max_size}, min_idle={pool_config.min_idle}")
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
    
    def acquire_from_pool(self, resource_type: str, config_id: Optional[str] = None, wait_ms: int = None) -> Optional[ResourceHandle]:
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
            logger.error(f"[ResourcePoolManager.acquire_from_pool] 资源池不存在: pool_key={pool_key}")
            raise ValueError(f"Pool not found for {pool_key}")

        logger.debug(f"[ResourcePoolManager.acquire_from_pool] 从资源池获取资源: pool_key={pool_key}")
        return pool.activate(wait_ms)

    def release_to_pool(self, handle: ResourceHandle) -> None:
        """
        释放资源到资源池

        Args:
            handle: 资源句柄
        """
        logger.debug(f"[ResourcePoolManager.release_to_pool] 释放资源: resource_id={handle.resource_id[:8]}..., type={handle.resource_type}")
        handle.release()

    def destroy(self, handle: ResourceHandle) -> None:
        """
        彻底销毁指定资源（从池中移除并关闭连接）

        与release()不同，destroy()不会将资源归还到空闲池，
        而是从池中彻底移除并关闭连接、释放资源。

        Args:
            handle: 资源句柄
        """
        resource_type = handle.resource_type
        resource_id = handle.resource_id
        log_arch_event(logger, component="ResourcePoolManager", stage="RESOURCE_POOL", event="destroy_resource", status="start", design_id="ARCH-6.2", resource_type=resource_type)
        logger.info(f"[ResourcePoolManager.destroy] 销毁资源: resource_type={resource_type}, resource_id={resource_id[:8]}...")

        # 根据资源类型查找对应的资源池
        for pool_key, pool in self._pools.items():
            if pool_key.startswith(f"{resource_type}:"):
                pool.destroy(handle)
                return

        logger.warning(f"[ResourcePoolManager.destroy] 未找到资源对应的池: resource_type={resource_type}, resource_id={resource_id[:8]}...")
    
    def destroy_all(self) -> None:
        """销毁所有资源池"""
        logger.info(f"[ResourcePoolManager.destroy_all] 开始销毁所有资源池: pool_count={len(self._pools)}")
        log_arch_event(logger, component="ResourcePoolManager", stage="RESOURCE_POOL", event="destroy_all_pools", status="start", design_id="ARCH-6.2", pool_count=len(self._pools))
        for pool in self._pools.values():
            pool.destroy_all()
        self._pools.clear()
        logger.info("[ResourcePoolManager.destroy_all] 所有资源池已销毁")
    
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
