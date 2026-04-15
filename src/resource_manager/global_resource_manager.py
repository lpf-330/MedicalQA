# -*- coding: utf-8 -*-
"""
全局资源管理器

提供全局统一的资源管理接口，是单例模式。
"""

import logging
from typing import Optional

from .resource_registry import ResourceRegistry
from .pool_manager import PoolManager
from .resource_handle import ResourceHandle
from src.config.global_config import GlobalConfig

logger = logging.getLogger(__name__)


class GlobalResourceManager:
    """
    全局资源管理器
    
    提供全局统一的资源管理接口，是单例模式。
    
    属性：
        INSTANCE: 单例实例
        _resourceRegistry: 资源工厂注册表
        _poolManager: 资源池管理器
        _initialized: 是否已初始化
    """
    
    INSTANCE: 'GlobalResourceManager' = None
    
    def __init__(self):
        """初始化全局资源管理器"""
        self._resourceRegistry = ResourceRegistry()
        self._poolManager = PoolManager()
        self._initialized = False
    
    def _init_global_resource_manager(self, global_config: GlobalConfig) -> None:
        """
        初始化全局资源管理器
        
        Args:
            global_config: 全局资源配置
        """
        if self._initialized:
            logger.warning("GlobalResourceManager already initialized")
            return
        
        logger.info("Initializing GlobalResourceManager...")
        
        for resource_type, pool_config in global_config.pool_configs.items():
            resource_config = global_config.get_resource_config(resource_type)
            if resource_config is None:
                logger.warning(f"No resource config found for {resource_type}")
                continue
            
            factory = self._resourceRegistry.get_factory(resource_type)
            if factory is None:
                logger.warning(f"No factory registered for {resource_type}")
                continue
            
            self._poolManager.create_pool(resource_type, pool_config, resource_config)
            
            pool = self._poolManager.get_pool(resource_type)
            pool.create_initial_resources(pool_config.min_idle)
            
            logger.info(f"Pool created for {resource_type}: min_idle={pool_config.min_idle}")
        
        self._initialized = True
        logger.info("GlobalResourceManager initialized successfully")
    
    def register_factory(self, resource_type: str, factory) -> None:
        """
        注册资源工厂
        
        Args:
            resource_type: 资源类型唯一标识
            factory: 资源工厂实例
        """
        self._resourceRegistry.register_factory(resource_type, factory)
        self._poolManager.register_factory(resource_type, factory)
        logger.info(f"Factory registered for {resource_type}")
    
    @classmethod
    def acquire(cls, resource_type: str, wait_ms: int = None) -> Optional[ResourceHandle]:
        """
        获取资源
        
        Args:
            resource_type: 资源类型
            wait_ms: 等待时间（毫秒）
            
        Returns:
            ResourceHandle: 资源句柄
        """
        if cls.INSTANCE is None:
            raise RuntimeError("GlobalResourceManager not initialized")
        
        return cls.INSTANCE._poolManager.acquire(resource_type, wait_ms)
    
    @classmethod
    def release(cls, handle: ResourceHandle) -> None:
        """
        释放资源
        
        Args:
            handle: 资源句柄
        """
        if cls.INSTANCE is None:
            raise RuntimeError("GlobalResourceManager not initialized")
        
        cls.INSTANCE._poolManager.release(handle)
    
    def shutdown(self) -> None:
        """关闭资源管理器，释放所有资源"""
        logger.info("Shutting down GlobalResourceManager...")
        self._poolManager.destroy_all()
        self._initialized = False
        logger.info("GlobalResourceManager shut down successfully")
    
    def get_stats(self) -> dict:
        """获取资源池统计信息"""
        return self._poolManager.get_pool_stats()
    
    @property
    def resourceRegistry(self) -> ResourceRegistry:
        """获取资源工厂注册表"""
        return self._resourceRegistry
    
    @property
    def poolManager(self) -> PoolManager:
        """获取资源池管理器"""
        return self._poolManager
    
    @property
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized


GlobalResourceManager.INSTANCE = GlobalResourceManager()
