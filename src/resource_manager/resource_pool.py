# AI辅助生成：GLM-5，2026-04-15
# -*- coding: utf-8 -*-
"""
资源池类

管理特定类型资源的获取、释放、创建、销毁。
"""

import time
import threading
import logging
from typing import Dict, Optional
from uuid import uuid4

from .resource import Resource
from .resource_config import ResourceConfig
from .resource_factory import ResourceFactory
from .resource_handle import ResourceHandle
from .resource_client import ResourceClient
from src.config.pool_config import PoolConfig

logger = logging.getLogger(__name__)


class ResourcePool:
    """
    资源池类
    
    管理特定类型资源的获取、释放、创建、销毁。
    
    属性：
        _resource_type: 资源类型
        _config: 资源池配置
        _factory: 资源工厂
        _resource_config: 资源配置
        _idle_resources: 空闲资源字典
        _active_resources: 活跃资源字典
        _lock: 线程锁
    """
    
    def __init__(
        self,
        resource_type: str,
        config: PoolConfig,
        factory: ResourceFactory,
        resource_config: ResourceConfig
    ):
        """
        初始化资源池
        
        Args:
            resource_type: 资源类型
            config: 资源池配置
            factory: 资源工厂
            resource_config: 资源配置
        """
        self._resource_type = resource_type
        self._config = config
        self._factory = factory
        self._resource_config = resource_config
        self._idle_resources: Dict[str, Resource] = {}
        self._active_resources: Dict[str, Resource] = {}
        self._lock = threading.Lock()
        logger.info(f"[ResourcePool] 初始化资源池: type={resource_type}, max_size={config.max_size}, min_idle={config.min_idle}")
    
    def acquire(self, wait_ms: int = None) -> Optional[ResourceHandle]:
        """
        获取资源
        
        Args:
            wait_ms: 等待时间（毫秒），None表示使用配置中的默认值
            
        Returns:
            ResourceHandle: 资源句柄，如果获取失败返回None
        """
        if wait_ms is None:
            wait_ms = self._config.max_wait_time
        
        logger.debug(f"[ResourcePool] 尝试获取资源: type={self._resource_type}, wait_ms={wait_ms}, idle={len(self._idle_resources)}, active={len(self._active_resources)}")
        
        start_time = time.time() * 1000
        
        while True:
            with self._lock:
                if self._idle_resources:
                    resource_id, resource = self._idle_resources.popitem()
                    self._active_resources[resource_id] = resource
                    logger.info(f"[ResourcePool] 从空闲池获取资源: type={self._resource_type}, resource_id={resource_id[:8]}..., idle={len(self._idle_resources)}, active={len(self._active_resources)}")
                    return ResourceHandle(resource_id, resource, self)
                
                if len(self._active_resources) < self._config.max_size:
                    logger.info(f"[ResourcePool] 空闲池为空，创建新资源: type={self._resource_type}")
                    resource = self._factory.create(self._resource_config)
                    resource_id = str(uuid4())
                    resource.activate()
                    self._active_resources[resource_id] = resource
                    logger.info(f"[ResourcePool] 新资源创建并激活成功: type={self._resource_type}, resource_id={resource_id[:8]}..., active={len(self._active_resources)}")
                    return ResourceHandle(resource_id, resource, self)
            
            if time.time() * 1000 - start_time >= wait_ms:
                logger.warning(f"[ResourcePool] 获取资源超时: type={self._resource_type}, wait_ms={wait_ms}")
                return None
            
            time.sleep(0.1)
    
    def release(self, handle: ResourceHandle) -> None:
        """
        释放资源
        
        Args:
            handle: 资源句柄
        """
        with self._lock:
            resource_id = handle.resource_id
            if resource_id in self._active_resources:
                resource = self._active_resources.pop(resource_id)
                self._idle_resources[resource_id] = resource
                logger.info(f"[ResourcePool] 资源释放归还: type={self._resource_type}, resource_id={resource_id[:8]}..., idle={len(self._idle_resources)}, active={len(self._active_resources)}")
            else:
                logger.warning(f"[ResourcePool] 释放资源失败，资源不存在: type={self._resource_type}, resource_id={resource_id[:8]}...")
    
    def create_initial_resources(self, count: int) -> None:
        """
        创建初始资源实例并激活
        
        Args:
            count: 要创建的资源数量
        """
        logger.info(f"[ResourcePool] 开始创建初始资源: type={self._resource_type}, count={count}")
        with self._lock:
            for i in range(count):
                resource = self._factory.create(self._resource_config)
                resource_id = str(uuid4())
                resource.activate()
                self._idle_resources[resource_id] = resource
                logger.debug(f"[ResourcePool] 初始资源创建并激活: type={self._resource_type}, index={i+1}/{count}, resource_id={resource_id[:8]}...")
        logger.info(f"[ResourcePool] 初始资源创建完成: type={self._resource_type}, total={len(self._idle_resources)}")
    
    def destroy_all(self) -> None:
        """销毁所有资源"""
        logger.info(f"[ResourcePool] 开始销毁所有资源: type={self._resource_type}, idle={len(self._idle_resources)}, active={len(self._active_resources)}")
        with self._lock:
            idle_count = len(self._idle_resources)
            active_count = len(self._active_resources)
            
            for resource_id, resource in self._idle_resources.items():
                try:
                    self._factory.destroy(resource)
                    logger.debug(f"[ResourcePool] 销毁空闲资源: type={self._resource_type}, resource_id={resource_id[:8]}...")
                except Exception as e:
                    logger.error(f"[ResourcePool] 销毁空闲资源失败: type={self._resource_type}, resource_id={resource_id[:8]}..., error={e}")
            
            for resource_id, resource in self._active_resources.items():
                try:
                    self._factory.destroy(resource)
                    logger.debug(f"[ResourcePool] 销毁活跃资源: type={self._resource_type}, resource_id={resource_id[:8]}...")
                except Exception as e:
                    logger.error(f"[ResourcePool] 销毁活跃资源失败: type={self._resource_type}, resource_id={resource_id[:8]}..., error={e}")
            
            self._idle_resources.clear()
            self._active_resources.clear()
        logger.info(f"[ResourcePool] 所有资源销毁完成: type={self._resource_type}, destroyed_idle={idle_count}, destroyed_active={active_count}")
    
    @property
    def idle_count(self) -> int:
        """获取空闲资源数量"""
        return len(self._idle_resources)
    
    @property
    def active_count(self) -> int:
        """获取活跃资源数量"""
        return len(self._active_resources)
    
    @property
    def total_count(self) -> int:
        """获取总资源数量"""
        return self.idle_count + self.active_count
