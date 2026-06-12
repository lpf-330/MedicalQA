# AI辅助生成：GLM-5，2026-04-15
# -*- coding: utf-8 -*-
"""
资源池类

管理特定类型资源的获取、释放、创建、销毁。
"""

import threading
import logging
from collections import deque
from typing import Dict, Optional
from uuid import uuid4

from .resource import Resource
from .resource_config import ResourceConfig
from .resource_factory import ResourceFactory
from .resource_handle import ResourceHandle
from .resource_request import ResourceRequest
from .resource_creation_guard import ResourceCreationGuard
from src.config.pool_config import PoolConfig
from src.utils.logger import log_arch_event

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
        _pending_requests: 等待队列（FIFO）
        _creation_guard: 资源创建保护器
        _creation_failures: 创建失败次数
        _total_created: 累计创建资源数
        _total_destroyed: 累计销毁资源数
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
        self._pending_requests: deque = deque()
        self._manager_ref = None
        self._creation_guard = ResourceCreationGuard(
            min_memory_mb=config.min_memory_mb,
            min_vram_mb=config.min_vram_mb,
            enabled=config.pre_create_check_enabled
        )
        self._creation_failures = 0
        self._total_created = 0
        self._total_destroyed = 0
        logger.info(f"[ResourcePool] 初始化资源池: type={self._resource_type}, max_size={config.max_size}, min_idle={config.min_idle}, max_wait_time={config.max_wait_time}, max_pending_requests={config.max_pending_requests}, pre_create_check_enabled={config.pre_create_check_enabled}, min_memory_mb={config.min_memory_mb}, min_vram_mb={config.min_vram_mb}, allow_dynamic_creation={config.allow_dynamic_creation}, creation_timeout={config.creation_timeout}")
        log_arch_event(logger, component="ResourcePool", stage="RESOURCE_POOL", event="pool_created", status="success", design_id="ARCH-6.2", resource_type=self._resource_type, max_size=config.max_size, min_idle=config.min_idle)
    
    def activate(self, wait_ms: int = None) -> Optional[ResourceHandle]:
        """
        激活并获取资源

        Args:
            wait_ms: 等待时间（毫秒），None表示使用配置中的默认值

        Returns:
            ResourceHandle: 资源句柄，如果获取失败返回None
        """
        if wait_ms is None:
            wait_ms = self._config.max_wait_time

        logger.debug(f"[ResourcePool] 尝试获取资源: type={self._resource_type}, wait_ms={wait_ms}, idle={len(self._idle_resources)}, active={len(self._active_resources)}")
        request = None
        wait_timeout_detail = ""

        with self._lock:
            if self._idle_resources:
                resource_id, resource = self._idle_resources.popitem()
                self._active_resources[resource_id] = resource
                log_arch_event(logger, component="ResourcePool", stage="RESOURCE_POOL", event="acquire_from_idle", status="success", design_id="ARCH-6.2", resource_type=self._resource_type)
                logger.info(f"[ResourcePool] 从空闲池获取资源: type={self._resource_type}, resource_id={resource_id[:8]}..., idle={len(self._idle_resources)}, active={len(self._active_resources)}")
                logger.info(f"[RESOURCE_ACQUIRE] type={self._resource_type}, from_idle=True, dynamic_creation=False, wait_queue_size={len(self._pending_requests)}")
                logger.info(f"[RESOURCE_ACQUIRE] type={self._resource_type}, from=idle, resource_id={resource_id}, wait_ms={wait_ms}")
                logger.info(f"[POOL_ACQUIRE] resource_type={self._resource_type}, path=idle_pool_direct, resource_id={resource_id[:8]}, idle_remaining={len(self._idle_resources)}, active={len(self._active_resources)}")
                return ResourceHandle(resource_id, resource, self, manager_ref=self._manager_ref)

            if len(self._active_resources) < self._config.max_size and self._config.allow_dynamic_creation:
                should_create = True
                if self._config.pre_create_check_enabled:
                    check_ok, check_reason = self._creation_guard.check_before_creation()
                    logger.info(f"[CREATION_GUARD] check_ok={check_ok}, reason={check_reason}, resource_type={self._resource_type}")
                    logger.info(f"[RESOURCE_GUARD] action=check, memory_mb={self._config.min_memory_mb}, vram_mb={self._config.min_vram_mb}, allowed={check_ok}")
                    if not check_ok:
                        logger.warning(f"[ResourcePool] 资源创建前检查失败，加入等待队列: type={self._resource_type}, reason={check_reason}")

                        if len(self._pending_requests) >= self._config.max_pending_requests:
                            logger.warning(f"[ResourcePool] 等待队列已满，无法加入: type={self._resource_type}, pending={len(self._pending_requests)}")
                            logger.warning(f"[POOL_WAIT_QUEUE] resource_type={self._resource_type}, queue_size={len(self._pending_requests)}, max_pending={self._config.max_pending_requests}, is_full=True, action=reject")
                            return None

                        request = ResourceRequest(timeout_ms=wait_ms)
                        self._pending_requests.append(request)
                        logger.info(f"[QUEUE_OPERATION] 入队(创建前检查失败): resource_type={self._resource_type}, queue_size={len(self._pending_requests)}, request_id={request.request_id}, reason={check_reason}")
                        logger.info(f"[POOL_WAIT_QUEUE] resource_type={self._resource_type}, action=enqueue_guard_failed, queue_size={len(self._pending_requests)}, max_pending={self._config.max_pending_requests}, is_full={len(self._pending_requests) >= self._config.max_pending_requests}, reason={check_reason}")
                        logger.debug(f"[ResourcePool] 创建前检查失败，加入等待队列: type={self._resource_type}, request_id={request.request_id[:8]}..., pending={len(self._pending_requests)}, reason={check_reason}")
                        logger.info(f"[RESOURCE_ACQUIRE] type={self._resource_type}, from_idle=False, dynamic_creation=False, wait_queue_size={len(self._pending_requests)}")
                        logger.info(f"[RESOURCE_ACQUIRE] type={self._resource_type}, from=wait, resource_id=pending, wait_ms={wait_ms}")

                        wait_timeout_detail = f", reason={check_reason}"
                        should_create = False

                if should_create:
                    logger.info(f"[ResourcePool] 空闲池为空，创建新资源: type={self._resource_type}")
                    result = self._try_create_resource()
                    if result is not None:
                        resource_id, resource = result
                        self._active_resources[resource_id] = resource
                        log_arch_event(logger, component="ResourcePool", stage="RESOURCE_POOL", event="acquire_from_creation", status="success", design_id="ARCH-6.2", resource_type=self._resource_type)
                        logger.info(f"[ResourcePool] 新资源创建并激活成功: type={self._resource_type}, resource_id={resource_id[:8]}..., active={len(self._active_resources)}")
                        logger.info(f"[RESOURCE_ACQUIRE] type={self._resource_type}, from_idle=False, dynamic_creation=True, wait_queue_size={len(self._pending_requests)}")
                        logger.info(f"[RESOURCE_ACQUIRE] type={self._resource_type}, from=create, resource_id={resource_id}, wait_ms={wait_ms}")
                        logger.info(f"[POOL_ACQUIRE] resource_type={self._resource_type}, path=dynamic_creation, resource_id={resource_id[:8]}, idle_remaining={len(self._idle_resources)}, active={len(self._active_resources)}")
                        return ResourceHandle(resource_id, resource, self, manager_ref=self._manager_ref)
                    else:
                        logger.warning(f"[ResourcePool] 资源创建或验证失败: type={self._resource_type}")
                        return None

            if request is None:
                if len(self._pending_requests) >= self._config.max_pending_requests:
                    logger.warning(f"[ResourcePool] 等待队列已满: type={self._resource_type}, pending={len(self._pending_requests)}")
                    logger.warning(f"[POOL_WAIT_QUEUE] resource_type={self._resource_type}, queue_size={len(self._pending_requests)}, max_pending={self._config.max_pending_requests}, is_full=True, action=reject")
                    return None

                request = ResourceRequest(timeout_ms=wait_ms)
                self._pending_requests.append(request)
                logger.info(f"[QUEUE_OPERATION] 入队: resource_type={self._resource_type}, queue_size={len(self._pending_requests)}, request_id={request.request_id}")
                logger.info(f"[POOL_WAIT_QUEUE] resource_type={self._resource_type}, action=enqueue_capacity_full, queue_size={len(self._pending_requests)}, max_pending={self._config.max_pending_requests}, is_full={len(self._pending_requests) >= self._config.max_pending_requests}")
                logger.debug(f"[ResourcePool] 加入等待队列: type={self._resource_type}, request_id={request.request_id[:8]}..., pending={len(self._pending_requests)}")
                logger.info(f"[RESOURCE_ACQUIRE] type={self._resource_type}, from_idle=False, dynamic_creation=False, wait_queue_size={len(self._pending_requests)}")
                logger.info(f"[RESOURCE_ACQUIRE] type={self._resource_type}, from=wait, resource_id=pending, wait_ms={wait_ms}")
                logger.info(f"[POOL_ACQUIRE] resource_type={self._resource_type}, path=wait_queue, resource_id=pending, wait_ms={wait_ms}, queue_size={len(self._pending_requests)}")

        if request is None:
            return None

        success = request.wait(wait_ms)

        if success and request.result is not None:
            logger.info(f"[ResourcePool] 从等待队列获取资源成功: type={self._resource_type}, request_id={request.request_id[:8]}...")
            return request.result
        else:
            with self._lock:
                try:
                    self._pending_requests.remove(request)
                except ValueError:
                    pass
            logger.warning(f"[ResourcePool] 等待资源超时: type={self._resource_type}, wait_ms={wait_ms}{wait_timeout_detail}")
            return None
    
    def release_to_pool(self, handle: ResourceHandle) -> None:
        """
        释放资源到池

        Args:
            handle: 资源句柄
        """
        with self._lock:
            resource_id = handle.resource_id
            if resource_id in self._active_resources:
                resource = self._active_resources.pop(resource_id)

                # 先清理过期请求
                self._cleanup_expired_requests()

                # 尝试通知等待者
                notified = self._notify_waiters(resource_id, resource)

                if not notified:
                    self._idle_resources[resource_id] = resource
                    log_arch_event(logger, component="ResourcePool", stage="RESOURCE_POOL", event="release_to_idle", status="success", design_id="ARCH-6.2", resource_type=self._resource_type)
                    logger.info(f"[ResourcePool] 资源释放归还: type={self._resource_type}, resource_id={resource_id[:8]}..., idle={len(self._idle_resources)}, active={len(self._active_resources)}")
                    logger.info(f"[RESOURCE_RELEASE] type={self._resource_type}, id={resource_id[:8]}, notified_waiter=False")
                    logger.info(f"[RESOURCE_RELEASE] type={self._resource_type}, resource_id={resource_id}, notify_waiter=False")
                    logger.info(f"[POOL_RELEASE] resource_type={self._resource_type}, resource_id={resource_id[:8]}, notified_waiter=False, idle={len(self._idle_resources)}, active={len(self._active_resources)}")
            else:
                logger.warning(f"[ResourcePool] 释放资源失败，资源不存在: type={self._resource_type}, resource_id={resource_id[:8]}...")

    def _notify_waiters(self, resource_id: str, resource: Resource) -> bool:
        """
        通知等待队列中的请求，将资源分配给队首未过期请求

        Args:
            resource_id: 资源ID
            resource: 资源实例

        Returns:
            bool: 是否成功分配给等待者
        """
        while self._pending_requests:
            request = self._pending_requests.popleft()
            logger.info(f"[QUEUE_OPERATION] 出队分配: resource_type={self._resource_type}, queue_size={len(self._pending_requests)}, request_id={request.request_id}")
            if not request.is_expired():
                self._active_resources[resource_id] = resource
                handle = ResourceHandle(resource_id, resource, self, manager_ref=self._manager_ref)
                request.set_result(handle)
                logger.info(f"[ResourcePool] 资源分配给等待请求: type={self._resource_type}, resource_id={resource_id[:8]}..., request_id={request.request_id[:8]}...")
                logger.info(f"[RESOURCE_RELEASE] type={self._resource_type}, id={resource_id[:8]}, notified_waiter=True")
                logger.info(f"[RESOURCE_RELEASE] type={self._resource_type}, resource_id={resource_id}, notify_waiter=True")
                logger.info(f"[POOL_RELEASE] resource_type={self._resource_type}, resource_id={resource_id[:8]}, notified_waiter=True, wait_queue_remaining={len(self._pending_requests)}")
                return True
            else:
                logger.debug(f"[ResourcePool] 跳过已过期请求: type={self._resource_type}, request_id={request.request_id[:8]}...")
        return False

    def _cleanup_expired_requests(self) -> int:
        """
        清理等待队列中的过期请求

        Returns:
            int: 清理的过期请求数量
        """
        cleaned = 0
        while self._pending_requests and self._pending_requests[0].is_expired():
            request = self._pending_requests.popleft()
            cleaned += 1
            logger.debug(f"[ResourcePool] 清理过期请求: type={self._resource_type}, request_id={request.request_id[:8]}...")
        if cleaned > 0:
            logger.info(f"[ResourcePool] 清理过期请求完成: type={self._resource_type}, cleaned={cleaned}, remaining={len(self._pending_requests)}")
        return cleaned
    
    def destroy(self, handle: ResourceHandle) -> None:
        """
        彻底销毁指定资源（从池中移除并关闭连接）

        与release()不同，destroy()不会将资源归还到空闲池，
        而是从池中彻底移除并调用工厂的destroy方法关闭连接、释放资源。

        Args:
            handle: 资源句柄
        """
        with self._lock:
            resource_id = handle.resource_id
            resource = None

            # 先从活跃资源中查找
            if resource_id in self._active_resources:
                resource = self._active_resources.pop(resource_id)
            # 再从空闲资源中查找
            elif resource_id in self._idle_resources:
                resource = self._idle_resources.pop(resource_id)

            if resource is not None:
                log_arch_event(logger, component="ResourcePool", stage="RESOURCE_POOL", event="resource_destroyed", status="success", design_id="ARCH-6.2", resource_type=self._resource_type)
                logger.info(f"[RESOURCE_DESTROY] 资源类型={self._resource_type}, 资源ID={resource_id}, 操作=销毁")
                try:
                    self._factory.destroy(resource)
                    self._total_destroyed += 1
                    logger.info(f"[ResourcePool] 资源已销毁: type={self._resource_type}, resource_id={resource_id[:8]}..., idle={len(self._idle_resources)}, active={len(self._active_resources)}")
                except Exception as e:
                    logger.error(f"[ResourcePool] 销毁资源失败: type={self._resource_type}, resource_id={resource_id[:8]}..., error={e}")
            else:
                logger.warning(f"[ResourcePool] 销毁资源失败，资源不存在: type={self._resource_type}, resource_id={resource_id[:8]}...")

    def create_initial_resources(self, count: int) -> None:
        """
        创建初始资源实例并激活

        通过_try_create_resource()创建，包含创建保护检查和创建后验证。

        Args:
            count: 要创建的资源数量
        """
        logger.info(f"[ResourcePool] 开始创建初始资源: type={self._resource_type}, count={count}")
        log_arch_event(logger, component="ResourcePool", stage="RESOURCE_POOL", event="create_initial_start", status="start", design_id="ARCH-6.2", resource_type=self._resource_type, count=count)
        created = 0
        with self._lock:
            for i in range(count):
                result = self._try_create_resource()
                if result is not None:
                    resource_id, resource = result
                    self._idle_resources[resource_id] = resource
                    created += 1
                    logger.debug(f"[ResourcePool] 初始资源创建并激活: type={self._resource_type}, index={i+1}/{count}, resource_id={resource_id[:8]}...")
                else:
                    logger.warning(f"[ResourcePool] 初始资源创建失败: type={self._resource_type}, index={i+1}/{count}")
        log_arch_event(logger, component="ResourcePool", stage="RESOURCE_POOL", event="create_initial_complete", status="success", design_id="ARCH-6.2", resource_type=self._resource_type, requested=count, created=created)
        logger.info(f"[ResourcePool] 初始资源创建完成: type={self._resource_type}, requested={count}, created={created}")
    
    def destroy_all(self) -> None:
        """销毁所有资源"""
        logger.info(f"[ResourcePool] 开始销毁所有资源: type={self._resource_type}, idle={len(self._idle_resources)}, active={len(self._active_resources)}")
        log_arch_event(logger, component="ResourcePool", stage="RESOURCE_POOL", event="destroy_all_start", status="start", design_id="ARCH-6.2", resource_type=self._resource_type)
        destroyed_count = 0
        with self._lock:
            idle_count = len(self._idle_resources)
            active_count = len(self._active_resources)
            
            for resource_id, resource in self._idle_resources.items():
                try:
                    self._factory.destroy(resource)
                    destroyed_count += 1
                    logger.info(f"[RESOURCE_DESTROY] 资源类型={self._resource_type}, 资源ID={resource_id}, 操作=销毁空闲资源")
                except Exception as e:
                    logger.error(f"[ResourcePool] 销毁空闲资源失败: type={self._resource_type}, resource_id={resource_id[:8]}..., error={e}")
            
            for resource_id, resource in self._active_resources.items():
                try:
                    self._factory.destroy(resource)
                    destroyed_count += 1
                    logger.info(f"[RESOURCE_DESTROY] 资源类型={self._resource_type}, 资源ID={resource_id}, 操作=销毁活跃资源")
                except Exception as e:
                    logger.error(f"[ResourcePool] 销毁活跃资源失败: type={self._resource_type}, resource_id={resource_id[:8]}..., error={e}")
            
            self._idle_resources.clear()
            self._active_resources.clear()
            self._total_destroyed += destroyed_count
        log_arch_event(logger, component="ResourcePool", stage="RESOURCE_POOL", event="destroy_all_complete", status="success", design_id="ARCH-6.2", resource_type=self._resource_type, destroyed_count=destroyed_count)
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
    
    @property
    def pending_count(self) -> int:
        """获取等待队列中的请求数量"""
        return len(self._pending_requests)
    
    def get_creation_guard_info(self) -> dict:
        """
        获取资源创建保护器信息
        
        Returns:
            dict: 资源创建保护器信息
        """
        return self._creation_guard.get_system_info()
    
    def _try_create_resource(self) -> Optional[tuple]:
        """
        尝试创建新资源（带创建保护检查和创建后验证）

        三层保护机制：
        1. 创建前资源预检：通过_creation_guard.check_before_creation()检查内存/显存
        2. 创建过程异常隔离：try/except捕获创建异常
        3. 创建后资源验证：调用_validate_created_resource()验证资源可用性

        Returns:
            Optional[tuple]: 验证通过返回(resource_id, resource)，失败返回None
        """
        # 创建前资源预检
        allowed, reason = self._creation_guard.check_before_creation(self._resource_config)
        if not allowed:
            logger.warning(f"[ResourcePool] 创建前检查未通过: type={self._resource_type}, reason={reason}")
            return None

        try:
            resource = self._factory.create(self._resource_config)
            resource_id = str(uuid4())
            resource.activate()
            self._total_created += 1
            
            # 创建后验证
            if self._validate_created_resource(resource):
                logger.info(f"[ResourcePool] 资源创建并验证通过: type={self._resource_type}, resource_id={resource_id[:8]}...")
                return (resource_id, resource)
            else:
                logger.warning(f"[ResourcePool] 资源创建后验证失败: type={self._resource_type}, resource_id={resource_id[:8]}...")
                self._cleanup_failed_resource(resource)
                self._creation_failures += 1
                return None
        except Exception as e:
            logger.error(f"[ResourcePool] 资源创建异常: type={self._resource_type}, error={e}")
            self._creation_failures += 1
            return None
    
    def _validate_created_resource(self, resource) -> bool:
        """
        创建后资源验证

        执行三层验证：
        1. 基础验证：资源对象非空
        2. 属性验证：如果资源有is_valid()方法，调用验证属性完整性
        3. 连通性验证：如果资源有health_check()方法，调用验证连通性

        Args:
            resource: 待验证的资源对象

        Returns:
            bool: 所有验证通过返回True，任一失败返回False
        """
        # 基础验证：资源对象非空
        logger.info(f"[RESOURCE_VALIDATE] layer=1/3, check=基础验证(资源对象非空), type={self._resource_type}")
        if resource is None:
            logger.warning(f"[ResourcePool] 资源验证失败-资源对象为空: type={self._resource_type}")
            return False
        logger.info(f"[RESOURCE_VALIDATE] layer=1/3, result=通过, type={self._resource_type}")

        # 属性验证：如果资源有is_valid()方法，调用验证属性完整性
        if hasattr(resource, 'is_valid') and callable(resource.is_valid):
            logger.info(f"[RESOURCE_VALIDATE] layer=2/3, check=属性完整性验证(is_valid), type={self._resource_type}")
            try:
                if not resource.is_valid():
                    logger.warning(f"[ResourcePool] 资源验证失败-属性完整性验证未通过: type={self._resource_type}")
                    return False
                logger.info(f"[RESOURCE_VALIDATE] layer=2/3, result=通过, type={self._resource_type}")
            except Exception as e:
                logger.warning(f"[ResourcePool] 资源验证失败-is_valid()调用异常: type={self._resource_type}, error={e}")
                return False
        else:
            logger.info(f"[RESOURCE_VALIDATE] layer=2/3, check=属性完整性验证(is_valid), skipped=方法不存在, type={self._resource_type}")

        # 连通性验证：如果资源有health_check()方法，调用验证连通性
        if hasattr(resource, 'health_check') and callable(resource.health_check):
            logger.info(f"[RESOURCE_VALIDATE] layer=3/3, check=连通性验证(health_check), type={self._resource_type}")
            try:
                if not resource.health_check():
                    logger.warning(f"[ResourcePool] 资源验证失败-连通性验证未通过: type={self._resource_type}")
                    return False
                logger.info(f"[RESOURCE_VALIDATE] layer=3/3, result=通过, type={self._resource_type}")
            except Exception as e:
                logger.warning(f"[ResourcePool] 资源验证失败-health_check()调用异常: type={self._resource_type}, error={e}")
                return False
        else:
            logger.info(f"[RESOURCE_VALIDATE] layer=3/3, check=连通性验证(health_check), skipped=方法不存在, type={self._resource_type}")

        log_arch_event(logger, component="ResourcePool", stage="RESOURCE_POOL", event="validate_resource", status="success", design_id="ARCH-6.2", resource_type=self._resource_type)
        logger.info(f"[RESOURCE_VALIDATE] 三层验证全部通过: type={self._resource_type}")
        return True
    
    def _cleanup_failed_resource(self, resource) -> None:
        """
        清理失败资源
        
        如果资源不为None且有destroy方法，调用destroy进行清理。
        
        Args:
            resource: 需要清理的失败资源
        """
        if resource is not None and hasattr(resource, 'destroy') and callable(resource.destroy):
            try:
                resource.destroy()
                logger.warning(f"[ResourcePool] 已清理失败资源: type={self._resource_type}")
            except Exception as e:
                logger.warning(f"[ResourcePool] 清理失败资源时异常: type={self._resource_type}, error={e}")
        else:
            logger.warning(f"[ResourcePool] 失败资源无法清理（无destroy方法或资源为None）: type={self._resource_type}")
    
    def get_health(self) -> Dict:
        """
        资源池健康检查
        
        返回包含以下字段的字典：
        - idle_count: 空闲资源数量
        - active_count: 活跃资源数量
        - total_count: 总资源数量
        - pending_count: 等待队列请求数量
        - health_status: 健康状态（healthy/degraded/critical）
        
        健康状态判断规则：
        - "healthy": idle_count > 0 且 pending_count == 0
        - "degraded": idle_count == 0 或 pending_count > 0
        - "critical": total_count == 0
        
        Returns:
            Dict: 健康检查结果字典
        """
        with self._lock:
            idle = len(self._idle_resources)
            active = len(self._active_resources)
            total = idle + active
            pending = len(self._pending_requests)
        
        # 判断健康状态
        if total == 0:
            health_status = "critical"
        elif idle > 0 and pending == 0:
            health_status = "healthy"
        else:
            health_status = "degraded"
        
        result = {
            "idle_count": idle,
            "active_count": active,
            "total_count": total,
            "pending_count": pending,
            "health_status": health_status
        }
        logger.info(f"[ResourcePool] 健康检查: type={self._resource_type}, status={health_status}, idle={idle}, active={active}, total={total}, pending={pending}")
        return result
    
    def get_stats(self) -> Dict:
        """
        资源池统计信息
        
        返回包含以下字段的字典：
        - utilization_rate: 资源使用率 (active_count / total_count)，total_count为0时返回0.0
        - pending_requests: 等待队列长度
        - creation_failures: 创建失败次数
        - total_created: 累计创建资源数
        - total_destroyed: 累计销毁资源数
        
        Returns:
            Dict: 统计信息字典
        """
        with self._lock:
            idle = len(self._idle_resources)
            active = len(self._active_resources)
            total = idle + active
            pending = len(self._pending_requests)
        
        utilization_rate = active / total if total > 0 else 0.0
        
        result = {
            "utilization_rate": utilization_rate,
            "pending_requests": pending,
            "creation_failures": self._creation_failures,
            "total_created": self._total_created,
            "total_destroyed": self._total_destroyed
        }
        logger.debug(f"[ResourcePool] 统计信息: type={self._resource_type}, utilization_rate={utilization_rate:.2f}, pending={pending}, failures={self._creation_failures}, created={self._total_created}, destroyed={self._total_destroyed}")
        return result
    
    def pre_warm(self, count: int) -> int:
        """
        资源池预热，预创建指定数量的资源实例

        通过_try_create_resource()创建，包含创建保护检查和创建后验证。

        Args:
            count: 预创建的资源数量

        Returns:
            int: 实际创建数量（可能因检查未通过或验证失败而少于请求的数量）
        """
        logger.info(f"[ResourcePool] 开始预热资源池: type={self._resource_type}, count={count}")
        log_arch_event(logger, component="ResourcePool", stage="RESOURCE_POOL", event="pre_warm_start", status="start", design_id="ARCH-6.2", resource_type=self._resource_type, count=count)
        created = 0
        with self._lock:
            for i in range(count):
                result = self._try_create_resource()
                if result is not None:
                    resource_id, resource = result
                    self._idle_resources[resource_id] = resource
                    created += 1
                    logger.info(f"[ResourcePool] 预热创建资源成功: type={self._resource_type}, index={i+1}/{count}, resource_id={resource_id[:8]}...")
                else:
                    logger.warning(f"[ResourcePool] 预热创建资源失败: type={self._resource_type}, index={i+1}/{count}")
        log_arch_event(logger, component="ResourcePool", stage="RESOURCE_POOL", event="pre_warm_complete", status="success", design_id="ARCH-6.2", resource_type=self._resource_type, requested=count, created=created)
        logger.info(f"[ResourcePool] 预热完成: type={self._resource_type}, requested={count}, created={created}")
        return created
    
    def evict_idle_resource(self, target_idle: int) -> int:
        """
        驱逐空闲资源，将空闲资源缩减到目标数量
        
        从idle_resources末尾取出资源，调用factory.destroy()销毁，
        返回实际销毁数量。
        
        Args:
            target_idle: 目标空闲资源数量
            
        Returns:
            int: 实际销毁数量
        """
        logger.info(f"[ResourcePool] 开始缩容资源池: type={self._resource_type}, target_idle={target_idle}")
        log_arch_event(logger, component="ResourcePool", stage="RESOURCE_POOL", event="evict_idle_start", status="start", design_id="ARCH-6.2", resource_type=self._resource_type, target_idle=target_idle)
        destroyed = 0
        with self._lock:
            to_destroy = len(self._idle_resources) - target_idle
            if to_destroy <= 0:
                logger.info(f"[ResourcePool] 无需缩容: type={self._resource_type}, current_idle={len(self._idle_resources)}, target_idle={target_idle}")
                return 0
            
            # 从idle_resources末尾取出资源销毁
            keys = list(self._idle_resources.keys())
            for i in range(to_destroy):
                if not self._idle_resources:
                    break
                key = keys[-(i + 1)]
                resource = self._idle_resources.pop(key)
                try:
                    self._factory.destroy(resource)
                    self._total_destroyed += 1
                    destroyed += 1
                    logger.info(f"[ResourcePool] 缩容销毁资源: type={self._resource_type}, resource_id={key[:8]}..., destroyed={destroyed}/{to_destroy}")
                except Exception as e:
                    logger.error(f"[ResourcePool] 缩容销毁资源失败: type={self._resource_type}, resource_id={key[:8]}..., error={e}")
                    # 销毁失败，将资源放回idle_resources
                    self._idle_resources[key] = resource
        
        log_arch_event(logger, component="ResourcePool", stage="RESOURCE_POOL", event="evict_idle_complete", status="success", design_id="ARCH-6.2", resource_type=self._resource_type, target_idle=target_idle, destroyed=destroyed)
        logger.info(f"[ResourcePool] 缩容完成: type={self._resource_type}, target_idle={target_idle}, destroyed={destroyed}")
        return destroyed
