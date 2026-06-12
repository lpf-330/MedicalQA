# -*- coding: utf-8 -*-
"""
资源申请请求类

封装资源申请请求信息，支持Event通知机制，用于等待队列中的资源分配。
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from .resource_handle import ResourceHandle

logger = logging.getLogger(__name__)


@dataclass
class ResourceRequest:
    """
    资源申请请求类
    
    封装资源申请请求信息，支持Event通知机制。
    用于ResourcePool的等待队列，实现资源的异步分配。
    
    Attributes:
        request_id: 请求唯一标识
        create_time: 请求创建时间戳
        event: 等待通知事件，用于阻塞等待资源
        result: 分配结果，初始为None
        timeout_ms: 超时时间（毫秒）
    """
    
    request_id: str = field(default_factory=lambda: str(uuid4()))
    create_time: float = field(default_factory=time.time)
    event: threading.Event = field(default_factory=threading.Event)
    result: Optional['ResourceHandle'] = None
    timeout_ms: int = 5000
    
    def __post_init__(self):
        """初始化后记录日志"""
        logger.debug(f"[ResourceRequest.__post_init__] 资源请求已创建: request_id={self.request_id[:8]}..., timeout_ms={self.timeout_ms}")
    
    def wait(self, timeout_ms: int = None) -> bool:
        """
        等待资源分配
        
        Args:
            timeout_ms: 超时时间（毫秒），None表示使用实例的timeout_ms
            
        Returns:
            bool: 是否成功获取资源（True表示成功，False表示超时）
        """
        if timeout_ms is None:
            timeout_ms = self.timeout_ms
        
        timeout_sec = timeout_ms / 1000.0
        logger.debug(f"[ResourceRequest.wait] 开始等待资源分配: request_id={self.request_id[:8]}..., timeout_ms={timeout_ms}")
        success = self.event.wait(timeout=timeout_sec)
        if not success:
            logger.warning(f"[ResourceRequest.wait] 等待资源分配超时: request_id={self.request_id[:8]}..., timeout_ms={timeout_ms}")
        return success
    
    def set_result(self, handle: 'ResourceHandle') -> None:
        """
        设置分配结果并通知
        
        Args:
            handle: 分配的资源句柄
        """
        self.result = handle
        self.event.set()
        logger.info(f"[ResourceRequest.set_result] 资源分配结果已设置: request_id={self.request_id[:8]}..., resource_id={handle.resource_id[:8]}...")
    
    def is_expired(self) -> bool:
        """
        检查请求是否已超时
        
        Returns:
            bool: 是否已超时
        """
        elapsed_ms = (time.time() - self.create_time) * 1000
        return elapsed_ms > self.timeout_ms
    
    def __repr__(self) -> str:
        """返回请求对象的字符串表示"""
        status = "expired" if self.is_expired() else "pending"
        has_result = "assigned" if self.result is not None else "waiting"
        return (f"ResourceRequest(request_id={self.request_id[:8]}..., "
                f"status={status}, result={has_result}, timeout_ms={self.timeout_ms})")
