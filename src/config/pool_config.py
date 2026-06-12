# AI辅助生成：GLM-5，2026-04-15
# -*- coding: utf-8 -*-
"""
资源池配置类

定义资源池的配置参数。
"""

from dataclasses import dataclass


@dataclass
class PoolConfig:
    """
    资源池配置类
    
    定义资源池的核心参数。
    
    基本属性：
        max_size: 资源池最大总容量
        min_idle: 最小空闲资源数
        idle_timeout: 空闲资源超时时间（毫秒）
        max_wait_time: 申请资源最大等待时间（毫秒）
    
    扩展属性（v2.3新增）：
        allow_dynamic_creation: 是否允许动态创建资源
        max_pending_requests: 最大等待请求数
        creation_timeout: 资源创建超时时间（毫秒）
        pre_create_check_enabled: 是否启用创建前检查
        min_memory_mb: 最小内存要求（MB）
        min_vram_mb: 最小显存要求（MB）
    """
    
    max_size: int = 10
    min_idle: int = 1
    idle_timeout: int = 300000
    max_wait_time: int = 5000
    
    allow_dynamic_creation: bool = True
    max_pending_requests: int = 100
    creation_timeout: int = 60000
    pre_create_check_enabled: bool = True
    min_memory_mb: int = 512
    min_vram_mb: int = 0
    
    def validate(self) -> bool:
        """
        验证配置有效性
        
        Returns:
            bool: 配置是否有效
        """
        if self.max_size < 1:
            return False
        if self.min_idle < 0:
            return False
        if self.min_idle > self.max_size:
            return False
        if self.idle_timeout < 0:
            return False
        if self.max_wait_time < 0:
            return False
        if self.max_pending_requests < 1:
            return False
        if self.creation_timeout < 0:
            return False
        if self.min_memory_mb < 0:
            return False
        if self.min_vram_mb < 0:
            return False
        return True
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "max_size": self.max_size,
            "min_idle": self.min_idle,
            "idle_timeout": self.idle_timeout,
            "max_wait_time": self.max_wait_time,
            "allow_dynamic_creation": self.allow_dynamic_creation,
            "max_pending_requests": self.max_pending_requests,
            "creation_timeout": self.creation_timeout,
            "pre_create_check_enabled": self.pre_create_check_enabled,
            "min_memory_mb": self.min_memory_mb,
            "min_vram_mb": self.min_vram_mb
        }
