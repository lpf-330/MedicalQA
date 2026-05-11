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
    
    属性：
        max_size: 资源池最大总容量
        min_idle: 最小空闲资源数
        idle_timeout: 空闲资源超时时间（毫秒）
        max_wait_time: 申请资源最大等待时间（毫秒）
    """
    
    max_size: int = 10
    min_idle: int = 1
    idle_timeout: int = 300000
    max_wait_time: int = 5000
    
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
        return True
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "max_size": self.max_size,
            "min_idle": self.min_idle,
            "idle_timeout": self.idle_timeout,
            "max_wait_time": self.max_wait_time
        }
