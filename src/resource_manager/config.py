"""
资源管理层配置类模块

本模块定义了资源管理层的配置类，包括资源池配置和全局资源配置。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .resource_config import ResourceConfig


@dataclass
class PoolConfig:
    """
    资源池配置类
    
    用于配置资源池的核心参数，包括容量控制、空闲资源管理、超时设置等。
    
    Attributes:
        max_size: 资源池的最大总容量，限制资源池内可同时存在的资源实例总数（含活跃+空闲）
        min_idle: 资源池维护的最小空闲资源数，用于保障快速获取资源的响应能力
        idle_timeout: 空闲资源的超时时间（毫秒），超过该时长未被使用的空闲资源会被驱逐
        max_wait_time: 申请资源时的最大等待时间（毫秒），当资源池无可用资源时的最长等待时长
    """
    
    max_size: int
    min_idle: int
    idle_timeout: int
    max_wait_time: int
    
    def __post_init__(self):
        """
        初始化后验证配置参数的有效性
        """
        self._validate()
    
    def _validate(self) -> None:
        """
        验证配置参数的有效性
        
        Raises:
            ValueError: 当配置参数不满足约束条件时抛出
        """
        if self.max_size <= 0:
            raise ValueError(f"max_size必须大于0，当前值: {self.max_size}")
        
        if self.min_idle < 0:
            raise ValueError(f"min_idle不能为负数，当前值: {self.min_idle}")
        
        if self.min_idle > self.max_size:
            raise ValueError(f"min_idle({self.min_idle})不能大于max_size({self.max_size})")
        
        if self.idle_timeout < 0:
            raise ValueError(f"idle_timeout不能为负数，当前值: {self.idle_timeout}")
        
        if self.max_wait_time < 0:
            raise ValueError(f"max_wait_time不能为负数，当前值: {self.max_wait_time}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典格式
        
        Returns:
            Dict[str, Any]: 配置参数的字典表示
        """
        return {
            'max_size': self.max_size,
            'min_idle': self.min_idle,
            'idle_timeout': self.idle_timeout,
            'max_wait_time': self.max_wait_time
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PoolConfig':
        """
        从字典创建PoolConfig实例
        
        Args:
            config_dict: 包含配置参数的字典
            
        Returns:
            PoolConfig: 配置实例
            
        Raises:
            KeyError: 当缺少必需的配置参数时抛出
        """
        return cls(
            max_size=config_dict['max_size'],
            min_idle=config_dict['min_idle'],
            idle_timeout=config_dict['idle_timeout'],
            max_wait_time=config_dict['max_wait_time']
        )
    
    def __repr__(self) -> str:
        """返回配置对象的字符串表示"""
        return (f"PoolConfig(max_size={self.max_size}, min_idle={self.min_idle}, "
                f"idle_timeout={self.idle_timeout}, max_wait_time={self.max_wait_time})")


@dataclass
class GlobalConfig:
    """
    全局资源配置类
    
    作为全局资源配置的总容器，统一管理所有资源的配置信息和资源池配置信息。
    
    Attributes:
        resource_configs: 全局资源配置总容器，key为资源类型唯一标识，value为对应资源的ResourceConfig配置对象
        pool_configs: 全局资源池配置总容器，key为资源类型唯一标识，value为对应资源池的PoolConfig配置对象
    """
    
    resource_configs: Dict[str, 'ResourceConfig'] = field(default_factory=dict)
    pool_configs: Dict[str, PoolConfig] = field(default_factory=dict)
    
    def add_resource_config(self, resource_type: str, config: 'ResourceConfig') -> None:
        """
        添加资源配置
        
        Args:
            resource_type: 资源类型唯一标识
            config: 资源配置对象
        """
        self.resource_configs[resource_type] = config
    
    def add_pool_config(self, resource_type: str, config: PoolConfig) -> None:
        """
        添加资源池配置
        
        Args:
            resource_type: 资源类型唯一标识
            config: 资源池配置对象
        """
        self.pool_configs[resource_type] = config
    
    def get_resource_config(self, resource_type: str) -> 'ResourceConfig':
        """
        获取指定类型的资源配置
        
        Args:
            resource_type: 资源类型唯一标识
            
        Returns:
            ResourceConfig: 资源配置对象
            
        Raises:
            KeyError: 当资源类型不存在时抛出
        """
        if resource_type not in self.resource_configs:
            raise KeyError(f"资源配置不存在: {resource_type}")
        return self.resource_configs[resource_type]
    
    def get_pool_config(self, resource_type: str) -> PoolConfig:
        """
        获取指定类型的资源池配置
        
        Args:
            resource_type: 资源类型唯一标识
            
        Returns:
            PoolConfig: 资源池配置对象
            
        Raises:
            KeyError: 当资源类型不存在时抛出
        """
        if resource_type not in self.pool_configs:
            raise KeyError(f"资源池配置不存在: {resource_type}")
        return self.pool_configs[resource_type]
    
    def has_resource_config(self, resource_type: str) -> bool:
        """
        检查是否存在指定类型的资源配置
        
        Args:
            resource_type: 资源类型唯一标识
            
        Returns:
            bool: 是否存在资源配置
        """
        return resource_type in self.resource_configs
    
    def has_pool_config(self, resource_type: str) -> bool:
        """
        检查是否存在指定类型的资源池配置
        
        Args:
            resource_type: 资源类型唯一标识
            
        Returns:
            bool: 是否存在资源池配置
        """
        return resource_type in self.pool_configs
    
    def remove_resource_config(self, resource_type: str) -> None:
        """
        移除指定类型的资源配置
        
        Args:
            resource_type: 资源类型唯一标识
            
        Raises:
            KeyError: 当资源类型不存在时抛出
        """
        if resource_type not in self.resource_configs:
            raise KeyError(f"资源配置不存在: {resource_type}")
        del self.resource_configs[resource_type]
    
    def remove_pool_config(self, resource_type: str) -> None:
        """
        移除指定类型的资源池配置
        
        Args:
            resource_type: 资源类型唯一标识
            
        Raises:
            KeyError: 当资源类型不存在时抛出
        """
        if resource_type not in self.pool_configs:
            raise KeyError(f"资源池配置不存在: {resource_type}")
        del self.pool_configs[resource_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将全局配置转换为字典格式
        
        Returns:
            Dict[str, Any]: 配置参数的字典表示
        """
        return {
            'resource_configs': {
                k: v.to_dict() for k, v in self.resource_configs.items()
            },
            'pool_configs': {
                k: v.to_dict() for k, v in self.pool_configs.items()
            }
        }
    
    def get_all_resource_types(self) -> list:
        """
        获取所有已配置的资源类型
        
        Returns:
            list: 资源类型列表
        """
        all_types = set(self.resource_configs.keys())
        all_types.update(self.pool_configs.keys())
        return list(all_types)
    
    def __repr__(self) -> str:
        """返回配置对象的字符串表示"""
        return (f"GlobalConfig(resource_configs_count={len(self.resource_configs)}, "
                f"pool_configs_count={len(self.pool_configs)})")
