# AI辅助生成：GLM-5，2026-04-15
# -*- coding: utf-8 -*-
"""
全局资源配置类

包含所有资源的配置信息和资源池配置信息。
"""

from typing import Dict, Any

from .pool_config import PoolConfig


class GlobalConfig:
    """
    全局资源配置类
    
    包含所有资源的配置信息和资源池配置信息。
    
    属性：
        resource_configs: 资源配置总容器，key为配置ID（config_id），value为ResourceConfig
        pool_configs: 资源池配置总容器，key为配置ID（config_id），value为PoolConfig
    """
    
    def __init__(self):
        """初始化全局配置"""
        self._resource_configs: Dict[str, Any] = {}
        self._pool_configs: Dict[str, PoolConfig] = {}

        # 服务端配置
        self._server_port: int = 8001
        self._vram_sufficient_gb: float = 8.0
        self._warmup_timeout: float = 30.0
        self._timeout_keep_alive: int = 60
    
    @property
    def resource_configs(self) -> Dict[str, Any]:
        """获取资源配置字典"""
        return self._resource_configs
    
    @property
    def pool_configs(self) -> Dict[str, PoolConfig]:
        """获取资源池配置字典"""
        return self._pool_configs

    @property
    def server_port(self) -> int:
        return self._server_port

    @property
    def vram_sufficient_gb(self) -> float:
        return self._vram_sufficient_gb

    @property
    def warmup_timeout(self) -> float:
        return self._warmup_timeout

    @property
    def timeout_keep_alive(self) -> int:
        return self._timeout_keep_alive
    
    def add_resource_config(self, config_id: str, config: Any) -> None:
        """
        添加资源配置
        
        Args:
            config_id: 配置ID（唯一标识）
            config: 资源配置对象
        """
        self._resource_configs[config_id] = config
    
    def get_resource_config(self, config_id: str) -> Any:
        """
        获取资源配置
        
        Args:
            config_id: 配置ID（唯一标识）
            
        Returns:
            资源配置对象
        """
        return self._resource_configs.get(config_id)
    
    def add_pool_config(self, config_id: str, config: PoolConfig) -> None:
        """
        添加资源池配置
        
        Args:
            config_id: 配置ID（唯一标识）
            config: 资源池配置对象
        """
        self._pool_configs[config_id] = config
    
    def get_pool_config(self, config_id: str) -> PoolConfig:
        """
        获取资源池配置
        
        Args:
            config_id: 配置ID（唯一标识）
            
        Returns:
            PoolConfig: 资源池配置对象
        """
        return self._pool_configs.get(config_id)
    
    def validate(self) -> bool:
        """
        验证配置有效性
        
        Returns:
            bool: 配置是否有效
        """
        for config in self._resource_configs.values():
            if hasattr(config, 'validate') and not config.validate():
                return False
        
        for config in self._pool_configs.values():
            if not config.validate():
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "server_port": self._server_port,
            "vram_sufficient_gb": self._vram_sufficient_gb,
            "warmup_timeout": self._warmup_timeout,
            "timeout_keep_alive": self._timeout_keep_alive,
            "resource_configs": {
                k: v.to_dict() if hasattr(v, 'to_dict') else str(v)
                for k, v in self._resource_configs.items()
            },
            "pool_configs": {
                k: v.to_dict() for k, v in self._pool_configs.items()
            }
        }
