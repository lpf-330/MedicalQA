# -*- coding: utf-8 -*-
"""
统一配置管理器

管理业务配置和资源配置的加载、验证、导出。
"""

import logging
from typing import Dict, Any, List, Optional, Set
from pathlib import Path

from src.config.base_config import BaseResourceConfig, BusinessConfig
from src.config.pool_config import PoolConfig
from src.config.global_config import GlobalConfig

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    统一配置管理器
    
    负责管理业务配置和资源配置的加载、验证、导出。
    
    属性：
        business_configs: 业务配置字典
        resource_configs: 资源配置字典
        pool_configs: 资源池配置字典
    """
    
    _instance: Optional['ConfigManager'] = None
    
    def __new__(cls) -> 'ConfigManager':
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化配置管理器"""
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._business_configs: Dict[str, BusinessConfig] = {}
        self._resource_configs: Dict[str, BaseResourceConfig] = {}
        self._pool_configs: Dict[str, PoolConfig] = {}
        
        self._initialized = True
        logger.info("[ConfigManager] 配置管理器初始化完成")
    
    @property
    def business_configs(self) -> Dict[str, BusinessConfig]:
        """获取所有业务配置"""
        return self._business_configs
    
    @property
    def resource_configs(self) -> Dict[str, BaseResourceConfig]:
        """获取所有资源配置"""
        return self._resource_configs
    
    @property
    def pool_configs(self) -> Dict[str, PoolConfig]:
        """获取所有资源池配置"""
        return self._pool_configs
    
    def load_all_configs(self) -> None:
        """
        加载所有配置
        
        按以下顺序加载：
        1. 扫描业务配置目录
        2. 解析业务配置，收集所需的资源配置文件名
        3. 资源配置去重
        4. 加载所需的资源配置
        """
        logger.info("[ConfigManager] 开始加载所有配置...")
        
        self._load_business_configs()
        
        required_resources = self._get_required_resource_configs()
        logger.info(f"[ConfigManager] 所需资源配置: {required_resources}")
        
        self._load_resource_configs(required_resources)
        
        logger.info("[ConfigManager] 所有配置加载完成")
    
    def _load_business_configs(self) -> None:
        """加载所有业务配置"""
        from src.config.business import get_all_business_configs, load_business_config
        
        business_config_files = get_all_business_configs()
        logger.info(f"[ConfigManager] 发现业务配置文件: {list(business_config_files.keys())}")
        
        for config_name in business_config_files:
            try:
                business_config = load_business_config(config_name)
                self._business_configs[config_name] = business_config
                logger.info(f"[ConfigManager] 业务配置加载成功: {config_name}")
            except Exception as e:
                logger.error(f"[ConfigManager] 业务配置加载失败: {config_name}, error={e}")
    
    def _get_required_resource_configs(self) -> Set[str]:
        """
        获取所有业务配置所需的资源配置文件名（去重）
        
        Returns:
            Set[str]: 资源配置文件名集合
        """
        required_resources = set()
        
        for config_name, business_config in self._business_configs.items():
            if hasattr(business_config, "resource_configs"):
                required_resources.update(business_config.resource_configs)
        
        return required_resources
    
    def _load_resource_configs(self, required_resources: Set[str]) -> None:
        """
        加载所需的资源配置
        
        Args:
            required_resources: 所需的资源配置文件名集合
        """
        from src.config.resources import load_resource_config
        
        for config_name in required_resources:
            try:
                config_data = load_resource_config(config_name)
                
                if config_data.get("resource_config"):
                    self._resource_configs[config_name] = config_data["resource_config"]
                
                if config_data.get("pool_config"):
                    self._pool_configs[config_name] = config_data["pool_config"]
                
                logger.info(f"[ConfigManager] 资源配置加载成功: {config_name}")
            except Exception as e:
                logger.error(f"[ConfigManager] 资源配置加载失败: {config_name}, error={e}")
    
    def get_business_config(self, business_id: str) -> Optional[BusinessConfig]:
        """
        获取指定业务配置
        
        Args:
            business_id: 业务ID
            
        Returns:
            BusinessConfig: 业务配置实例
        """
        return self._business_configs.get(business_id)
    
    def get_resource_config(self, config_id: str) -> Optional[BaseResourceConfig]:
        """
        获取指定资源配置
        
        Args:
            config_id: 资源配置ID
            
        Returns:
            BaseResourceConfig: 资源配置实例
        """
        return self._resource_configs.get(config_id)
    
    def get_pool_config(self, config_id: str) -> Optional[PoolConfig]:
        """
        获取指定资源池配置
        
        Args:
            config_id: 资源配置ID
            
        Returns:
            PoolConfig: 资源池配置实例
        """
        return self._pool_configs.get(config_id)
    
    def validate(self) -> bool:
        """
        验证所有配置有效性
        
        Returns:
            bool: 所有配置是否有效
        """
        errors = []
        
        for config_id, resource_config in self._resource_configs.items():
            if not resource_config.validate():
                errors.append(f"资源配置 {config_id} 验证失败")
        
        for business_id, business_config in self._business_configs.items():
            if not business_config.validate():
                errors.append(f"业务配置 {business_id} 验证失败")
        
        if errors:
            for error in errors:
                logger.error(f"[ConfigManager] 配置验证失败: {error}")
            return False
        
        logger.info("[ConfigManager] 所有配置验证通过")
        return True
    
    def to_global_config(self) -> GlobalConfig:
        """
        转换为GlobalConfig实例
        
        Returns:
            GlobalConfig: 全局资源配置实例
        """
        global_config = GlobalConfig()
        
        for config_id, resource_config in self._resource_configs.items():
            resource_type = resource_config.resource_type
            
            if resource_type == "neo4j_connection":
                from src.resource_manager.neo4j_connection import Neo4jConnectionConfig
                
                neo4j_config = Neo4jConnectionConfig(
                    uri=resource_config.uri,
                    user=resource_config.user,
                    password=resource_config.password,
                    database=getattr(resource_config, "database", "neo4j")
                )
                global_config.add_resource_config(resource_type, neo4j_config)
                
            elif resource_type == "vllm_model":
                from src.resource_manager.vllm_model import VLLMModelConfig
                
                vllm_config = VLLMModelConfig(
                    model_path=resource_config.model_path,
                    model_name=getattr(resource_config, "model_name", ""),
                    tensor_parallel_size=getattr(resource_config, "tensor_parallel_size", 1),
                    max_model_len=getattr(resource_config, "max_model_len", 8192),
                    gpu_memory_utilization=getattr(resource_config, "gpu_memory_utilization", 0.9)
                )
                global_config.add_resource_config(resource_type, vllm_config)
        
        for config_id, pool_config in self._pool_configs.items():
            if config_id in self._resource_configs:
                resource_type = self._resource_configs[config_id].resource_type
                global_config.add_pool_config(resource_type, pool_config)
        
        return global_config
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"ConfigManager("
            f"business_configs={len(self._business_configs)}, "
            f"resource_configs={len(self._resource_configs)}, "
            f"pool_configs={len(self._pool_configs)})"
        )


_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """
    获取配置管理器实例（单例模式）
    
    Returns:
        ConfigManager: 配置管理器实例
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
        _config_manager.load_all_configs()
    return _config_manager
