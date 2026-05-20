# AI辅助生成：GLM-5，2026-04-15
# -*- coding: utf-8 -*-
"""
全局资源管理器

提供全局统一的资源管理接口，是单例模式。
"""

import logging
import importlib
from typing import Optional, TYPE_CHECKING

from .resource_registry import ResourceRegistry
from .pool_manager import ResourcePoolManager
from .resource_handle import ResourceHandle
from src.config.global_config import GlobalConfig

if TYPE_CHECKING:
    from src.config.config_manager import ConfigManager

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
        self._poolManager = ResourcePoolManager()
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
        
        self._register_default_factories()
        
        for config_id, pool_config in global_config.pool_configs.items():
            resource_config = global_config.get_resource_config(config_id)
            if resource_config is None:
                logger.warning(f"No resource config found for config_id: {config_id}")
                continue
            
            resource_type = resource_config.resource_type
            
            if self._poolManager.has_pool(resource_type, config_id):
                logger.info(f"Pool already exists for {resource_type}:{config_id}, sharing...")
                continue
            
            factory = self._resourceRegistry.get_factory(resource_type)
            if factory is None:
                logger.warning(f"No factory registered for {resource_type}")
                continue
            
            self._poolManager.create_pool(resource_type, pool_config, resource_config, config_id)
            
            pool = self._poolManager.get_pool(resource_type, config_id)
            pool.create_initial_resources(pool_config.min_idle)
            
            logger.info(f"Pool created for {resource_type}:{config_id}: min_idle={pool_config.min_idle}")
        
        self._initialized = True
        logger.info("GlobalResourceManager initialized successfully")
    
    def _register_default_factories(self) -> None:
        """
        注册默认的资源工厂
        
        根据项目架构设计，自动注册所有资源类型的工厂类。
        这样可以避免在main.py中手动注册，更符合封装原则。
        
        使用延迟导入（lazy import）避免在初始化时加载所有依赖。
        """
        logger.info("Registering default resource factories...")
        
        factory_mapping = {
            "neo4j_connection": ("src.resource_manager.neo4j_connection", "Neo4jConnectionFactory"),
            "vllm_model": ("src.resource_manager.vllm_model", "VLLMModelFactory"),
            "milvus_connection": ("src.resource_manager.milvus_connection", "MilvusConnectionFactory"),
            "vector_model": ("src.resource_manager.vector_model", "VectorModelFactory"),
            "intent_model": ("src.resource_manager.intent_model", "IntentModelFactory"),
        }
        
        registered_count = 0
        for resource_type, (module_name, class_name) in factory_mapping.items():
            try:
                module = importlib.import_module(module_name)
                factory_class = getattr(module, class_name)
                self.register_factory(resource_type, factory_class())
                registered_count += 1
                logger.debug(f"Factory registered for {resource_type}")
            except Exception as e:
                logger.warning(f"Failed to register factory for {resource_type}: {str(e)}")
        
        logger.info(f"Default factories registered: {registered_count}/{len(factory_mapping)}")
    
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
    def acquire(cls, resource_type: str, config_id: str = None, wait_ms: int = None) -> Optional[ResourceHandle]:
        """
        获取资源
        
        Args:
            resource_type: 资源类型
            config_id: 配置ID，如果为None则使用默认值
            wait_ms: 等待时间（毫秒）
            
        Returns:
            ResourceHandle: 资源句柄
        """
        if cls.INSTANCE is None:
            raise RuntimeError("GlobalResourceManager not initialized")
        
        return cls.INSTANCE._poolManager.acquire(resource_type, config_id, wait_ms)
    
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
    def poolManager(self) -> ResourcePoolManager:
        """获取资源池管理器"""
        return self._poolManager
    
    @property
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized
    
    def initialize_from_config_manager(self, config_manager: 'ConfigManager') -> None:
        """
        从ConfigManager初始化资源管理器
        
        Args:
            config_manager: 配置管理器实例
        """
        global_config = config_manager.to_global_config()
        self._init_global_resource_manager(global_config)
    
    @classmethod
    def initialize(cls) -> 'ConfigManager':
        """
        统一的初始化接口（类方法）
        
        自动完成以下步骤：
        1. 加载配置（通过ConfigManager）
        2. 验证配置
        3. 初始化资源管理器
        4. 注册资源工厂
        5. 创建资源池
        
        Returns:
            ConfigManager: 配置管理器实例
            
        Example:
            >>> config_manager = GlobalResourceManager.initialize()
            >>> # 现在所有资源都已初始化完成
        """
        from src.config.config_manager import get_config_manager
        
        logger.info("=" * 60)
        logger.info("开始初始化GlobalResourceManager...")
        logger.info("=" * 60)
        
        # 1. 加载配置
        logger.info("步骤1: 加载配置...")
        config_manager = get_config_manager()
        config_manager.load_all_configs()
        
        logger.info(f"  业务配置: {list(config_manager.business_configs.keys())}")
        logger.info(f"  资源配置: {list(config_manager.resource_configs.keys())}")
        logger.info(f"  资源池配置: {list(config_manager.pool_configs.keys())}")
        
        # 2. 验证配置
        logger.info("步骤2: 验证配置...")
        if not config_manager.validate():
            raise RuntimeError("配置验证失败")
        
        # 3. 初始化资源管理器
        logger.info("步骤3: 初始化资源管理器...")
        cls.INSTANCE.initialize_from_config_manager(config_manager)
        
        # 4. 获取统计信息
        stats = cls.INSTANCE.get_stats()
        logger.info(f"步骤4: 资源初始化完成: {stats}")
        
        logger.info("=" * 60)
        logger.info("GlobalResourceManager初始化完成")
        logger.info("=" * 60)
        
        return config_manager


GlobalResourceManager.INSTANCE = GlobalResourceManager()
