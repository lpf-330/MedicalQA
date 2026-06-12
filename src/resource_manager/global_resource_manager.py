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
from src.utils.logger import log_arch_event

if TYPE_CHECKING:
    from src.config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class GlobalResourceManager:
    """
    全局资源管理器
    
    提供全局统一的资源管理接口，是单例模式。
    
    属性：
        INSTANCE: 单例实例
        _resource_registry: 资源工厂注册表
        _pool_manager: 资源池管理器
        _initialized: 是否已初始化
    """
    
    INSTANCE: 'GlobalResourceManager' = None
    
    def __init__(self):
        """初始化全局资源管理器"""
        self._resource_registry = ResourceRegistry()
        self._pool_manager = ResourcePoolManager()
        self._client_type_map: dict = {}
        self._initialized = False
    
    def _convert_resource_config(self, resource_config) -> object:
        """
        将BaseResourceConfig转换为ResourceManager层具体的Config类

        ConfigManager.to_global_config()仅传递BaseResourceConfig原始数据，
        此方法在ResourceManager层内部完成到具体Config类的转换，
        避免Config层反向依赖ResourceManager层。

        Args:
            resource_config: BaseResourceConfig实例

        Returns:
            具体的资源Config类实例（如Neo4jConnectionConfig、ReasoningModelConfig等）

        Raises:
            ValueError: 未知资源类型
        """
        resource_type = resource_config.resource_type

        if resource_type == "neo4j_connection":
            from src.resource_manager.neo4j_connection import Neo4jConnectionConfig
            return Neo4jConnectionConfig(
                uri=resource_config.uri,
                user=resource_config.user,
                password=resource_config.password,
                database=getattr(resource_config, "database", "neo4j")
            )

        elif resource_type == "reasoning_model":
            from src.resource_manager.reasoning_model import ReasoningModelConfig
            return ReasoningModelConfig(
                base_url=resource_config.base_url,
                model_name=getattr(resource_config, "model_name", ""),
                default_temperature=getattr(resource_config, "default_temperature", 0.0),
                default_max_tokens=getattr(resource_config, "default_max_tokens", 1),
                default_top_p=getattr(resource_config, "default_top_p", 0.0),
                default_repetition_penalty=getattr(resource_config, "default_repetition_penalty", 1.15),
                timeout=getattr(resource_config, "timeout", 600.0),
                auto_start=getattr(resource_config, "auto_start", False),
                model_path=getattr(resource_config, "model_path", ""),
                launch_host=getattr(resource_config, "launch_host", "0.0.0.0"),
                launch_port=getattr(resource_config, "launch_port", 30000),
                launch_args=getattr(resource_config, "launch_args", ""),
                startup_timeout=getattr(resource_config, "startup_timeout", 300),
                health_check_interval=getattr(resource_config, "health_check_interval", 5.0),
                shutdown_timeout=getattr(resource_config, "shutdown_timeout", 30),
            )

        elif resource_type == "health_assessment_model":
            from src.resource_manager.health_assessment_model import HealthAssessmentModelConfig
            return HealthAssessmentModelConfig(
                base_url=resource_config.base_url,
                model_name=getattr(resource_config, "model_name", ""),
                default_temperature=getattr(resource_config, "default_temperature", 0.0),
                default_max_tokens=getattr(resource_config, "default_max_tokens", 1),
                default_top_p=getattr(resource_config, "default_top_p", 0.0),
                default_repetition_penalty=getattr(resource_config, "default_repetition_penalty", 1.15),
                timeout=getattr(resource_config, "timeout", 600.0),
                auto_start=getattr(resource_config, "auto_start", False),
                model_path=getattr(resource_config, "model_path", ""),
                launch_host=getattr(resource_config, "launch_host", "0.0.0.0"),
                launch_port=getattr(resource_config, "launch_port", 30001),
                launch_args=getattr(resource_config, "launch_args", ""),
                startup_timeout=getattr(resource_config, "startup_timeout", 300),
                health_check_interval=getattr(resource_config, "health_check_interval", 5.0),
                shutdown_timeout=getattr(resource_config, "shutdown_timeout", 30),
            )

        elif resource_type == "milvus_connection":
            from src.resource_manager.milvus_connection import MilvusConnectionConfig
            return MilvusConnectionConfig(
                uri=resource_config.uri,
                user=resource_config.user,
                password=resource_config.password,
                token=getattr(resource_config, "token", "")
            )

        elif resource_type == "intent_model":
            from src.resource_manager.intent_model import IntentModelConfig
            return IntentModelConfig(
                model_path=resource_config.model_path,
                model_name=getattr(resource_config, "model_name", ""),
                device=getattr(resource_config, "device", ""),
                max_length=getattr(resource_config, "max_length", 128)
            )

        elif resource_type == "vector_model":
            from src.resource_manager.vector_model import VectorModelConfig
            return VectorModelConfig(
                model_path=resource_config.model_path,
                model_name=getattr(resource_config, "model_name", ""),
                device=getattr(resource_config, "device", ""),
                dimension=getattr(resource_config, "dimension", 1024),
                batch_size=getattr(resource_config, "batch_size", 32)
            )

        elif resource_type == "ner_model":
            from src.resource_manager.ner_model import NerModelConfig
            return NerModelConfig(
                model_path=resource_config.model_path,
                model_name=getattr(resource_config, "model_name", ""),
                device=getattr(resource_config, "device", ""),
                max_length=getattr(resource_config, "max_length", 512)
            )

        else:
            raise ValueError(f"Unknown resource type: {resource_type}")

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
        log_arch_event(logger, component="GlobalResourceManager", stage="RESOURCE_LIFECYCLE", event="init_start", status="start", design_id="ARCH-6.1")

        self._register_default_factories()

        for config_id, pool_config in global_config.pool_configs.items():
            resource_config = global_config.get_resource_config(config_id)
            if resource_config is None:
                logger.warning(f"No resource config found for config_id: {config_id}")
                continue

            resource_type = resource_config.resource_type

            if self._pool_manager.has_pool(resource_type, config_id):
                logger.info(f"Pool already exists for {resource_type}:{config_id}, sharing...")
                continue

            factory = self._resource_registry.get_factory(resource_type)
            if factory is None:
                logger.warning(f"No factory registered for {resource_type}")
                continue

            converted_config = self._convert_resource_config(resource_config)
            self._pool_manager.create_pool(resource_type, pool_config, converted_config, config_id)
            logger.info(f"[RESOURCE_POOL_CREATE] type={resource_type}, max_size={pool_config.max_size}, min_idle={pool_config.min_idle}")

            logger.info(f"[CONFIG_POOL_CREATE] resource_type={resource_type}, config_id={config_id}, max_size={pool_config.max_size}, min_idle={pool_config.min_idle}, idle_timeout={pool_config.idle_timeout}, max_wait_time={pool_config.max_wait_time}, allow_dynamic_creation={pool_config.allow_dynamic_creation}, max_pending_requests={pool_config.max_pending_requests}, creation_timeout={pool_config.creation_timeout}, pre_create_check_enabled={pool_config.pre_create_check_enabled}, min_memory_mb={pool_config.min_memory_mb}, min_vram_mb={pool_config.min_vram_mb}")

            logger.info(f"[PoolConfig] resource_type={resource_type}, config_id={config_id}, max_size={pool_config.max_size}, min_idle={pool_config.min_idle}, max_wait_time={pool_config.max_wait_time}, max_pending_requests={pool_config.max_pending_requests}, pre_create_check_enabled={pool_config.pre_create_check_enabled}, min_memory_mb={pool_config.min_memory_mb}, min_vram_mb={pool_config.min_vram_mb}, allow_dynamic_creation={pool_config.allow_dynamic_creation}, creation_timeout={pool_config.creation_timeout}")

            pool = self._pool_manager.get_pool(resource_type, config_id)
            pool._manager_ref = self
            pool.create_initial_resources(pool_config.min_idle)

            logger.info(f"Pool created for {resource_type}:{config_id}: min_idle={pool_config.min_idle}")

        self._initialized = True
        log_arch_event(logger, component="GlobalResourceManager", stage="RESOURCE_LIFECYCLE", event="init_complete", status="success", design_id="ARCH-6.1", pool_count=len(self._pool_manager._pools))
        logger.info("GlobalResourceManager initialized successfully")
    
    def _register_default_factories(self) -> None:
        """
        注册默认的资源工厂
        
        根据项目架构设计，自动注册所有资源类型的工厂类。
        这样可以避免在main.py中手动注册，更符合封装原则。
        
        使用延迟导入（lazy import）避免在初始化时加载所有依赖。
        
        同时注册资源类型对应的客户端类到ResourceHandle的客户端注册表，
        实现业务层通过接口获取客户端，而非直接依赖实现类。
        """
        logger.info("Registering default resource factories...")
        log_arch_event(logger, component="GlobalResourceManager", stage="RESOURCE_LIFECYCLE", event="register_factories_start", status="start", design_id="ARCH-6.1")
        
        factory_mapping = {
            "neo4j_connection": ("src.resource_manager.neo4j_connection", "Neo4jConnectionFactory"),
            "reasoning_model": ("src.resource_manager.reasoning_model", "ReasoningModelFactory"),
            "health_assessment_model": ("src.resource_manager.health_assessment_model", "HealthAssessmentModelFactory"),
            "milvus_connection": ("src.resource_manager.milvus_connection", "MilvusConnectionFactory"),
            "vector_model": ("src.resource_manager.vector_model", "VectorModelFactory"),
            "intent_model": ("src.resource_manager.intent_model", "IntentModelFactory"),
            "ner_model": ("src.resource_manager.ner_model", "NerModelFactory"),
        }
        
        # 资源类型到客户端类的映射，用于ResourceHandle客户端注册表
        client_mapping = {
            "reasoning_model": ("src.resource_manager.reasoning_model", "ReasoningModelClient"),
            "health_assessment_model": ("src.resource_manager.health_assessment_model", "HealthAssessmentModelClient"),
            "neo4j_connection": ("src.resource_manager.neo4j_connection", "Neo4jConnectionClient"),
            "milvus_connection": ("src.resource_manager.milvus_connection", "MilvusConnectionClient"),
            "vector_model": ("src.resource_manager.vector_model", "VectorModelClient"),
            "intent_model": ("src.resource_manager.intent_model", "IntentModelClient"),
            "ner_model": ("src.resource_manager.ner_model", "NerModelClient"),
        }
        
        registered_count = 0
        for resource_type, (module_name, class_name) in factory_mapping.items():
            try:
                module = importlib.import_module(module_name)
                factory_class = getattr(module, class_name)
                self.register_factory(resource_type, factory_class())
                registered_count += 1
                logger.debug(f"Factory registered for {resource_type}")
                logger.info(f"[FACTORY_REGISTER] type={resource_type}, success=True")
                logger.info(f"[RESOURCE_FACTORY_REGISTER] type={resource_type}, factory_class={class_name}")
            except Exception as e:
                logger.warning(f"Failed to register factory for {resource_type}: {str(e)}")
                logger.info(f"[FACTORY_REGISTER] type={resource_type}, success=False")
        
        logger.info(f"Default factories registered: {registered_count}/{len(factory_mapping)}")
        
        # 注册客户端类到ResourceHandle和client_type_map
        logger.info("Registering resource clients to ResourceHandle...")
        client_registered_count = 0
        for resource_type, (module_name, class_name) in client_mapping.items():
            try:
                module = importlib.import_module(module_name)
                client_class = getattr(module, class_name)
                ResourceHandle.register_client(resource_type, client_class)
                self._client_type_map[resource_type] = client_class
                client_registered_count += 1
                logger.debug(f"Client registered for {resource_type}: {class_name}")
            except Exception as e:
                logger.warning(f"Failed to register client for {resource_type}: {str(e)}")

        log_arch_event(logger, component="GlobalResourceManager", stage="RESOURCE_LIFECYCLE", event="register_factories_complete", status="success", design_id="ARCH-6.1", factory_count=registered_count, client_count=client_registered_count)
        logger.info(f"Resource clients registered: {client_registered_count}/{len(client_mapping)}")
    
    def register_factory(self, resource_type: str, factory) -> None:
        """
        注册资源工厂
        
        Args:
            resource_type: 资源类型唯一标识
            factory: 资源工厂实例
        """
        self._resource_registry.register_factory(resource_type, factory)
        self._pool_manager.register_factory(resource_type, factory)
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
        
        logger.info(f"[RESOURCE_ACQUIRE] resource_type={resource_type}, config_id={config_id}, wait_ms={wait_ms}")
        handle = cls.INSTANCE._pool_manager.acquire_from_pool(resource_type, config_id, wait_ms)
        if handle is not None:
            log_arch_event(logger, component="GlobalResourceManager", stage="RESOURCE_LIFECYCLE", event="acquire", status="success", design_id="ARCH-6.1", resource_type=resource_type, config_id=config_id)
            logger.info(f"[RESOURCE_ACQUIRE] resource_type={resource_type}, config_id={config_id}, resource_id={handle.resource_id}, status=success")
        else:
            log_arch_event(logger, component="GlobalResourceManager", stage="RESOURCE_LIFECYCLE", event="acquire", status="failed", design_id="ARCH-6.1", resource_type=resource_type, config_id=config_id)
            logger.warning(f"[RESOURCE_ACQUIRE] resource_type={resource_type}, config_id={config_id}, status=failed")
        return handle
    
    @classmethod
    def release(cls, handle: ResourceHandle) -> None:
        """
        释放资源
        
        Args:
            handle: 资源句柄
        """
        if cls.INSTANCE is None:
            raise RuntimeError("GlobalResourceManager not initialized")
        
        log_arch_event(logger, component="GlobalResourceManager", stage="RESOURCE_LIFECYCLE", event="release", status="success", design_id="ARCH-6.1", resource_type=handle.resource_type)
        logger.info(f"[RESOURCE_RELEASE] resource_type={handle.resource_type}, resource_id={handle.resource_id}")
        cls.INSTANCE._pool_manager.release_to_pool(handle)

    @classmethod
    def destroy(cls, handle: ResourceHandle) -> None:
        """
        彻底销毁资源（从池中移除并关闭连接）

        与release()不同，destroy()不会将资源归还到空闲池，
        而是从池中彻底移除并关闭连接、释放资源。
        适用于需要断开连接、不再复用的场景。

        Args:
            handle: 资源句柄
        """
        if cls.INSTANCE is None:
            raise RuntimeError("GlobalResourceManager not initialized")

        log_arch_event(logger, component="GlobalResourceManager", stage="RESOURCE_LIFECYCLE", event="destroy", status="success", design_id="ARCH-6.1", resource_type=handle.resource_type)
        logger.info(f"[RESOURCE_DESTROY] GlobalResourceManager.destroy调用, 资源ID={handle.resource_id}")
        cls.INSTANCE._pool_manager.destroy(handle)
    
    def shutdown(self) -> None:
        """关闭资源管理器，释放所有资源"""
        logger.info("Shutting down GlobalResourceManager...")
        log_arch_event(logger, component="GlobalResourceManager", stage="RESOURCE_LIFECYCLE", event="shutdown", status="start", design_id="ARCH-6.1")
        self._pool_manager.destroy_all()
        self._initialized = False
        log_arch_event(logger, component="GlobalResourceManager", stage="RESOURCE_LIFECYCLE", event="shutdown", status="success", design_id="ARCH-6.1")
        logger.info("GlobalResourceManager shut down successfully")
    
    def get_stats(self) -> dict:
        """获取资源池统计信息"""
        return self._pool_manager.get_pool_stats()
    
    @property
    def resource_registry(self) -> ResourceRegistry:
        """获取资源工厂注册表"""
        return self._resource_registry

    @property
    def pool_manager(self) -> ResourcePoolManager:
        """获取资源池管理器"""
        return self._pool_manager

    @property
    def client_type_map(self) -> dict:
        """获取资源客户端类型映射"""
        return self._client_type_map
    
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
        log_arch_event(logger, component="GlobalResourceManager", stage="RESOURCE_LIFECYCLE", event="initialize", status="start", design_id="ARCH-6.1")

        # 1. 加载配置
        logger.info("步骤1: 加载配置...")
        config_manager = get_config_manager()
        # 注意：get_config_manager()内部已调用load_all_configs()，无需重复调用

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

        log_arch_event(logger, component="GlobalResourceManager", stage="RESOURCE_LIFECYCLE", event="initialize", status="success", design_id="ARCH-6.1", stats=str(stats))
        logger.info("=" * 60)
        logger.info("GlobalResourceManager初始化完成")
        logger.info("=" * 60)

        return config_manager


GlobalResourceManager.INSTANCE = GlobalResourceManager()
