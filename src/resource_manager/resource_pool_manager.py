# AI辅助生成：GLM-5，2026-04-15
"""
资源池管理器模块

负责管理多个资源池，协调资源的创建和分配。
"""

from typing import Dict, Tuple, TYPE_CHECKING

from .config import PoolConfig
from .resource_pool import ResourcePool

if TYPE_CHECKING:
    from .resource import Resource
    from .resource_handle import ResourceHandle
    from .resource_factory import ResourceFactory
    from .resource_registry import ResourceRegistry
    from .resource_config import ResourceConfig


class ResourcePoolManager:
    """
    资源池管理器类
    
    负责管理多个资源池，协调资源的创建和分配。
    是资源池生命周期的核心管理者。
    
    核心职责：
    - 创建和管理不同类型的资源池
    - 提供资源申请和释放的统一接口
    - 协调资源工厂和资源池的协作
    - 维护资源池的健康状态
    
    设计说明：
    - ResourcePoolManager管理所有类型的资源池
    - 通过resource_type作为key，实现资源池的快速查找
    - 支持动态创建资源池，可在运行时添加新的资源类型
    - 与ResourceRegistry协作，获取资源工厂创建资源
    """
    
    def __init__(self, resource_registry: 'ResourceRegistry'):
        """
        初始化资源池管理器
        
        Args:
            resource_registry: 资源注册器实例，用于获取资源工厂
        """
        self._pools: Dict[str, ResourcePool] = {}
        self._resource_registry = resource_registry
        self._resource_configs: Dict[str, 'ResourceConfig'] = {}
    
    @property
    def pools(self) -> Dict[str, ResourcePool]:
        """
        获取资源池缓存容器（只读）
        
        Returns:
            Dict[str, ResourcePool]: 资源池映射字典
        """
        return self._pools.copy()
    
    def set_resource_config(self, resource_type: str, config: 'ResourceConfig') -> None:
        """
        设置资源配置
        
        用于资源池创建新资源时获取配置信息。
        
        Args:
            resource_type: 资源类型唯一标识
            config: 资源配置对象
        """
        self._resource_configs[resource_type] = config
    
    def create_pool(self, resource_type: str, pool_config: PoolConfig) -> ResourcePool:
        """
        根据指定的资源类型和池化配置，创建并注册对应类型的ResourcePool资源池
        
        完成资源池的初始化、参数配置与生命周期绑定。
        
        Args:
            resource_type: 资源类型唯一标识
            pool_config: 资源池配置对象
            
        Returns:
            ResourcePool: 创建的资源池实例
            
        Raises:
            ValueError: 如果资源类型已存在对应的资源池
            KeyError: 如果资源类型未注册对应的资源工厂
            
        Example:
            >>> manager = ResourcePoolManager(registry)
            >>> pool_config = PoolConfig(max_size=10, min_idle=2, idle_timeout=60000, max_wait_time=5000)
            >>> pool = manager.create_pool('neo4j_database', pool_config)
        """
        if resource_type in self._pools:
            raise ValueError(f"资源类型 '{resource_type}' 已存在对应的资源池")
        
        # 获取资源工厂
        factory = self._resource_registry.get_factory(resource_type)
        
        # 创建资源池
        pool = ResourcePool(resource_type, pool_config, factory)
        
        # 注册资源池
        self._pools[resource_type] = pool
        
        return pool
    
    def get_factory(self, resource_type: str) -> 'ResourceFactory':
        """
        根据资源类型获取对应的ResourceFactory实例
        
        用于资源的实例化创建、扩容补充，
        是ResourcePool生成可用资源的核心入口。
        
        Args:
            resource_type: 资源类型唯一标识
            
        Returns:
            ResourceFactory: 对应资源类型的工厂实例
            
        Raises:
            KeyError: 当资源类型不存在时抛出
            
        Example:
            >>> manager = ResourcePoolManager(registry)
            >>> factory = manager.get_factory('neo4j_database')
        """
        return self._resource_registry.get_factory(resource_type)
    
    def acquire_from_pool(self, resource_type: str) -> Tuple[str, 'Resource']:
        """
        从指定类型的ResourcePool中申请获取可用资源实例
        
        返回包含resource_id和原始Resource对象的元组，
        供ResourceHandle封装后交付业务层使用。
        
        如果资源池没有空闲资源且未满，则创建新资源。
        
        Args:
            resource_type: 资源类型唯一标识
            
        Returns:
            Tuple[str, Resource]: 包含资源ID和资源实例的元组
            
        Raises:
            KeyError: 当资源类型不存在对应的资源池时抛出
            RuntimeError: 当资源池已满且无空闲资源时抛出
            RuntimeError: 当资源创建失败时抛出
            
        Example:
            >>> manager = ResourcePoolManager(registry)
            >>> resource_id, resource = manager.acquire_from_pool('neo4j_database')
        """
        if resource_type not in self._pools:
            raise KeyError(f"资源类型 '{resource_type}' 不存在对应的资源池")
        
        pool = self._pools[resource_type]
        
        # 检查是否有空闲资源
        if pool.has_available_resource():
            try:
                return pool.activate()
            except RuntimeError:
                # 没有空闲资源，需要创建新资源
                pass
        
        # 没有空闲资源且资源池未满，创建新资源
        if not pool.is_full():
            # 获取资源工厂和配置
            factory = self.get_factory(resource_type)
            config = self._resource_configs.get(resource_type)
            
            if config is None:
                raise RuntimeError(f"资源类型 '{resource_type}' 缺少资源配置")
            
            # 创建新资源
            resource = factory.create(config)
            
            # 添加到资源池的空闲资源中
            resource_id = pool.add_idle_resource(resource)
            
            # 激活资源
            return pool.activate()
        
        # 资源池已满且无空闲资源
        raise RuntimeError(f"资源池已满，无法获取资源。资源类型: {resource_type}")
    
    def release_to_pool(self, resource_type: str, resource_id: str) -> None:
        """
        将使用完毕的资源归还至对应资源池
        
        完成资源的状态重置、回收与复用，
        支撑池化资源的循环利用与生命周期闭环。
        
        Args:
            resource_type: 资源类型唯一标识
            resource_id: 资源的唯一标识ID
            
        Raises:
            KeyError: 当资源类型不存在对应的资源池时抛出
            KeyError: 当资源ID不存在于活跃资源中时抛出
            
        Example:
            >>> manager = ResourcePoolManager(registry)
            >>> resource_id, resource = manager.acquire_from_pool('neo4j_database')
            >>> # 使用资源...
            >>> manager.release_to_pool('neo4j_database', resource_id)
        """
        if resource_type not in self._pools:
            raise KeyError(f"资源类型 '{resource_type}' 不存在对应的资源池")
        
        pool = self._pools[resource_type]
        pool.release(resource_id)
    
    def get_pool(self, resource_type: str) -> ResourcePool:
        """
        获取指定类型的资源池
        
        Args:
            resource_type: 资源类型唯一标识
            
        Returns:
            ResourcePool: 资源池实例
            
        Raises:
            KeyError: 当资源类型不存在对应的资源池时抛出
        """
        if resource_type not in self._pools:
            raise KeyError(f"资源类型 '{resource_type}' 不存在对应的资源池")
        
        return self._pools[resource_type]
    
    def has_pool(self, resource_type: str) -> bool:
        """
        检查是否存在指定类型的资源池
        
        Args:
            resource_type: 资源类型唯一标识
            
        Returns:
            bool: 是否存在对应的资源池
        """
        return resource_type in self._pools
    
    def remove_pool(self, resource_type: str) -> None:
        """
        移除指定类型的资源池
        
        注意：此操作会清空资源池中的所有资源
        
        Args:
            resource_type: 资源类型唯一标识
            
        Raises:
            KeyError: 当资源类型不存在对应的资源池时抛出
        """
        if resource_type not in self._pools:
            raise KeyError(f"资源类型 '{resource_type}' 不存在对应的资源池")
        
        pool = self._pools.pop(resource_type)
        pool.clear()
    
    def get_all_resource_types(self) -> list:
        """
        获取所有已创建资源池的资源类型列表
        
        Returns:
            list: 资源类型列表
        """
        return list(self._pools.keys())
    
    def evict_idle_resources_all(self) -> Dict[str, int]:
        """
        驱逐所有资源池中的空闲资源
        
        Returns:
            Dict[str, int]: 各资源池驱逐的资源数量映射
        """
        evicted_counts = {}
        for resource_type, pool in self._pools.items():
            evicted_count = pool.evict_idle_resources()
            evicted_counts[resource_type] = evicted_count
        return evicted_counts
    
    def shutdown(self) -> None:
        """
        关闭资源池管理器
        
        清空所有资源池，释放所有资源。
        注意：此操作不可逆，谨慎使用。
        """
        for pool in self._pools.values():
            try:
                pool.clear()
            except Exception as e:
                print(f"清空资源池失败: {e}")
        
        self._pools.clear()
    
    def __repr__(self) -> str:
        """返回资源池管理器的字符串表示"""
        return f"ResourcePoolManager(pools_count={len(self._pools)})"
