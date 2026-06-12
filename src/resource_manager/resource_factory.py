# AI辅助生成：GLM-5，2026-04-15
"""
资源工厂接口模块

定义资源工厂的基本行为，包括创建资源、销毁资源等。
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .resource import Resource
    from .resource_config import ResourceConfig


class ResourceFactory(ABC):
    """
    资源工厂接口
    
    定义资源工厂的基本行为，所有资源工厂类必须实现此接口。
    资源工厂负责资源实例的创建和销毁，是资源生命周期的核心管理者。
    
    核心职责：
    - 根据资源配置创建资源实例
    - 销毁资源实例，释放相关资源
    - 管理资源的初始化和清理过程
    
    设计说明：
    - 每种资源类型对应一个ResourceFactory实现类
    - ResourceFactory由ResourceRegistry注册和管理
    - ResourceFactory被ResourcePool使用，用于资源池的扩容和资源补充
    """
    
    @abstractmethod
    def create(self, config: 'ResourceConfig') -> 'Resource':
        """
        根据传入的ResourceConfig资源配置，创建并初始化一个全新的Resource资源实例
        
        是资源实例化的核心入口，用于资源池的初始化、空闲资源补充、池化扩容等场景，
        返回可直接投入使用的资源对象。
        
        Args:
            config: ResourceConfig实例，包含资源的配置信息
                - resource_type: 资源类型唯一标识
                - resource_name: 资源业务名称
                - config_protocol: 资源个性化配置协议
                
        Returns:
            Resource: 创建并初始化完成的资源实例
            
        Raises:
            ResourceException: 如果资源创建失败
            ConfigException: 如果配置无效或缺少必要参数
            
        Example:
            >>> factory = Neo4jResourceFactory()
            >>> config = Neo4jConnectionConfig(...)
            >>> resource = factory.create(config)
            >>> resource.get_type()
            'neo4j_database'
        """
        pass
    
    @abstractmethod
    def destroy(self, resource: 'Resource') -> None:
        """
        销毁指定的Resource资源实例
        
        执行资源的关闭、连接释放、内存清理、状态重置等收尾操作，
        用于资源池驱逐超时/冗余空闲资源、系统停机时的全量资源回收，
        保障资源安全释放，避免资源泄漏。
        
        Args:
            resource: 要销毁的Resource实例
            
        注意：
        - 销毁前应确保资源处于空闲状态（未被激活）
        - 销毁操作不可逆，销毁后资源实例不可再使用
        - 销毁过程会释放所有底层资源
        
        Raises:
            ResourceException: 如果资源销毁失败
            ResourceException: 如果资源处于活跃状态，无法销毁
            
        Example:
            >>> factory = Neo4jResourceFactory()
            >>> resource = factory.create(config)
            >>> # 使用资源...
            >>> resource.deactivate()  # 先停用资源
            >>> factory.destroy(resource)  # 再销毁资源
        """
        pass
