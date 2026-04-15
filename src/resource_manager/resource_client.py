"""
资源客户端接口模块

定义资源客户端的基本行为，包括获取资源类型、获取原始资源等。
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .resource import Resource


class ResourceClient(ABC):
    """
    资源客户端接口
    
    定义资源客户端的基本行为，所有资源客户端类必须实现此接口。
    资源客户端是对资源的封装，为业务层提供统一的资源访问接口，
    隐藏资源的底层实现细节。
    
    核心职责：
    - 提供资源类型的唯一标识
    - 提供对原始资源实例的访问能力
    - 为业务层提供统一的资源操作接口
    
    设计说明：
    - ResourceClient是业务层访问资源的统一入口
    - ResourceClient封装了Resource实例，提供业务友好的接口
    - ResourceClient由ResourceHandle持有，通过ResourceHandle进行生命周期管理
    """
    
    @abstractmethod
    def get_resource_type(self) -> str:
        """
        获取当前资源客户端对应的资源类型唯一标识
        
        返回的字符串用于GlobalResourceManager的资源类型匹配、注册校验与生命周期调度，
        是资源分类管理的核心标识。
        
        Returns:
            str: 资源类型的唯一标识字符串
            
        Example:
            >>> client.get_resource_type()
            'neo4j_database'
        """
        pass
    
    @abstractmethod
    def get_raw_resource(self) -> 'Resource':
        """
        获取资源客户端封装的原始资源实例
        
        返回底层的Resource核心对象，供业务层进行原生能力的扩展调用，
        暴露资源的底层核心执行能力。
        
        注意：
        - 此方法返回的是原始Resource实例，使用时需要注意资源状态管理
        - 建议优先使用ResourceClient提供的业务方法，而非直接操作原始资源
        - 直接操作原始资源可能绕过客户端的状态管理和安全检查
        
        Returns:
            Resource: 原始资源实例
            
        Example:
            >>> raw_resource = client.get_raw_resource()
            >>> raw_resource.get_type()
            'neo4j_database'
        """
        pass
