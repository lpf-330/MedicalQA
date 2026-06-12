"""
资源配置接口模块

定义资源配置的基本属性，包括资源类型、资源名称、配置协议等。
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar


# 定义配置协议的泛型类型变量
T = TypeVar('T')


class ResourceConfig(ABC, Generic[T]):
    """
    资源配置接口
    
    定义资源配置的基本属性，所有资源配置类必须实现此接口。
    资源配置用于存储资源的初始化参数，为资源实例化、工厂创建提供配置支撑。
    
    核心职责：
    - 提供资源的唯一类型标识
    - 提供资源的业务名称（用于日志、监控、运维界面的人工识别）
    - 提供资源的个性化配置协议对象（适配不同类型资源的专属配置）
    
    泛型说明：
    - T: 资源的个性化配置协议类型，不同资源类型可以有不同的配置协议
    """
    
    @property
    @abstractmethod
    def resource_type(self) -> str:
        """
        获取资源的唯一类型标识
        
        用于全局资源的注册、分类与路由匹配，关联对应资源池、资源工厂，
        是资源在全局体系中的核心身份标识。
        
        Returns:
            str: 资源的唯一类型标识字符串
            
        Example:
            >>> config.resource_type
            'neo4j_database'
        """
        pass
    
    @property
    @abstractmethod
    def resource_name(self) -> str:
        """
        获取资源的业务名称/显示名称
        
        用于日志、监控、运维界面的人工识别与管理，补充唯一标识的可读性，
        方便运维排查。
        
        Returns:
            str: 资源的业务名称字符串
            
        Example:
            >>> config.resource_name
            'Neo4j图数据库-生产环境'
        """
        pass
    
    @property
    @abstractmethod
    def config_protocol(self) -> T:
        """
        获取资源的个性化配置协议对象
        
        泛型T适配不同类型资源的专属配置（如数据库JDBC配置、HTTP客户端配置等），
        为资源实例化、初始化提供参数支撑，实现异构资源配置的统一封装。
        
        Returns:
            T: 资源的个性化配置协议对象
            
        Example:
            >>> config.config_protocol
            Neo4jConnectionConfig(uri='bolt://localhost:7687', user='neo4j', password='***')
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> dict:
        """
        将资源配置转换为字典格式
        
        用于配置的序列化和持久化，方便配置的存储和传输。
        
        Returns:
            dict: 包含资源配置信息的字典
            
        Example:
            >>> config.to_dict()
            {
                'resource_type': 'neo4j_database',
                'resource_name': 'Neo4j图数据库-生产环境',
                'config_protocol': {...}
            }
        """
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """
        验证资源配置的有效性
        
        检查资源配置的各项参数是否合法、完整，确保配置可用于资源实例化。
        
        Returns:
            bool: 配置是否有效
                - True: 配置有效，可用于资源实例化
                - False: 配置无效，缺少必要参数或参数不合法
                
        Raises:
            ConfigException: 如果配置验证失败且需要抛出异常
            
        Example:
            >>> config.validate()
            True
        """
        pass
