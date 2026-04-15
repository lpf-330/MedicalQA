# -*- coding: utf-8 -*-
"""
Neo4j连接资源封装

封装Neo4j数据库连接资源，提供统一的资源管理接口。
"""

import time
from typing import Any, Dict, List, Optional

from src.resource_manager.resource import Resource
from src.resource_manager.resource_config import ResourceConfig
from src.resource_manager.resource_factory import ResourceFactory
from src.resource_manager.resource_client import ResourceClient
from src.adapters import Neo4jAdapterImpl


class Neo4jConnectionResource(Resource):
    """
    Neo4j连接资源类
    
    封装Neo4j数据库连接，实现Resource接口。
    
    属性：
        _config: Neo4j连接配置
        _adapter: Neo4j适配器实例
        _last_used_time: 最后使用时间戳
        _is_active: 资源活跃状态
    """
    
    def __init__(self, config: 'Neo4jConnectionConfig'):
        """
        初始化Neo4j连接资源
        
        Args:
            config: Neo4j连接配置
        """
        self._config = config
        self._adapter: Optional[Neo4jAdapterImpl] = None
        self._last_used_time = int(time.time() * 1000)
        self._is_active = False
    
    def get_type(self) -> str:
        """获取资源类型标识"""
        return "neo4j_connection"
    
    def get_last_used_time(self) -> int:
        """获取最后使用时间戳"""
        return self._last_used_time
    
    def is_activate(self) -> bool:
        """校验资源活跃状态"""
        return self._is_active
    
    def activate(self) -> None:
        """激活资源"""
        if self._is_active:
            return
        
        config_protocol = self._config.config_protocol
        self._adapter = Neo4jAdapterImpl(
            uri=config_protocol["uri"],
            user=config_protocol["user"],
            password=config_protocol["password"]
        )
        self._adapter.connect()
        self._is_active = True
        self._last_used_time = int(time.time() * 1000)
    
    def deactivate(self) -> None:
        """停用资源"""
        if not self._is_active:
            return
        
        if self._adapter is not None:
            self._adapter.disconnect()
        self._is_active = False
    
    def destroy(self) -> None:
        """销毁资源"""
        self.deactivate()
        self._adapter = None
    
    def get_adapter(self) -> Optional[Neo4jAdapterImpl]:
        """
        获取Neo4j适配器实例
        
        Returns:
            Neo4jAdapterImpl: Neo4j适配器实例
        """
        return self._adapter


class Neo4jConnectionConfig(ResourceConfig[Dict[str, str]]):
    """
    Neo4j连接配置类
    
    实现ResourceConfig接口，存储Neo4j连接配置。
    
    属性：
        _resource_type: 资源类型标识
        _resource_name: 资源业务名称
        _config_protocol: 个性化配置协议
    """
    
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        resource_name: str = "Neo4j图数据库"
    ):
        """
        初始化Neo4j连接配置
        
        Args:
            uri: Neo4j连接URI
            user: 用户名
            password: 密码
            database: 数据库名
            resource_name: 资源业务名称
        """
        self._resource_type = "neo4j_connection"
        self._resource_name = resource_name
        self._config_protocol: Dict[str, str] = {
            "uri": uri,
            "user": user,
            "password": password,
            "database": database
        }
    
    @property
    def resource_type(self) -> str:
        """获取资源类型标识"""
        return self._resource_type
    
    @property
    def resource_name(self) -> str:
        """获取资源业务名称"""
        return self._resource_name
    
    @property
    def config_protocol(self) -> Dict[str, str]:
        """获取个性化配置协议"""
        return self._config_protocol
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "resource_type": self._resource_type,
            "resource_name": self._resource_name,
            "config_protocol": self._config_protocol
        }
    
    def validate(self) -> bool:
        """验证配置有效性"""
        if not self._config_protocol.get("uri"):
            return False
        if not self._config_protocol.get("user"):
            return False
        if not self._config_protocol.get("password"):
            return False
        return True


class Neo4jConnectionFactory(ResourceFactory):
    """
    Neo4j连接工厂类
    
    实现ResourceFactory接口，负责Neo4j连接资源的创建和销毁。
    """
    
    def create(self, config: ResourceConfig) -> Resource:
        """
        创建Neo4j连接资源
        
        Args:
            config: 资源配置
            
        Returns:
            Resource: Neo4j连接资源实例
        """
        if not isinstance(config, Neo4jConnectionConfig):
            raise TypeError(f"Expected Neo4jConnectionConfig, got {type(config)}")
        
        return Neo4jConnectionResource(config)
    
    def destroy(self, resource: Resource) -> None:
        """
        销毁Neo4j连接资源
        
        Args:
            resource: 要销毁的资源实例
        """
        if not isinstance(resource, Neo4jConnectionResource):
            raise TypeError(f"Expected Neo4jConnectionResource, got {type(resource)}")
        
        resource.destroy()


class Neo4jConnectionClient(ResourceClient):
    """
    Neo4j连接客户端类
    
    实现ResourceClient接口，为业务层提供统一的Neo4j操作接口。
    
    属性：
        _resource: 封装的Neo4j连接资源
    """
    
    def __init__(self, resource: Neo4jConnectionResource):
        """
        初始化Neo4j连接客户端
        
        Args:
            resource: Neo4j连接资源实例
        """
        self._resource = resource
    
    def get_resource_type(self) -> str:
        """获取资源类型标识"""
        return self._resource.get_type()
    
    def get_raw_resource(self) -> Resource:
        """获取原始资源实例"""
        return self._resource
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        执行Cypher查询
        
        Args:
            query: Cypher查询语句
            
        Returns:
            查询结果列表
        """
        adapter = self._resource.get_adapter()
        if adapter is None:
            raise RuntimeError("Neo4j adapter not initialized")
        return adapter.execute_query(query)
    
    def execute_query_with_params(
        self, 
        query: str, 
        params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        执行带参数的Cypher查询
        
        Args:
            query: Cypher查询语句
            params: 参数字典
            
        Returns:
            查询结果列表
        """
        adapter = self._resource.get_adapter()
        if adapter is None:
            raise RuntimeError("Neo4j adapter not initialized")
        return adapter.execute_query_with_params(query, params)
    
    def get_disease_info(self, disease_name: str) -> Optional[Dict[str, Any]]:
        """
        获取疾病信息
        
        Args:
            disease_name: 疾病名称
            
        Returns:
            疾病信息字典
        """
        query = """
        MATCH (d:Disease {name: $name})
        RETURN d.name as name, d.desc as desc, d.cause as cause, 
               d.prevent as prevent, d.cure_lasttime as cure_lasttime,
               d.cured_prob as cured_prob, d.easy_get as easy_get
        """
        results = self.execute_query_with_params(query, {"name": disease_name})
        return results[0] if results else None
    
    def get_symptoms_by_disease(self, disease_name: str) -> List[Dict[str, Any]]:
        """
        获取疾病的症状列表
        
        Args:
            disease_name: 疾病名称
            
        Returns:
            症状列表
        """
        query = """
        MATCH (d:Disease {name: $name})-[:has_symptom]->(s:Symptom)
        RETURN s.name as symptom_name
        """
        return self.execute_query_with_params(query, {"name": disease_name})
    
    def get_drugs_by_disease(self, disease_name: str) -> List[Dict[str, Any]]:
        """
        获取疾病的常用药物列表
        
        Args:
            disease_name: 疾病名称
            
        Returns:
            药物列表
        """
        query = """
        MATCH (d:Disease {name: $name})-[:common_drug]->(dr:Drug)
        RETURN dr.name as drug_name
        """
        return self.execute_query_with_params(query, {"name": disease_name})
