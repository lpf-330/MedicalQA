# -*- coding: utf-8 -*-
"""
Neo4j连接客户端封装

封装Neo4j数据库连接客户端，为业务层提供统一的Neo4j操作接口。
"""

from typing import Any, Dict, List, Optional

from src.resource_manager.resource import Resource
from src.resource_manager.resource_client import ResourceClient
from src.resource_manager.neo4j_connection.neo4j_connection_resource import Neo4jConnectionResource


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
