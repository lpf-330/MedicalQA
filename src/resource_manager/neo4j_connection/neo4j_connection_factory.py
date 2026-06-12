# -*- coding: utf-8 -*-
"""
Neo4j连接工厂封装

封装Neo4j数据库连接资源的创建和销毁，实现ResourceFactory接口。
"""

from src.resource_manager.resource import Resource
from src.resource_manager.resource_config import ResourceConfig
from src.resource_manager.resource_factory import ResourceFactory
from src.resource_manager.neo4j_connection.neo4j_connection_config import Neo4jConnectionConfig
from src.resource_manager.neo4j_connection.neo4j_connection_resource import Neo4jConnectionResource


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
