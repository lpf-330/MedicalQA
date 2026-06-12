# -*- coding: utf-8 -*-

"""Milvus连接资源工厂类。"""

from src.resource_manager.resource import Resource
from src.resource_manager.resource_config import ResourceConfig
from src.resource_manager.resource_factory import ResourceFactory
from src.resource_manager.milvus_connection.milvus_connection_config import MilvusConnectionConfig
from src.resource_manager.milvus_connection.milvus_connection_resource import MilvusConnectionResource


class MilvusConnectionFactory(ResourceFactory):

    def create(self, config: ResourceConfig) -> Resource:
        if not isinstance(config, MilvusConnectionConfig):
            raise TypeError(f"Expected MilvusConnectionConfig, got {type(config)}")

        return MilvusConnectionResource(config)

    def destroy(self, resource: Resource) -> None:
        if not isinstance(resource, MilvusConnectionResource):
            raise TypeError(f"Expected MilvusConnectionResource, got {type(resource)}")

        resource.destroy()
