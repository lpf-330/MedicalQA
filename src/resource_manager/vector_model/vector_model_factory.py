# -*- coding: utf-8 -*-
"""向量模型资源工厂类。"""

from src.resource_manager.resource import Resource
from src.resource_manager.resource_config import ResourceConfig
from src.resource_manager.resource_factory import ResourceFactory
from src.resource_manager.vector_model.vector_model_config import VectorModelConfig
from src.resource_manager.vector_model.vector_model_resource import VectorModelResource


class VectorModelFactory(ResourceFactory):

    def create(self, config: ResourceConfig) -> Resource:
        if not isinstance(config, VectorModelConfig):
            raise TypeError(f"Expected VectorModelConfig, got {type(config)}")

        return VectorModelResource(config)

    def destroy(self, resource: Resource) -> None:
        if not isinstance(resource, VectorModelResource):
            raise TypeError(f"Expected VectorModelResource, got {type(resource)}")

        resource.destroy()
