# -*- coding: utf-8 -*-
"""
SGLang模型工厂封装

实现ResourceFactory接口，负责SGLang模型资源的创建和销毁。
"""

from src.resource_manager.resource import Resource
from src.resource_manager.resource_config import ResourceConfig
from src.resource_manager.resource_factory import ResourceFactory
from src.resource_manager.reasoning_model.reasoning_model_config import ReasoningModelConfig
from src.resource_manager.reasoning_model.reasoning_model_resource import ReasoningModelResource


class ReasoningModelFactory(ResourceFactory):
    """
    SGLang模型工厂类

    实现ResourceFactory接口，负责SGLang模型资源的创建和销毁。
    """

    def create(self, config: ResourceConfig) -> Resource:
        if not isinstance(config, ReasoningModelConfig):
            raise TypeError(f"Expected ReasoningModelConfig, got {type(config)}")
        return ReasoningModelResource(config)

    def destroy(self, resource: Resource) -> None:
        if not isinstance(resource, ReasoningModelResource):
            raise TypeError(f"Expected ReasoningModelResource, got {type(resource)}")
        resource.destroy()
