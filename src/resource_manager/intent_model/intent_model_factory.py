# -*- coding: utf-8 -*-
"""
意图模型资源工厂类

负责意图模型资源的创建和销毁，实现ResourceFactory接口。
通过IntentModelConfig创建IntentModelResource实例。
"""

from src.resource_manager.resource import Resource
from src.resource_manager.resource_config import ResourceConfig
from src.resource_manager.resource_factory import ResourceFactory
from src.resource_manager.intent_model.intent_model_config import IntentModelConfig
from src.resource_manager.intent_model.intent_model_resource import IntentModelResource


class IntentModelFactory(ResourceFactory):

    def create(self, config: ResourceConfig) -> Resource:
        if not isinstance(config, IntentModelConfig):
            raise TypeError(f"Expected IntentModelConfig, got {type(config)}")

        return IntentModelResource(config)

    def destroy(self, resource: Resource) -> None:
        if not isinstance(resource, IntentModelResource):
            raise TypeError(f"Expected IntentModelResource, got {type(resource)}")

        resource.destroy()
