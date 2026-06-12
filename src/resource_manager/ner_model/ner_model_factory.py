# -*- coding: utf-8 -*-
"""
NER模型资源工厂类

负责NER模型资源的创建和销毁，实现ResourceFactory接口。
通过NerModelConfig创建NerModelResource实例。
"""

from src.resource_manager.resource import Resource
from src.resource_manager.resource_config import ResourceConfig
from src.resource_manager.resource_factory import ResourceFactory
from src.resource_manager.ner_model.ner_model_config import NerModelConfig
from src.resource_manager.ner_model.ner_model_resource import NerModelResource


class NerModelFactory(ResourceFactory):

    def create(self, config: ResourceConfig) -> Resource:
        if not isinstance(config, NerModelConfig):
            raise TypeError(f"Expected NerModelConfig, got {type(config)}")

        return NerModelResource(config)

    def destroy(self, resource: Resource) -> None:
        if not isinstance(resource, NerModelResource):
            raise TypeError(f"Expected NerModelResource, got {type(resource)}")

        resource.destroy()