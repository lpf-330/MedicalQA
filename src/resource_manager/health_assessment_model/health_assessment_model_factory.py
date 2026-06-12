# -*- coding: utf-8 -*-
"""
健康评估模型工厂封装

实现ResourceFactory接口，负责健康评估模型资源的创建和销毁。
"""

from src.resource_manager.resource import Resource
from src.resource_manager.resource_config import ResourceConfig
from src.resource_manager.resource_factory import ResourceFactory
from src.resource_manager.health_assessment_model.health_assessment_model_config import HealthAssessmentModelConfig
from src.resource_manager.health_assessment_model.health_assessment_model_resource import HealthAssessmentModelResource


class HealthAssessmentModelFactory(ResourceFactory):
    """
    健康评估模型工厂类

    实现ResourceFactory接口，负责健康评估模型资源的创建和销毁。
    """

    def create(self, config: ResourceConfig) -> Resource:
        """
        创建健康评估模型资源

        Args:
            config: 资源配置

        Returns:
            Resource: 健康评估模型资源实例
        """
        if not isinstance(config, HealthAssessmentModelConfig):
            raise TypeError(f"Expected HealthAssessmentModelConfig, got {type(config)}")

        return HealthAssessmentModelResource(config)

    def destroy(self, resource: Resource) -> None:
        """
        销毁健康评估模型资源

        Args:
            resource: 要销毁的资源实例
        """
        if not isinstance(resource, HealthAssessmentModelResource):
            raise TypeError(f"Expected HealthAssessmentModelResource, got {type(resource)}")

        resource.destroy()
