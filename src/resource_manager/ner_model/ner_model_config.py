# -*- coding: utf-8 -*-
"""
NER模型资源配置类

定义NER模型资源的配置信息，包括模型路径、设备、最大长度等参数。
继承ResourceConfig，提供资源配置的标准接口实现。
"""

from typing import Any, Dict

from src.resource_manager.resource_config import ResourceConfig


class NerModelConfig(ResourceConfig[Dict[str, Any]]):

    def __init__(
        self,
        model_path: str,
        model_name: str = "ner-cmeee",
        device: str = "cpu",
        max_length: int = 512,
        resource_name: str = "医学命名实体识别模型"
    ):
        self._resource_type = "ner_model"
        self._resource_name = resource_name
        self._config_protocol: Dict[str, Any] = {
            "model_path": model_path,
            "model_name": model_name,
            "device": device,
            "max_length": max_length
        }

    @property
    def resource_type(self) -> str:
        return self._resource_type

    @property
    def resource_name(self) -> str:
        return self._resource_name

    @property
    def config_protocol(self) -> Dict[str, Any]:
        return self._config_protocol

    def to_dict(self) -> dict:
        return {
            "resource_type": self._resource_type,
            "resource_name": self._resource_name,
            "config_protocol": self._config_protocol
        }

    def validate(self) -> bool:
        if not self._config_protocol.get("model_path"):
            return False
        return True