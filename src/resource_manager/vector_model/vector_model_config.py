# -*- coding: utf-8 -*-
"""向量模型资源配置类。"""

from typing import Any, Dict

from src.resource_manager.resource_config import ResourceConfig


class VectorModelConfig(ResourceConfig[Dict[str, Any]]):

    def __init__(
        self,
        model_path: str,
        model_name: str = "vector-embedding",
        device: str = "cpu",
        dimension: int = 1024,
        batch_size: int = 32,
        resource_name: str = "向量编码模型"
    ):
        self._resource_type = "vector_model"
        self._resource_name = resource_name
        self._config_protocol: Dict[str, Any] = {
            "model_path": model_path,
            "model_name": model_name,
            "device": device,
            "dimension": dimension,
            "batch_size": batch_size
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
        if self._config_protocol.get("dimension", 1024) < 1:
            return False
        return True
