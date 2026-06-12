# -*- coding: utf-8 -*-
"""
NER模型资源配置文件

定义NER命名实体识别模型的配置参数和资源池配置。
"""

from dataclasses import dataclass
from typing import Any, Dict

from src.config.base_config import BaseResourceConfig
from src.config.pool_config import PoolConfig


@dataclass
class NerModelConfig(BaseResourceConfig):
    """
    NER模型资源配置类

    属性：
        config_id: 配置ID（文件名作为唯一标识）
        resource_type: 资源类型
        model_path: 模型路径
        model_name: 模型名称
        device: 运行设备
        max_length: 最大序列长度
    """

    config_id: str = "ner_model_config"
    resource_type: str = "ner_model"
    model_path: str = ""
    model_name: str = ""
    device: str = ""
    max_length: int = 512

    def validate(self) -> bool:
        if not super().validate():
            return False

        if not self.model_path:
            print("警告: 模型路径不能为空")
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "model_path": self.model_path,
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
        })
        return base_dict


resource_config = NerModelConfig()

resource_type = "ner_model"

pool_config = PoolConfig()