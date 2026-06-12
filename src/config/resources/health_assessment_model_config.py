# -*- coding: utf-8 -*-
"""
健康评估模型资源配置文件

定义健康评估模型的配置参数和资源池配置。
健康评估模型通过SGLang实例（:30001）提供服务。
所有运行期值由application.yaml覆盖，此处只保存空值或占位默认值。
"""

from dataclasses import dataclass
from typing import Any, Dict

from src.config.base_config import BaseResourceConfig
from src.config.pool_config import PoolConfig


@dataclass
class HealthAssessmentModelConfig(BaseResourceConfig):
    """
    健康评估模型资源配置类

    属性：
        config_id: 配置ID
        resource_type: 资源类型
        base_url: 健康评估模型推理引擎HTTP服务地址（运行期由yaml覆盖）
        model_name: 模型名称（运行期由yaml覆盖）
        default_temperature: 默认温度（占位默认值，运行期由yaml覆盖）
        default_max_tokens: 默认最大token数（占位默认值，运行期由yaml覆盖）
        default_top_p: 默认top_p（占位默认值，运行期由yaml覆盖）
    """

    config_id: str = "health_assessment_model_config"
    resource_type: str = "health_assessment_model"
    base_url: str = ""
    model_name: str = ""
    default_temperature: float = 0.0
    default_max_tokens: int = 1
    default_top_p: float = 0.0
    default_repetition_penalty: float = 1.15
    timeout: float = 600.0
    # 服务启动参数
    auto_start: bool = False
    model_path: str = ""
    launch_host: str = "0.0.0.0"
    launch_port: int = 30001
    launch_args: str = ""
    startup_timeout: int = 300
    health_check_interval: float = 5.0
    shutdown_timeout: int = 30

    def validate(self) -> bool:
        if not super().validate():
            return False
        if not self.base_url:
            print("警告: base_url不能为空")
            return False
        if not self.model_name:
            print("警告: model_name不能为空")
            return False
        if self.auto_start and not self.model_path:
            print("警告: auto_start=True时model_path不能为空")
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "base_url": self.base_url,
            "model_name": self.model_name,
            "default_temperature": self.default_temperature,
            "default_max_tokens": self.default_max_tokens,
            "default_top_p": self.default_top_p,
            "timeout": self.timeout,
            "auto_start": self.auto_start,
            "model_path": self.model_path,
            "launch_host": self.launch_host,
            "launch_port": self.launch_port,
            "launch_args": self.launch_args,
            "startup_timeout": self.startup_timeout,
            "health_check_interval": self.health_check_interval,
            "shutdown_timeout": self.shutdown_timeout,
        })
        return base_dict


resource_config = HealthAssessmentModelConfig()

resource_type = "health_assessment_model"

pool_config = PoolConfig()
