# -*- coding: utf-8 -*-
"""
健康评估模型配置封装

实现ResourceConfig接口，存储健康评估模型配置。
健康评估模型-4B-AWQ通过SGLang HTTP服务提供健康评估能力。
auto_start=True时，资源管理层自动启动SGLang服务子进程。
"""

from typing import Any, Dict

from src.resource_manager.resource_config import ResourceConfig


class HealthAssessmentModelConfig(ResourceConfig[Dict[str, Any]]):
    """
    健康评估模型健康评估模型配置类

    实现ResourceConfig接口，存储健康评估模型配置。

    属性：
        _resource_type: 资源类型标识
        _resource_name: 资源业务名称
        _config_protocol: 个性化配置协议
    """

    def __init__(
        self,
        base_url: str,
        model_name: str = "",
        default_temperature: float = 0.0,
        default_max_tokens: int = 1,
        default_top_p: float = 0.0,
        default_repetition_penalty: float = 1.15,
        timeout: float = 300.0,
        auto_start: bool = False,
        model_path: str = "",
        launch_host: str = "0.0.0.0",
        launch_port: int = 30001,
        launch_args: str = "",
        startup_timeout: int = 300,
        health_check_interval: float = 5.0,
        shutdown_timeout: int = 30,
        resource_name: str = "健康评估模型资源"
    ):
        self._resource_type = "health_assessment_model"
        self._resource_name = resource_name
        self._config_protocol: Dict[str, Any] = {
            "base_url": base_url,
            "model_name": model_name,
            "default_temperature": default_temperature,
            "default_max_tokens": default_max_tokens,
            "default_top_p": default_top_p,
            "default_repetition_penalty": default_repetition_penalty,
            "timeout": timeout,
            "auto_start": auto_start,
            "model_path": model_path,
            "launch_host": launch_host,
            "launch_port": launch_port,
            "launch_args": launch_args,
            "startup_timeout": startup_timeout,
            "health_check_interval": health_check_interval,
            "shutdown_timeout": shutdown_timeout,
        }

    @property
    def resource_type(self) -> str:
        """获取资源类型标识"""
        return self._resource_type

    @property
    def resource_name(self) -> str:
        """获取资源业务名称"""
        return self._resource_name

    @property
    def config_protocol(self) -> Dict[str, Any]:
        """获取个性化配置协议"""
        return self._config_protocol

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "resource_type": self._resource_type,
            "resource_name": self._resource_name,
            "config_protocol": self._config_protocol
        }

    def validate(self) -> bool:
        """验证配置有效性"""
        if not self._config_protocol.get("base_url"):
            return False
        timeout = self._config_protocol.get("timeout", 300.0)
        if timeout <= 0:
            return False
        if self._config_protocol.get("auto_start", False) and not self._config_protocol.get("model_path"):
            return False
        return True
