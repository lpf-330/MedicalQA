# AI辅助生成：GLM-5，2026-04-15
# -*- coding: utf-8 -*-
"""
意图模型资源配置文件

定义意图识别模型的配置参数和资源池配置。
"""

from dataclasses import dataclass, field
from typing import Any, Dict
from pathlib import Path

from src.config.base_config import BaseResourceConfig
from src.config.pool_config import PoolConfig


def _get_default_model_path() -> str:
    """获取默认模型路径"""
    project_root = Path(__file__).parent.parent.parent.parent
    return str(project_root / "base_models" / "Apollo-0.5B")


@dataclass
class IntentModelResourceConfig(BaseResourceConfig):
    """
    意图模型资源配置类

    属性：
        config_id: 配置ID（文件名作为唯一标识）
        resource_type: 资源类型
        model_path: 模型路径
        model_name: 模型名称
        device: 运行设备
        max_length: 最大序列长度
    """

    config_id: str = "intent_model_config"
    resource_type: str = "intent_model"
    model_path: str = field(default_factory=_get_default_model_path)
    model_name: str = "FreedomIntelligence/Apollo-0.5B"
    device: str = "cuda"
    max_length: int = 512

    def validate(self) -> bool:
        """
        验证配置有效性

        Returns:
            bool: 配置是否有效
        """
        if not super().validate():
            return False

        if not self.model_path:
            print("警告: 模型路径不能为空")
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        导出配置为字典

        Returns:
            Dict[str, Any]: 配置字典
        """
        base_dict = super().to_dict()
        base_dict.update({
            "model_path": self.model_path,
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
        })
        return base_dict


resource_config = IntentModelResourceConfig()

resource_type = "intent_model"

pool_config = PoolConfig(
    max_size=1,
    min_idle=1,
    idle_timeout=600000,
    max_wait_time=30000
)
