# -*- coding: utf-8 -*-
"""
VLLM资源配置文件

定义VLLM模型的配置参数和资源池配置。
"""

from dataclasses import dataclass, field
from typing import Any, Dict
from pathlib import Path

from src.config.base_config import BaseResourceConfig
from src.config.pool_config import PoolConfig


def _get_default_model_path() -> str:
    """获取默认模型路径"""
    project_root = Path(__file__).parent.parent.parent.parent
    return str(project_root / "base_models" / "Qwen3-4B-Instruct-2507")


@dataclass
class VLLMResourceConfig(BaseResourceConfig):
    """
    VLLM资源配置类
    
    属性：
        config_id: 配置ID（文件名作为唯一标识）
        resource_type: 资源类型
        model_path: 模型路径
        model_name: 模型名称
        tensor_parallel_size: 张量并行大小
        max_model_len: 最大模型长度
        gpu_memory_utilization: GPU内存利用率
    """
    
    config_id: str = "vllm_config"
    resource_type: str = "vllm_model"
    model_path: str = field(default_factory=_get_default_model_path)
    model_name: str = "Qwen3-4B-Instruct-2507"
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.8
    
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
        if not Path(self.model_path).exists():
            print(f"警告: 模型路径不存在: {self.model_path}")
            return False
        if self.tensor_parallel_size < 1:
            print("警告: tensor_parallel_size 必须 >= 1")
            return False
        if not 0 < self.gpu_memory_utilization <= 1:
            print("警告: gpu_memory_utilization 必须在 (0, 1] 范围内")
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
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        })
        return base_dict


resource_config = VLLMResourceConfig()

resource_type = "vllm_model"

pool_config = PoolConfig(
    max_size=1,
    min_idle=1,
    idle_timeout=600000,
    max_wait_time=30000
)
