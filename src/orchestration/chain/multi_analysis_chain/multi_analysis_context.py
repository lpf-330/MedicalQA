# -*- coding: utf-8 -*-
"""
多维度分析Chain策略专属输入数据类
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class MultiAnalysisContextBody:
    """
    多维度分析Chain策略专属输入数据类

    Attributes:
        validated_data: 校验后的数据
        degradation_level: 降级级别
    """
    validated_data: Dict = field(default_factory=dict)
    degradation_level: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "validated_data": self.validated_data,
            "degradation_level": self.degradation_level
        }
