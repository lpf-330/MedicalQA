# -*- coding: utf-8 -*-
"""
数据准备Chain策略专属输出数据类
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DataPrepareResultData:
    """
    数据准备Chain策略专属输出数据类

    Attributes:
        validated_data: 校验后的数据
        degradation_level: 降级级别（0-3）
        missing_fields: 缺失字段列表
        data_completeness: 数据完整度（0.0-1.0）
    """
    validated_data: Dict[str, Any] = field(default_factory=dict)
    degradation_level: int = 0
    missing_fields: List[str] = field(default_factory=list)
    data_completeness: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "validated_data": self.validated_data,
            "degradation_level": self.degradation_level,
            "missing_fields": self.missing_fields,
            "data_completeness": self.data_completeness
        }
