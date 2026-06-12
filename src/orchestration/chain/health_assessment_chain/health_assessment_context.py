# -*- coding: utf-8 -*-
"""
健康评估Chain策略专属输入数据类
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class HealthAssessmentContextBody:
    """
    健康评估Chain策略专属输入数据类

    Attributes:
        dimension_summaries: 8维度结构化摘要
        anomalies: 异常指标列表
        risk_factors: 风险因子列表
        medical_entities: 医疗实体列表
        user_profile: 用户档案
    """
    dimension_summaries: Dict[str, Dict] = field(default_factory=dict)
    anomalies: List[Dict] = field(default_factory=list)
    risk_factors: List[Dict] = field(default_factory=list)
    medical_entities: Dict[str, List] = field(default_factory=dict)
    user_profile: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "dimension_summaries": self.dimension_summaries,
            "anomalies": self.anomalies,
            "risk_factors": self.risk_factors,
            "medical_entities": self.medical_entities,
            "user_profile": self.user_profile
        }
