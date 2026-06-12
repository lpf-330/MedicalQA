# -*- coding: utf-8 -*-
"""
多维度分析Chain策略专属输出数据类
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class MultiAnalysisResultData:
    """
    多维度分析Chain策略专属输出数据类

    Attributes:
        anomalies: 异常指标列表，包含指标名、异常类型、异常值、参考范围
        risk_factors: 风险因子列表，包含因子名、风险等级、依据
        medical_entities: 医疗实体字典，按类型分类存储
        analysis_summary: 分析摘要
    """
    anomalies: List[Dict] = field(default_factory=list)
    risk_factors: List[Dict] = field(default_factory=list)
    medical_entities: Dict[str, List] = field(default_factory=dict)
    analysis_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "anomalies": self.anomalies,
            "risk_factors": self.risk_factors,
            "medical_entities": self.medical_entities,
            "analysis_summary": self.analysis_summary
        }
