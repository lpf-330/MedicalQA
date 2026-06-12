# -*- coding: utf-8 -*-
"""
报告知识检索Chain策略专属输入数据类
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ReportKnowledgeRetrievalContextBody:
    """
    报告知识检索Chain策略专属输入数据类

    Attributes:
        anomalies: 异常指标列表
        medical_entities: 医疗实体列表
        risk_diseases: 风险疾病列表
    """
    anomalies: List[Dict] = field(default_factory=list)
    medical_entities: List[Dict] = field(default_factory=list)
    risk_diseases: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "anomalies": self.anomalies,
            "medical_entities": self.medical_entities,
            "risk_diseases": self.risk_diseases
        }
