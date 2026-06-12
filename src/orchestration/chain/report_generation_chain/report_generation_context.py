# -*- coding: utf-8 -*-
"""
报告生成Chain策略专属输入数据类
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ReportGenerationContextBody:
    """
    报告生成Chain策略专属输入数据类

    Attributes:
        report_materials: 报告素材
        health_score: 健康评分
        health_level: 健康等级
        risk_level: 风险等级
        risk_diseases: 风险疾病
        user_profile: 用户档案
        monitoring_data: 监测数据
    """
    report_materials: Dict = field(default_factory=dict)
    health_score: float = 0.0
    health_level: str = ""
    risk_level: str = ""
    risk_diseases: List[Dict] = field(default_factory=list)
    user_profile: Dict = field(default_factory=dict)
    monitoring_data: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "report_materials": self.report_materials,
            "health_score": self.health_score,
            "health_level": self.health_level,
            "risk_level": self.risk_level,
            "risk_diseases": self.risk_diseases,
            "user_profile": self.user_profile,
            "monitoring_data": self.monitoring_data
        }
