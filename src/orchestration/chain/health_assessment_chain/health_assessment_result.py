# -*- coding: utf-8 -*-
"""
健康评估Chain策略专属输出数据类
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class HealthAssessmentResultData:
    """
    健康评估Chain策略专属输出数据类

    Attributes:
        health_score: 健康综合评分(0-100)
        health_level: 健康等级(优秀/良好/一般/较差/差)
        risk_level: 风险等级(低/轻/中/高)
        disease_risks: 疾病风险评分列表
        score_breakdown: 评分明细(各子指标评分+理由，可追溯)
        reasoning: 医学推理过程汇总
        degraded: 是否降级执行
        degraded_reason: 降级原因
    """
    health_score: float = 0.0
    health_level: str = "一般"
    risk_level: str = "低"
    disease_risks: List[Dict] = field(default_factory=list)
    score_breakdown: Dict = field(default_factory=dict)
    reasoning: str = ""
    degraded: bool = False
    degraded_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "health_score": self.health_score,
            "health_level": self.health_level,
            "risk_level": self.risk_level,
            "disease_risks": self.disease_risks,
            "score_breakdown": self.score_breakdown,
            "reasoning": self.reasoning,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason
        }
