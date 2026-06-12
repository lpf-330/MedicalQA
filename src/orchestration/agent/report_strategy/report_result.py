# -*- coding: utf-8 -*-
"""
健康报告生成策略结果数据类

该模块定义ReportResultData数据类，用于健康报告生成策略的结果数据传递。
基于设计文档《项目业务详细设计v5》第3.2节的设计实现。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReportResultData:
    """
    报告策略结果数据类

    Attributes:
        report: 报告内容
        health_score: 健康评分
        health_level: 健康等级
        risk_level: 风险等级
        risk_diseases: 风险疾病
        sources: 知识来源
        word_count: 报告字数
        session_id: 会话ID
        dimension_summaries: 各维度评估结果
        health_assessment: 健康评估结果
        error_code: 错误码
        error_message: 错误消息
        degraded: 是否降级
        degraded_reason: 降级原因
    """
    report: str = ""
    health_score: float = 0.0
    health_level: str = ""
    risk_level: str = ""
    risk_diseases: List[Dict] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    word_count: int = 0
    session_id: str = ""
    dimension_summaries: Dict = field(default_factory=dict)
    health_assessment: Optional[Dict] = None
    error_code: int = 0
    error_message: str = ""
    degraded: bool = False
    degraded_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "report": self.report,
            "health_score": self.health_score,
            "health_level": self.health_level,
            "risk_level": self.risk_level,
            "risk_diseases": self.risk_diseases,
            "sources": self.sources,
            "word_count": self.word_count,
            "session_id": self.session_id,
            "dimension_summaries": self.dimension_summaries,
            "health_assessment": self.health_assessment,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason
        }
