# -*- coding: utf-8 -*-
"""
综合健康分析策略结果数据类

该模块定义ComprehensiveHealthAnalysisResultData数据类，
用于综合健康分析Agent策略的结果数据传递。
基于设计文档《项目业务详细设计v5.16》第3.3节的设计实现。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ComprehensiveHealthAnalysisResultData:
    """
    综合健康分析Agent结果数据
    
    输出数据提交给REPORT_GENERATION阶段：
    - user_profile: 用户基本信息
    - anomalies: 异常指标列表
    - risk_factors: 风险因子列表
    - medical_entities: 医疗实体列表
    - dimension_summaries: 8维度知识检索结果摘要
    - health_assessment: 健康评估结果
    """
    # 用户基本信息
    user_profile: Dict[str, Any] = field(default_factory=dict)
    
    # 异常指标列表
    anomalies: List[Dict] = field(default_factory=list)
    
    # 风险因子列表
    risk_factors: List[Dict] = field(default_factory=list)
    
    # 医疗实体列表
    medical_entities: Dict[str, List] = field(default_factory=dict)
    
    # 8维度知识检索结果摘要
    dimension_summaries: Dict[str, Dict] = field(default_factory=dict)
    
    # 健康评估结果
    health_assessment: Optional[Dict[str, Any]] = None
    
    # 检索统计信息
    retrieval_stats: Dict[str, Any] = field(default_factory=dict)
    
    # 错误处理
    error_code: int = 0
    error_message: str = ""
    degraded: bool = False
    degraded_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_profile": self.user_profile,
            "anomalies": self.anomalies,
            "risk_factors": self.risk_factors,
            "medical_entities": self.medical_entities,
            "dimension_summaries": self.dimension_summaries,
            "health_assessment": self.health_assessment,
            "retrieval_stats": self.retrieval_stats,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
        }
