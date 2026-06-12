# -*- coding: utf-8 -*-
"""
健康报告生成策略上下文数据类

该模块定义ReportContextBody数据类，用于健康报告生成策略的上下文数据传递。
基于设计文档《项目业务详细设计v5》第3.2节的设计实现。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReportContextBody:
    """
    报告策略上下文数据类

    Attributes:
        task_id: 任务ID
        monitoring_data: 监测数据
        user_profile: 用户档案
        session_id: 会话ID
        current_state: 当前状态，默认"INITIAL"
        validated_data: 校验后的数据
        degradation_level: 降级级别
        anomalies: 异常指标
        risk_factors: 风险因子
        medical_entities: 医疗实体
        dimension_summaries: 8维度知识检索结果摘要
        knowledge_results: 知识检索结果
        health_assessment: 健康评估结果
        health_score: 健康评分
        health_level: 健康等级
        risk_level: 风险等级
        risk_diseases: 风险疾病
        report_content: 报告内容
        sources: 知识来源
        error_code: 错误码
        error_message: 错误消息
        stream_generator: 流式生成器
        is_streaming: 是否流式输出
        degraded: 是否降级
        degraded_reason: 降级原因
        report_generation_retry_count: 报告生成重试次数（最多2次，基于设计文档状态转换规则）
    """
    task_id: str = ""
    monitoring_data: Dict = field(default_factory=dict)
    user_profile: Dict = field(default_factory=dict)
    session_id: str = ""
    current_state: str = "INITIAL"
    validated_data: Dict = field(default_factory=dict)
    degradation_level: int = 0
    anomalies: List[Dict] = field(default_factory=list)
    risk_factors: List[Dict] = field(default_factory=list)
    medical_entities: Dict[str, List] = field(default_factory=dict)
    dimension_summaries: Dict[str, Dict] = field(default_factory=dict)
    knowledge_results: List[Dict] = field(default_factory=list)
    health_assessment: Optional[Dict] = None
    health_score: float = 0.0
    health_level: str = ""
    risk_level: str = ""
    risk_diseases: List[Dict] = field(default_factory=list)
    report_content: str = ""
    sources: List[str] = field(default_factory=list)
    error_code: int = 0
    error_message: str = ""
    stream_generator: Any = None
    is_streaming: bool = False
    degraded: bool = False
    degraded_reason: str = ""
    report_generation_retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "monitoring_data": self.monitoring_data,
            "user_profile": self.user_profile,
            "session_id": self.session_id,
            "current_state": self.current_state,
            "validated_data": self.validated_data,
            "degradation_level": self.degradation_level,
            "anomalies": self.anomalies,
            "risk_factors": self.risk_factors,
            "medical_entities": self.medical_entities,
            "dimension_summaries": self.dimension_summaries,
            "knowledge_results": self.knowledge_results,
            "health_assessment": self.health_assessment,
            "health_score": self.health_score,
            "health_level": self.health_level,
            "risk_level": self.risk_level,
            "risk_diseases": self.risk_diseases,
            "report_content": self.report_content,
            "sources": self.sources,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "stream_generator": self.stream_generator,
            "is_streaming": self.is_streaming,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
            "report_generation_retry_count": self.report_generation_retry_count
        }
