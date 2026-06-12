# -*- coding: utf-8 -*-
"""
综合健康分析策略上下文数据类

该模块定义ComprehensiveHealthAnalysisContextBody及其辅助数据类，
用于综合健康分析Agent策略的上下文数据传递。
基于设计文档《项目业务详细设计v5.16》第3.3节的设计实现。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from src.orchestration.agent.comprehensive_health_analysis_strategy.retrieval_planner import RetrievalPlan
from src.orchestration.agent.comprehensive_health_analysis_strategy.sufficiency_evaluator import DimensionSufficiency
from src.orchestration.agent.comprehensive_health_analysis_strategy.vector_prescan import VectorPrescanResult


@dataclass
class DimensionKnowledge:
    """维度知识数据结构"""
    dimension_name: str
    query: str = ""
    raw_knowledge: List[Dict] = field(default_factory=list)
    refined_knowledge: List[Dict] = field(default_factory=list)
    candidate_knowledge: List[Dict] = field(default_factory=list)
    selected_knowledge: List[Dict] = field(default_factory=list)
    summary: str = ""
    score: float = 0.0
    is_sufficient: bool = False
    gaps: List[str] = field(default_factory=list)
    dimension_user_relevance: float = 0.0
    dimension_dim_relevance: float = 0.0
    retrieve_attempts: int = 0
    relevance_result: Optional[Dict[str, Any]] = None
    hybrid_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension_name": self.dimension_name,
            "query": self.query,
            "raw_knowledge": self.raw_knowledge,
            "refined_knowledge": self.refined_knowledge,
            "candidate_knowledge": self.candidate_knowledge,
            "selected_knowledge": self.selected_knowledge,
            "summary": self.summary,
            "score": self.score,
            "is_sufficient": self.is_sufficient,
            "gaps": self.gaps,
            "dimension_user_relevance": self.dimension_user_relevance,
            "dimension_dim_relevance": self.dimension_dim_relevance,
            "retrieve_attempts": self.retrieve_attempts,
            "relevance_result": self.relevance_result,
            "hybrid_scores": self.hybrid_scores,
        }


@dataclass
class SharedMemory:
    """维度间共享内存"""
    common_knowledge: Dict[str, List[Dict]] = field(default_factory=dict)
    cross_references: Dict[str, List[str]] = field(default_factory=dict)
    shared_entities: Dict[str, Dict] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "common_knowledge": self.common_knowledge,
            "cross_references": self.cross_references,
            "shared_entities": self.shared_entities,
        }


@dataclass
class RetrievalStats:
    """检索统计信息"""
    total_retrieval_count: int = 0
    supplement_retrieval_count: int = 0
    total_time_ms: float = 0.0
    quality_score: float = 0.0
    dimension_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_retrieval_count": self.total_retrieval_count,
            "supplement_retrieval_count": self.supplement_retrieval_count,
            "total_time_ms": self.total_time_ms,
            "quality_score": self.quality_score,
            "dimension_scores": self.dimension_scores,
        }


@dataclass
class HealthAssessment:
    """健康评估结果数据结构"""
    health_score: Optional[float] = None
    health_level: Optional[str] = None
    risk_level: Optional[str] = None
    disease_risks: List[Dict] = field(default_factory=list)
    score_breakdown: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    degraded: bool = False
    degraded_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "health_score": self.health_score,
            "health_level": self.health_level,
            "risk_level": self.risk_level,
            "disease_risks": self.disease_risks,
            "score_breakdown": self.score_breakdown,
            "reasoning": self.reasoning,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
        }


@dataclass
class ComprehensiveHealthAnalysisContextBody:
    """
    综合健康分析Agent上下文数据
    
    输入数据来自DATA_PARSE阶段：
    - anomalies: 异常指标列表
    - risk_factors: 风险因子列表
    - medical_entities: 医疗实体列表
    - user_profile: 用户档案
    """
    # 输入数据
    anomalies: List[Dict] = field(default_factory=list)
    risk_factors: List[Dict] = field(default_factory=list)
    medical_entities: Dict[str, List] = field(default_factory=dict)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    
    # 状态管理
    current_state: str = "BuildQueries"
    current_step: int = 0

    # Agent检索状态
    retrieval_plan: Dict[str, RetrievalPlan] = field(default_factory=dict)
    prescan_results: Dict[str, VectorPrescanResult] = field(default_factory=dict)
    dimension_used_paths: Dict[str, List[str]] = field(default_factory=dict)
    supplement_round: int = 0
    iteration_round: int = 0
    sufficiency_results: Dict[str, DimensionSufficiency] = field(default_factory=dict)
    ner_entities: Dict[str, List[str]] = field(default_factory=dict)
    knowledge_blacklist: Set[str] = field(default_factory=set)
    is_partial_retrieve: bool = False
    vacancy_dimensions: List[str] = field(default_factory=list)
    retained_knowledge: Dict[str, List[Dict]] = field(default_factory=dict)

    # v8 循环保护计数器
    chain_loop_count: int = 0
    agent_retrieval_loop_count: int = 0

    # v8 知识表（跨维度引用记录）
    # 维度表（主表）：维度名 → 该维度保留的知识项neo4j ID列表
    dimension_table: Dict[str, List[str]] = field(default_factory=dict)
    # 知识表（附表）：neo4j ID → 它同时属于哪些维度的记录
    knowledge_cross_refs: Dict[str, List[str]] = field(default_factory=dict)

    # 中间结果
    dimension_queries: Dict[str, str] = field(default_factory=dict)
    dimension_knowledge: Dict[str, DimensionKnowledge] = field(default_factory=dict)
    shared_memory: SharedMemory = field(default_factory=SharedMemory)
    retrieval_stats: RetrievalStats = field(default_factory=RetrievalStats)
    
    # 最终结果
    dimension_summaries: Dict[str, Dict] = field(default_factory=dict)
    health_assessment: Optional[HealthAssessment] = None
    
    # 错误处理
    error_code: int = 0
    error_message: str = ""
    degraded: bool = False
    degraded_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomalies": self.anomalies,
            "risk_factors": self.risk_factors,
            "medical_entities": self.medical_entities,
            "user_profile": self.user_profile,
            "current_state": self.current_state,
            "current_step": self.current_step,
            "retrieval_plan": {k: v.to_dict() for k, v in self.retrieval_plan.items()},
            "prescan_results": {k: v.to_dict() for k, v in self.prescan_results.items()},
            "dimension_used_paths": self.dimension_used_paths,
            "supplement_round": self.supplement_round,
            "iteration_round": self.iteration_round,
            "sufficiency_results": {k: v.to_dict() for k, v in self.sufficiency_results.items()},
            "ner_entities": self.ner_entities,
            "knowledge_blacklist": list(self.knowledge_blacklist),
            "is_partial_retrieve": self.is_partial_retrieve,
            "vacancy_dimensions": self.vacancy_dimensions,
            "retained_knowledge": self.retained_knowledge,
            "chain_loop_count": self.chain_loop_count,
            "agent_retrieval_loop_count": self.agent_retrieval_loop_count,
            "dimension_table": self.dimension_table,
            "knowledge_cross_refs": self.knowledge_cross_refs,
            "dimension_queries": self.dimension_queries,
            "dimension_knowledge": {k: v.to_dict() for k, v in self.dimension_knowledge.items()},
            "shared_memory": self.shared_memory.to_dict(),
            "retrieval_stats": self.retrieval_stats.to_dict(),
            "dimension_summaries": self.dimension_summaries,
            "health_assessment": self.health_assessment.to_dict() if self.health_assessment else None,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
        }
