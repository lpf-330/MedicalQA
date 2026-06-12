# AI辅助生成：GLM-5，2026-04-15
# -*- coding: utf-8 -*-
"""
健康报告生成业务配置文件

定义健康报告生成业务的配置参数和所需的资源配置引用。
合并了原ReportBusinessConfig、HealthAssessmentConfig、ComprehensiveHealthAnalysisConfig的所有字段。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.config.base_config import BusinessConfig
import logging

_config_logger = logging.getLogger(__name__)


@dataclass
class ReportServiceConfig(BusinessConfig):
    """
    健康报告生成业务配置类

    合并了原ReportBusinessConfig、HealthAssessmentConfig、
    ComprehensiveHealthAnalysisConfig的字段，统一由ConfigManager加载。
    """

    business_id: str = "report_service"
    resource_configs: List[str] = field(default_factory=lambda: [
        "neo4j_connection_config", "reasoning_model_config", "health_assessment_model_config", "milvus_connection_config",
        "vector_model_config", "intent_model_config", "ner_model_config"
    ])

    # ========================================================================
    # 基础参数
    # ========================================================================
    max_retries: int = 3
    timeout: int = 300

    # 各阶段超时
    data_prepare_timeout: int = 5
    multi_analysis_timeout: int = 30
    parallel_processing_timeout: int = 120
    report_generation_timeout: int = 240

    # FSM控制参数
    report_max_iterations: int = 30

    # 各状态超时配置
    state_timeouts: Dict[str, int] = field(default_factory=lambda: {
        "INITIAL": 15, "DATA_PREPARE": 10, "DATA_PARSE": 30,
        "COMPREHENSIVE_HEALTH_ANALYSIS": 1200, "REPORT_GENERATION": 240,
        "STREAMING": 30, "ASSEMBLY": 10, "FINISHED": 5,
    })

    # ========================================================================
    # 知识检索参数
    # ========================================================================
    enable_knowledge_retrieval: bool = True
    knowledge_retrieval_top_k: int = 30
    knowledge_fusion_threshold: float = 0.6
    vector_entity_weight: float = 0.40
    vector_attribute_weight: float = 0.30
    vector_relation_weight: float = 0.30
    knowledge_merge_limit: int = 30
    prompt_truncation_chars: int = 4000
    max_prompt_chars: int = 4500
    max_knowledge_chars: int = 3000

    # ========================================================================
    # 报告生成参数
    # ========================================================================
    report_min_length: int = 800
    report_max_length: int = 4000
    max_report_retries: int = 2
    report_generation_max_tokens: int = 4000
    report_generation_temperature: float = 0.0
    value_text_max_chars: int = 500
    summary_text_max_chars: int = 2000
    basis_text_max_chars: int = 200
    family_history_limit: int = 100
    references_limit: int = 100
    past_medical_history_limit: int = 100
    template_summary_truncate_len: int = 200
    anomaly_deviation_threshold: int = 10

    # ========================================================================
    # 健康评估参数
    # ========================================================================
    health_score_thresholds: Dict[str, int] = field(default_factory=lambda: {
        "excellent": 90, "good": 80, "normal": 70, "poor": 60
    })
    risk_level_thresholds: Dict[str, int] = field(default_factory=lambda: {
        "low": 20, "mild": 40, "moderate": 60
    })

    # 健康维度定义（5维度）
    health_dimensions: Dict[str, Dict] = field(default_factory=lambda: {
        "D1": {"name": "生理指标", "weight": 0.35, "sub_indicators": ["血压", "血糖", "血脂", "BMI", "心率"]},
        "D2": {"name": "生活方式", "weight": 0.20, "sub_indicators": ["运动", "饮食", "睡眠", "吸烟", "饮酒"]},
        "D3": {"name": "病史风险", "weight": 0.25, "sub_indicators": ["既往病史", "家族史", "用药史"]},
        "D4": {"name": "心理状态", "weight": 0.10, "sub_indicators": ["压力水平", "情绪状态"]},
        "D5": {"name": "预防措施", "weight": 0.10, "sub_indicators": ["体检频率", "疫苗接种", "筛查情况"]},
    })

    # 疾病风险因子定义（6因子）
    disease_risk_factors: Dict[str, Dict] = field(default_factory=lambda: {
        "F1": {"name": "异常指标风险", "weight": 0.30, "description": "异常指标数量和严重程度"},
        "F2": {"name": "病史风险", "weight": 0.25, "description": "既往病史数量和类型"},
        "F3": {"name": "家族史风险", "weight": 0.15, "description": "家族病史情况"},
        "F4": {"name": "生活方式风险", "weight": 0.15, "description": "不良生活习惯数量"},
        "F5": {"name": "年龄风险", "weight": 0.10, "description": "年龄相关风险"},
        "F6": {"name": "并发症风险", "weight": 0.05, "description": "潜在并发症风险"},
    })

    # 风险等级划分标准（4等级，含分数区间和描述）
    risk_level_standards: Dict[str, Dict] = field(default_factory=lambda: {
        "低": {"min": 0, "max": 29, "description": "无异常指标,无病史,生活方式良好", "advice": "保持现状,定期体检"},
        "轻": {"min": 30, "max": 49, "description": "1-2个轻微异常,或1个可控病史", "advice": "改善生活方式,关注异常指标"},
        "中": {"min": 50, "max": 69, "description": "3-5个异常,或2个病史,或生活方式差", "advice": "积极干预,就医咨询"},
        "高": {"min": 70, "max": 100, "description": "多个严重异常,或多病共存,或高危病史", "advice": "立即就医,密切监测"},
    })

    # 规则引擎扣分值
    deduction_severe: int = 15
    deduction_moderate: int = 10
    deduction_mild: int = 5
    deduction_risk_factor: int = 5
    deduction_disease: int = 8

    # 评估默认值
    default_dimension_score: float = 0.5
    default_confidence: float = 0.7
    rule_engine_confidence: float = 0.8
    base_health_score: int = 100

    # 风险评估阈值
    disease_count_mild: int = 2
    anomaly_count_mild: int = 2
    anomaly_count_moderate: int = 5
    disease_risk_weight: float = 0.5

    # 查询构建截断限制
    query_entity_limit: int = 2
    query_disease_limit: int = 3
    query_anomaly_limit: int = 3

    # Prompt预算参数
    prompt_budget_buffer: int = 50
    prompt_budget_minimum: int = 400
    knowledge_budget_ratio: float = 0.55
    user_info_budget_ratio: float = 0.45
    knowledge_item_limit: int = 3
    knowledge_content_truncate_len: int = 600

    # 用户信息截断
    max_user_info_chars: int = 800
    max_risk_factor_user_info_chars: int = 600

    # 健康评估模型调用参数
    health_assessment_context_length: int = 8192
    health_assessment_max_tokens: int = 3072
    health_assessment_batch_max_tokens: int = 3072
    health_assessment_max_retries: int = 1
    health_assessment_enable_thinking: bool = True
    health_assessment_repetition_penalty: float = 1.15

    # 批量评估调用参数（HybridRelevance等结构化JSON输出，关闭thinking避免格式瑕疵）
    batch_evaluation_enable_thinking: bool = False

    # Qwen3结构化输出自修复调用参数（关闭thinking，修复重试也是结构化JSON输出）
    reasoning_repair_enable_thinking: bool = False
    reasoning_repair_repetition_penalty: float = 1.15

    # 疾病风险评估参数
    multi_disease_threshold: int = 3
    elderly_age_threshold: int = 65
    anomaly_match_score_increment: int = 15
    history_weight_multiplier: float = 1.5
    family_history_weight_multiplier: float = 1.2
    disease_risk_top_n: int = 5
    base_disease_risk_score: int = 30  # 疾病存在本身的基础风险分

    # ========================================================================
    # 综合健康分析参数
    # ========================================================================
    max_steps: int = 5
    sufficiency_threshold: float = 0.6
    # v8 HybridRelevance评分系数（主流程）
    relevance_alpha: float = 0.50
    relevance_beta: float = 0.30
    relevance_gamma: float = 0.20
    # v8 HybridRelevance降级评分系数
    degraded_alpha: float = 0.50
    degraded_beta: float = 0.30
    degraded_gamma: float = 0.20
    relevance_threshold: float = 0.4
    max_retrieve_attempts: int = 2
    vector_candidate_top_k: int = 3
    vector_candidate_threshold: float = 0.6
    vector_default_score: float = 0.5
    low_quality_min_content_len: int = 10
    dimension_weights: Dict[str, float] = field(default_factory=lambda: {
        "disease_risk": 0.20, "medication": 0.15, "treatment": 0.15,
        "dietary": 0.10, "checkup": 0.10, "complication": 0.15,
        "prevention": 0.10, "susceptible": 0.05,
    })
    batch_evaluation_max_tokens: int = 512
    batch_refine_max_tokens: int = 512
    dimension_max_knowledge_items: int = 3
    analysis_sequential_top_k: int = 20

    # 分析FSM控制参数
    analysis_max_iterations: int = 20

    # 批量推理并发度（call_model_batch 的 ThreadPoolExecutor max_workers）
    batch_max_workers: int = 4

    # 分析各状态超时配置
    analysis_state_timeouts: Dict[str, int] = field(default_factory=lambda: {
        "BuildQueries": 10, "PlanRetrieval": 120, "InitRetrievalContext": 5,
        "ParallelDimensionRetrieve": 180, "InterDimensionSync": 30,
        "HybridRelevance": 300, "EvaluateSufficiency": 300,
        "RefineKnowledge": 240, "HealthAssess": 420, "Output": 10,
    })

    # 分析过程截断限制
    rule_knowledge_item_limit: int = 3
    rule_content_truncate_len: int = 600
    knowledge_item_display_limit: int = 3
    evaluation_content_truncate_len: int = 600
    refine_content_truncate_len: int = 700
    suggested_keyword_limit: int = 3
    batch_refine_timeout: int = 120
    batch_evaluation_timeout: int = 120
    fuzzy_match_symptom_limit: int = 3
    disease_per_symptom_limit: int = 5
    rule_entity_limit: int = 5

    # 规则引擎充分性评分映射（知识数量 → 充分性分数）
    sufficiency_count_high: int = 5
    sufficiency_count_medium: int = 3
    sufficiency_count_low: int = 1
    sufficiency_score_high: float = 0.8
    sufficiency_score_medium: float = 0.6
    sufficiency_score_low: float = 0.4
    sufficiency_score_none: float = 0.2

    # ========================================================================
    # 数据准备降级参数
    # ========================================================================
    degradation_completeness_high: float = 0.9
    degradation_completeness_medium: float = 0.7
    degradation_completeness_low: float = 0.5
    degradation_core_missing_mild: int = 2
    degradation_core_missing_moderate: int = 4

    # ========================================================================
    # Agent检索参数
    # ========================================================================
    agent_max_paths_per_dimension: int = 3
    agent_max_supplement_rounds: int = 3
    agent_vector_prescan_top_k: int = 10
    agent_vector_prescan_threshold: float = 0.5
    agent_plan_temperature: float = 0.0
    agent_plan_max_tokens: int = 512
    agent_evaluate_max_tokens: int = 256
    agent_evaluate_timeout: int = 120

    # 候选检索与知识选取参数
    agent_candidate_retrieve_limit: int = 20
    agent_knowledge_selection_temperature: float = 0.1
    agent_knowledge_selection_max_tokens: int = 512
    agent_knowledge_selection_content_truncate_len: int = 150

    # 混合相关性评估参数
    agent_hybrid_relevance_user_weight: float = 0.60
    agent_hybrid_relevance_dim_weight: float = 0.40
    agent_hybrid_relevance_temperature: float = 0.1
    agent_hybrid_relevance_max_tokens: int = 768
    agent_hybrid_relevance_content_truncate_len: int = 150

    # 补丁路径规划参数
    agent_patch_path_temperature: float = 0.0
    agent_patch_path_max_tokens: int = 256

    # 迭代控制参数
    agent_max_iteration_rounds: int = 2

    # v8 循环保护参数
    max_chain_loops: int = 2
    max_agent_retrieval_loops: int = 2

    def validate(self) -> bool:
        if not super().validate():
            return False

        if not self.resource_configs:
            _config_logger.warning('resource_configs 不能为空')
            return False
        if self.max_retries < 0:
            _config_logger.warning('max_retries 不能为负数')
            return False
        if self.timeout < 0:
            _config_logger.warning('timeout 不能为负数')
            return False
        if self.data_prepare_timeout < 0:
            _config_logger.warning('data_prepare_timeout 不能为负数')
            return False
        if self.multi_analysis_timeout < 0:
            _config_logger.warning('multi_analysis_timeout 不能为负数')
            return False
        if self.parallel_processing_timeout < 0:
            _config_logger.warning('parallel_processing_timeout 不能为负数')
            return False
        if self.report_generation_timeout < 0:
            _config_logger.warning('report_generation_timeout 不能为负数')
            return False
        if self.knowledge_retrieval_top_k < 1:
            _config_logger.warning('knowledge_retrieval_top_k 必须 >= 1')
            return False
        if not 0 <= self.knowledge_fusion_threshold <= 1:
            _config_logger.warning('knowledge_fusion_threshold 必须在 [0, 1] 范围内')
            return False
        if abs(self.vector_entity_weight + self.vector_attribute_weight + self.vector_relation_weight - 1.0) > 0.01:
            _config_logger.warning('向量权重之和必须等于1')
            return False
        if not self.health_score_thresholds:
            _config_logger.warning('health_score_thresholds 不能为空')
            return False
        if not self.risk_level_thresholds:
            _config_logger.warning('risk_level_thresholds 不能为空')
            return False
        if self.report_min_length < 0:
            _config_logger.warning('report_min_length 不能为负数')
            return False
        if self.report_max_length < self.report_min_length:
            _config_logger.warning('report_max_length 不能小于 report_min_length')
            return False
        if self.max_report_retries < 0:
            _config_logger.warning('max_report_retries 不能为负数')
            return False
        if self.report_max_iterations < 1:
            _config_logger.warning('report_max_iterations 必须 >= 1')
            return False
        if self.analysis_max_iterations < 1:
            _config_logger.warning('analysis_max_iterations 必须 >= 1')
            return False

        max_sub_timeout = max(
            self.data_prepare_timeout,
            self.multi_analysis_timeout,
            self.parallel_processing_timeout,
            self.report_generation_timeout
        )
        if max_sub_timeout > self.timeout:
            _config_logger.warning(f"警告: 最长阶段超时时间({max_sub_timeout}秒)超过总超时时间({self.timeout}秒)")
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "data_prepare_timeout": self.data_prepare_timeout,
            "multi_analysis_timeout": self.multi_analysis_timeout,
            "parallel_processing_timeout": self.parallel_processing_timeout,
            "report_generation_timeout": self.report_generation_timeout,
            "report_max_iterations": self.report_max_iterations,
            "state_timeouts": self.state_timeouts,
            "enable_knowledge_retrieval": self.enable_knowledge_retrieval,
            "knowledge_retrieval_top_k": self.knowledge_retrieval_top_k,
            "knowledge_fusion_threshold": self.knowledge_fusion_threshold,
            "vector_entity_weight": self.vector_entity_weight,
            "vector_attribute_weight": self.vector_attribute_weight,
            "vector_relation_weight": self.vector_relation_weight,
            "knowledge_merge_limit": self.knowledge_merge_limit,
            "prompt_truncation_chars": self.prompt_truncation_chars,
            "max_prompt_chars": self.max_prompt_chars,
            "max_knowledge_chars": self.max_knowledge_chars,
            "report_min_length": self.report_min_length,
            "report_max_length": self.report_max_length,
            "max_report_retries": self.max_report_retries,
            "report_generation_max_tokens": self.report_generation_max_tokens,
            "value_text_max_chars": self.value_text_max_chars,
            "summary_text_max_chars": self.summary_text_max_chars,
            "basis_text_max_chars": self.basis_text_max_chars,
            "family_history_limit": self.family_history_limit,
            "references_limit": self.references_limit,
            "past_medical_history_limit": self.past_medical_history_limit,
            "template_summary_truncate_len": self.template_summary_truncate_len,
            "anomaly_deviation_threshold": self.anomaly_deviation_threshold,
            "health_score_thresholds": self.health_score_thresholds,
            "risk_level_thresholds": self.risk_level_thresholds,
            "health_dimensions": self.health_dimensions,
            "disease_risk_factors": self.disease_risk_factors,
            "risk_level_standards": self.risk_level_standards,
            "deduction_severe": self.deduction_severe,
            "deduction_moderate": self.deduction_moderate,
            "deduction_mild": self.deduction_mild,
            "deduction_risk_factor": self.deduction_risk_factor,
            "deduction_disease": self.deduction_disease,
            "default_dimension_score": self.default_dimension_score,
            "default_confidence": self.default_confidence,
            "rule_engine_confidence": self.rule_engine_confidence,
            "base_health_score": self.base_health_score,
            "disease_count_mild": self.disease_count_mild,
            "anomaly_count_mild": self.anomaly_count_mild,
            "anomaly_count_moderate": self.anomaly_count_moderate,
            "disease_risk_weight": self.disease_risk_weight,
            "query_entity_limit": self.query_entity_limit,
            "query_disease_limit": self.query_disease_limit,
            "query_anomaly_limit": self.query_anomaly_limit,
            "prompt_budget_buffer": self.prompt_budget_buffer,
            "prompt_budget_minimum": self.prompt_budget_minimum,
            "knowledge_budget_ratio": self.knowledge_budget_ratio,
            "user_info_budget_ratio": self.user_info_budget_ratio,
            "knowledge_item_limit": self.knowledge_item_limit,
            "knowledge_content_truncate_len": self.knowledge_content_truncate_len,
            "max_user_info_chars": self.max_user_info_chars,
            "max_risk_factor_user_info_chars": self.max_risk_factor_user_info_chars,
            "health_assessment_context_length": self.health_assessment_context_length,
            "health_assessment_max_tokens": self.health_assessment_max_tokens,
            "health_assessment_batch_max_tokens": self.health_assessment_batch_max_tokens,
            "health_assessment_max_retries": self.health_assessment_max_retries,
            "health_assessment_enable_thinking": self.health_assessment_enable_thinking,
            "health_assessment_repetition_penalty": self.health_assessment_repetition_penalty,
            "batch_evaluation_enable_thinking": self.batch_evaluation_enable_thinking,
            "reasoning_repair_enable_thinking": self.reasoning_repair_enable_thinking,
            "reasoning_repair_repetition_penalty": self.reasoning_repair_repetition_penalty,
            "multi_disease_threshold": self.multi_disease_threshold,
            "elderly_age_threshold": self.elderly_age_threshold,
            "anomaly_match_score_increment": self.anomaly_match_score_increment,
            "history_weight_multiplier": self.history_weight_multiplier,
            "family_history_weight_multiplier": self.family_history_weight_multiplier,
            "disease_risk_top_n": self.disease_risk_top_n,
            "base_disease_risk_score": self.base_disease_risk_score,
            "max_steps": self.max_steps,
            "sufficiency_threshold": self.sufficiency_threshold,
            "beta": self.relevance_beta,
            "gamma": self.relevance_gamma,
            "degraded_alpha": self.degraded_alpha,
            "degraded_beta": self.degraded_beta,
            "degraded_gamma": self.degraded_gamma,
            "relevance_threshold": self.relevance_threshold,
            "max_retrieve_attempts": self.max_retrieve_attempts,
            "vector_candidate_top_k": self.vector_candidate_top_k,
            "vector_candidate_threshold": self.vector_candidate_threshold,
            "vector_default_score": self.vector_default_score,
            "low_quality_min_content_len": self.low_quality_min_content_len,
            "dimension_weights": self.dimension_weights,
            "batch_evaluation_max_tokens": self.batch_evaluation_max_tokens,
            "batch_refine_max_tokens": self.batch_refine_max_tokens,
            "dimension_max_knowledge_items": self.dimension_max_knowledge_items,
            "analysis_sequential_top_k": self.analysis_sequential_top_k,
            "analysis_max_iterations": self.analysis_max_iterations,
            "analysis_state_timeouts": self.analysis_state_timeouts,
            "rule_knowledge_item_limit": self.rule_knowledge_item_limit,
            "rule_content_truncate_len": self.rule_content_truncate_len,
            "knowledge_item_display_limit": self.knowledge_item_display_limit,
            "evaluation_content_truncate_len": self.evaluation_content_truncate_len,
            "refine_content_truncate_len": self.refine_content_truncate_len,
            "suggested_keyword_limit": self.suggested_keyword_limit,
            "batch_refine_timeout": self.batch_refine_timeout,
            "batch_evaluation_timeout": self.batch_evaluation_timeout,
            "fuzzy_match_symptom_limit": self.fuzzy_match_symptom_limit,
            "disease_per_symptom_limit": self.disease_per_symptom_limit,
            "rule_entity_limit": self.rule_entity_limit,
            "sufficiency_count_high": self.sufficiency_count_high,
            "sufficiency_count_medium": self.sufficiency_count_medium,
            "sufficiency_count_low": self.sufficiency_count_low,
            "sufficiency_score_high": self.sufficiency_score_high,
            "sufficiency_score_medium": self.sufficiency_score_medium,
            "sufficiency_score_low": self.sufficiency_score_low,
            "sufficiency_score_none": self.sufficiency_score_none,
            "degradation_completeness_high": self.degradation_completeness_high,
            "degradation_completeness_medium": self.degradation_completeness_medium,
            "degradation_completeness_low": self.degradation_completeness_low,
            "degradation_core_missing_mild": self.degradation_core_missing_mild,
            "degradation_core_missing_moderate": self.degradation_core_missing_moderate,
            "agent_max_paths_per_dimension": self.agent_max_paths_per_dimension,
            "agent_max_supplement_rounds": self.agent_max_supplement_rounds,
            "agent_vector_prescan_top_k": self.agent_vector_prescan_top_k,
            "agent_vector_prescan_threshold": self.agent_vector_prescan_threshold,
            "agent_plan_temperature": self.agent_plan_temperature,
            "agent_plan_max_tokens": self.agent_plan_max_tokens,
            "agent_evaluate_max_tokens": self.agent_evaluate_max_tokens,
            "agent_evaluate_timeout": self.agent_evaluate_timeout,
            "agent_candidate_retrieve_limit": self.agent_candidate_retrieve_limit,
            "agent_knowledge_selection_temperature": self.agent_knowledge_selection_temperature,
            "agent_knowledge_selection_max_tokens": self.agent_knowledge_selection_max_tokens,
            "agent_knowledge_selection_content_truncate_len": self.agent_knowledge_selection_content_truncate_len,
            "agent_hybrid_relevance_user_weight": self.agent_hybrid_relevance_user_weight,
            "agent_hybrid_relevance_dim_weight": self.agent_hybrid_relevance_dim_weight,
            "agent_hybrid_relevance_temperature": self.agent_hybrid_relevance_temperature,
            "agent_hybrid_relevance_max_tokens": self.agent_hybrid_relevance_max_tokens,
            "agent_hybrid_relevance_content_truncate_len": self.agent_hybrid_relevance_content_truncate_len,
            "agent_patch_path_temperature": self.agent_patch_path_temperature,
            "agent_patch_path_max_tokens": self.agent_patch_path_max_tokens,
            "agent_max_iteration_rounds": self.agent_max_iteration_rounds,
            "max_chain_loops": self.max_chain_loops,
            "max_agent_retrieval_loops": self.max_agent_retrieval_loops,
            "batch_max_workers": self.batch_max_workers,
        })
        return base_dict


def get_runtime_config() -> ReportServiceConfig:
    """获取运行期合并后的报告业务配置，优先从ConfigManager获取"""
    from src.config.config_manager import ConfigManager
    config = ConfigManager().get_business_config("report_service_config")
    if config is not None:
        return config  # type: ignore[return-value]
    return ReportServiceConfig()


business_config = ReportServiceConfig()
