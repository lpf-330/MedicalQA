# AI辅助生成：GLM-5，2026-04-15
# -*- coding: utf-8 -*-
"""
健康咨询业务配置文件

定义健康咨询业务的配置参数和所需的资源配置引用。
合并了原ConsultBusinessConfig的所有字段。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.config.base_config import BusinessConfig
import logging

_config_logger = logging.getLogger(__name__)


@dataclass
class ConsultServiceConfig(BusinessConfig):
    """
    健康咨询业务配置类

    合并了原ConsultBusinessConfig的字段，统一由ConfigManager加载。
    """

    business_id: str = "consult_service"
    resource_configs: List[str] = field(default_factory=lambda: [
        "neo4j_connection_config", "reasoning_model_config", "health_assessment_model_config", "milvus_connection_config", "vector_model_config"
    ])

    # 基础参数
    max_retries: int = 3
    timeout: int = 60
    enable_knowledge_retrieval: bool = True
    enable_model_consultation: bool = True

    # 知识检索参数
    knowledge_retrieval_top_k: int = 20
    knowledge_fusion_threshold: float = 0.6
    knowledge_integration_threshold: float = 0.6
    vector_entity_weight: float = 0.40
    vector_attribute_weight: float = 0.30
    vector_relation_weight: float = 0.30
    knowledge_merge_limit: int = 30
    knowledge_sufficiency_min_count: int = 3
    vector_search_timeout: int = 10
    sequential_top_k: int = 20
    sequential_collection_weights: dict = None

    def __post_init__(self):
        if self.sequential_collection_weights is None:
            object.__setattr__(self, 'sequential_collection_weights', {
                "medical_entity": 0.40,
                "entity_attributes": 0.30,
                "entity_relations": 0.30,
            })

    # 回答生成参数
    answer_min_length: int = 200
    answer_max_length: int = 800
    max_knowledge_chars: int = 6000
    max_assistant_len: int = 200
    max_dialogue_rounds: int = 2
    reasoning_enable_thinking: bool = True
    reasoning_repetition_penalty: float = 1.15

    # 超时参数
    query_parse_timeout: int = 10
    knowledge_retrieval_timeout: int = 120
    answer_generation_timeout: int = 120

    # 意图分类与置信度参数
    intent_classification_threshold: float = 0.5
    confidence_high: float = 0.8
    confidence_medium: float = 0.5
    confidence_low: float = 0.3

    # 实体与关键词提取限制
    anchored_entity_limit: int = 10
    symptom_keyword_limit: int = 5
    fuzzy_match_symptom_limit: int = 3
    disease_per_symptom_limit: int = 5
    follow_up_limit: int = 3
    template_knowledge_limit: int = 3
    question_max_length: int = 1000

    # FSM控制参数
    consult_max_iterations: int = 20

    # 批量评估参数
    batch_evaluation_max_tokens: int = 256
    batch_evaluation_timeout: int = 120

    # 知识检索Agent参数（ReAct模式）
    knowledge_retrieval_max_steps: int = 5
    knowledge_retrieval_max_prompt_chars: int = 4000
    sufficiency_count_weight: float = 0.4
    sufficiency_entity_weight: float = 0.3
    sufficiency_relevance_weight: float = 0.3
    sufficiency_count_denominator: float = 10.0
    sufficiency_entity_denominator: float = 5.0
    neo4j_keyword_search_limit: int = 5
    neo4j_degraded_search_score: float = 0.5

    # 咨询知识链参数
    knowledge_context_display_limit: int = 5
    consult_with_knowledge_high_confidence: float = 0.85
    consult_with_knowledge_low_confidence: float = 0.5

    # 默认评分参数
    neo4j_default_score: float = 0.8
    neo4j_degraded_score: float = 0.6
    source_default_confidence: float = 0.5

    # 各状态超时配置
    state_timeouts: Dict[str, int] = field(default_factory=lambda: {
        "INITIAL": 5, "QUERY_PARSE": 10, "KNOWLEDGE_RETRIEVAL": 120,
        "KNOWLEDGE_INTEGRATION": 10, "ANSWER_GENERATION": 120,
        "STREAMING": 30, "FINISHED": 5,
    })

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
        if self.knowledge_retrieval_top_k < 1:
            _config_logger.warning('knowledge_retrieval_top_k 必须 >= 1')
            return False
        if not 0 <= self.knowledge_fusion_threshold <= 1:
            _config_logger.warning('knowledge_fusion_threshold 必须在 [0, 1] 范围内')
            return False
        if not 0 <= self.knowledge_integration_threshold <= 1:
            _config_logger.warning('knowledge_integration_threshold 必须在 [0, 1] 范围内')
            return False
        if abs(self.vector_entity_weight + self.vector_attribute_weight + self.vector_relation_weight - 1.0) > 0.01:
            _config_logger.warning('向量权重之和必须等于1')
            return False
        if self.sequential_top_k < 1:
            _config_logger.warning('sequential_top_k 必须 >= 1')
            return False
        if self.answer_min_length < 0:
            _config_logger.warning('answer_min_length 不能为负数')
            return False
        if self.answer_max_length < self.answer_min_length:
            _config_logger.warning('answer_max_length 不能小于 answer_min_length')
            return False
        if self.query_parse_timeout < 0:
            _config_logger.warning('query_parse_timeout 不能为负数')
            return False
        if self.knowledge_retrieval_timeout < 0:
            _config_logger.warning('knowledge_retrieval_timeout 不能为负数')
            return False
        if self.answer_generation_timeout < 0:
            _config_logger.warning('answer_generation_timeout 不能为负数')
            return False
        if self.consult_max_iterations < 1:
            _config_logger.warning('consult_max_iterations 必须 >= 1')
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "enable_knowledge_retrieval": self.enable_knowledge_retrieval,
            "enable_model_consultation": self.enable_model_consultation,
            "knowledge_retrieval_top_k": self.knowledge_retrieval_top_k,
            "knowledge_fusion_threshold": self.knowledge_fusion_threshold,
            "knowledge_integration_threshold": self.knowledge_integration_threshold,
            "sequential_top_k": self.sequential_top_k,
            "sequential_collection_weights": self.sequential_collection_weights,
            "vector_entity_weight": self.vector_entity_weight,
            "vector_attribute_weight": self.vector_attribute_weight,
            "vector_relation_weight": self.vector_relation_weight,
            "knowledge_merge_limit": self.knowledge_merge_limit,
            "knowledge_sufficiency_min_count": self.knowledge_sufficiency_min_count,
            "vector_search_timeout": self.vector_search_timeout,
            "answer_min_length": self.answer_min_length,
            "answer_max_length": self.answer_max_length,
            "max_knowledge_chars": self.max_knowledge_chars,
            "max_assistant_len": self.max_assistant_len,
            "max_dialogue_rounds": self.max_dialogue_rounds,
            "reasoning_enable_thinking": self.reasoning_enable_thinking,
            "reasoning_repetition_penalty": self.reasoning_repetition_penalty,
            "query_parse_timeout": self.query_parse_timeout,
            "knowledge_retrieval_timeout": self.knowledge_retrieval_timeout,
            "answer_generation_timeout": self.answer_generation_timeout,
            "intent_classification_threshold": self.intent_classification_threshold,
            "confidence_high": self.confidence_high,
            "confidence_medium": self.confidence_medium,
            "confidence_low": self.confidence_low,
            "anchored_entity_limit": self.anchored_entity_limit,
            "symptom_keyword_limit": self.symptom_keyword_limit,
            "fuzzy_match_symptom_limit": self.fuzzy_match_symptom_limit,
            "disease_per_symptom_limit": self.disease_per_symptom_limit,
            "follow_up_limit": self.follow_up_limit,
            "template_knowledge_limit": self.template_knowledge_limit,
            "question_max_length": self.question_max_length,
            "consult_max_iterations": self.consult_max_iterations,
            "batch_evaluation_max_tokens": self.batch_evaluation_max_tokens,
            "batch_evaluation_timeout": self.batch_evaluation_timeout,
            "knowledge_retrieval_max_steps": self.knowledge_retrieval_max_steps,
            "knowledge_retrieval_max_prompt_chars": self.knowledge_retrieval_max_prompt_chars,
            "sufficiency_count_weight": self.sufficiency_count_weight,
            "sufficiency_entity_weight": self.sufficiency_entity_weight,
            "sufficiency_relevance_weight": self.sufficiency_relevance_weight,
            "sufficiency_count_denominator": self.sufficiency_count_denominator,
            "sufficiency_entity_denominator": self.sufficiency_entity_denominator,
            "neo4j_keyword_search_limit": self.neo4j_keyword_search_limit,
            "neo4j_degraded_search_score": self.neo4j_degraded_search_score,
            "knowledge_context_display_limit": self.knowledge_context_display_limit,
            "consult_with_knowledge_high_confidence": self.consult_with_knowledge_high_confidence,
            "consult_with_knowledge_low_confidence": self.consult_with_knowledge_low_confidence,
            "neo4j_default_score": self.neo4j_default_score,
            "neo4j_degraded_score": self.neo4j_degraded_score,
            "source_default_confidence": self.source_default_confidence,
            "state_timeouts": self.state_timeouts,
        })
        return base_dict


def get_runtime_config() -> ConsultServiceConfig:
    """获取运行期合并后的咨询业务配置，优先从ConfigManager获取"""
    from src.config.config_manager import ConfigManager
    config = ConfigManager().get_business_config("consult_service_config")
    if config is not None:
        return config  # type: ignore[return-value]
    return ConsultServiceConfig()


business_config = ConsultServiceConfig()
