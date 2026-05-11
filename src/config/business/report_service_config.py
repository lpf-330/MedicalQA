# AI辅助生成：GLM-5，2026-04-15
# -*- coding: utf-8 -*-
"""
健康报告生成业务配置文件

定义健康报告生成业务的配置参数和所需的资源配置引用。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.config.base_config import BusinessConfig


@dataclass
class ReportServiceConfig(BusinessConfig):
    """
    健康报告生成业务配置类

    属性：
        business_id: 业务ID（文件名作为唯一标识）
        resource_configs: 所需的资源配置文件名列表
        max_retries: 最大重试次数
        timeout: 总超时时间（秒）
        data_prepare_timeout: 数据准备超时时间（秒）
        multi_analysis_timeout: 多维度分析超时时间（秒）
        parallel_processing_timeout: 并行处理超时时间（秒）
        report_generation_timeout: 报告生成超时时间（秒）
        enable_knowledge_retrieval: 是否启用知识检索
        knowledge_retrieval_top_k: 知识检索返回Top-K结果数
        knowledge_fusion_threshold: 知识融合阈值
        vector_entity_weight: 向量实体权重
        vector_attribute_weight: 向量属性权重
        vector_relation_weight: 向量关系权重
        health_score_thresholds: 健康评分阈值配置
        risk_level_thresholds: 风险等级阈值配置
        report_min_length: 报告最小长度
        report_max_length: 报告最大长度
        max_report_retries: 报告生成最大重试次数
    """

    business_id: str = "report_service"
    resource_configs: List[str] = field(default_factory=lambda: ["neo4j_config", "vllm_config", "milvus_config", "vector_model_config"])

    max_retries: int = 3
    timeout: int = 300
    data_prepare_timeout: int = 5
    multi_analysis_timeout: int = 30
    parallel_processing_timeout: int = 60
    report_generation_timeout: int = 240
    enable_knowledge_retrieval: bool = True
    knowledge_retrieval_top_k: int = 30
    knowledge_fusion_threshold: float = 0.6
    vector_entity_weight: float = 0.40
    vector_attribute_weight: float = 0.30
    vector_relation_weight: float = 0.30
    health_score_thresholds: Dict[str, int] = field(default_factory=lambda: {"excellent": 90, "good": 80, "normal": 70, "poor": 60})
    risk_level_thresholds: Dict[str, int] = field(default_factory=lambda: {"low": 20, "mild": 40, "moderate": 60})
    report_min_length: int = 1000
    report_max_length: int = 5000
    max_report_retries: int = 2

    def validate(self) -> bool:
        """
        验证配置有效性

        Returns:
            bool: 配置是否有效
        """
        if not super().validate():
            return False

        if not self.resource_configs:
            print("警告: resource_configs 不能为空")
            return False
        if self.max_retries < 0:
            print("警告: max_retries 不能为负数")
            return False
        if self.timeout < 0:
            print("警告: timeout 不能为负数")
            return False
        if self.data_prepare_timeout < 0:
            print("警告: data_prepare_timeout 不能为负数")
            return False
        if self.multi_analysis_timeout < 0:
            print("警告: multi_analysis_timeout 不能为负数")
            return False
        if self.parallel_processing_timeout < 0:
            print("警告: parallel_processing_timeout 不能为负数")
            return False
        if self.report_generation_timeout < 0:
            print("警告: report_generation_timeout 不能为负数")
            return False
        if self.knowledge_retrieval_top_k < 1:
            print("警告: knowledge_retrieval_top_k 必须 >= 1")
            return False
        if not 0 <= self.knowledge_fusion_threshold <= 1:
            print("警告: knowledge_fusion_threshold 必须在 [0, 1] 范围内")
            return False
        if abs(self.vector_entity_weight + self.vector_attribute_weight + self.vector_relation_weight - 1.0) > 0.01:
            print("警告: 向量权重之和必须等于1")
            return False
        if not self.health_score_thresholds:
            print("警告: health_score_thresholds 不能为空")
            return False
        if not self.risk_level_thresholds:
            print("警告: risk_level_thresholds 不能为空")
            return False
        if self.report_min_length < 0:
            print("警告: report_min_length 不能为负数")
            return False
        if self.report_max_length < self.report_min_length:
            print("警告: report_max_length 不能小于 report_min_length")
            return False
        if self.max_report_retries < 0:
            print("警告: max_report_retries 不能为负数")
            return False

        # 验证总超时时间是否足够（应大于等于最长的阶段超时时间）
        max_sub_timeout = max(
            self.data_prepare_timeout,
            self.multi_analysis_timeout,
            self.parallel_processing_timeout,
            self.report_generation_timeout
        )
        if max_sub_timeout > self.timeout:
            print(f"警告: 最长阶段超时时间({max_sub_timeout}秒)超过总超时时间({self.timeout}秒)")
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        导出配置为字典

        Returns:
            Dict[str, Any]: 配置字典
        """
        base_dict = super().to_dict()
        base_dict.update({
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "data_prepare_timeout": self.data_prepare_timeout,
            "multi_analysis_timeout": self.multi_analysis_timeout,
            "parallel_processing_timeout": self.parallel_processing_timeout,
            "report_generation_timeout": self.report_generation_timeout,
            "enable_knowledge_retrieval": self.enable_knowledge_retrieval,
            "knowledge_retrieval_top_k": self.knowledge_retrieval_top_k,
            "knowledge_fusion_threshold": self.knowledge_fusion_threshold,
            "vector_entity_weight": self.vector_entity_weight,
            "vector_attribute_weight": self.vector_attribute_weight,
            "vector_relation_weight": self.vector_relation_weight,
            "health_score_thresholds": self.health_score_thresholds,
            "risk_level_thresholds": self.risk_level_thresholds,
            "report_min_length": self.report_min_length,
            "report_max_length": self.report_max_length,
            "max_report_retries": self.max_report_retries,
        })
        return base_dict


business_config = ReportServiceConfig()
