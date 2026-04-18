# -*- coding: utf-8 -*-
"""
健康咨询业务配置文件

定义健康咨询业务的配置参数和所需的资源配置引用。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.config.base_config import BusinessConfig


@dataclass
class ConsultServiceConfig(BusinessConfig):
    """
    健康咨询业务配置类

    属性：
        business_id: 业务ID（文件名作为唯一标识）
        resource_configs: 所需的资源配置文件名列表
        max_retries: 最大重试次数
        timeout: 超时时间（秒）
        enable_knowledge_retrieval: 是否启用知识检索
        enable_model_consultation: 是否启用模型咨询
        knowledge_retrieval_top_k: 知识检索返回Top-K结果数
        knowledge_fusion_threshold: 知识融合阈值
        vector_entity_weight: 向量实体权重
        vector_attribute_weight: 向量属性权重
        vector_relation_weight: 向量关系权重
        answer_min_length: 回答最小长度
        answer_max_length: 回答最大长度
        query_parse_timeout: 查询解析超时时间（秒）
        knowledge_retrieval_timeout: 知识检索超时时间（秒）
        answer_generation_timeout: 回答生成超时时间（秒）
    """

    business_id: str = "consult_service"
    resource_configs: List[str] = field(default_factory=lambda: ["neo4j_config", "vllm_config", "milvus_config", "vector_model_config"])

    max_retries: int = 3
    timeout: int = 60
    enable_knowledge_retrieval: bool = True
    enable_model_consultation: bool = True

    knowledge_retrieval_top_k: int = 20
    knowledge_fusion_threshold: float = 0.6
    vector_entity_weight: float = 0.40
    vector_attribute_weight: float = 0.30
    vector_relation_weight: float = 0.30
    answer_min_length: int = 200
    answer_max_length: int = 800
    query_parse_timeout: int = 10
    knowledge_retrieval_timeout: int = 20
    answer_generation_timeout: int = 60

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
        if self.knowledge_retrieval_top_k < 1:
            print("警告: knowledge_retrieval_top_k 必须 >= 1")
            return False
        if not 0 <= self.knowledge_fusion_threshold <= 1:
            print("警告: knowledge_fusion_threshold 必须在 [0, 1] 范围内")
            return False
        if abs(self.vector_entity_weight + self.vector_attribute_weight + self.vector_relation_weight - 1.0) > 0.01:
            print("警告: 向量权重之和必须等于1")
            return False
        if self.answer_min_length < 0:
            print("警告: answer_min_length 不能为负数")
            return False
        if self.answer_max_length < self.answer_min_length:
            print("警告: answer_max_length 不能小于 answer_min_length")
            return False
        if self.query_parse_timeout < 0:
            print("警告: query_parse_timeout 不能为负数")
            return False
        if self.knowledge_retrieval_timeout < 0:
            print("警告: knowledge_retrieval_timeout 不能为负数")
            return False
        if self.answer_generation_timeout < 0:
            print("警告: answer_generation_timeout 不能为负数")
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
            "enable_knowledge_retrieval": self.enable_knowledge_retrieval,
            "enable_model_consultation": self.enable_model_consultation,
            "knowledge_retrieval_top_k": self.knowledge_retrieval_top_k,
            "knowledge_fusion_threshold": self.knowledge_fusion_threshold,
            "vector_entity_weight": self.vector_entity_weight,
            "vector_attribute_weight": self.vector_attribute_weight,
            "vector_relation_weight": self.vector_relation_weight,
            "answer_min_length": self.answer_min_length,
            "answer_max_length": self.answer_max_length,
            "query_parse_timeout": self.query_parse_timeout,
            "knowledge_retrieval_timeout": self.knowledge_retrieval_timeout,
            "answer_generation_timeout": self.answer_generation_timeout,
        })
        return base_dict


business_config = ConsultServiceConfig()
