# -*- coding: utf-8 -*-
"""
报告知识检索Chain策略专属资源类
"""

from dataclasses import dataclass
from typing import Any, Optional

from src.orchestration.tool_call_handler.Impl.vector_retrieval_handler import VectorRetrievalHandler
from src.orchestration.tool_call_handler.Impl.neo4j_medical_handler import Neo4jMedicalHandler


@dataclass
class ReportKnowledgeRetrievalResource:
    """
    报告知识检索Chain策略专属资源类

    Attributes:
        vector_handler: 向量检索Handler（复用健康咨询的Handler）
        neo4j_handler: Neo4j医疗Handler（复用健康咨询的Handler）
        vector_encode_service: 向量编码服务（复用健康咨询的Service）
    """
    vector_handler: Optional[VectorRetrievalHandler] = None
    neo4j_handler: Optional[Neo4jMedicalHandler] = None
    vector_encode_service: Optional[Any] = None
