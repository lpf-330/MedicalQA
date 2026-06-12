# -*- coding: utf-8 -*-
"""
知识检索Chain策略专属资源类
"""

from dataclasses import dataclass
from typing import Optional

from src.orchestration.tool_call_handler.Impl.vector_retrieval_handler import VectorRetrievalHandler
from src.orchestration.tool_call_handler.Impl.neo4j_medical_handler import Neo4jMedicalHandler


@dataclass
class KnowledgeRetrievalResource:
    """
    知识检索Chain策略专属资源类

    Attributes:
        vector_handler: 向量检索Handler
        neo4j_handler: Neo4j医疗Handler
    """
    vector_handler: Optional[VectorRetrievalHandler] = None
    neo4j_handler: Optional[Neo4jMedicalHandler] = None
