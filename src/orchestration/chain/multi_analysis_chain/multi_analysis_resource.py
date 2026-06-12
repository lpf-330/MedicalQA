# -*- coding: utf-8 -*-
"""
多维度分析Chain策略专属资源类
"""

from dataclasses import dataclass
from typing import Any, Optional

from src.orchestration.tool_call_handler.Impl.intent_classification_handler import IntentClassificationHandler
from src.orchestration.tool_call_handler.Impl.ner_model_handler import NerModelHandler


@dataclass
class MultiAnalysisResource:
    """
    多维度分析Chain策略专属资源类

    Attributes:
        intent_handler: 意图分类Handler（仅用于意图分类，不用于实体提取）
        ner_handler: NER模型Handler（用于医学实体提取）
        vector_encode_service: 向量编码服务（复用健康咨询的Service）
    """
    intent_handler: Optional[IntentClassificationHandler] = None
    ner_handler: Optional[NerModelHandler] = None
    vector_encode_service: Optional[Any] = None
