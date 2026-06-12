# -*- coding: utf-8 -*-
"""
知识检索Chain策略专属输入数据类
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class KnowledgeRetrievalContextBody:
    """
    知识检索Chain策略专属输入数据类

    Attributes:
        query_text: 查询文本
        extracted_entities: 已提取的医疗实体
        intent_label: 意图标签
    """
    query_text: str
    extracted_entities: List[Dict] = field(default_factory=list)
    intent_label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "query_text": self.query_text,
            "extracted_entities": self.extracted_entities,
            "intent_label": self.intent_label
        }
