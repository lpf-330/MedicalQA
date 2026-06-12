# -*- coding: utf-8 -*-
"""
回答生成Chain策略专属输入数据类
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AnswerGenerationContextBody:
    """
    回答生成Chain策略专属输入数据类

    Attributes:
        query_text: 用户查询文本
        knowledge_context: 整合后的知识素材文本
        intent_label: 意图标签
        chat_history: 对话历史
    """
    query_text: str
    knowledge_context: str = ""
    intent_label: str = ""
    chat_history: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "query_text": self.query_text,
            "knowledge_context": self.knowledge_context,
            "intent_label": self.intent_label,
            "chat_history": self.chat_history
        }
