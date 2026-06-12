# -*- coding: utf-8 -*-
"""
回答生成Chain策略专属输出数据类
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AnswerGenerationResultData:
    """
    回答生成Chain策略专属输出数据类

    Attributes:
        answer_text: 生成的回答文本
        sources: 知识来源引用列表
        word_count: 回答字数
        has_disclaimer: 是否包含免责声明
    """
    answer_text: str
    sources: List[str] = field(default_factory=list)
    word_count: int = 0
    has_disclaimer: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "answer_text": self.answer_text,
            "sources": self.sources,
            "word_count": self.word_count,
            "has_disclaimer": self.has_disclaimer
        }
