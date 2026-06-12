# -*- coding: utf-8 -*-
"""
健康咨询策略结果数据类

该模块定义ConsultResultData数据类，用于健康咨询策略的结果数据传递。
基于设计文档《项目业务详细设计v5》第2节的设计实现。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ConsultResultData:
    answer: str
    suggestions: List[str] = field(default_factory=list)
    related_knowledge: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    confidence: float = 0.0
    session_id: str = ""
    sources: List[str] = field(default_factory=list)
    word_count: int = 0
    is_health_consultation: bool = True
    error_code: int = 0
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "suggestions": self.suggestions,
            "related_knowledge": self.related_knowledge,
            "follow_up_questions": self.follow_up_questions,
            "confidence": self.confidence,
            "session_id": self.session_id,
            "sources": self.sources,
            "word_count": self.word_count,
            "is_health_consultation": self.is_health_consultation,
            "error_code": self.error_code,
            "error_message": self.error_message
        }
