# -*- coding: utf-8 -*-
"""
健康咨询策略上下文数据类

该模块定义ConsultContextBody数据类，用于健康咨询策略的上下文数据传递。
基于设计文档《项目业务详细设计v5》第2节的设计实现。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ConsultContextBody:
    question: str
    session_id: str = ""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    current_state: str = "INITIAL"
    extracted_entities: List[Dict] = field(default_factory=list)
    intent_label: str = ""
    knowledge_results: List[Dict] = field(default_factory=list)
    answer_text: str = ""
    sources: List[str] = field(default_factory=list)
    knowledge_context: str = ""
    is_health_consultation: bool = True
    rewritten_query: str = ""
    error_code: int = 0
    error_message: str = ""
    stream_generator: Any = None
    is_streaming: bool = False
    degraded: bool = False
    degraded_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "session_id": self.session_id,
            "conversation_history": self.conversation_history,
            "user_profile": self.user_profile,
            "current_state": self.current_state,
            "extracted_entities": self.extracted_entities,
            "intent_label": self.intent_label,
            "knowledge_results": self.knowledge_results,
            "answer_text": self.answer_text,
            "sources": self.sources,
            "knowledge_context": self.knowledge_context,
            "is_health_consultation": self.is_health_consultation,
            "rewritten_query": self.rewritten_query,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "stream_generator": self.stream_generator,
            "is_streaming": self.is_streaming,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason
        }
