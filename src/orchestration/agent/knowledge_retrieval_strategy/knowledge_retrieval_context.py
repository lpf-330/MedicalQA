# -*- coding: utf-8 -*-
"""
知识检索策略上下文数据类

该模块定义KnowledgeRetrievalContextBody和RetrievalStep数据类，
用于知识检索Agent策略的上下文数据传递。
基于设计文档《项目业务详细设计v5》第2.3节的设计实现。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RetrievalStep:
    """单步检索记录"""
    step_num: int
    thought: str = ""
    action: str = ""
    action_params: Dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    results: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_num": self.step_num,
            "thought": self.thought,
            "action": self.action,
            "action_params": self.action_params,
            "observation": self.observation,
            "results_count": len(self.results)
        }


@dataclass
class KnowledgeRetrievalContextBody:
    """
    知识检索Agent上下文数据
    
    输入数据来自QUERY_PARSE阶段：
    - query_text: 查询文本
    - extracted_entities: 已提取的医疗实体
    - intent_label: 意图标签
    """
    query_text: str
    extracted_entities: List[Dict] = field(default_factory=list)
    intent_label: str = ""
    
    current_state: str = "Thought"
    current_step: int = 0
    
    all_results: List[Dict] = field(default_factory=list)
    anchored_entities: List[Dict] = field(default_factory=list)
    anchored_relations: List[Dict] = field(default_factory=list)
    
    step_history: List[RetrievalStep] = field(default_factory=list)
    
    is_sufficient: bool = False
    sufficiency_score: float = 0.0
    
    degraded: bool = False
    degraded_reason: str = ""
    error_code: int = 0
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_text": self.query_text,
            "extracted_entities": self.extracted_entities,
            "intent_label": self.intent_label,
            "current_state": self.current_state,
            "current_step": self.current_step,
            "all_results_count": len(self.all_results),
            "anchored_entities_count": len(self.anchored_entities),
            "anchored_relations_count": len(self.anchored_relations),
            "step_history": [s.to_dict() for s in self.step_history],
            "is_sufficient": self.is_sufficient,
            "sufficiency_score": self.sufficiency_score,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
            "error_code": self.error_code,
            "error_message": self.error_message
        }
