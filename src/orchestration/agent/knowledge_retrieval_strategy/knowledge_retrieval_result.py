# -*- coding: utf-8 -*-
"""
知识检索策略结果数据类

该模块定义KnowledgeRetrievalResultData数据类，
用于知识检索Agent策略的结果数据传递。
基于设计文档《项目业务详细设计v5》第2.3节的设计实现。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class KnowledgeRetrievalResultData:
    """
    知识检索Agent结果数据
    
    输出数据提交给KNOWLEDGE_INTEGRATION阶段：
    - merged_results: 合并去重后的最终知识素材
    - anchored_entities: 锚定实体列表
    - anchored_relations: 锚定关系列表
    """
    merged_results: List[Dict] = field(default_factory=list)
    anchored_entities: List[Dict] = field(default_factory=list)
    anchored_relations: List[Dict] = field(default_factory=list)
    
    total_steps: int = 0
    sufficiency_score: float = 0.0
    is_sufficient: bool = False
    
    degraded: bool = False
    degraded_reason: str = ""
    error_code: int = 0
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "merged_results": self.merged_results,
            "anchored_entities": self.anchored_entities,
            "anchored_relations": self.anchored_relations,
            "total_steps": self.total_steps,
            "sufficiency_score": self.sufficiency_score,
            "is_sufficient": self.is_sufficient,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
            "error_code": self.error_code,
            "error_message": self.error_message
        }
