# -*- coding: utf-8 -*-
"""
报告知识检索Chain策略专属输出数据类
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ReportKnowledgeRetrievalResultData:
    """
    报告知识检索Chain策略专属输出数据类

    Attributes:
        vector_results: 向量检索原始结果
        knowledge_results: 图谱查询增强结果
        merged_results: 合并去重后的最终知识素材
    """
    vector_results: List[Dict] = field(default_factory=list)
    knowledge_results: List[Dict] = field(default_factory=list)
    merged_results: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "vector_results": self.vector_results,
            "knowledge_results": self.knowledge_results,
            "merged_results": self.merged_results
        }
