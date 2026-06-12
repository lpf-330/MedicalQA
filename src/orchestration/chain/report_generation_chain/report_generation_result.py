# -*- coding: utf-8 -*-
"""
报告生成Chain策略专属输出数据类
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ReportGenerationResultData:
    """
    报告生成Chain策略专属输出数据类

    Attributes:
        report_content: 报告内容（Markdown格式）
        word_count: 报告字数
        has_disclaimer: 是否包含免责声明
        sources: 知识来源
    """
    report_content: str = ""
    word_count: int = 0
    has_disclaimer: bool = False
    sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "report_content": self.report_content,
            "word_count": self.word_count,
            "has_disclaimer": self.has_disclaimer,
            "sources": self.sources
        }
