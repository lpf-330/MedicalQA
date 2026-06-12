# -*- coding: utf-8 -*-
"""
报告生成Chain策略专属资源类
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ReportGenerationResource:
    """
    报告生成Chain策略专属资源类

    Attributes:
        model_service: 报告模型服务（将在后续实现ReportModelService）
    """
    model_service: Optional[Any] = None

    def get_model_result(self, messages: List[Dict[str, str]], temperature: float = None, max_tokens: int = None) -> str:
        """
        获取模型生成结果

        Args:
            messages: 消息列表
            temperature: 温度参数（None时使用服务默认值）
            max_tokens: 最大token数（None时使用服务默认值）

        Returns:
            模型生成的回复
        """
        if self.model_service is None:
            return "模型服务未初始化"
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return self.model_service.call_model(messages, **kwargs)
