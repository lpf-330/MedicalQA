# -*- coding: utf-8 -*-
"""
回答生成Chain策略专属资源类
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.orchestration.model_business_service.Impl.consult_model_service import ConsultModelService


@dataclass
class AnswerGenerationResource:
    """
    回答生成Chain策略专属资源类

    Attributes:
        model_service: 咨询模型服务
    """
    model_service: Optional[ConsultModelService] = None

    def get_model_result(self, messages: List[Dict[str, str]]) -> str:
        """
        获取模型生成结果

        Args:
            messages: 消息列表

        Returns:
            模型生成的回复
        """
        if self.model_service is None:
            return "模型服务未初始化"
        return self.model_service.call_model(messages)
