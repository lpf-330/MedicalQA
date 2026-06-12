# -*- coding: utf-8 -*-
"""
健康评估Chain策略专属资源类
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class HealthAssessmentResource:
    """
    健康评估Chain策略专属资源类

    Attributes:
        health_assessment_model: 健康评估模型模型实例
        call_router: MCP调用路由器(用于调用向量检索、图谱查询等工具)
    """
    health_assessment_model: Optional[Any] = None
    call_router: Optional[Any] = None
