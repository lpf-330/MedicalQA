# -*- coding: utf-8 -*-
"""
编排层自定义异常

定义编排层内部使用的自定义异常类，避免编排层直接依赖适配层异常，
遵循分层架构原则。
"""

from src.errors.exceptions import ServiceDegradationException
from src.errors.error_codes import ErrorCode


class EngineUnavailableError(ServiceDegradationException):
    """
    编排层自定义异常：模型引擎不可用

    当模型引擎崩溃或不可用时，适配层的EngineDeadException
    被转换为此编排层异常，避免编排层直接依赖适配层。
    """

    def __init__(self, message: str = "模型引擎不可用", cause: Exception = None):
        super().__init__(message, ErrorCode.SGLANG_ENGINE_DEAD, cause=cause)
