# -*- coding: utf-8 -*-
"""
统一错误码与异常定义包

对外暴露所有公开符号，使用方式：
    from src.errors import ErrorCode, MedicalQAException, LLMServiceError, ExceptionHandler
"""

from src.errors.error_codes import ErrorCode
from src.errors.exceptions import (
    MedicalQAException,
    ParamException,
    BusinessException,
    ResourceException,
    DataException,
    NetworkException,
    ServiceDegradationException,
    LLMServiceError,
    Neo4jConnectionError,
    MilvusUnavailableError,
    SGLangEngineError,
    HealthAssessmentError,
    DataPrepareError,
    DataParseError,
    MultiAnalysisError,
    ComprehensiveAnalysisError,
)
from src.errors.handler import (
    ExceptionHandler,
    get_exception_handler,
    catch_exception,
)

__all__ = [
    'ErrorCode',
    'MedicalQAException',
    'ParamException',
    'BusinessException',
    'ResourceException',
    'DataException',
    'NetworkException',
    'ServiceDegradationException',
    'LLMServiceError',
    'Neo4jConnectionError',
    'MilvusUnavailableError',
    'SGLangEngineError',
    'HealthAssessmentError',
    'DataPrepareError',
    'DataParseError',
    'MultiAnalysisError',
    'ComprehensiveAnalysisError',
    'ExceptionHandler',
    'get_exception_handler',
    'catch_exception',
]
