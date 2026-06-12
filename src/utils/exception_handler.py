# -*- coding: utf-8 -*-
"""
异常处理类（重导出模块）

原 ErrorCode、MedicalQAException 及其子类、ExceptionHandler 等已迁移至 src.errors 包，
本模块保留为向后兼容重导出。新代码请使用：from src.errors import ...
"""

from src.errors.error_codes import ErrorCode  # noqa: F401
from src.errors.exceptions import (  # noqa: F401
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
from src.errors.handler import (  # noqa: F401
    ExceptionHandler,
    get_exception_handler,
    catch_exception,
)
