# -*- coding: utf-8 -*-
"""
统一异常类定义

包含 MedicalQAException 基础异常、通用异常子类和 ServiceDegradationException 降级异常层级。
降级异常用于编排层 strategy 的 isinstance 判断，替代原有的字符串匹配降级逻辑。
"""

from typing import Any, Dict, Optional, Type, Union

from src.errors.error_codes import ErrorCode


class MedicalQAException(Exception):
    """MedicalQA 系统基础异常类"""

    def __init__(
        self,
        error_code: Union[ErrorCode, int],
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.error_code = error_code if isinstance(error_code, ErrorCode) else ErrorCode(error_code)
        self.message = message
        self.details = details or {}
        self.cause = cause
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'error_code': self.error_code.value,
            'error_name': self.error_code.name,
            'message': self.message,
            'details': self.details
        }
        if self.cause:
            result['cause'] = str(self.cause)
        return result

    def __str__(self) -> str:
        return f"[{self.error_code.name}] {self.message}"


# ============================================================================
# 通用异常子类
# ============================================================================

class ParamException(MedicalQAException):
    """参数异常"""

    def __init__(
        self,
        message: str = "参数错误",
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(ErrorCode.PARAM_ERROR, message, details, cause)


class BusinessException(MedicalQAException):
    """业务异常"""

    def __init__(
        self,
        message: str = "业务处理错误",
        error_code: ErrorCode = ErrorCode.BUSINESS_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(error_code, message, details, cause)


class ResourceException(MedicalQAException):
    """资源异常"""

    def __init__(
        self,
        message: str = "资源错误",
        error_code: ErrorCode = ErrorCode.RESOURCE_UNAVAILABLE,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(error_code, message, details, cause)


class DataException(MedicalQAException):
    """数据异常"""

    def __init__(
        self,
        message: str = "数据错误",
        error_code: ErrorCode = ErrorCode.DATA_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(error_code, message, details, cause)


class NetworkException(MedicalQAException):
    """网络异常"""

    def __init__(
        self,
        message: str = "网络错误",
        error_code: ErrorCode = ErrorCode.NETWORK_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(error_code, message, details, cause)


# ============================================================================
# 降级异常层级
# ============================================================================

class ServiceDegradationException(MedicalQAException):
    """
    服务降级基类

    编排层 strategy 使用 isinstance(error, ServiceDegradationException) 或其子类
    判断降级分支，替代原有的字符串匹配逻辑。
    """

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(error_code, message, details, cause)


class LLMServiceError(ServiceDegradationException):
    """LLM/模型服务不可用"""

    def __init__(
        self,
        message: str = "LLM服务不可用",
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message, ErrorCode.LLM_FAILURE, details, cause)


class Neo4jConnectionError(ServiceDegradationException):
    """Neo4j 连接失败"""

    def __init__(
        self,
        message: str = "Neo4j连接失败",
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message, ErrorCode.NEO4J_UNAVAILABLE, details, cause)


class MilvusUnavailableError(ServiceDegradationException):
    """Milvus 不可用"""

    def __init__(
        self,
        message: str = "Milvus服务不可用",
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message, ErrorCode.MILVUS_UNAVAILABLE, details, cause)


class SGLangEngineError(ServiceDegradationException):
    """SGLang 引擎故障"""

    def __init__(
        self,
        message: str = "SGLang引擎故障",
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message, ErrorCode.SGLANG_ENGINE_DEAD, details, cause)


class HealthAssessmentError(ServiceDegradationException):
    """健康评估失败"""

    def __init__(
        self,
        message: str = "健康评估失败",
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message, ErrorCode.HEALTH_ASSESS_FAILURE, details, cause)


class DataPrepareError(ServiceDegradationException):
    """数据准备阶段失败"""

    def __init__(
        self,
        message: str = "数据准备失败",
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message, ErrorCode.REPORT_DATA_PREPARE_TIMEOUT, details, cause)


class DataParseError(ServiceDegradationException):
    """数据解析阶段失败"""

    def __init__(
        self,
        message: str = "数据解析失败",
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message, ErrorCode.REPORT_DATA_PARSE_TIMEOUT, details, cause)


class MultiAnalysisError(ServiceDegradationException):
    """多维分析阶段失败"""

    def __init__(
        self,
        message: str = "多维分析失败",
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message, ErrorCode.REPORT_COMPREHENSIVE_ANALYSIS_TIMEOUT, details, cause)


class ComprehensiveAnalysisError(ServiceDegradationException):
    """综合分析阶段失败"""

    def __init__(
        self,
        message: str = "综合分析失败",
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message, ErrorCode.REPORT_COMPREHENSIVE_ANALYSIS_TIMEOUT, details, cause)
