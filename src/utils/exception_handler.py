"""
异常处理类

提供统一的异常处理功能，如异常捕获、异常记录、异常转换等。
"""

import traceback
import sys
from typing import Any, Callable, Dict, Optional, Type, Union
from functools import wraps
from enum import Enum

from .logger import Logger, get_logger


class ErrorCode(Enum):
    """
    错误码枚举类
    
    定义系统中所有可能的错误码
    """
    # 系统级错误 (1000-1999)
    UNKNOWN_ERROR = 1000
    SYSTEM_ERROR = 1001
    CONFIG_ERROR = 1002
    INIT_ERROR = 1003
    
    # 参数错误 (2000-2999)
    PARAM_ERROR = 2000
    PARAM_MISSING = 2001
    PARAM_INVALID = 2002
    PARAM_FORMAT_ERROR = 2003
    
    # 业务错误 (3000-3999)
    BUSINESS_ERROR = 3000
    RESOURCE_NOT_FOUND = 3001
    RESOURCE_ALREADY_EXISTS = 3002
    RESOURCE_UNAVAILABLE = 3003
    OPERATION_FAILED = 3004
    
    # 数据错误 (4000-4999)
    DATA_ERROR = 4000
    DATA_NOT_FOUND = 4001
    DATA_FORMAT_ERROR = 4002
    DATA_INTEGRITY_ERROR = 4003
    
    # 网络错误 (5000-5999)
    NETWORK_ERROR = 5000
    CONNECTION_ERROR = 5001
    TIMEOUT_ERROR = 5002
    
    # 权限错误 (6000-6999)
    PERMISSION_ERROR = 6000
    AUTHENTICATION_ERROR = 6001
    AUTHORIZATION_ERROR = 6002


class MedicalQAException(Exception):
    """
    MedicalQA系统基础异常类
    
    所有自定义异常的基类
    
    Attributes:
        error_code: 错误码
        message: 错误消息
        details: 错误详情
        cause: 原始异常
    """
    
    def __init__(
        self,
        error_code: Union[ErrorCode, int],
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        初始化异常
        
        Args:
            error_code: 错误码
            message: 错误消息
            details: 错误详情
            cause: 原始异常
        """
        self.error_code = error_code if isinstance(error_code, ErrorCode) else ErrorCode(error_code)
        self.message = message
        self.details = details or {}
        self.cause = cause
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将异常转换为字典格式
        
        Returns:
            包含异常信息的字典
        """
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


class ExceptionHandler:
    """
    异常处理类
    
    提供统一的异常处理功能，如异常捕获、异常记录、异常转换等。
    
    Attributes:
        logger: 日志记录器
        exception_handlers: 异常处理器映射
    """
    
    def __init__(self, logger: Optional[Logger] = None):
        """
        初始化异常处理器
        
        Args:
            logger: 日志记录器，如果为None则创建默认日志记录器
        """
        self.logger = logger or get_logger('ExceptionHandler')
        self.exception_handlers: Dict[Type[Exception], Callable] = {}
    
    def register_handler(
        self,
        exception_class: Type[Exception],
        handler: Callable[[Exception], Any]
    ) -> None:
        """
        注册异常处理器
        
        Args:
            exception_class: 异常类
            handler: 处理函数
        """
        self.exception_handlers[exception_class] = handler
    
    def handle(
        self,
        exception: Exception,
        reraise: bool = False,
        default_return: Any = None
    ) -> Any:
        """
        处理异常
        
        Args:
            exception: 异常对象
            reraise: 是否重新抛出异常
            default_return: 默认返回值
        
        Returns:
            处理结果或默认返回值
        
        Raises:
            Exception: 如果reraise为True，则重新抛出异常
        """
        # 记录异常
        self._log_exception(exception)
        
        # 查找并执行处理器
        for exc_class, handler in self.exception_handlers.items():
            if isinstance(exception, exc_class):
                try:
                    return handler(exception)
                except Exception as e:
                    self.logger.error(f"异常处理器执行失败: {e}")
                    if reraise:
                        raise
                    return default_return
        
        # 如果没有找到处理器
        if reraise:
            raise exception
        return default_return
    
    def _log_exception(self, exception: Exception) -> None:
        """
        记录异常日志
        
        Args:
            exception: 异常对象
        """
        if isinstance(exception, MedicalQAException):
            self.logger.error(
                f"MedicalQA异常: {exception}",
                exc_info=True,
                error_code=exception.error_code.value,
                details=exception.details
            )
        else:
            self.logger.error(
                f"系统异常: {type(exception).__name__}: {str(exception)}",
                exc_info=True
            )
    
    def catch(
        self,
        *exception_classes: Type[Exception],
        default_return: Any = None,
        reraise: bool = False
    ) -> Callable:
        """
        异常捕获装饰器
        
        Args:
            *exception_classes: 要捕获的异常类
            default_return: 默认返回值
            reraise: 是否重新抛出异常
        
        Returns:
            装饰器函数
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except exception_classes as e:
                    return self.handle(e, reraise=reraise, default_return=default_return)
                except Exception as e:
                    self._log_exception(e)
                    if reraise:
                        raise
                    return default_return
            return wrapper
        return decorator
    
    @staticmethod
    def convert_to_medical_qa_exception(
        exception: Exception,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        message: Optional[str] = None
    ) -> MedicalQAException:
        """
        将普通异常转换为MedicalQA异常
        
        Args:
            exception: 原始异常
            error_code: 错误码
            message: 错误消息，如果为None则使用原始异常的消息
        
        Returns:
            MedicalQAException实例
        """
        return MedicalQAException(
            error_code=error_code,
            message=message or str(exception),
            cause=exception
        )
    
    @staticmethod
    def get_exception_info(exception: Exception) -> Dict[str, Any]:
        """
        获取异常详细信息
        
        Args:
            exception: 异常对象
        
        Returns:
            包含异常详细信息的字典
        """
        exc_type, exc_value, exc_traceback = sys.exc_info()
        
        info = {
            'type': type(exception).__name__,
            'message': str(exception),
            'traceback': traceback.format_exception(
                type(exception),
                exception,
                exception.__traceback__
            )
        }
        
        if isinstance(exception, MedicalQAException):
            info.update(exception.to_dict())
        
        return info
    
    @staticmethod
    def format_exception_traceback(exception: Exception) -> str:
        """
        格式化异常堆栈信息
        
        Args:
            exception: 异常对象
        
        Returns:
            格式化的堆栈信息字符串
        """
        return ''.join(traceback.format_exception(
            type(exception),
            exception,
            exception.__traceback__
        ))


# 全局异常处理器实例
_global_exception_handler: Optional[ExceptionHandler] = None


def get_exception_handler(logger: Optional[Logger] = None) -> ExceptionHandler:
    """
    获取全局异常处理器实例（单例模式）
    
    Args:
        logger: 日志记录器
    
    Returns:
        ExceptionHandler实例
    """
    global _global_exception_handler
    if _global_exception_handler is None:
        _global_exception_handler = ExceptionHandler(logger)
    return _global_exception_handler


def catch_exception(
    *exception_classes: Type[Exception],
    default_return: Any = None,
    reraise: bool = False
) -> Callable:
    """
    异常捕获装饰器（便捷函数）
    
    Args:
        *exception_classes: 要捕获的异常类
        default_return: 默认返回值
        reraise: 是否重新抛出异常
    
    Returns:
        装饰器函数
    """
    return get_exception_handler().catch(
        *exception_classes,
        default_return=default_return,
        reraise=reraise
    )
