# -*- coding: utf-8 -*-
"""
异常处理器

提供统一的异常处理功能，包括异常捕获、异常记录、异常转换等。
从 src/utils/exception_handler.py 迁入，内部引用改为 src.errors 包。
"""

import traceback
import sys
import logging
from typing import Any, Callable, Dict, Optional, Type, Union
from functools import wraps

from src.errors.error_codes import ErrorCode
from src.errors.exceptions import MedicalQAException


class ExceptionHandler:
    """异常处理类"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger('ExceptionHandler')
        self.exception_handlers: Dict[Type[Exception], Callable] = {}

    def register_handler(
        self,
        exception_class: Type[Exception],
        handler: Callable[[Exception], Any]
    ) -> None:
        self.exception_handlers[exception_class] = handler

    def handle(
        self,
        exception: Exception,
        reraise: bool = False,
        default_return: Any = None
    ) -> Any:
        self._log_exception(exception)

        for exc_class, handler in self.exception_handlers.items():
            if isinstance(exception, exc_class):
                try:
                    return handler(exception)
                except Exception as e:
                    self.logger.error(f"异常处理器执行失败: {e}")
                    if reraise:
                        raise
                    return default_return

        if reraise:
            raise exception
        return default_return

    def _log_exception(self, exception: Exception) -> None:
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
        return MedicalQAException(
            error_code=error_code,
            message=message or str(exception),
            cause=exception
        )

    @staticmethod
    def get_exception_info(exception: Exception) -> Dict[str, Any]:
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
        return ''.join(traceback.format_exception(
            type(exception),
            exception,
            exception.__traceback__
        ))


# 全局异常处理器实例
_global_exception_handler: Optional[ExceptionHandler] = None


def get_exception_handler(logger: Optional[logging.Logger] = None) -> ExceptionHandler:
    global _global_exception_handler
    if _global_exception_handler is None:
        _global_exception_handler = ExceptionHandler(logger)
    return _global_exception_handler


def catch_exception(
    *exception_classes: Type[Exception],
    default_return: Any = None,
    reraise: bool = False
) -> Callable:
    return get_exception_handler().catch(
        *exception_classes,
        default_return=default_return,
        reraise=reraise
    )
