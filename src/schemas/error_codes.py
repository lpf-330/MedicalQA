# -*- coding: utf-8 -*-
"""
错误码枚举（重导出模块）

原 ErrorCode 枚举已迁移至 src.errors.error_codes，本模块保留为向后兼容重导出。
新代码请使用：from src.errors import ErrorCode
"""

from src.errors.error_codes import ErrorCode  # noqa: F401

__all__ = ['ErrorCode']
