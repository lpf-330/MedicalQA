# -*- coding: utf-8 -*-
"""
Transformers适配器

为项目各层级、各类提供统一的Transformers模型操作接口。
"""

from .transformers_adapter import TransformersAdapter
from .transformers_adapter_impl import TransformersAdapterImpl

__all__ = ['TransformersAdapter', 'TransformersAdapterImpl']
