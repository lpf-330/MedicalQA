# -*- coding: utf-8 -*-
"""
SGLang适配器

为项目各层级提供统一的SGLang推理引擎操作接口。
SGLang是独立HTTP服务，兼容OpenAI API。

双实例架构：
- :30000 Qwen3-4B-AWQ 主推理
- :30001 健康评估模型-4B-AWQ 健康评估
"""

from .sglang_adapter import SGLangAdapter
from .sglang_adapter_impl import SGLangAdapterImpl

__all__ = [
    'SGLangAdapter',
    'SGLangAdapterImpl',
]
