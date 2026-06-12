# -*- coding: utf-8 -*-
"""
SGLang适配器接口

为项目各层级提供统一的SGLang推理引擎操作接口。
SGLang是独立HTTP服务，兼容OpenAI Chat Completions API。

关键设计：
- 使用messages格式（chat completion）
- 通过HTTP连接外部服务，不需要load_model
- 支持connect/disconnect管理连接生命周期
"""

from abc import abstractmethod
from typing import AsyncIterator, Iterator, List, Optional

from src.adapters.base_adapter import BaseAdapter


class SGLangAdapter(BaseAdapter):
    """
    SGLang适配器接口

    定义SGLang推理引擎操作的标准接口，为项目各层级提供统一的访问方式。
    SGLang是独立HTTP服务，通过OpenAI兼容API交互。
    """

    @abstractmethod
    def connect(self, base_url: str, **kwargs) -> None:
        """
        连接SGLang HTTP服务

        Args:
            base_url: SGLang服务地址（如 http://localhost:30000）
            **kwargs: 其他连接参数
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """断开与SGLang服务的连接"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        检查是否已连接SGLang服务

        Returns:
            bool: 是否已连接
        """
        pass

    @abstractmethod
    def generate(
        self,
        messages: List[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        enable_thinking: Optional[bool] = None,
        **kwargs
    ) -> str:
        """
        生成文本（非流式）

        Args:
            messages: 对话消息列表，格式为 [{"role": "system/user/assistant", "content": "..."}]
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top_p参数
            repetition_penalty: 重复惩罚系数
            enable_thinking: 是否启用thinking模式（Qwen3架构专用）
            **kwargs: 其他生成参数

        Returns:
            生成的文本
        """
        pass

    @abstractmethod
    def generate_batch(
        self,
        messages_list: List[List[dict]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> List[str]:
        """
        批量生成文本（非流式）

        Args:
            messages_list: 对话消息列表的列表
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top_p参数
            **kwargs: 其他生成参数

        Returns:
            生成的文本列表
        """
        pass

    @abstractmethod
    def stream_generate(
        self,
        messages: List[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        流式生成文本（用于SSE）

        Args:
            messages: 对话消息列表
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top_p参数
            **kwargs: 其他生成参数

        Yields:
            生成的文本片段
        """
        pass

    @abstractmethod
    async def async_stream_generate(
        self,
        messages: List[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        异步流式生成文本

        Args:
            messages: 对话消息列表
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top_p参数
            **kwargs: 其他生成参数

        Yields:
            生成的文本片段
        """
        pass
