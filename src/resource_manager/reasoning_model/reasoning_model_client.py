# -*- coding: utf-8 -*-
"""
SGLang模型客户端封装

实现ModelResourceClient接口，为业务层提供统一的模型推理操作接口。
SGLang模型基于SGLang框架部署，使用messages格式（chat completion）。
SGLang是外部HTTP服务，引擎崩溃不影响客户端进程。
"""

import logging
from typing import AsyncIterator, Iterator, List, Optional

from src.resource_manager.resource import Resource
from src.resource_manager.resource_client import ModelResourceClient
from src.resource_manager.reasoning_model.reasoning_model_resource import ReasoningModelResource

logger = logging.getLogger(__name__)


class ReasoningModelClient(ModelResourceClient):
    """
    SGLang模型客户端类

    实现ModelResourceClient接口，为业务层提供统一的模型推理操作接口。
    使用messages格式（chat completion）与SGLang HTTP服务交互。

    属性：
        _resource: 封装的SGLang模型资源
    """

    def __init__(self, resource: ReasoningModelResource):
        self._resource = resource

    def get_resource_type(self) -> str:
        """获取资源类型标识"""
        return self._resource.get_type()

    def get_raw_resource(self) -> Resource:
        """获取原始资源实例"""
        return self._resource

    def generate(
        self,
        messages: List[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        **kwargs
    ) -> str:
        """生成文本（非流式）"""
        logger.info("[STAGE_ENTER] ReasoningModelClient.generate")
        adapter = self._resource.get_adapter()
        if adapter is None:
            logger.info("[STAGE_EXIT] ReasoningModelClient.generate")
            raise RuntimeError("SGLang adapter not initialized")
        result = adapter.generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            **kwargs
        )
        logger.info("[STAGE_EXIT] ReasoningModelClient.generate")
        return result

    def generate_batch(
        self,
        messages_list: List[List[dict]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> List[str]:
        """批量生成文本（非流式）"""
        adapter = self._resource.get_adapter()
        if adapter is None:
            raise RuntimeError("SGLang adapter not initialized")
        return adapter.generate_batch(
            messages_list=messages_list,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )

    def stream_generate(
        self,
        messages: List[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> Iterator[str]:
        """流式生成文本"""
        adapter = self._resource.get_adapter()
        if adapter is None:
            raise RuntimeError("SGLang adapter not initialized")
        return adapter.stream_generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )

    async def async_stream_generate(
        self,
        messages: List[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """异步流式生成文本"""
        adapter = self._resource.get_async_adapter()
        if adapter is None:
            raise RuntimeError("SGLang async adapter not initialized")
        async for chunk in adapter.async_stream_generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        ):
            yield chunk

    def is_model_loaded(self) -> bool:
        """检查SGLang服务是否已连接"""
        adapter = self._resource.get_adapter()
        if adapter is None:
            return False
        return adapter.is_connected()

    def mark_engine_dead(self) -> None:
        """标记模型引擎为不可用状态（SGLang外部服务，仅记录日志）"""
        logger.warning(
            "[RESOURCE_LIFECYCLE] mark_engine_dead: SGLang是外部HTTP服务，"
            "引擎崩溃不影响客户端进程，仅记录日志"
        )
