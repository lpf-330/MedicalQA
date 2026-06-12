# -*- coding: utf-8 -*-
"""
健康评估模型客户端封装

实现ModelResourceClient接口，为业务层提供统一的健康评估模型健康评估模型推理操作接口。
健康评估模型-4B-AWQ通过SGLang HTTP服务提供健康评估能力。
"""

import logging
from typing import AsyncIterator, Iterator, List, Optional

from src.resource_manager.resource import Resource
from src.resource_manager.resource_client import ModelResourceClient
from src.resource_manager.health_assessment_model.health_assessment_model_resource import HealthAssessmentModelResource
from src.adapters.sglang.sglang_adapter import SGLangAdapter

logger = logging.getLogger(__name__)


class HealthAssessmentModelClient(ModelResourceClient):
    """
    健康评估模型健康评估模型客户端类

    实现ModelResourceClient接口，为业务层提供统一的模型推理操作接口。
    使用messages格式（chat completion）与SGLang服务交互。

    属性：
        _resource: 封装的健康评估模型资源
    """

    def __init__(self, resource: HealthAssessmentModelResource):
        """
        初始化健康评估模型客户端

        Args:
            resource: 健康评估模型资源实例
        """
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
        **kwargs
    ) -> str:
        """
        生成文本（非流式）

        Args:
            messages: 对话消息列表，格式为 [{"role": "system/user/assistant", "content": "..."}]
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top_p参数
            **kwargs: 其他生成参数

        Returns:
            生成的文本
        """
        logger.info("[STAGE_ENTER] HealthAssessmentModelClient.generate")
        adapter = self._resource.get_adapter()
        if adapter is None:
            logger.info("[STAGE_EXIT] HealthAssessmentModelClient.generate")
            raise RuntimeError("健康评估模型 SGLang adapter not initialized")
        try:
            result = adapter.generate(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            logger.info("[STAGE_EXIT] HealthAssessmentModelClient.generate")
            return result
        except Exception as e:
            logger.info(f"[STAGE_EXIT] HealthAssessmentModelClient.generate, error: {e}")
            raise

    def generate_batch(
        self,
        messages_list: List[List[dict]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> List[str]:
        """
        批量生成文本

        Args:
            messages_list: 对话消息列表的列表
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top_p参数
            **kwargs: 其他生成参数

        Returns:
            生成的文本列表
        """
        adapter = self._resource.get_adapter()
        if adapter is None:
            raise RuntimeError("健康评估模型 SGLang adapter not initialized")
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
        """
        流式生成文本

        Args:
            messages: 对话消息列表
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top_p参数
            **kwargs: 其他生成参数

        Yields:
            生成的文本片段
        """
        adapter = self._resource.get_adapter()
        if adapter is None:
            raise RuntimeError("健康评估模型 SGLang adapter not initialized")
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
        adapter = self._resource.get_adapter()
        if adapter is None:
            raise RuntimeError("健康评估模型 SGLang adapter not initialized")
        async for chunk in adapter.async_stream_generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        ):
            yield chunk

    def is_model_loaded(self) -> bool:
        """
        检查模型是否已加载

        Returns:
            bool: 模型是否已加载（SGLang服务是否已连接）
        """
        adapter = self._resource.get_adapter()
        if adapter is None:
            return False
        return adapter.is_connected()

    def mark_engine_dead(self) -> None:
        """
        标记模型引擎为不可用状态

        SGLang是外部HTTP服务，无法通过客户端标记引擎状态，
        仅记录日志供排查。
        """
        logger.warning(
            "[RESOURCE_LIFECYCLE] mark_engine_dead: "
            "SGLang是外部服务，无法标记引擎状态，仅记录日志"
        )
