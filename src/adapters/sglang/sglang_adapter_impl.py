# -*- coding: utf-8 -*-
"""
SGLang适配器实现类

通过OpenAI兼容REST API与SGLang HTTP服务交互。
使用openai库作为HTTP客户端，SGLang原生兼容OpenAI Chat Completions API。
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import AsyncIterator, Iterator, List, Optional

import openai

from src.adapters.sglang.sglang_adapter import SGLangAdapter
from src.utils.logger import log_arch_event, truncate_for_log

logger = logging.getLogger(__name__)

_THINK_START = "<think>"
_THINK_END = "</think>"



class SGLangAdapterImpl(SGLangAdapter):
    """
    SGLang适配器实现类

    通过OpenAI兼容API与SGLang HTTP服务交互。
    支持双实例架构（:30000主推理 + :30001健康评估）。
    """

    def __init__(self):
        super().__init__()
        self._client: Optional[openai.OpenAI] = None
        self._async_client: Optional[openai.AsyncOpenAI] = None
        self._base_url: Optional[str] = None
        self._model_name: Optional[str] = None
        self._connected: bool = False
        self._default_temperature: float = 0.0
        self._default_max_tokens: int = 1
        self._default_top_p: float = 0.0
        self._default_repetition_penalty: float = 1.15
        logger.debug("[SGLangAdapter] 初始化SGLang适配器")

    def connect(self, base_url: str, **kwargs) -> None:
        """
        连接SGLang HTTP服务

        Args:
            base_url: SGLang服务地址
            **kwargs: 可选参数
                - model_name: 模型名称（用于API请求）
                - default_temperature: 默认温度
                - default_max_tokens: 默认最大token数
                - default_top_p: 默认top_p
                - default_repetition_penalty: 默认重复惩罚
                - timeout: 请求超时时间（秒）
        """
        if self._connected:
            logger.debug(f"[SGLangAdapter] 已连接，跳过: base_url={base_url}")
            return

        self._base_url = base_url
        self._model_name = kwargs.get("model_name", "")
        self._default_temperature = kwargs.get("default_temperature", 0.0)
        self._default_max_tokens = kwargs.get("default_max_tokens", 1)
        self._default_top_p = kwargs.get("default_top_p", 0.0)
        self._default_repetition_penalty = kwargs.get("default_repetition_penalty", 1.15)
        timeout = kwargs.get("timeout", 120.0)

        logger.info(
            f"[SGLangAdapter] 连接SGLang服务: base_url={base_url}, "
            f"model_name={self._model_name}, timeout={timeout}"
        )

        try:
            self._client = openai.OpenAI(
                base_url=f"{base_url}/v1",
                api_key="empty",
                timeout=timeout,
            )
            self._async_client = openai.AsyncOpenAI(
                base_url=f"{base_url}/v1",
                api_key="empty",
                timeout=timeout,
            )
            self._connected = True
            self._set_initialized(True)
            log_arch_event(logger, component="SGLangAdapter", stage="ADAPTER", event="connect", status="success", design_id="ARCH-7.4", base_url=base_url)
            logger.info(f"[SGLangAdapter] 连接成功: base_url={base_url}")
        except Exception as e:
            self._connected = False
            logger.error(f"[SGLangAdapter] 连接失败: base_url={base_url}, error={e}")
            raise

    def disconnect(self) -> None:
        """断开与SGLang服务的连接"""
        if not self._connected:
            logger.debug("[SGLangAdapter] 未连接，跳过断开")
            return

        logger.info(f"[SGLangAdapter] 断开连接: base_url={self._base_url}")

        try:
            if self._client:
                self._client.close()
            if self._async_client:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(self._async_client.close())
                    else:
                        loop.run_until_complete(self._async_client.close())
                except RuntimeError:
                    pass
        except Exception as e:
            logger.warning(f"[SGLangAdapter] 断开连接时异常: {e}")
        finally:
            self._client = None
            self._async_client = None
            self._connected = False
            self._base_url = None
            self._model_name = None
            self._set_initialized(False)
        log_arch_event(logger, component="SGLangAdapter", stage="ADAPTER", event="disconnect", status="success", design_id="ARCH-7.4")

    def is_connected(self) -> bool:
        return self._connected

    def is_initialized(self) -> bool:
        return self._connected

    def _build_completion_params(
        self,
        messages: List[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        stream: bool = False,
        enable_thinking: Optional[bool] = None,
    ) -> dict:
        """构建chat completion请求参数"""
        params = {
            "model": self._model_name or "default",
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self._default_max_tokens,
            "temperature": temperature if temperature is not None else self._default_temperature,
            "top_p": top_p if top_p is not None else self._default_top_p,
            "stream": stream,
        }
        extra_body = {}
        rep_penalty = repetition_penalty if repetition_penalty is not None else self._default_repetition_penalty
        if rep_penalty > 0.0:
            extra_body["repetition_penalty"] = rep_penalty
        if enable_thinking is not None:
            extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        if extra_body:
            params["extra_body"] = extra_body
        return params

    def _extract_content(self, response) -> str:
        """从chat completion响应中提取文本内容（优先利用reasoning-parser的分离结果，降级手动剥离thinking标签）"""
        if not response.choices:
            logger.warning("[SGLangAdapter] 响应无choices")
            return ""
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            logger.warning(f"[SGLangAdapter] 输出被截断: finish_reason=length, completion_tokens可能达到max_tokens上限")
        message = choice.message
        content = message.content or ""

        # 记录reasoning_content（如有），便于调试和日志可观测性
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content:
            logger.debug(f"[SGLangAdapter] reasoning_content已由SGLang分离, 长度={len(reasoning_content)}")

        # 防御性处理：当reasoning-parser未生效时，content可能仍含think标签
        if _THINK_START in content and _THINK_END in content:
            logger.warning("[SGLangAdapter] content仍含thinking标签（reasoning-parser可能未生效），执行手动剥离")
            content = content.split(_THINK_END, 1)[1].strip()
        elif content.startswith(_THINK_START):
            logger.warning("[SGLangAdapter] content含未闭合thinking标签（reasoning-parser可能未生效），执行手动剥离")
            content = content[len(_THINK_START):].strip()

        return content

    @staticmethod
    def _filter_stream_chunk(buffer: str, in_thinking: bool, enable_thinking: bool):
        """过滤流式chunk中的thinking内容，返回(filtered_text, new_in_thinking)"""
        if enable_thinking:
            return buffer, False

        text = buffer
        # When in_thinking, skip everything until THINK_END
        if in_thinking:
            end_idx = text.find(_THINK_END)
            if end_idx != -1:
                text = text[end_idx + len(_THINK_END):]
                in_thinking = False
                # Fall through to process remaining text
            else:
                return "", True

        # Not in thinking: look for THINK_START
        idx = text.find(_THINK_START)
        if idx == -1:
            return text, False

        before = text[:idx]
        after = text[idx + len(_THINK_START):]
        end_idx = after.find(_THINK_END)
        if end_idx != -1:
            # Complete think block within this chunk
            remaining = after[end_idx + len(_THINK_END):]
            # Recursively process remaining
            result, new_state = SGLangAdapterImpl._filter_stream_chunk(
                before + remaining, False, False
            )
            return result, new_state
        else:
            # Think block started but not ended
            return before, True

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
        """生成文本（非流式）"""
        if not self._connected or not self._client:
            raise RuntimeError("[SGLangAdapter] 未连接SGLang服务")

        params = self._build_completion_params(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stream=False,
            enable_thinking=enable_thinking,
        )

        logger.debug(
            f"[SGLangAdapter] generate: model={params['model']}, "
            f"max_tokens={params['max_tokens']}, temp={params['temperature']}"
        )
        logger.debug(f"[SGLangAdapter] request: {truncate_for_log(repr(params), 500)}")

        try:
            extra_body = params.pop("extra_body", None)
            response = self._client.chat.completions.create(**params, extra_body=extra_body)
            result = self._extract_content(response)
            logger.debug(f"[SGLangAdapter] response: {truncate_for_log(repr(result), 500)}")
            usage = getattr(response, "usage", None)
            if usage:
                logger.debug(
                    f"[SGLangAdapter] generate完成: "
                    f"prompt_tokens={usage.prompt_tokens}, "
                    f"completion_tokens={usage.completion_tokens}"
                )
            log_arch_event(logger, component="SGLangAdapter", stage="ADAPTER", event="generate", status="success", design_id="ARCH-7.4")
            return result
        except openai.APIConnectionError as e:
            logger.error(f"[SGLangAdapter] 连接错误: {e}")
            raise
        except openai.APIStatusError as e:
            logger.error(f"[SGLangAdapter] API错误: status={e.status_code}, message={e.message}")
            raise
        except Exception as e:
            logger.error(f"[SGLangAdapter] 生成失败: {e}")
            raise

    def generate_batch(
        self,
        messages_list: List[List[dict]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_workers: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """批量生成文本（线程并发调用，利用SGLang服务端continuous batching并行推理）

        Args:
            messages_list: 多组消息列表
            max_tokens: 最大生成token数
            temperature: 采样温度
            top_p: top_p采样参数
            max_workers: 最大并发线程数，默认为len(messages_list)即全并发
        """
        if not self._connected or not self._client:
            raise RuntimeError("[SGLangAdapter] 未连接SGLang服务")

        total = len(messages_list)
        workers = max_workers or total
        logger.debug(f"[SGLangAdapter] generate_batch: total={total}, workers={workers}")

        if total <= 1 or workers <= 1:
            return self._generate_batch_sequential(
                messages_list, max_tokens, temperature, top_p, **kwargs
            )

        results = [None] * total

        def _call(index: int) -> tuple:
            try:
                result = self.generate(
                    messages=messages_list[index],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **kwargs
                )
                return (index, result)
            except Exception as e:
                logger.error(f"[SGLangAdapter] batch[{index}/{total}] 失败: {e}")
                return (index, "")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_call, i): i for i in range(total)}
            for future in as_completed(futures):
                index, result = future.result()
                results[index] = result

        success_count = sum(1 for r in results if r)
        logger.debug(f"[SGLangAdapter] generate_batch完成: total={total}, success={success_count}")
        return results

    def _generate_batch_sequential(
        self,
        messages_list: List[List[dict]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> List[str]:
        """批量生成文本（串行回退）"""
        results = []
        total = len(messages_list)
        for i, messages in enumerate(messages_list):
            try:
                result = self.generate(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                logger.error(f"[SGLangAdapter] batch[{i}/{total}] 失败: {e}")
                results.append("")
        success_count = sum(1 for r in results if r)
        logger.debug(f"[SGLangAdapter] generate_batch_sequential完成: total={total}, success={success_count}")
        return results

    def stream_generate(
        self,
        messages: List[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> Iterator[str]:
        """流式生成文本"""
        if not self._connected or not self._client:
            raise RuntimeError("[SGLangAdapter] 未连接SGLang服务")

        params = self._build_completion_params(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            enable_thinking=kwargs.get("enable_thinking"),
        )

        logger.debug(
            f"[SGLangAdapter] stream_generate: model={params['model']}, "
            f"max_tokens={params['max_tokens']}"
        )
        logger.debug(f"[SGLangAdapter] request: {truncate_for_log(repr(params), 500)}")

        try:
            extra_body = params.pop("extra_body", None)
            enable_thinking = (extra_body or {}).get("enable_thinking", False)
            stream = self._client.chat.completions.create(**params, extra_body=extra_body)
            in_thinking = False
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    content = delta.content or ""
                    if content:
                        filtered, in_thinking = self._filter_stream_chunk(
                            content, in_thinking, enable_thinking
                        )
                        if filtered:
                            yield filtered
        except openai.APIConnectionError as e:
            logger.error(f"[SGLangAdapter] 流式连接错误: {e}")
            raise
        except Exception as e:
            logger.error(f"[SGLangAdapter] 流式生成失败: {e}")
            raise

    async def async_stream_generate(
        self,
        messages: List[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """异步流式生成文本"""
        if not self._connected or not self._async_client:
            raise RuntimeError("[SGLangAdapter] 未连接SGLang服务")

        params = self._build_completion_params(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            enable_thinking=kwargs.get("enable_thinking"),
        )

        logger.debug(
            f"[SGLangAdapter] async_stream_generate: model={params['model']}, "
            f"max_tokens={params['max_tokens']}"
        )
        logger.debug(f"[SGLangAdapter] request: {truncate_for_log(repr(params), 500)}")

        try:
            extra_body = params.pop("extra_body", None)
            enable_thinking = (extra_body or {}).get("enable_thinking", False)
            stream = await self._async_client.chat.completions.create(**params, extra_body=extra_body)
            in_thinking = False
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    content = delta.content or ""
                    if content:
                        filtered, in_thinking = self._filter_stream_chunk(
                            content, in_thinking, enable_thinking
                        )
                        if filtered:
                            yield filtered
        except openai.APIConnectionError as e:
            logger.error(f"[SGLangAdapter] 异步流式连接错误: {e}")
            raise
        except Exception as e:
            logger.error(f"[SGLangAdapter] 异步流式生成失败: {e}")
            raise
