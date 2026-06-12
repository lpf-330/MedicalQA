# -*- coding: utf-8 -*-
"""
健康咨询模型业务服务

提供健康咨询业务场景下的模型服务。

资源获取时机说明：
- 系统启动时：Pool根据min_idle配置预创建资源实例（处于空闲状态）
- 处理请求时：调用acquire()获取资源，处理完成后立即释放
- 禁止在初始化时获取资源并长期持有
"""

import logging
import time
from typing import AsyncIterator, Dict, Iterator, List, Optional

from src.orchestration.exceptions import EngineUnavailableError
from src.orchestration.model_business_service.model_business_service import ModelBusinessService
from src.resource_manager.global_resource_manager import GlobalResourceManager
from src.schemas.resource_type import ResourceType, ConfigId
from src.config.business.consult_service_config import get_runtime_config
from src.utils.logger import log_arch_event, log_llm_input_final

logger = logging.getLogger(__name__)


class _LazyConsultConfig:
    """延迟加载配置代理，每次属性访问时从ConfigManager获取真实配置"""
    def __getattr__(self, name):
        return getattr(get_runtime_config(), name)


_consult_config = _LazyConsultConfig()


class ConsultModelService(ModelBusinessService[List[Dict[str, str]], str]):

    def __init__(
        self,
        model_path: str = "",
        system_prompt: str = "你是一个专业的健康咨询助手，请根据用户的描述提供专业的健康建议。 /no_think"
    ):
        self._model_path = model_path
        self._system_prompt = system_prompt
        self._init_model()

    def _init_model(self) -> None:
        """
        初始化模型

        初始化模型相关配置，不获取资源。
        资源在call_model方法中临时获取，使用后立即释放。
        """
        logger.debug(f"[ConsultModelService] _init_model called, model_path={self._model_path}")
        # 初始化配置验证
        if not self._system_prompt:
            logger.warning("[ConsultModelService] system_prompt is empty, using default")
            self._system_prompt = "你是一个专业的健康咨询助手，请根据用户的描述提供专业的健康建议。 /no_think"

    def _build_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        构建完整的messages列表，确保包含system消息

        Args:
            messages: 原始消息列表

        Returns:
            包含system消息的完整消息列表
        """
        has_system_message = any(msg.get("role") == "system" for msg in messages)

        if not has_system_message:
            result = [{"role": "system", "content": self._system_prompt}]
            result.extend(messages)
            return result

        return list(messages)

    def call_model(self, messages: List[Dict[str, str]], timeout: float = None) -> str:
        """
        调用模型 - 在需要时获取资源，处理完成后立即释放

        Args:
            messages: 消息列表
            timeout: 超时时间（秒），默认从配置类读取
        """
        if timeout is None:
            timeout = float(_consult_config.state_timeouts.get("ANSWER_GENERATION", 60))
        logger.info(f"[ConsultModelService] call_model called, message_count={len(messages)}")
        logger.info(f"[LLM_CALL_PURPOSE] purpose=模型调用, message_count={len(messages)}")
        start_time = time.time()

        try:
            with GlobalResourceManager.acquire(ResourceType.REASONING_MODEL, ConfigId.REASONING_CONFIG) as handle:
                if handle is None:
                    raise RuntimeError("Failed to acquire reasoning_model resource")

                adapter = handle.get_client()

                full_messages = self._build_messages(messages)
                prompt_for_log = str(full_messages)
                logger.info(f"[ConsultModelService] 构建的messages数量={len(full_messages)}")

                log_llm_input_final(
                    logger,
                    component="ConsultModelService",
                    model_operation="call_model",
                    messages=full_messages,
                    prompt=prompt_for_log,
                )
                log_arch_event(
                    logger,
                    component="ConsultModelService",
                    stage="MODEL_SERVICE",
                    event="call_model",
                    status="before_generate",
                    design_id="ARCH-3.3",
                )

                llm_start = time.time()
                result = adapter.generate(messages=full_messages, enable_thinking=_consult_config.reasoning_enable_thinking, repetition_penalty=_consult_config.reasoning_repetition_penalty)
                llm_duration = time.time() - llm_start
                elapsed = time.time() - start_time
                logger.info(f"[LLM_OUTPUT] {result}")
                logger.info(f"[LLM_DURATION] duration={llm_duration:.2f}s")
                logger.info(f"[ConsultModelService] call_model completed, elapsed={elapsed:.3f}s, response_length={len(result)}")
                logger.info(f"[LLM_CALL_SUMMARY] input_messages={len(full_messages)}, output_length={len(result)}, elapsed={elapsed:.3f}s")
                return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[ConsultModelService] call_model failed, elapsed={elapsed:.3f}s, error={str(e)}")
            error_name = type(e).__name__
            error_msg = str(e)
            if "EngineDead" in error_name or "EngineDead" in error_msg:
                try:
                    adapter.mark_engine_dead()
                    logger.error(f"[ENGINE_DEAD] 检测到引擎崩溃，标记引擎不可用: {error_name}: {error_msg}")
                except Exception as e:
                    logger.debug(f"[ConsultModelService] 标记引擎不可用失败: {e}")
                raise EngineUnavailableError(f"SGLang引擎已不可用: {error_name}: {error_msg}") from e
            raise

    def call_model_batch(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """
        批量调用模型 - 咨询场景暂不使用批量推理，降级为串行调用

        Args:
            prompts: 输入提示列表
            max_tokens: 最大生成token数，默认从配置类读取
            timeout: 单个prompt超时时间（秒），默认从配置类读取
            **kwargs: 其他参数

        Returns:
            List[str]: 每个prompt对应的生成结果列表
        """
        if max_tokens is None:
            max_tokens = _consult_config.batch_evaluation_max_tokens
        if timeout is None:
            timeout = _consult_config.batch_evaluation_timeout
        logger.info(f"[ConsultModelService] call_model_batch called, batch_size={len(prompts)}, 降级为串行调用")
        results = []
        for i, prompt in enumerate(prompts):
            messages = [{"role": "user", "content": prompt}]
            result = self.call_model(messages, timeout=float(timeout))
            results.append(result)
            logger.info(f"[ConsultModelService] call_model_batch 串行调用[{i}]完成")
        return results

    def generate_with_context(
        self,
        user_query: str,
        knowledge_context: str
    ) -> str:
        logger.info(f"[ConsultModelService] generate_with_context called, query_length={len(user_query)}, context_length={len(knowledge_context)}")
        logger.info(f"[LLM_CALL_PURPOSE] purpose=知识增强回答生成, query_length={len(user_query)}, context_length={len(knowledge_context)}")
        start_time = time.time()
        try:
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "system", "content": f"参考知识：\n{knowledge_context}"},
                {"role": "user", "content": user_query}
            ]
            logger.info(f"[LLM_INPUT] messages_count={len(messages)}")
            for i, msg in enumerate(messages):
                logger.info(f"[LLM_INPUT] Message[{i}] role={msg.get('role')}: {msg.get('content', '')}")
            llm_start = time.time()
            result = self.call_model(messages)
            llm_duration = time.time() - llm_start
            elapsed = time.time() - start_time
            logger.info(f"[LLM_OUTPUT] {result}")
            logger.info(f"[LLM_DURATION] duration={llm_duration:.2f}s")
            logger.info(f"[ConsultModelService] generate_with_context completed, elapsed={elapsed:.3f}s, response_length={len(result)}")
            logger.info(f"[LLM_CALL_SUMMARY] input_query_length={len(user_query)}, input_context_length={len(knowledge_context)}, output_length={len(result)}, elapsed={elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[ConsultModelService] generate_with_context failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def stream_generate(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        """
        流式生成 - 接受完整的messages列表，由调用方控制prompt结构
        """
        logger.info(f"[ConsultModelService] stream_generate called, message_count={len(messages)}")
        logger.info(f"[LLM_CALL_PURPOSE] purpose=流式模型调用, message_count={len(messages)}")
        start_time = time.time()

        try:
            with GlobalResourceManager.acquire(ResourceType.REASONING_MODEL, ConfigId.REASONING_CONFIG) as handle:
                if handle is None:
                    raise RuntimeError("Failed to acquire reasoning_model resource")

                adapter = handle.get_client()
                full_messages = self._build_messages(messages)
                prompt_for_log = str(full_messages)
                logger.info(f"[ConsultModelService] 构建的messages数量={len(full_messages)}")

                log_llm_input_final(
                    logger,
                    component="ConsultModelService",
                    model_operation="stream_generate",
                    messages=full_messages,
                    prompt=prompt_for_log,
                )
                log_arch_event(
                    logger,
                    component="ConsultModelService",
                    stage="MODEL_SERVICE",
                    event="stream_generate",
                    status="before_generate",
                    design_id="ARCH-3.3",
                )

                llm_start = time.time()
                collected_content = []
                for chunk in adapter.stream_generate(messages=full_messages, enable_thinking=_consult_config.reasoning_enable_thinking, repetition_penalty=_consult_config.reasoning_repetition_penalty):
                    collected_content.append(chunk)
                    yield chunk

                llm_duration = time.time() - llm_start
                elapsed = time.time() - start_time
                full_content = ''.join(collected_content)
                logger.info(f"[LLM_OUTPUT] {full_content}")
                logger.info(f"[LLM_DURATION] duration={llm_duration:.2f}s")
                logger.info(f"[ConsultModelService] stream_generate completed, elapsed={elapsed:.3f}s")
                logger.info(f"[LLM_OUTPUT_STREAM] total_tokens={len(full_content)}, content_preview={full_content[:200]}")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[ConsultModelService] stream_generate failed, elapsed={elapsed:.3f}s, error={str(e)}")
            error_name = type(e).__name__
            error_msg = str(e)
            if "EngineDead" in error_name or "EngineDead" in error_msg:
                try:
                    adapter.mark_engine_dead()
                    logger.error(f"[ENGINE_DEAD] 检测到引擎崩溃，标记引擎不可用: {error_name}: {error_msg}")
                except Exception as e:
                    logger.debug(f"[ConsultModelService] 标记引擎不可用失败: {e}")
                raise EngineUnavailableError(f"SGLang引擎已不可用: {error_name}: {error_msg}") from e
            raise

    async def async_stream_generate(self, messages: List[Dict[str, str]]) -> 'AsyncIterator[str]':
        """
        异步流式生成 - 接受完整的messages列表，使用AsyncLLM实现真正的实时流式输出
        """
        logger.info(f"[ConsultModelService] async_stream_generate called, message_count={len(messages)}")
        logger.info(f"[LLM_CALL_PURPOSE] purpose=异步流式模型调用, message_count={len(messages)}")
        start_time = time.time()

        try:
            with GlobalResourceManager.acquire(ResourceType.REASONING_MODEL, ConfigId.REASONING_CONFIG) as handle:
                if handle is None:
                    raise RuntimeError("Failed to acquire reasoning_model resource")

                adapter = handle.get_client()
                full_messages = self._build_messages(messages)
                prompt_for_log = str(full_messages)
                logger.info(f"[ConsultModelService] 构建的messages数量={len(full_messages)}")

                log_llm_input_final(
                    logger,
                    component="ConsultModelService",
                    model_operation="async_stream_generate",
                    messages=full_messages,
                    prompt=prompt_for_log,
                )
                log_arch_event(
                    logger,
                    component="ConsultModelService",
                    stage="MODEL_SERVICE",
                    event="async_stream_generate",
                    status="before_generate",
                    design_id="ARCH-3.3",
                )

                llm_start = time.time()
                collected_content = []
                async for chunk in adapter.async_stream_generate(messages=full_messages, enable_thinking=_consult_config.reasoning_enable_thinking, repetition_penalty=_consult_config.reasoning_repetition_penalty):
                    collected_content.append(chunk)
                    yield chunk

                llm_duration = time.time() - llm_start
                elapsed = time.time() - start_time
                full_content = ''.join(collected_content)
                logger.info(f"[LLM_OUTPUT] {full_content}")
                logger.info(f"[LLM_DURATION] duration={llm_duration:.2f}s")
                logger.info(f"[ConsultModelService] async_stream_generate completed, elapsed={elapsed:.3f}s")
                logger.info(f"[LLM_OUTPUT_STREAM] total_tokens={len(full_content)}, content_preview={full_content[:200]}")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[ConsultModelService] async_stream_generate failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def stream_generate_with_context(self, user_query: str, knowledge_context: str) -> Iterator[str]:
        """
        流式生成 - 在需要时获取资源，流式输出完成后释放
        """
        logger.info(f"[ConsultModelService] stream_generate_with_context called, query_length={len(user_query)}, context_length={len(knowledge_context)}")
        logger.info(f"[LLM_CALL_PURPOSE] purpose=知识增强流式回答生成, query_length={len(user_query)}, context_length={len(knowledge_context)}")
        start_time = time.time()

        try:
            with GlobalResourceManager.acquire(ResourceType.REASONING_MODEL, ConfigId.REASONING_CONFIG) as handle:
                if handle is None:
                    raise RuntimeError("Failed to acquire reasoning_model resource")

                adapter = handle.get_client()

                messages = [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "system", "content": f"参考知识：\n{knowledge_context}"},
                    {"role": "user", "content": user_query}
                ]
                logger.info(f"[ConsultModelService] 构建的messages数量={len(messages)}")

                log_llm_input_final(
                    logger,
                    component="ConsultModelService",
                    model_operation="stream_generate_with_context",
                    messages=messages,
                    prompt=str(messages),
                )
                log_arch_event(
                    logger,
                    component="ConsultModelService",
                    stage="MODEL_SERVICE",
                    event="stream_generate_with_context",
                    status="before_generate",
                    design_id="ARCH-3.3",
                )

                llm_start = time.time()
                collected_content = []
                for chunk in adapter.stream_generate(messages=messages, enable_thinking=_consult_config.reasoning_enable_thinking, repetition_penalty=_consult_config.reasoning_repetition_penalty):
                    collected_content.append(chunk)
                    yield chunk

                llm_duration = time.time() - llm_start
                elapsed = time.time() - start_time
                full_content = ''.join(collected_content)
                logger.info(f"[LLM_OUTPUT] {full_content}")
                logger.info(f"[LLM_DURATION] duration={llm_duration:.2f}s")
                logger.info(f"[ConsultModelService] stream_generate_with_context completed, elapsed={elapsed:.3f}s")
                logger.info(f"[LLM_OUTPUT_STREAM] total_tokens={len(full_content)}, content_preview={full_content[:200]}")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[ConsultModelService] stream_generate_with_context failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def release_model(self) -> None:
        """
        释放资源 - 由于资源在每次调用后已释放，此方法保持兼容性但无需操作
        """
        logger.info("[ConsultModelService] release_model called (no-op, resources released after each call)")
