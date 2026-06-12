# -*- coding: utf-8 -*-
"""
健康报告生成模型服务

提供健康报告生成业务场景下的模型服务。

资源获取时机说明：
- 系统启动时：Pool根据min_idle配置预创建资源实例（处于空闲状态）
- 处理请求时：调用acquire()获取资源，处理完成后立即释放
- 禁止在初始化时获取资源并长期持有
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import AsyncIterator, Dict, Iterator, List, Tuple

from src.orchestration.exceptions import EngineUnavailableError
from src.orchestration.model_business_service.model_business_service import ModelBusinessService
from src.resource_manager.global_resource_manager import GlobalResourceManager
from src.schemas.resource_type import ResourceType, ConfigId
from src.config.business.report_service_config import get_runtime_config
from src.utils.logger import log_arch_event, log_llm_input_final

logger = logging.getLogger(__name__)


class _LazyReportServiceConfig:
    """延迟加载配置代理，每次属性访问时从ConfigManager获取真实配置"""
    def __getattr__(self, name):
        return getattr(get_runtime_config(), name)


_report_service_config = _LazyReportServiceConfig()


class ReportModelService(ModelBusinessService[List[Dict[str, str]], str]):
    """
    健康报告生成模型服务类

    继承ModelBusinessService接口，为健康报告生成业务提供模型服务。

    Token预算分配：
        - Prompt模板：约2000 Token
        - 知识素材：约3000 Token
        - 用户数据摘要：约1500 Token
        - 报告输出：约3500 Token
    """

    DEFAULT_SYSTEM_PROMPT = """你是一位专业的医疗健康评估助手。请严格按照用户提供的报告模板结构生成健康评估报告。

输出要求：
1. 直接以"# 健康评估报告"开头，以免责声明结束
2. 严格按模板输出六个章节，不得增减
3. 第一至第四节：专业学术分析风格，语言严谨准确，不使用图标、特殊符号、emoji表情，仅进行数据分析，不提出建议
4. 第五节：针对老年用户，采用适老化表达，直接、明确地提出建议

禁止输出：
- 禁止输出"分析部分"、"建议部分"等分类标题
- 禁止输出"全文完"、"报告完毕"、"总字数："等结束语或统计信息
- 禁止输出任何报告正文之外的说明、总结、提示等内容
- 禁止输出对本次生成任务的完成汇报

只输出报告正文内容，不输出任何附加内容。 /no_think"""

    def __init__(
        self,
        model_path: str = "",
        system_prompt: str = None
    ):
        self._model_path = model_path
        self._system_prompt = system_prompt if system_prompt is not None else self.DEFAULT_SYSTEM_PROMPT
        self._init_model()

    def _init_model(self) -> None:
        """
        初始化模型

        初始化模型相关配置，不获取资源。
        资源在call_model方法中临时获取，使用后立即释放。
        """
        logger.debug(f"[ReportModelService] _init_model called, model_path={self._model_path}")
        # 初始化配置验证
        if not self._system_prompt:
            logger.warning("[ReportModelService] system_prompt is empty, using default")
            self._system_prompt = self.DEFAULT_SYSTEM_PROMPT

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
        调用模型生成报告内容 - 在需要时获取资源，处理完成后立即释放

        Args:
            messages: 消息列表
            timeout: 超时时间（秒），默认从配置类读取
        """
        if timeout is None:
            timeout = float(_report_service_config.report_generation_timeout)
        logger.info(f"[ReportModelService] call_model called, message_count={len(messages)}")
        logger.info(f"[LLM_CALL_PURPOSE] purpose=报告生成模型调用, message_count={len(messages)}")
        start_time = time.time()
        adapter = None
        try:
            with GlobalResourceManager.acquire(ResourceType.REASONING_MODEL, ConfigId.REASONING_CONFIG) as handle:
                if handle is None:
                    raise RuntimeError("Failed to acquire reasoning_model resource")

                adapter = handle.get_client()

                full_messages = self._build_messages(messages)
                prompt_for_log = str(full_messages)
                logger.info(f"[ReportModelService] 构建的messages数量={len(full_messages)}")

                log_llm_input_final(
                    logger,
                    component="ReportModelService",
                    model_operation="call_model",
                    messages=full_messages,
                    prompt=prompt_for_log,
                )
                log_arch_event(
                    logger,
                    component="ReportModelService",
                    stage="MODEL_SERVICE",
                    event="call_model",
                    status="before_generate",
                    design_id="ARCH-3.3",
                )

                llm_start = time.time()
                result = adapter.generate(messages=full_messages, max_tokens=_report_service_config.report_generation_max_tokens, enable_thinking=_report_service_config.batch_evaluation_enable_thinking, repetition_penalty=_report_service_config.health_assessment_repetition_penalty)
                llm_duration = time.time() - llm_start
                elapsed = time.time() - start_time
                logger.info(f"[LLM_OUTPUT] output_length={len(result)}")
                logger.info(f"[LLM_DURATION] duration={llm_duration:.2f}s")
                logger.info(f"[ReportModelService] call_model completed, elapsed={elapsed:.3f}s, response_length={len(result)}")
                logger.info(f"[LLM_CALL_SUMMARY] input_messages={len(full_messages)}, output_length={len(result)}, elapsed={elapsed:.3f}s")
                return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[ReportModelService] call_model failed, elapsed={elapsed:.3f}s, error={str(e)}")
            error_name = type(e).__name__
            error_msg = str(e)
            if "EngineDead" in error_name or "EngineDead" in error_msg:
                if adapter is not None:
                    try:
                        adapter.mark_engine_dead()
                        logger.error(f"[ENGINE_DEAD] 检测到引擎崩溃，标记引擎不可用: {error_name}: {error_msg}")
                    except Exception as e:
                        logger.debug(f"[ReportModelService] 标记引擎不可用失败: {e}")
                raise EngineUnavailableError(f"推理引擎已不可用: {error_name}: {error_msg}") from e
            raise

    def call_model_batch(
        self,
        prompts: List[str],
        max_tokens: int = None,
        timeout: int = None,
        **kwargs
    ) -> List[str]:
        """
        批量调用模型生成内容 - 在需要时获取资源，处理完成后立即释放

        将多个prompt一次性提交给推理引擎进行批量推理，
        利用推理引擎的continuous batching机制共享forward pass，减少引擎运行次数。

        Args:
            prompts: 输入提示列表，每个元素是一个独立的评估prompt
            max_tokens: 最大生成token数，默认从配置类读取
            timeout: 单个prompt超时时间（秒），默认从配置类读取
            **kwargs: 其他参数

        Returns:
            List[str]: 每个prompt对应的生成结果列表

        Raises:
            EngineUnavailableError: 当推理引擎已崩溃时抛出
            TimeoutError: 当批量推理超时时抛出
        """
        if max_tokens is None:
            max_tokens = _report_service_config.report_generation_max_tokens
        if timeout is None:
            timeout = _report_service_config.timeout
        batch_size = len(prompts)
        logger.info(f"[ReportModelService] call_model_batch called, batch_size={batch_size}, max_tokens={max_tokens}")
        start_time = time.time()

        # 截断超长prompt
        truncated_prompts = []
        for i, prompt in enumerate(prompts):
            if len(prompt) > _report_service_config.prompt_truncation_chars:
                truncated_prompts.append(prompt[:_report_service_config.prompt_truncation_chars])
                logger.warning(f"[ReportModelService] call_model_batch: "
                              f"prompt[{i}]被截断至{_report_service_config.prompt_truncation_chars}字符")
            else:
                truncated_prompts.append(prompt)
            logger.info(f"[ReportModelService] 批量Prompt[{i}] (长度={len(prompt)})")

        # 标准标签日志 - [LLM_CALL_PURPOSE] 记录批量调用目的
        logger.info(f"[LLM_CALL_PURPOSE] purpose=批量报告评估调用, batch_size={len(prompts)}")
        for i, prompt in enumerate(prompts):
            logger.info(f"[LLM_CALL_PURPOSE] Batch[{i}] prompt_length={len(prompt)}")

        max_workers = _report_service_config.batch_max_workers
        max_workers = min(batch_size, max_workers)
        enable_thinking = _report_service_config.batch_evaluation_enable_thinking
        repetition_penalty = _report_service_config.health_assessment_repetition_penalty

        def _generate_single(item: Tuple[int, str]) -> Tuple[int, str]:
            """单prompt生成（线程安全：每个线程独立 acquire→generate→release）"""
            idx, prompt_text = item
            adapter = None
            try:
                with GlobalResourceManager.acquire(ResourceType.REASONING_MODEL, ConfigId.REASONING_CONFIG) as handle:
                    if handle is None:
                        logger.error(f"[ReportModelService] call_model_batch[{idx}/{batch_size}] 获取资源失败")
                        return (idx, "")

                    adapter = handle.get_client()
                    messages = [
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": prompt_text}
                    ]

                    log_llm_input_final(
                        logger,
                        component="ReportModelService",
                        model_operation="call_model_batch",
                        messages=messages,
                        prompt=prompt_text,
                    )
                    result = adapter.generate(
                        messages=messages,
                        max_tokens=max_tokens,
                        enable_thinking=enable_thinking,
                        repetition_penalty=repetition_penalty,
                    )
                    return (idx, result)
            except Exception as e:
                logger.error(f"[ReportModelService] call_model_batch[{idx}/{batch_size}] 失败: {str(e)}")
                error_name = type(e).__name__
                error_msg = str(e)
                if "EngineDead" in error_name or "EngineDead" in error_msg:
                    if adapter is not None:
                        try:
                            adapter.mark_engine_dead()
                        except Exception as e:
                            logger.debug(f"[ReportModelService] 标记引擎不可用失败: {e}")
                    logger.error(f"[ENGINE_DEAD] 检测到引擎崩溃，标记引擎不可用: {error_name}: {error_msg}")
                    raise EngineUnavailableError(f"推理引擎已不可用: {error_name}: {error_msg}") from e
                return (idx, "")

        # 并发执行：ThreadPoolExecutor + 资源池并发控制
        results: List[str] = [""] * batch_size
        engine_dead = False
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_generate_single, (i, prompt)): i
                for i, prompt in enumerate(truncated_prompts)
            }
            for future in as_completed(futures):
                try:
                    idx, result = future.result(timeout=timeout)
                    results[idx] = result
                except EngineUnavailableError:
                    engine_dead = True
                except concurrent.futures.TimeoutError:
                    idx = futures[future]
                    logger.error(f"[ReportModelService] call_model_batch[{idx}] 超时(timeout={timeout}s)")
                except Exception as e:
                    idx = futures[future]
                    logger.error(f"[ReportModelService] call_model_batch[{idx}] future异常: {str(e)}")

        if engine_dead:
            raise EngineUnavailableError("推理引擎已不可用: batch中检测到EngineDead")

        elapsed = time.time() - start_time
        logger.info(f"[ReportModelService] call_model_batch completed, batch_size={batch_size}, max_workers={max_workers}, elapsed={elapsed:.3f}s")

        # 标准标签日志 - [LLM_CALL_SUMMARY] 记录批量调用摘要
        for i, result in enumerate(results):
            logger.info(f"[LLM_CALL_SUMMARY] Batch[{i}] output_length={len(result) if result else 0}")
        logger.info(f"[LLM_CALL_SUMMARY] batch_size={len(results)}, elapsed={elapsed:.3f}s")

        return results

    def generate_report(
        self,
        prompt: str,
        max_tokens: int = None
    ) -> str:
        """
        生成报告 - 在需要时获取资源，处理完成后立即释放
        """
        if max_tokens is None:
            max_tokens = _report_service_config.report_generation_max_tokens
        logger.debug(f"[ReportModelService] generate_report called, prompt_length={len(prompt)}, max_tokens={max_tokens}")
        start_time = time.time()

        try:
            with GlobalResourceManager.acquire(ResourceType.REASONING_MODEL, ConfigId.REASONING_CONFIG) as handle:
                if handle is None:
                    raise RuntimeError("Failed to acquire reasoning_model resource")

                adapter = handle.get_client()

                messages = [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": prompt}
                ]

                logger.info(f"[LLM_CALL_PURPOSE] purpose=报告生成, prompt_length={len(prompt)}, max_tokens={max_tokens}")
                log_llm_input_final(
                    logger,
                    component="ReportModelService",
                    model_operation="generate_report",
                    messages=messages,
                    prompt=prompt,
                )
                log_arch_event(
                    logger,
                    component="ReportModelService",
                    stage="MODEL_SERVICE",
                    event="generate_report",
                    status="before_generate",
                    design_id="ARCH-3.3",
                )
                llm_start = time.time()
                result = adapter.generate(messages=messages, max_tokens=max_tokens, enable_thinking=_report_service_config.batch_evaluation_enable_thinking, repetition_penalty=_report_service_config.health_assessment_repetition_penalty)
                llm_duration = time.time() - llm_start
                elapsed = time.time() - start_time
                logger.info(f"[LLM_OUTPUT] output_length={len(result)}")
                logger.info(f"[LLM_DURATION] duration={llm_duration:.2f}s")
                logger.info(f"[ReportModelService] generate_report completed, elapsed={elapsed:.3f}s, response_length={len(result)}")
                logger.info(f"[LLM_CALL_SUMMARY] input_length={len(prompt)}, output_length={len(result)}, elapsed={elapsed:.3f}s")
                return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[ReportModelService] generate_report failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def stream_generate(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        """
        流式生成报告内容 - 接受完整的messages列表，由调用方控制prompt结构
        """
        logger.info(f"[ReportModelService] stream_generate called, message_count={len(messages)}")
        logger.info(f"[LLM_CALL_PURPOSE] purpose=流式报告生成, message_count={len(messages)}")
        start_time = time.time()

        try:
            with GlobalResourceManager.acquire(ResourceType.REASONING_MODEL, ConfigId.REASONING_CONFIG) as handle:
                if handle is None:
                    raise RuntimeError("Failed to acquire reasoning_model resource")

                adapter = handle.get_client()
                full_messages = self._build_messages(messages)
                prompt_for_log = str(full_messages)
                logger.info(f"[ReportModelService] 构建的messages数量={len(full_messages)}")

                log_llm_input_final(
                    logger,
                    component="ReportModelService",
                    model_operation="stream_generate",
                    messages=full_messages,
                    prompt=prompt_for_log,
                )
                log_arch_event(
                    logger,
                    component="ReportModelService",
                    stage="MODEL_SERVICE",
                    event="stream_generate",
                    status="before_generate",
                    design_id="ARCH-3.3",
                )

                llm_start = time.time()
                collected_content = []
                for chunk in adapter.stream_generate(messages=full_messages, max_tokens=_report_service_config.report_generation_max_tokens, enable_thinking=_report_service_config.health_assessment_enable_thinking, repetition_penalty=_report_service_config.health_assessment_repetition_penalty):
                    collected_content.append(chunk)
                    yield chunk

                llm_duration = time.time() - llm_start
                elapsed = time.time() - start_time
                full_content = ''.join(collected_content)
                logger.info(f"[LLM_OUTPUT] output_length={len(full_content)}")
                logger.info(f"[LLM_DURATION] duration={llm_duration:.2f}s")
                logger.info(f"[ReportModelService] stream_generate completed, elapsed={elapsed:.3f}s")
                logger.info(f"[LLM_OUTPUT_STREAM] total_tokens={len(full_content)}")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[ReportModelService] stream_generate failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    async def async_stream_generate(self, messages: List[Dict[str, str]]) -> 'AsyncIterator[str]':
        """
        异步流式生成报告内容 - 接受完整的messages列表，使用AsyncLLM实现真正的实时流式输出
        """
        logger.info(f"[ReportModelService] async_stream_generate called, message_count={len(messages)}")
        logger.info(f"[LLM_CALL_PURPOSE] purpose=异步流式报告生成, message_count={len(messages)}")
        start_time = time.time()

        try:
            with GlobalResourceManager.acquire(ResourceType.REASONING_MODEL, ConfigId.REASONING_CONFIG) as handle:
                if handle is None:
                    raise RuntimeError("Failed to acquire reasoning_model resource")

                adapter = handle.get_client()
                full_messages = self._build_messages(messages)
                prompt_for_log = str(full_messages)
                logger.info(f"[ReportModelService] 构建的messages数量={len(full_messages)}")

                log_llm_input_final(
                    logger,
                    component="ReportModelService",
                    model_operation="async_stream_generate",
                    messages=full_messages,
                    prompt=prompt_for_log,
                )
                log_arch_event(
                    logger,
                    component="ReportModelService",
                    stage="MODEL_SERVICE",
                    event="async_stream_generate",
                    status="before_generate",
                    design_id="ARCH-3.3",
                )

                llm_start = time.time()
                collected_content = []
                async for chunk in adapter.async_stream_generate(messages=full_messages, max_tokens=_report_service_config.report_generation_max_tokens, enable_thinking=_report_service_config.health_assessment_enable_thinking, repetition_penalty=_report_service_config.health_assessment_repetition_penalty):
                    collected_content.append(chunk)
                    yield chunk

                llm_duration = time.time() - llm_start
                elapsed = time.time() - start_time
                full_content = ''.join(collected_content)
                logger.info(f"[LLM_OUTPUT] output_length={len(full_content)}")
                logger.info(f"[LLM_DURATION] duration={llm_duration:.2f}s")
                logger.info(f"[ReportModelService] async_stream_generate completed, elapsed={elapsed:.3f}s")
                logger.info(f"[LLM_OUTPUT_STREAM] total_tokens={len(full_content)}")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[ReportModelService] async_stream_generate failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def stream_generate_with_context(
        self,
        user_query: str,
        knowledge_context: str = ""
    ) -> Iterator[str]:
        """
        带知识上下文的流式生成 - 在需要时获取资源，流式输出完成后释放

        Args:
            user_query: 用户查询/请求
            knowledge_context: 知识上下文

        Yields:
            生成的文本片段
        """
        logger.info(f"[ReportModelService] stream_generate_with_context called, query_length={len(user_query)}, context_length={len(knowledge_context)}")
        start_time = time.time()

        try:
            with GlobalResourceManager.acquire(ResourceType.REASONING_MODEL, ConfigId.REASONING_CONFIG) as handle:
                if handle is None:
                    raise RuntimeError("Failed to acquire reasoning_model resource")

                adapter = handle.get_client()

                messages = [
                    {"role": "system", "content": self._system_prompt}
                ]

                if knowledge_context:
                    messages.append({"role": "system", "content": f"参考知识素材：\n{knowledge_context}"})

                messages.append({"role": "user", "content": user_query})

                logger.info(f"[LLM_CALL_PURPOSE] purpose=知识增强流式报告生成, query_length={len(user_query)}, context_length={len(knowledge_context)}")
                log_llm_input_final(
                    logger,
                    component="ReportModelService",
                    model_operation="stream_generate_with_context",
                    messages=messages,
                    prompt=user_query,
                )
                log_arch_event(
                    logger,
                    component="ReportModelService",
                    stage="MODEL_SERVICE",
                    event="stream_generate_with_context",
                    status="before_generate",
                    design_id="ARCH-3.3",
                )

                llm_start = time.time()
                collected_content = []
                for chunk in adapter.stream_generate(messages=messages, max_tokens=_report_service_config.report_generation_max_tokens, enable_thinking=_report_service_config.health_assessment_enable_thinking, repetition_penalty=_report_service_config.health_assessment_repetition_penalty):
                    collected_content.append(chunk)
                    yield chunk

                llm_duration = time.time() - llm_start
                elapsed = time.time() - start_time
                full_content = ''.join(collected_content)
                logger.info(f"[LLM_OUTPUT] output_length={len(full_content)}")
                logger.info(f"[LLM_DURATION] duration={llm_duration:.2f}s")
                logger.info(f"[ReportModelService] stream_generate_with_context completed, elapsed={elapsed:.3f}s")
                logger.info(f"[LLM_OUTPUT_STREAM] total_tokens={len(full_content)}")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[ReportModelService] stream_generate_with_context failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def release_model(self) -> None:
        """
        释放资源 - 由于资源在每次调用后已释放，此方法保持兼容性但无需操作
        """
        logger.info("[ReportModelService] release_model called (no-op, resources released after each call)")
