# -*- coding: utf-8 -*-
"""
健康评估模型服务

提供健康评估业务场景下的模型服务。
健康评估模型作为健康评估引擎，对每个子指标进行医学推理评估。

资源获取时机说明：
- 系统启动时：Pool根据min_idle配置预创建资源实例（处于空闲状态）
- 处理请求时：调用acquire()获取资源，处理完成后立即释放
- 禁止在初始化时获取资源并长期持有
"""

import logging
import time
from typing import List, Dict, Optional

from src.orchestration.exceptions import EngineUnavailableError
from src.orchestration.model_business_service.model_business_service import ModelBusinessService
from src.resource_manager.global_resource_manager import GlobalResourceManager
from src.schemas.resource_type import ResourceType, ConfigId
from src.config.business.report_service_config import get_runtime_config
from src.utils.logger import log_arch_event

logger = logging.getLogger(__name__)


class HealthAssessmentModelService(ModelBusinessService[str, str]):
    """
    健康评估模型服务类

    继承ModelBusinessService接口，为健康评估业务提供模型服务。

    设计说明：
        健康评估模型（基于Qwen3-4B-Thinking的AWQ量化版）是医学专用小模型，具备医学推理能力，
        用于健康评估框架中各子指标的医学推理评估。

        健康评估模型context_length由SGLang启动参数决定，max_tokens受此限制。
        使用/no_think模式（enable_thinking=False）跳过推理过程，直接输出JSON评估结果。

    调用方式：
        HealthAssessmentChain通过 health_assessment_model.generate(prompt) 调用本服务，
        本服务内部通过GlobalResourceManager临时获取health_assessment_model资源，调用后立即释放。
    """

    # 健康评估专用的系统提示词
    HEALTH_ASSESSMENT_SYSTEM_PROMPT = """你是一位全科医生，擅长精炼评估。请在3秒内、不超过50字完成思考，然后直接输出JSON。"""

    # 通过配置类集中管理默认参数（惰性代理，避免模块导入时ConfigManager未初始化）
    class _LazyHealthAssessmentConfig:
        """延迟加载配置代理，每次属性访问时从ConfigManager获取真实配置"""
        def __getattr__(self, name):
            return getattr(get_runtime_config(), name)

    _health_assessment_config = _LazyHealthAssessmentConfig()
    DEFAULT_MAX_TOKENS = _health_assessment_config.health_assessment_max_tokens

    def __init__(self):
        """初始化健康评估模型服务"""
        self._system_prompt = self.HEALTH_ASSESSMENT_SYSTEM_PROMPT
        self._init_model()

    def _init_model(self) -> None:
        """
        初始化模型配置

        不获取资源，资源在generate方法中临时获取，使用后立即释放。
        """
        logger.info("[HealthAssessment_MODEL_LOAD] 模型服务初始化完成（health_assessment_model资源池，运行时按需获取）")
        logger.info("[HealthAssessmentModelService] 初始化完成")

    def generate(self, prompt: str, max_tokens: Optional[int] = None, timeout: Optional[float] = None) -> str:
        """
        调用健康评估模型生成评估结果

        在需要时获取health_assessment_model资源，处理完成后立即释放。

        Args:
            prompt: 评估Prompt（已包含评估维度、子指标、用户数据等）
            max_tokens: 最大生成token数，默认从配置类读取
            timeout: 超时时间（秒），默认从配置类读取

        Returns:
            str: 模型生成的评估结果（JSON格式字符串）

        Raises:
            RuntimeError: 资源获取失败时抛出
        """
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if timeout is None:
            timeout = float(self._health_assessment_config.timeout)
        context_length = self._health_assessment_config.health_assessment_context_length
        if max_tokens > context_length:
            logger.warning(f"[HealthAssessmentModelService] max_tokens={max_tokens}超过健康评估模型上下文长度限制，调整为{context_length}")
            max_tokens = context_length
        logger.info(f"[HealthAssessmentModelService] generate调用, prompt长度={len(prompt)}, max_tokens={max_tokens}")
        log_arch_event(
            logger,
            component="HealthAssessmentModelService",
            stage="MODEL_SERVICE",
            event="generate",
            status="before_generate",
            design_id="ARCH-3.3",
        )

        start_time = 0
        try:
            with GlobalResourceManager.acquire(ResourceType.HEALTH_ASSESSMENT_MODEL, ConfigId.HEALTH_ASSESSMENT_CONFIG) as handle:
                if handle is None:
                    raise RuntimeError("Failed to acquire health_assessment_model resource")

                adapter = handle.get_client()
                logger.info("[HealthAssessment_MODEL_LOAD] health_assessment_model资源获取成功，模型已就绪")

                # 构建messages格式
                messages = [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": prompt}
                ]

                # 记录健康评估模型输入
                logger.info(f"[HealthAssessment_INPUT] system_prompt: {self._system_prompt}")
                logger.info(f"[HealthAssessment_INPUT] user_prompt(完整): {prompt}")

                start_time = time.time()
                result = adapter.generate(messages=messages, max_tokens=max_tokens, repetition_penalty=self._health_assessment_config.health_assessment_repetition_penalty, enable_thinking=self._health_assessment_config.batch_evaluation_enable_thinking)
                elapsed = time.time() - start_time

                logger.info(f"[HealthAssessmentModelService] generate完成, elapsed={elapsed:.3f}s, response_length={len(result) if result else 0}")
                logger.info(f"[HealthAssessment_OUTPUT] {result}")
                logger.info(f"[HealthAssessment_DURATION] {elapsed:.3f}s")

                return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[HealthAssessmentModelService] generate失败, elapsed={elapsed:.3f}s, error={str(e)}")
            error_name = type(e).__name__
            error_msg = str(e)
            if "EngineDead" in error_name or "EngineDead" in error_msg:
                try:
                    adapter.mark_engine_dead()
                    logger.error(f"[ENGINE_DEAD] 检测到引擎崩溃，标记引擎不可用: {error_name}: {error_msg}")
                except Exception as e:
                    logger.debug(f"[HealthAssessmentModelService] 标记引擎不可用失败: {e}")
                raise EngineUnavailableError(f"推理引擎已不可用: {error_name}: {error_msg}") from e
            raise

    def call_model_batch(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """
        批量调用健康评估模型生成评估结果

        将多个评估prompt一次性提交给推理引擎进行批量推理，
        利用推理引擎的continuous batching机制共享forward pass，减少引擎运行次数。

        在需要时获取health_assessment_model资源，处理完成后立即释放。

        Args:
            prompts: 评估Prompt列表（每个prompt已包含评估维度、子指标、用户数据等）
            max_tokens: 最大生成token数，默认从配置类读取
            timeout: 单个prompt超时时间（秒），默认从配置类读取
            **kwargs: 其他参数

        Returns:
            List[str]: 每个prompt对应的模型生成结果列表（JSON格式字符串）

        Raises:
            EngineUnavailableError: 当推理引擎已崩溃时抛出
            TimeoutError: 当批量推理超时时抛出
        """
        if max_tokens is None:
            max_tokens = self._health_assessment_config.health_assessment_batch_max_tokens
        if timeout is None:
            timeout = self._health_assessment_config.timeout
        context_length = self._health_assessment_config.health_assessment_context_length
        if max_tokens > context_length:
            logger.warning(f"[HealthAssessmentModelService] batch max_tokens={max_tokens}超过健康评估模型上下文长度限制，调整为{context_length}")
            max_tokens = context_length
        batch_size = len(prompts)
        logger.info(f"[HealthAssessmentModelService] call_model_batch调用, batch_size={batch_size}, max_tokens={max_tokens}")
        log_arch_event(
            logger,
            component="HealthAssessmentModelService",
            stage="MODEL_SERVICE",
            event="call_model_batch",
            status="before_generate",
            design_id="ARCH-3.3",
        )

        start_time = 0
        try:
            with GlobalResourceManager.acquire(ResourceType.HEALTH_ASSESSMENT_MODEL, ConfigId.HEALTH_ASSESSMENT_CONFIG) as handle:
                if handle is None:
                    raise RuntimeError("Failed to acquire health_assessment_model resource for batch")

                adapter = handle.get_client()
                logger.info("[HealthAssessment_MODEL_LOAD] health_assessment_model资源获取成功，模型已就绪（批量推理）")

                # 为每个prompt构建messages格式
                messages_list = [
                    [
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    for prompt in prompts
                ]

                # 记录健康评估模型批量输入
                logger.info(f"[HealthAssessment_INPUT] 批量推理prompt数量={batch_size}")
                for i, prompt in enumerate(prompts):
                    logger.info(f"[HealthAssessment_INPUT] Batch[{i}] prompt_len={len(prompt)}")
                    logger.info(f"[HealthAssessment_INPUT] Batch[{i}] prompt(完整): {prompt}")

                # 批量推理过程日志
                logger.info(f"[BATCH_INFERENCE] prompt_count={batch_size}, prompt_lengths={[len(p) for p in prompts]}")

                start_time = time.time()
                results = adapter.generate_batch(
                    messages_list=messages_list,
                    max_tokens=max_tokens,
                    enable_thinking=self._health_assessment_config.batch_evaluation_enable_thinking,
                    repetition_penalty=self._health_assessment_config.health_assessment_repetition_penalty,
                    **kwargs
                )
                elapsed = time.time() - start_time

                logger.info(f"[HealthAssessmentModelService] call_model_batch完成, batch_size={batch_size}, elapsed={elapsed:.3f}s")

                # 记录健康评估模型批量输出
                for i, result in enumerate(results):
                    logger.info(f"[HealthAssessment_OUTPUT] Batch[{i}] result_len={len(result) if result else 0}")
                    logger.info(f"[HealthAssessment_OUTPUT] Batch[{i}] result(完整): {result}")
                logger.info(f"[HealthAssessment_DURATION] 批量推理 batch_size={batch_size} duration={elapsed:.3f}s")

                # 批量推理结果和耗时日志
                logger.info(f"[BATCH_RESULT] result_count={len(results)}, result_lengths={[len(r) if r else 0 for r in results]}")

                return results

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[HealthAssessmentModelService] call_model_batch失败, batch_size={batch_size}, elapsed={elapsed:.3f}s, error={str(e)}")
            logger.warning(f"[DEGRADE_TRIGGER] 批量推理失败，降级为串行推理: {e}")
            error_name = type(e).__name__
            error_msg = str(e)
            if "EngineDead" in error_name or "EngineDead" in error_msg:
                try:
                    adapter.mark_engine_dead()
                    logger.error(f"[ENGINE_DEAD] 检测到引擎崩溃，标记引擎不可用: {error_name}: {error_msg}")
                except Exception as e:
                    logger.debug(f"[HealthAssessmentModelService] 标记引擎不可用失败: {e}")
                raise EngineUnavailableError(f"推理引擎已不可用: {error_name}: {error_msg}") from e
            raise

    def call_model(self, messages: List[Dict[str, str]], timeout: float = None) -> str:
        """
        调用模型 - 实现ModelBusinessService接口

        Args:
            messages: 消息列表
            timeout: 超时时间（秒），默认从配置类读取

        Returns:
            str: 模型生成的文本
        """
        if timeout is None:
            timeout = float(self._health_assessment_config.timeout)
        logger.info(f"[HealthAssessmentModelService] call_model调用, message_count={len(messages)}")
        log_arch_event(
            logger,
            component="HealthAssessmentModelService",
            stage="MODEL_SERVICE",
            event="call_model",
            status="before_generate",
            design_id="ARCH-3.3",
        )

        start_time = 0
        try:
            with GlobalResourceManager.acquire(ResourceType.HEALTH_ASSESSMENT_MODEL, ConfigId.HEALTH_ASSESSMENT_CONFIG) as handle:
                if handle is None:
                    raise RuntimeError("Failed to acquire health_assessment_model resource")

                adapter = handle.get_client()
                logger.info("[HealthAssessment_MODEL_LOAD] health_assessment_model资源获取成功，模型已就绪")

                # 确保messages包含system消息
                has_system = any(msg.get("role") == "system" for msg in messages)
                if not has_system:
                    full_messages = [{"role": "system", "content": self._system_prompt}]
                    full_messages.extend(messages)
                else:
                    full_messages = list(messages)

                # 记录健康评估模型输入
                _sys_parts = [m.get('content', '') for m in messages if m.get('role') == 'system']
                logger.info(f"[HealthAssessment_INPUT] system_prompt: {'|'.join(_sys_parts) if _sys_parts else self._system_prompt}")
                logger.info(f"[HealthAssessment_INPUT] messages(完整): {messages}")

                start_time = time.time()
                result = adapter.generate(messages=full_messages, max_tokens=self.DEFAULT_MAX_TOKENS, enable_thinking=self._health_assessment_config.batch_evaluation_enable_thinking, repetition_penalty=self._health_assessment_config.health_assessment_repetition_penalty)
                elapsed = time.time() - start_time

                logger.info(f"[HealthAssessmentModelService] call_model完成, elapsed={elapsed:.3f}s, response_length={len(result) if result else 0}")
                logger.info(f"[HealthAssessment_OUTPUT] {result}")
                logger.info(f"[HealthAssessment_DURATION] {elapsed:.3f}s")

                return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[HealthAssessmentModelService] call_model失败, elapsed={elapsed:.3f}s, error={str(e)}")
            error_name = type(e).__name__
            error_msg = str(e)
            if "EngineDead" in error_name or "EngineDead" in error_msg:
                try:
                    adapter.mark_engine_dead()
                    logger.error(f"[ENGINE_DEAD] 检测到引擎崩溃，标记引擎不可用: {error_name}: {error_msg}")
                except Exception as e:
                    logger.debug(f"[HealthAssessmentModelService] 标记引擎不可用失败: {e}")
                raise EngineUnavailableError(f"推理引擎已不可用: {error_name}: {error_msg}") from e
            raise

    def release_model(self) -> None:
        """
        释放资源 - 由于资源在每次调用后已释放，此方法保持兼容性但无需操作
        """
        logger.info("[HealthAssessmentModelService] release_model called (no-op, resources released after each call)")
