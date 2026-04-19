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
from typing import Dict, Iterator, List, Optional

from src.orchestration.model_business_service.model_business_service import ModelBusinessService
from src.resource_manager.global_resource_manager import GlobalResourceManager
from src.resource_manager.resource_handle import ResourceHandle
from src.resource_manager.vllm_model.vllm_model_resource import VLLMModelClient

logger = logging.getLogger(__name__)


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

只输出报告正文内容，不输出任何附加内容。"""

    def __init__(
        self,
        model_path: str = "",
        system_prompt: str = None
    ):
        self._model_path = model_path
        self._system_prompt = system_prompt if system_prompt is not None else self.DEFAULT_SYSTEM_PROMPT

    def call_model(self, messages: List[Dict[str, str]]) -> str:
        """
        调用模型生成报告内容 - 在需要时获取资源，处理完成后立即释放
        """
        logger.debug(f"[ReportModelService] call_model called, message_count={len(messages)}")
        logger.debug(f"[ReportModelService] LLM输入 - messages: {messages}")
        start_time = time.time()
        
        handle = None
        try:
            handle = GlobalResourceManager.acquire("vllm_model", "vllm_config")
            if handle is None:
                raise RuntimeError("Failed to acquire vllm_model resource")
            
            model_client = VLLMModelClient(handle.resource)
            
            prompt = self._build_prompt(messages)
            logger.debug(f"[ReportModelService] LLM输入 - 构建的prompt:\n{prompt[:2000]}{'...' if len(prompt) > 2000 else ''}")

            result = model_client.generate(prompt, max_tokens=5000)
            elapsed = time.time() - start_time
            logger.info(f"[ReportModelService] call_model completed, elapsed={elapsed:.3f}s, response_length={len(result)}")
            logger.debug(f"[ReportModelService] LLM输出 - 内容:\n{result[:2000]}{'...' if len(result) > 2000 else ''}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[ReportModelService] call_model failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise
        finally:
            if handle is not None:
                GlobalResourceManager.release(handle)

    def generate_report(
        self,
        prompt: str,
        max_tokens: int = 5000
    ) -> str:
        """
        生成报告 - 在需要时获取资源，处理完成后立即释放
        """
        logger.debug(f"[ReportModelService] generate_report called, prompt_length={len(prompt)}, max_tokens={max_tokens}")
        start_time = time.time()
        
        handle = None
        try:
            handle = GlobalResourceManager.acquire("vllm_model", "vllm_config")
            if handle is None:
                raise RuntimeError("Failed to acquire vllm_model resource")
            
            model_client = VLLMModelClient(handle.resource)

            result = model_client.generate(prompt, max_tokens=max_tokens)
            elapsed = time.time() - start_time
            logger.info(f"[ReportModelService] generate_report completed, elapsed={elapsed:.3f}s, response_length={len(result)}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[ReportModelService] generate_report failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise
        finally:
            if handle is not None:
                GlobalResourceManager.release(handle)

    def stream_generate(self, prompt: str) -> Iterator[str]:
        """
        流式生成报告内容 - 在需要时获取资源，流式输出完成后释放
        """
        logger.info(f"[ReportModelService] stream_generate called, prompt_length={len(prompt)}")
        start_time = time.time()
        
        handle = None
        try:
            handle = GlobalResourceManager.acquire("vllm_model", "vllm_config")
            if handle is None:
                raise RuntimeError("Failed to acquire vllm_model resource")
            
            model_client = VLLMModelClient(handle.resource)

            logger.info(f"[ReportModelService] ========== LLM完整输入 ==========")
            logger.info(f"[ReportModelService] System Prompt: {self._system_prompt}")
            logger.info(f"[ReportModelService] 构建的完整Prompt (长度={len(prompt)}):")
            logger.info(f"{prompt}")
            logger.info(f"[ReportModelService] ==============================")

            for chunk in model_client.stream_generate(prompt):
                yield chunk
            
            elapsed = time.time() - start_time
            logger.info(f"[ReportModelService] stream_generate completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[ReportModelService] stream_generate failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise
        finally:
            if handle is not None:
                GlobalResourceManager.release(handle)

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
        
        handle = None
        try:
            handle = GlobalResourceManager.acquire("vllm_model", "vllm_config")
            if handle is None:
                raise RuntimeError("Failed to acquire vllm_model resource")
            
            model_client = VLLMModelClient(handle.resource)
            
            prompt_parts = [self._system_prompt]
            
            if knowledge_context:
                prompt_parts.append(f"\n参考知识素材：\n{knowledge_context}")
            
            prompt_parts.append(f"\n用户：{user_query}")
            prompt_parts.append("助手：")
            
            full_prompt = "\n".join(prompt_parts)

            for chunk in model_client.stream_generate(full_prompt, max_tokens=5000):
                yield chunk
            
            elapsed = time.time() - start_time
            logger.info(f"[ReportModelService] stream_generate_with_context completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[ReportModelService] stream_generate_with_context failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise
        finally:
            if handle is not None:
                GlobalResourceManager.release(handle)

    def release(self) -> None:
        """
        释放资源 - 由于资源在每次调用后已释放，此方法保持兼容性但无需操作
        """
        logger.info("[ReportModelService] release called (no-op, resources released after each call)")

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt_parts = [self._system_prompt]

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.insert(0, content)
            elif role == "user":
                prompt_parts.append(f"用户：{content}")
            elif role == "assistant":
                prompt_parts.append(f"助手：{content}")

        prompt_parts.append("助手：")
        return "\n".join(prompt_parts)
