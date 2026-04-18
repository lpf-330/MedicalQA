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
from typing import Any, Dict, Iterator, List, Optional

from src.orchestration.model_business_service.model_business_service import ModelBusinessService
from src.resource_manager.global_resource_manager import GlobalResourceManager
from src.resource_manager.resource_handle import ResourceHandle
from src.resource_manager.vllm_model import VLLMModelClient, VLLMModelResource

logger = logging.getLogger(__name__)


class ConsultModelService(ModelBusinessService[List[Dict[str, str]], str]):

    def __init__(
        self,
        model_path: str = "",
        system_prompt: str = "你是一个专业的健康咨询助手，请根据用户的描述提供专业的健康建议。"
    ):
        self._model_path = model_path
        self._system_prompt = system_prompt

    def call_model(self, messages: List[Dict[str, str]]) -> str:
        """
        调用模型 - 在需要时获取资源，处理完成后立即释放
        """
        logger.debug(f"[ConsultModelService] call_model called, message_count={len(messages)}")
        logger.debug(f"[ConsultModelService] LLM输入 - messages: {messages}")
        start_time = time.time()
        
        handle = None
        try:
            handle = GlobalResourceManager.acquire("vllm_model", "vllm_config")
            if handle is None:
                raise RuntimeError("Failed to acquire vllm_model resource")
            
            model_client = VLLMModelClient(handle.resource)
            
            prompt = self._build_prompt(messages)
            logger.debug(f"[ConsultModelService] LLM输入 - 构建的prompt:\n{prompt[:2000]}{'...' if len(prompt) > 2000 else ''}")
            result = model_client.generate(prompt)
            elapsed = time.time() - start_time
            logger.info(f"[ConsultModelService] call_model completed, elapsed={elapsed:.3f}s, response_length={len(result)}")
            logger.debug(f"[ConsultModelService] LLM输出 - 内容:\n{result[:2000]}{'...' if len(result) > 2000 else ''}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[ConsultModelService] call_model failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise
        finally:
            if handle is not None:
                GlobalResourceManager.release(handle)

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

    def generate_with_context(
        self,
        user_query: str,
        knowledge_context: str
    ) -> str:
        logger.debug(f"[ConsultModelService] generate_with_context called, query_length={len(user_query)}, context_length={len(knowledge_context)}")
        start_time = time.time()
        try:
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "system", "content": f"参考知识：\n{knowledge_context}"},
                {"role": "user", "content": user_query}
            ]
            result = self.call_model(messages)
            elapsed = time.time() - start_time
            logger.info(f"[ConsultModelService] generate_with_context completed, elapsed={elapsed:.3f}s, response_length={len(result)}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[ConsultModelService] generate_with_context failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def stream_generate_with_context(self, user_query: str, knowledge_context: str) -> Iterator[str]:
        """
        流式生成 - 在需要时获取资源，流式输出完成后释放
        """
        logger.info(f"[ConsultModelService] stream_generate_with_context called, query_length={len(user_query)}, context_length={len(knowledge_context)}")
        start_time = time.time()
        
        handle = None
        try:
            handle = GlobalResourceManager.acquire("vllm_model", "vllm_config")
            if handle is None:
                raise RuntimeError("Failed to acquire vllm_model resource")
            
            model_client = VLLMModelClient(handle.resource)
            
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "system", "content": f"参考知识：\n{knowledge_context}"},
                {"role": "user", "content": user_query}
            ]
            prompt = self._build_prompt(messages)
            
            logger.info(f"[ConsultModelService] ========== LLM完整输入 ==========")
            logger.info(f"[ConsultModelService] System Prompt: {self._system_prompt}")
            logger.info(f"[ConsultModelService] Knowledge Context (长度={len(knowledge_context)}):")
            logger.info(f"{knowledge_context[:3000]}{'...' if len(knowledge_context) > 3000 else ''}")
            logger.info(f"[ConsultModelService] User Query: {user_query}")
            logger.info(f"[ConsultModelService] 构建的完整Prompt (长度={len(prompt)}):")
            logger.info(f"{prompt}")
            logger.info(f"[ConsultModelService] ==============================")
            
            for chunk in model_client.stream_generate(prompt):
                yield chunk
            
            elapsed = time.time() - start_time
            logger.info(f"[ConsultModelService] stream_generate_with_context completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[ConsultModelService] stream_generate_with_context failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise
        finally:
            if handle is not None:
                GlobalResourceManager.release(handle)

    def release(self) -> None:
        """
        释放资源 - 由于资源在每次调用后已释放，此方法保持兼容性但无需操作
        """
        logger.info("[ConsultModelService] release called (no-op, resources released after each call)")
