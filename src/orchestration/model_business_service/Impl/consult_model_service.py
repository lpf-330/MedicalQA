# -*- coding: utf-8 -*-

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
        self._model_handle: Optional[ResourceHandle] = None
        self._model_client: Optional[VLLMModelClient] = None

    def _init_model(self) -> None:
        if self._model_handle is not None:
            logger.debug("[ConsultModelService] _init_model skipped, already initialized")
            return

        logger.info("[ConsultModelService] _init_model started")
        start_time = time.time()
        try:
            self._model_handle = GlobalResourceManager.acquire("vllm_model")
            if self._model_handle is None:
                raise RuntimeError("Failed to acquire vllm_model resource")

            self._model_client = VLLMModelClient(self._model_handle.resource)
            elapsed = time.time() - start_time
            logger.info(f"[ConsultModelService] _init_model completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[ConsultModelService] _init_model failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def call_model(self, messages: List[Dict[str, str]]) -> str:
        logger.debug(f"[ConsultModelService] call_model called, message_count={len(messages)}")
        logger.debug(f"[ConsultModelService] LLM输入 - messages: {messages}")
        start_time = time.time()
        try:
            if self._model_client is None:
                raise RuntimeError("Model not initialized, call _init_model first")

            prompt = self._build_prompt(messages)
            logger.debug(f"[ConsultModelService] LLM输入 - 构建的prompt:\n{prompt[:2000]}{'...' if len(prompt) > 2000 else ''}")
            result = self._model_client.generate(prompt)
            elapsed = time.time() - start_time
            logger.info(f"[ConsultModelService] call_model completed, elapsed={elapsed:.3f}s, response_length={len(result)}")
            logger.debug(f"[ConsultModelService] LLM输出 - 内容:\n{result[:2000]}{'...' if len(result) > 2000 else ''}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[ConsultModelService] call_model failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

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
        logger.info(f"[ConsultModelService] stream_generate_with_context called, query_length={len(user_query)}, context_length={len(knowledge_context)}")
        start_time = time.time()
        try:
            if self._model_client is None:
                raise RuntimeError("Model not initialized, call _init_model first")

            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "system", "content": f"参考知识：\n{knowledge_context}"},
                {"role": "user", "content": user_query}
            ]
            prompt = self._build_prompt(messages)
            
            # 记录完整的LLM输入
            logger.info(f"[ConsultModelService] ========== LLM完整输入 ==========")
            logger.info(f"[ConsultModelService] System Prompt: {self._system_prompt}")
            logger.info(f"[ConsultModelService] Knowledge Context (长度={len(knowledge_context)}):")
            logger.info(f"{knowledge_context[:3000]}{'...' if len(knowledge_context) > 3000 else ''}")
            logger.info(f"[ConsultModelService] User Query: {user_query}")
            logger.info(f"[ConsultModelService] 构建的完整Prompt (长度={len(prompt)}):")
            logger.info(f"{prompt}")
            logger.info(f"[ConsultModelService] ==============================")
            
            result = self._model_client.stream_generate(prompt)
            elapsed = time.time() - start_time
            logger.info(f"[ConsultModelService] stream_generate_with_context initiated, elapsed={elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[ConsultModelService] stream_generate_with_context failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def release(self) -> None:
        logger.info("[ConsultModelService] release started")
        start_time = time.time()
        try:
            if self._model_handle is not None:
                GlobalResourceManager.release(self._model_handle)
                self._model_handle = None
                self._model_client = None
            elapsed = time.time() - start_time
            logger.info(f"[ConsultModelService] release completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[ConsultModelService] release failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise
