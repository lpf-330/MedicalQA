# -*- coding: utf-8 -*-
import logging
import time
import asyncio
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

from .vllm_adapter import VLLMAdapter

logger = logging.getLogger(__name__)


class VLLMAdapterImpl(VLLMAdapter):
    
    def __init__(self):
        self._async_llm: Optional[AsyncLLM] = None
        self._model_path: Optional[str] = None
        self._is_loaded: bool = False
        logger.debug("[VLLMAdapter] 初始化VLLM适配器")
    
    def load_model(
        self, 
        model_path: str, 
        tensor_parallel_size: int = 1,
        **kwargs
    ) -> None:
        if self._is_loaded:
            logger.debug(f"[VLLMAdapter] 模型已加载，跳过: model_path={model_path}")
            return
        
        logger.info(f"[VLLMAdapter] 开始加载模型: model_path={model_path}, tensor_parallel_size={tensor_parallel_size}, kwargs={kwargs}")
        start_time = time.time()
        
        try:
            async_engine_args = AsyncEngineArgs(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                enforce_eager=True,
                max_model_len=kwargs.get('max_model_len', 8192),
                gpu_memory_utilization=kwargs.get('gpu_memory_utilization', 0.9),
                disable_log_stats=True,
            )
            self._async_llm = AsyncLLM.from_engine_args(async_engine_args)
            self._model_path = model_path
            self._is_loaded = True
            
            elapsed = time.time() - start_time
            logger.info(f"[VLLMAdapter] 模型加载完成: model_path={model_path}, elapsed={elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VLLMAdapter] 模型加载失败: model_path={model_path}, elapsed={elapsed:.2f}s")
            logger.error(f"[VLLMAdapter] 错误类型: {type(e).__name__}")
            logger.error(f"[VLLMAdapter] 错误信息: {str(e)}")
            raise
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        if not self._is_loaded or self._async_llm is None:
            raise RuntimeError("Model not loaded")
        
        logger.debug(f"[VLLMAdapter] LLM输入 - prompt长度: {len(prompt)}字符")
        
        start_time = time.time()
        
        full_text = ""
        loop = asyncio.new_event_loop()
        try:
            async def _collect():
                nonlocal full_text
                import uuid
                sampling_params = SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    output_kind=RequestOutputKind.CUMULATIVE,
                    **kwargs
                )
                request_id = f"sync-gen-{uuid.uuid4().hex[:8]}"
                async for output in self._async_llm.generate(
                    request_id=request_id,
                    prompt=prompt,
                    sampling_params=sampling_params
                ):
                    for completion in output.outputs:
                        full_text = completion.text
                    if output.finished:
                        break
            loop.run_until_complete(_collect())
        finally:
            loop.close()
        
        elapsed = time.time() - start_time
        logger.info(f"[VLLMAdapter] 生成完成: output_len={len(full_text)}, elapsed={elapsed:.2f}s")
        return full_text
    
    def generate_batch(
        self, 
        prompts: List[str], 
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> List[str]:
        if not self._is_loaded or self._async_llm is None:
            raise RuntimeError("Model not loaded")
        
        logger.debug(f"[VLLMAdapter] 开始批量生成: batch_size={len(prompts)}, max_tokens={max_tokens}")
        
        results = []
        for prompt in prompts:
            result = self.generate(prompt, max_tokens, temperature, top_p, **kwargs)
            results.append(result)
        
        return results
    
    def stream_generate(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Iterator[str]:
        if not self._is_loaded or self._async_llm is None:
            raise RuntimeError("Model not loaded")
        
        logger.debug(f"[VLLMAdapter] LLM流式输入 - prompt长度: {len(prompt)}字符")
        
        full_text = self.generate(prompt, max_tokens, temperature, top_p, **kwargs)
        for char in full_text:
            yield char
    
    async def async_stream_generate(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> AsyncIterator[str]:
        if self._async_llm is None:
            raise RuntimeError("AsyncLLM not loaded")
        
        logger.debug(f"[VLLMAdapter] AsyncLLM流式输入 - prompt长度: {len(prompt)}字符")
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            output_kind=RequestOutputKind.DELTA,
            **kwargs
        )
        
        import uuid
        request_id = f"async-stream-{uuid.uuid4().hex[:8]}"
        
        async for output in self._async_llm.generate(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params
        ):
            for completion in output.outputs:
                new_text = completion.text
                if new_text:
                    yield new_text
            
            if output.finished:
                break
    
    def unload_model(self) -> None:
        if self._async_llm is not None:
            logger.info(f"[VLLMAdapter] 开始卸载模型: model_path={self._model_path}")
            self._async_llm.shutdown()
            self._async_llm = None
            logger.info("[VLLMAdapter] 模型卸载完成")
        self._model_path = None
        self._is_loaded = False
    
    def is_model_loaded(self) -> bool:
        return self._is_loaded and self._async_llm is not None
    
    def __enter__(self) -> 'VLLMAdapterImpl':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unload_model()
