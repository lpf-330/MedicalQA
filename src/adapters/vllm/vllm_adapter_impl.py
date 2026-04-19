# -*- coding: utf-8 -*-
"""
VLLM适配器实现类

转接适配VLLM引擎，为项目各层级提供统一的模型推理操作接口。
"""

import logging
import time
from typing import Any, Dict, Iterator, List, Optional

from vllm import LLM, SamplingParams

from .vllm_adapter import VLLMAdapter

logger = logging.getLogger(__name__)


class VLLMAdapterImpl(VLLMAdapter):
    """
    VLLM适配器实现类
    
    封装vllm库，为项目提供统一的模型推理操作接口。
    
    属性：
        _llm: VLLM LLM实例
        _model_path: 模型路径
        _is_loaded: 模型是否已加载
    """
    
    def __init__(self):
        """初始化VLLM适配器"""
        self._llm: Optional[LLM] = None
        self._model_path: Optional[str] = None
        self._is_loaded: bool = False
        logger.debug("[VLLMAdapter] 初始化VLLM适配器")
    
    def load_model(
        self, 
        model_path: str, 
        tensor_parallel_size: int = 1,
        **kwargs
    ) -> None:
        """
        加载模型
        
        Args:
            model_path: 模型路径
            tensor_parallel_size: 张量并行大小
            **kwargs: 其他VLLM参数
        """
        if self._is_loaded:
            logger.debug(f"[VLLMAdapter] 模型已加载，跳过: model_path={model_path}")
            return
        
        logger.info(f"[VLLMAdapter] 开始加载模型: model_path={model_path}, tensor_parallel_size={tensor_parallel_size}, kwargs={kwargs}")
        start_time = time.time()
        
        try:
            self._llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                enforce_eager=True,
                **kwargs
            )
            self._model_path = model_path
            self._is_loaded = True
            
            elapsed = time.time() - start_time
            logger.info(f"[VLLMAdapter] 模型加载完成: model_path={model_path}, elapsed={elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VLLMAdapter] 模型加载失败: model_path={model_path}, elapsed={elapsed:.2f}s")
            logger.error(f"[VLLMAdapter] 错误类型: {type(e).__name__}")
            logger.error(f"[VLLMAdapter] 错误信息: {str(e)}")
            logger.error(f"[VLLMAdapter] 配置参数: tensor_parallel_size={tensor_parallel_size}, kwargs={kwargs}")
            logger.error(f"[VLLMAdapter] GPU内存利用率: {kwargs.get('gpu_memory_utilization', '未设置')}")
            logger.error(f"[VLLMAdapter] 最大模型长度: {kwargs.get('max_model_len', '未设置')}")
            raise
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top_p参数
            **kwargs: 其他生成参数
            
        Returns:
            生成的文本
        """
        if not self._is_loaded or self._llm is None:
            logger.error("[VLLMAdapter] 生成失败，模型未加载")
            raise RuntimeError("Model not loaded")
        
        logger.debug(f"[VLLMAdapter] LLM输入 - prompt长度: {len(prompt)}字符")
        logger.debug(f"[VLLMAdapter] LLM输入 - prompt内容:\n{prompt[:2000]}{'...' if len(prompt) > 2000 else ''}")
        logger.debug(f"[VLLMAdapter] LLM参数 - max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}, kwargs={kwargs}")
        
        start_time = time.time()
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        outputs = self._llm.generate([prompt], sampling_params)
        result = outputs[0].outputs[0].text
        
        elapsed = time.time() - start_time
        logger.info(f"[VLLMAdapter] 生成完成: output_len={len(result)}, elapsed={elapsed:.2f}s")
        logger.debug(f"[VLLMAdapter] LLM输出 - 内容:\n{result[:2000]}{'...' if len(result) > 2000 else ''}")
        return result
    
    def generate_batch(
        self, 
        prompts: List[str], 
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> List[str]:
        """
        批量生成文本
        
        Args:
            prompts: 输入提示列表
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top_p参数
            **kwargs: 其他生成参数
            
        Returns:
            生成的文本列表
        """
        if not self._is_loaded or self._llm is None:
            logger.error("[VLLMAdapter] 批量生成失败，模型未加载")
            raise RuntimeError("Model not loaded")
        
        logger.debug(f"[VLLMAdapter] 开始批量生成: batch_size={len(prompts)}, max_tokens={max_tokens}")
        start_time = time.time()
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        outputs = self._llm.generate(prompts, sampling_params)
        results = [output.outputs[0].text for output in outputs]
        
        elapsed = time.time() - start_time
        logger.info(f"[VLLMAdapter] 批量生成完成: batch_size={len(results)}, elapsed={elapsed:.2f}s")
        return results
    
    def stream_generate(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Iterator[str]:
        if not self._is_loaded or self._llm is None:
            raise RuntimeError("Model not loaded")
        
        logger.debug(f"[VLLMAdapter] LLM流式输入 - prompt长度: {len(prompt)}字符")
        logger.debug(f"[VLLMAdapter] LLM流式输入 - prompt内容:\n{prompt[:2000]}{'...' if len(prompt) > 2000 else ''}")
        logger.debug(f"[VLLMAdapter] LLM流式参数 - max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}, kwargs={kwargs}")
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        outputs = self._llm.generate([prompt], sampling_params, use_tqdm=False)
        if outputs:
            generated_text = outputs[0].outputs[0].text
            logger.debug(f"[VLLMAdapter] LLM流式输出 - 内容长度: {len(generated_text)}")
            
            for char in generated_text:
                yield char
    
    def unload_model(self) -> None:
        """卸载模型，释放资源"""
        if self._llm is not None:
            logger.info(f"[VLLMAdapter] 开始卸载模型: model_path={self._model_path}")
            del self._llm
            self._llm = None
            logger.info("[VLLMAdapter] 模型卸载完成")
        self._model_path = None
        self._is_loaded = False
    
    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._is_loaded
    
    def __enter__(self) -> 'VLLMAdapterImpl':
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器退出"""
        self.unload_model()
