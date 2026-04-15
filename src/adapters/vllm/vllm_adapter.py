# -*- coding: utf-8 -*-
"""
VLLM适配器接口

为项目各层级、各类提供统一的VLLM模型推理引擎操作接口。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List


class VLLMAdapter(ABC):
    """
    VLLM适配器接口
    
    定义VLLM模型推理引擎操作的标准接口，为项目各层级提供统一的访问方式。
    
    使用示例：
        adapter = VLLMAdapterImpl()
        adapter.load_model(model_path="/path/to/model")
        result = adapter.generate(prompt="你好")
        adapter.unload_model()
    """
    
    @abstractmethod
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
            
        Raises:
            ModelLoadError: 当模型加载失败时抛出
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def generate_batch(
        self, 
        prompts: List[str], 
        **kwargs
    ) -> List[str]:
        """
        批量生成文本
        
        Args:
            prompts: 输入提示列表
            **kwargs: 其他生成参数
            
        Returns:
            生成的文本列表
        """
        pass
    
    @abstractmethod
    def stream_generate(
        self, 
        prompt: str, 
        **kwargs
    ) -> Iterator[str]:
        """
        流式生成文本（用于SSE）
        
        Args:
            prompt: 输入提示
            **kwargs: 其他生成参数
            
        Yields:
            生成的文本片段
        """
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """
        卸载模型，释放资源
        """
        pass
    
    @abstractmethod
    def is_model_loaded(self) -> bool:
        """
        检查模型是否已加载
        
        Returns:
            bool: 模型是否已加载
        """
        pass
