# -*- coding: utf-8 -*-
"""
VLLM模型资源封装

封装VLLM模型推理资源，提供统一的资源管理接口。
"""

import time
from typing import Any, Dict, Iterator, List, Optional

from src.resource_manager.resource import Resource
from src.resource_manager.resource_config import ResourceConfig
from src.resource_manager.resource_factory import ResourceFactory
from src.resource_manager.resource_client import ResourceClient
from src.adapters import VLLMAdapterImpl


class VLLMModelResource(Resource):
    """
    VLLM模型资源类
    
    封装VLLM模型推理资源，实现Resource接口。
    
    属性：
        _config: VLLM模型配置
        _adapter: VLLM适配器实例
        _last_used_time: 最后使用时间戳
        _is_active: 资源活跃状态
    """
    
    def __init__(self, config: 'VLLMModelConfig'):
        """
        初始化VLLM模型资源
        
        Args:
            config: VLLM模型配置
        """
        self._config = config
        self._adapter: Optional[VLLMAdapterImpl] = None
        self._last_used_time = int(time.time() * 1000)
        self._is_active = False
    
    def get_type(self) -> str:
        """获取资源类型标识"""
        return "vllm_model"
    
    def get_last_used_time(self) -> int:
        """获取最后使用时间戳"""
        return self._last_used_time
    
    def is_activate(self) -> bool:
        """校验资源活跃状态"""
        return self._is_active
    
    def activate(self) -> None:
        """激活资源"""
        if self._is_active:
            return
        
        config_protocol = self._config.config_protocol
        self._adapter = VLLMAdapterImpl()
        self._adapter.load_model(
            model_path=config_protocol["model_path"],
            tensor_parallel_size=config_protocol.get("tensor_parallel_size", 1),
            max_model_len=config_protocol.get("max_model_len", 8192),
            gpu_memory_utilization=config_protocol.get("gpu_memory_utilization", 0.9)
        )
        self._is_active = True
        self._last_used_time = int(time.time() * 1000)
    
    def deactivate(self) -> None:
        """
        停用资源（释放回池，保持连接）
        
        语义：资源从活跃状态变为空闲状态，归还到资源池
        行为：仅标记状态，不断开连接
        场景：资源使用完毕，释放回资源池复用
        """
        if not self._is_active:
            return
        
        self._is_active = False
    
    def destroy(self) -> None:
        """
        销毁资源（彻底释放）
        
        语义：资源彻底销毁，从资源池移除
        行为：断开连接，释放所有资源
        场景：资源池关闭、资源过期、资源异常需销毁
        """
        if self._adapter is not None:
            self._adapter.unload_model()
        self._adapter = None
        self._is_active = False
    
    def get_adapter(self) -> Optional[VLLMAdapterImpl]:
        """
        获取VLLM适配器实例
        
        Returns:
            VLLMAdapterImpl: VLLM适配器实例
        """
        return self._adapter


class VLLMModelConfig(ResourceConfig[Dict[str, Any]]):
    """
    VLLM模型配置类
    
    实现ResourceConfig接口，存储VLLM模型配置。
    
    属性：
        _resource_type: 资源类型标识
        _resource_name: 资源业务名称
        _config_protocol: 个性化配置协议
    """
    
    def __init__(
        self,
        model_path: str,
        model_name: str = "Qwen3-4B-Instruct-2507",
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.9,
        resource_name: str = "VLLM模型推理资源"
    ):
        """
        初始化VLLM模型配置
        
        Args:
            model_path: 模型路径
            model_name: 模型名称
            tensor_parallel_size: 张量并行大小
            max_model_len: 最大模型长度
            gpu_memory_utilization: GPU内存利用率
            resource_name: 资源业务名称
        """
        self._resource_type = "vllm_model"
        self._resource_name = resource_name
        self._config_protocol: Dict[str, Any] = {
            "model_path": model_path,
            "model_name": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization
        }
    
    @property
    def resource_type(self) -> str:
        """获取资源类型标识"""
        return self._resource_type
    
    @property
    def resource_name(self) -> str:
        """获取资源业务名称"""
        return self._resource_name
    
    @property
    def config_protocol(self) -> Dict[str, Any]:
        """获取个性化配置协议"""
        return self._config_protocol
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "resource_type": self._resource_type,
            "resource_name": self._resource_name,
            "config_protocol": self._config_protocol
        }
    
    def validate(self) -> bool:
        """验证配置有效性"""
        if not self._config_protocol.get("model_path"):
            return False
        if self._config_protocol.get("tensor_parallel_size", 1) < 1:
            return False
        gpu_util = self._config_protocol.get("gpu_memory_utilization", 0.9)
        if gpu_util <= 0 or gpu_util > 1:
            return False
        return True


class VLLMModelFactory(ResourceFactory):
    """
    VLLM模型工厂类
    
    实现ResourceFactory接口，负责VLLM模型资源的创建和销毁。
    """
    
    def create(self, config: ResourceConfig) -> Resource:
        """
        创建VLLM模型资源
        
        Args:
            config: 资源配置
            
        Returns:
            Resource: VLLM模型资源实例
        """
        if not isinstance(config, VLLMModelConfig):
            raise TypeError(f"Expected VLLMModelConfig, got {type(config)}")
        
        return VLLMModelResource(config)
    
    def destroy(self, resource: Resource) -> None:
        """
        销毁VLLM模型资源
        
        Args:
            resource: 要销毁的资源实例
        """
        if not isinstance(resource, VLLMModelResource):
            raise TypeError(f"Expected VLLMModelResource, got {type(resource)}")
        
        resource.destroy()


class VLLMModelClient(ResourceClient):
    """
    VLLM模型客户端类
    
    实现ResourceClient接口，为业务层提供统一的模型推理操作接口。
    
    属性：
        _resource: 封装的VLLM模型资源
    """
    
    def __init__(self, resource: VLLMModelResource):
        """
        初始化VLLM模型客户端
        
        Args:
            resource: VLLM模型资源实例
        """
        self._resource = resource
    
    def get_resource_type(self) -> str:
        """获取资源类型标识"""
        return self._resource.get_type()
    
    def get_raw_resource(self) -> Resource:
        """获取原始资源实例"""
        return self._resource
    
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
        adapter = self._resource.get_adapter()
        if adapter is None:
            raise RuntimeError("VLLM adapter not initialized")
        return adapter.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
    
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
        adapter = self._resource.get_adapter()
        if adapter is None:
            raise RuntimeError("VLLM adapter not initialized")
        return adapter.generate_batch(
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
    
    def stream_generate(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Iterator[str]:
        """
        流式生成文本
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top_p参数
            **kwargs: 其他生成参数
            
        Yields:
            生成的文本片段
        """
        adapter = self._resource.get_adapter()
        if adapter is None:
            raise RuntimeError("VLLM adapter not initialized")
        return adapter.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
    
    def is_model_loaded(self) -> bool:
        """
        检查模型是否已加载
        
        Returns:
            bool: 模型是否已加载
        """
        adapter = self._resource.get_adapter()
        if adapter is None:
            return False
        return adapter.is_model_loaded()
