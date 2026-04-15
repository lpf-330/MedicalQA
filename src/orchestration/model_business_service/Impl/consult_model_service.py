# -*- coding: utf-8 -*-
"""
健康咨询模型业务服务

封装健康咨询业务场景下的模型服务，实现ModelBusinessService接口。
"""

from typing import Any, Dict, List, Optional

from src.orchestration.model_business_service.model_business_service import ModelBusinessService
from src.resource_manager.vllm_model import VLLMModelClient, VLLMModelResource, VLLMModelConfig


class ConsultModelService(ModelBusinessService[List[Dict[str, str]], str]):
    """
    健康咨询模型业务服务类
    
    实现ModelBusinessService接口，为健康咨询业务场景提供模型服务。
    
    属性：
        _model_client: VLLM模型客户端
        _model_resource: VLLM模型资源
        _system_prompt: 系统提示词
    """
    
    def __init__(
        self,
        model_path: str,
        system_prompt: str = "你是一个专业的健康咨询助手，请根据用户的描述提供专业的健康建议。"
    ):
        """
        初始化健康咨询模型业务服务
        
        Args:
            model_path: 模型路径
            system_prompt: 系统提示词
        """
        self._model_path = model_path
        self._system_prompt = system_prompt
        self._model_client: Optional[VLLMModelClient] = None
        self._model_resource: Optional[VLLMModelResource] = None
    
    def _init_model(self) -> None:
        """初始化模型"""
        if self._model_resource is not None:
            return
        
        config = VLLMModelConfig(model_path=self._model_path)
        self._model_resource = VLLMModelResource(config)
        self._model_resource.activate()
        self._model_client = VLLMModelClient(self._model_resource)
    
    def call_model(self, messages: List[Dict[str, str]]) -> str:
        """
        使用模型服务
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "..."}]
            
        Returns:
            模型生成的回复
        """
        if self._model_client is None:
            raise RuntimeError("Model not initialized, call _init_model first")
        
        prompt = self._build_prompt(messages)
        return self._model_client.generate(prompt)
    
    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        构建模型输入提示
        
        Args:
            messages: 消息列表
            
        Returns:
            构建好的提示字符串
        """
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
        """
        带知识上下文的生成
        
        Args:
            user_query: 用户查询
            knowledge_context: 知识上下文
            
        Returns:
            模型生成的回复
        """
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "system", "content": f"参考知识：\n{knowledge_context}"},
            {"role": "user", "content": user_query}
        ]
        return self.call_model(messages)
    
    def release(self) -> None:
        """释放模型资源"""
        if self._model_resource is not None:
            self._model_resource.destroy()
            self._model_resource = None
            self._model_client = None
