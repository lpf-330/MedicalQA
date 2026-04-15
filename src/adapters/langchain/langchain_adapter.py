# -*- coding: utf-8 -*-
"""
Langchain适配器接口

为项目各层级、各类提供统一的Langchain框架操作接口。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class LangchainAdapter(ABC):
    """
    Langchain适配器接口
    
    定义Langchain框架操作的标准接口，为项目各层级提供统一的访问方式。
    
    使用示例：
        adapter = LangchainAdapterImpl()
        chain = adapter.create_chain("llm_chain", config)
        result = chain.run(question="什么是糖尿病？")
    """
    
    @abstractmethod
    def create_chain(
        self, 
        chain_type: str, 
        config: Dict[str, Any]
    ) -> 'InternalChain':
        """
        创建链式调用
        
        Args:
            chain_type: 链类型（如"llm_chain", "sequential_chain"等）
            config: 链配置
            
        Returns:
            内部链对象
        """
        pass
    
    @abstractmethod
    def create_tool(
        self, 
        tool_type: str, 
        config: Dict[str, Any]
    ) -> 'InternalTool':
        """
        创建工具
        
        Args:
            tool_type: 工具类型
            config: 工具配置
            
        Returns:
            内部工具对象
        """
        pass
    
    @abstractmethod
    def create_memory(
        self, 
        memory_type: str, 
        config: Dict[str, Any]
    ) -> 'InternalMemory':
        """
        创建记忆管理
        
        Args:
            memory_type: 记忆类型（如"buffer", "summary"等）
            config: 记忆配置
            
        Returns:
            内部记忆对象
        """
        pass
    
    @abstractmethod
    def create_agent(
        self, 
        agent_type: str, 
        config: Dict[str, Any]
    ) -> 'InternalAgent':
        """
        创建Agent
        
        Args:
            agent_type: Agent类型
            config: Agent配置（包含tools、llm等）
            
        Returns:
            内部Agent对象
        """
        pass


class InternalChain(ABC):
    """
    内部链接口
    
    封装langchain的链式调用，为项目提供统一的链操作接口。
    """
    
    @abstractmethod
    def run(self, **kwargs) -> str:
        """
        执行链式调用
        
        Args:
            **kwargs: 链的输入参数
            
        Returns:
            执行结果
        """
        pass
    
    @abstractmethod
    async def arun(self, **kwargs) -> str:
        """
        异步执行链式调用
        
        Args:
            **kwargs: 链的输入参数
            
        Returns:
            执行结果
        """
        pass


class InternalTool(ABC):
    """
    内部工具接口
    
    封装langchain的工具，为项目提供统一的工具操作接口。
    """
    
    @abstractmethod
    def run(self, **kwargs) -> str:
        """
        执行工具
        
        Args:
            **kwargs: 工具的输入参数
            
        Returns:
            执行结果
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""
        pass


class InternalMemory(ABC):
    """
    内部记忆接口
    
    封装langchain的记忆管理，为项目提供统一的记忆操作接口。
    """
    
    @abstractmethod
    def save_context(self, inputs: Dict, outputs: Dict) -> None:
        """
        保存上下文
        
        Args:
            inputs: 输入
            outputs: 输出
        """
        pass
    
    @abstractmethod
    def load_memory_variables(self, inputs: Dict) -> Dict:
        """
        加载记忆变量
        
        Args:
            inputs: 输入
            
        Returns:
            记忆变量
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空记忆"""
        pass


class InternalAgent(ABC):
    """
    内部Agent接口
    
    封装langchain的Agent，为项目提供统一的Agent操作接口。
    """
    
    @abstractmethod
    def run(self, input_text: str) -> str:
        """
        执行Agent
        
        Args:
            input_text: 输入文本
            
        Returns:
            执行结果
        """
        pass
    
    @abstractmethod
    async def arun(self, input_text: str) -> str:
        """
        异步执行Agent
        
        Args:
            input_text: 输入文本
            
        Returns:
            执行结果
        """
        pass
