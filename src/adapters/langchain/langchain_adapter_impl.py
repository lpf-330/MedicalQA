# -*- coding: utf-8 -*-
"""
Langchain适配器实现类

转接适配Langchain框架，为项目各层级提供统一的框架操作接口。

注意：Langchain 1.x版本中，部分组件已移至langchain_community包。
本适配器使用延迟导入，在需要时才导入相关组件。
"""

from typing import Any, Dict, Optional

from .langchain_adapter import (
    LangchainAdapter, 
    InternalChain, 
    InternalTool, 
    InternalMemory, 
    InternalAgent
)


class InternalChainImpl(InternalChain):
    """内部链实现类"""
    
    def __init__(self, chain):
        self._chain = chain
    
    def run(self, **kwargs) -> str:
        return self._chain.run(**kwargs)
    
    async def arun(self, **kwargs) -> str:
        return await self._chain.arun(**kwargs)


class InternalToolImpl(InternalTool):
    """内部工具实现类"""
    
    def __init__(self, tool):
        self._tool = tool
    
    def run(self, **kwargs) -> str:
        return self._tool.run(**kwargs)
    
    @property
    def name(self) -> str:
        return self._tool.name
    
    @property
    def description(self) -> str:
        return self._tool.description


class InternalMemoryImpl(InternalMemory):
    """内部记忆实现类"""
    
    def __init__(self, memory):
        self._memory = memory
    
    def save_context(self, inputs: Dict, outputs: Dict) -> None:
        self._memory.save_context(inputs, outputs)
    
    def load_memory_variables(self, inputs: Dict) -> Dict:
        return self._memory.load_memory_variables(inputs)
    
    def clear(self) -> None:
        self._memory.clear()


class InternalAgentImpl(InternalAgent):
    """内部Agent实现类"""
    
    def __init__(self, agent_executor):
        self._agent_executor = agent_executor
    
    def run(self, input_text: str) -> str:
        return self._agent_executor.run(input_text)
    
    async def arun(self, input_text: str) -> str:
        return await self._agent_executor.arun(input_text)


class LangchainAdapterImpl(LangchainAdapter):
    """
    Langchain适配器实现类
    
    封装langchain库，为项目提供统一的框架操作接口。
    使用延迟导入，在需要时才导入langchain相关组件。
    
    属性：
        _llm: 语言模型实例
    """
    
    def __init__(self, llm=None):
        """
        初始化Langchain适配器
        
        Args:
            llm: 语言模型实例（可选）
        """
        self._llm = llm
    
    def set_llm(self, llm) -> None:
        """
        设置语言模型
        
        Args:
            llm: 语言模型实例
        """
        self._llm = llm
    
    def create_chain(
        self, 
        chain_type: str, 
        config: Dict[str, Any]
    ) -> InternalChain:
        """
        创建链式调用
        
        Args:
            chain_type: 链类型
            config: 链配置
            
        Returns:
            内部链对象
        """
        if chain_type == "llm_chain":
            try:
                from langchain.chains import LLMChain
                from langchain_core.prompts import PromptTemplate
            except ImportError:
                try:
                    from langchain_community.chains import LLMChain
                    from langchain_core.prompts import PromptTemplate
                except ImportError:
                    raise ImportError(
                        "Langchain components not found. "
                        "Please install langchain-community: pip install langchain-community"
                    )
            
            prompt = PromptTemplate(
                template=config.get("prompt_template", "{input}"),
                input_variables=config.get("input_variables", ["input"])
            )
            
            chain = LLMChain(
                llm=self._llm,
                prompt=prompt
            )
            return InternalChainImpl(chain)
        
        raise ValueError(f"Unsupported chain type: {chain_type}")
    
    def create_tool(
        self, 
        tool_type: str, 
        config: Dict[str, Any]
    ) -> InternalTool:
        """
        创建工具
        
        Args:
            tool_type: 工具类型
            config: 工具配置
            
        Returns:
            内部工具对象
        """
        try:
            from langchain.tools import Tool
        except ImportError:
            try:
                from langchain_community.tools import Tool
            except ImportError:
                raise ImportError(
                    "Langchain Tool not found. "
                    "Please install langchain-community: pip install langchain-community"
                )
        
        tool = Tool(
            name=config.get("name", tool_type),
            description=config.get("description", ""),
            func=config.get("func")
        )
        return InternalToolImpl(tool)
    
    def create_memory(
        self, 
        memory_type: str, 
        config: Dict[str, Any]
    ) -> InternalMemory:
        """
        创建记忆管理
        
        Args:
            memory_type: 记忆类型
            config: 记忆配置
            
        Returns:
            内部记忆对象
        """
        if memory_type == "buffer":
            try:
                from langchain.memory import ConversationBufferMemory
            except ImportError:
                try:
                    from langchain_community.chat_message_histories import ConversationBufferMemory
                except ImportError:
                    raise ImportError(
                        "Langchain ConversationBufferMemory not found. "
                        "Please install langchain-community: pip install langchain-community"
                    )
            
            memory = ConversationBufferMemory(
                memory_key=config.get("memory_key", "history"),
                return_messages=config.get("return_messages", False)
            )
            return InternalMemoryImpl(memory)
        
        raise ValueError(f"Unsupported memory type: {memory_type}")
    
    def create_agent(
        self, 
        agent_type: str, 
        config: Dict[str, Any]
    ) -> InternalAgent:
        """
        创建Agent
        
        Args:
            agent_type: Agent类型
            config: Agent配置
            
        Returns:
            内部Agent对象
        """
        try:
            from langchain.agents import AgentExecutor, initialize_agent
        except ImportError:
            try:
                from langchain_community.agents import AgentExecutor, initialize_agent
            except ImportError:
                raise ImportError(
                    "Langchain Agent components not found. "
                    "Please install langchain-community: pip install langchain-community"
                )
        
        tools = config.get("tools", [])
        
        agent_executor = initialize_agent(
            tools=tools,
            llm=self._llm,
            agent=agent_type,
            verbose=config.get("verbose", False)
        )
        return InternalAgentImpl(agent_executor)
