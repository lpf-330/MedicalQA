# -*- coding: utf-8 -*-
"""
Langchain适配器实现类

转接适配Langchain框架，为项目各层级提供统一的框架操作接口。

注意：Langchain 1.x版本中，部分组件已移至langchain_core包。
本适配器使用langchain_core和langgraph组件。
"""

import logging
from typing import Any, Dict, List

from .langchain_adapter import (
    LangchainAdapter,
    InternalChain,
    InternalTool,
    InternalMemory,
    InternalAgent
)
from src.utils.logger import log_arch_event

logger = logging.getLogger(__name__)


class InternalChainImpl(InternalChain):
    """内部链实现类 - 使用LCEL (LangChain Expression Language)"""
    
    def __init__(self, chain):
        self._chain = chain
    
    def run(self, **kwargs) -> str:
        result = self._chain.invoke(kwargs)
        if isinstance(result, str):
            return result
        elif hasattr(result, 'content'):
            return result.content
        return str(result)
    
    async def arun(self, **kwargs) -> str:
        result = await self._chain.ainvoke(kwargs)
        if isinstance(result, str):
            return result
        elif hasattr(result, 'content'):
            return result.content
        return str(result)


class InternalToolImpl(InternalTool):
    """内部工具实现类"""
    
    def __init__(self, tool):
        self._tool = tool
    
    def run(self, **kwargs) -> str:
        result = self._tool.invoke(kwargs)
        if isinstance(result, str):
            return result
        return str(result)
    
    @property
    def name(self) -> str:
        return self._tool.name
    
    @property
    def description(self) -> str:
        return self._tool.description


class InternalMemoryImpl(InternalMemory):
    """内部记忆实现类 - 使用BaseChatMessageHistory"""
    
    def __init__(self, chat_history):
        self._chat_history = chat_history
        self._messages: List[Dict] = []
    
    def save_context(self, inputs: Dict, outputs: Dict) -> None:
        from langchain_core.messages import HumanMessage, AIMessage
        
        for key, value in inputs.items():
            self._messages.append({"role": "human", "content": value})
        for key, value in outputs.items():
            self._messages.append({"role": "ai", "content": value})
        
        if hasattr(self._chat_history, 'add_message'):
            for key, value in inputs.items():
                self._chat_history.add_message(HumanMessage(content=value))
            for key, value in outputs.items():
                self._chat_history.add_message(AIMessage(content=value))
    
    def load_memory_variables(self, inputs: Dict) -> Dict:
        return {"history": self._messages}
    
    def clear(self) -> None:
        self._messages = []
        if hasattr(self._chat_history, 'clear'):
            self._chat_history.clear()


class InternalAgentImpl(InternalAgent):
    """内部Agent实现类 - 使用langgraph"""
    
    def __init__(self, agent_executor):
        self._agent_executor = agent_executor
    
    def run(self, input_text: str) -> str:
        result = self._agent_executor.invoke({"messages": [("user", input_text)]})
        if isinstance(result, dict):
            if "messages" in result:
                messages = result["messages"]
                if messages:
                    last_message = messages[-1]
                    if hasattr(last_message, 'content'):
                        return last_message.content
                    return str(last_message)
            return str(result)
        return str(result)
    
    async def arun(self, input_text: str) -> str:
        result = await self._agent_executor.ainvoke({"messages": [("user", input_text)]})
        if isinstance(result, dict):
            if "messages" in result:
                messages = result["messages"]
                if messages:
                    last_message = messages[-1]
                    if hasattr(last_message, 'content'):
                        return last_message.content
                    return str(last_message)
            return str(result)
        return str(result)


class LangchainAdapterImpl(LangchainAdapter):
    """
    Langchain适配器实现类
    
    封装langchain库，为项目提供统一的框架操作接口。
    使用langchain_core和langgraph组件。
    
    属性：
        _llm: 语言模型实例
    """
    
    def __init__(self, llm=None):
        """
        初始化Langchain适配器

        Args:
            llm: 语言模型实例（可选）
        """
        super().__init__()
        self._llm = llm
    
    def set_llm(self, llm) -> None:
        """
        设置语言模型

        Args:
            llm: 语言模型实例
        """
        self._llm = llm
        self._set_initialized(True)
    
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
        logger.info(f"[LangchainAdapterImpl.create_chain] 创建链式调用: chain_type={chain_type}")
        
        if chain_type == "llm_chain":
            from langchain_core.prompts import PromptTemplate
            
            prompt = PromptTemplate(
                template=config.get("prompt_template", "{input}"),
                input_variables=config.get("input_variables", ["input"])
            )
            
            if self._llm is None:
                logger.error("[LangchainAdapterImpl.create_chain] LLM未设置")
                raise ValueError("LLM not set. Call set_llm() first.")
            
            chain = prompt | self._llm
            log_arch_event(logger, component="LangchainAdapter", stage="ADAPTER", event="create_chain", status="success", design_id="ARCH-7.5", chain_type=chain_type)
            logger.info(f"[LangchainAdapterImpl.create_chain] LLM链创建成功: chain_type={chain_type}")
            return InternalChainImpl(chain)
        
        logger.error(f"[LangchainAdapterImpl.create_chain] 不支持的链类型: chain_type={chain_type}")
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
        logger.info(f"[LangchainAdapterImpl.create_tool] 创建工具: tool_type={tool_type}, name={config.get('name', tool_type)}")
        from langchain_core.tools import Tool
        
        tool = Tool(
            name=config.get("name", tool_type),
            description=config.get("description", ""),
            func=config.get("func")
        )
        log_arch_event(logger, component="LangchainAdapter", stage="ADAPTER", event="create_tool", status="success", design_id="ARCH-7.5", tool_type=tool_type)
        logger.info(f"[LangchainAdapterImpl.create_tool] 工具创建成功: name={tool.name}")
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
            from langchain_core.chat_history import BaseChatMessageHistory
            
            class SimpleChatHistory(BaseChatMessageHistory):
                def __init__(self):
                    self._messages = []
                
                @property
                def messages(self):
                    return self._messages
                
                def add_message(self, message):
                    self._messages.append(message)
                
                def clear(self):
                    self._messages = []
            
            chat_history = SimpleChatHistory()
            return InternalMemoryImpl(chat_history)
        
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
            config: Agent配置（包含tools、llm等）
            
        Returns:
            内部Agent对象
        """
        logger.info(f"[LangchainAdapterImpl.create_agent] 创建Agent: agent_type={agent_type}")
        from langgraph.prebuilt import create_react_agent
        
        tools = config.get("tools", [])
        llm = config.get("llm", self._llm)
        
        if llm is None:
            logger.error("[LangchainAdapterImpl.create_agent] LLM未设置")
            raise ValueError("LLM not set. Call set_llm() or provide llm in config.")
        
        agent_executor = create_react_agent(
            model=llm,
            tools=tools,
            state_modifier=config.get("system_prompt", None)
        )
        log_arch_event(logger, component="LangchainAdapter", stage="ADAPTER", event="create_agent", status="success", design_id="ARCH-7.5", agent_type=agent_type, tools_count=len(tools))
        logger.info(f"[LangchainAdapterImpl.create_agent] Agent创建成功: agent_type={agent_type}, tools_count={len(tools)}")
        return InternalAgentImpl(agent_executor)
