"""
编排层

编排层是系统的核心大脑，负责整个业务流程的状态机驱动和Agent策略执行。

核心职责：
    - 实现基于有限状态机(FSM)的业务逻辑控制
    - 封装业务策略：针对不同业务场景（健康咨询/健康报告）特殊封装业务策略逻辑
    - 驱动工具调用和内容生成，协调各阶段的输入输出
    - 管理执行状态，控制流式生成节奏
    - 使用MCP代理层的mcp代理工具

包结构：
    - state_machine: 状态机，为agent策略提供状态管理支持
    - agent: Agent模式，包含Agent、AgentStrategy、AgentContext、AgentResult、AgentResource
    - chain: Chain模式，包含Chain、ChainContext、ChainResult
    - tool_call_handler: Tool调用处理器接口
    - model_business_service: 模型业务服务接口
"""

# 导入状态机
from src.orchestration.state_machine import StateMachine

# 导入Chain模式
from src.orchestration.chain import Chain, ChainContext, ChainResult

# 导入Agent模式
from src.orchestration.agent import (
    Agent,
    AgentStrategy,
    AgentContext,
    AgentResult,
    AgentResource
)

# 导入服务接口
from src.orchestration.tool_call_handler import ToolCallHandler
from src.orchestration.model_business_service import ModelBusinessService

__all__ = [
    # 状态机
    'StateMachine',
    # Chain模式
    'Chain',
    'ChainContext',
    'ChainResult',
    # Agent模式
    'Agent',
    'AgentStrategy',
    'AgentContext',
    'AgentResult',
    'AgentResource',
    # 服务接口
    'ToolCallHandler',
    'ModelBusinessService'
]
