"""
编排层Agent模式包

该包提供agent模式的设计规范与组合包装，包括：
- AgentContext: agent输入数据容器类
- AgentResult: agent输出数据容器类
- AgentResource: agent资源类
- AgentStrategy: agent策略接口
- Agent: agent组合容器类
"""

from src.orchestration.agent.data_classes import AgentContext, AgentResult
from src.orchestration.agent.agent_resource import AgentResource
from src.orchestration.agent.agent_strategy import AgentStrategy
from src.orchestration.agent.agent import Agent

__all__ = [
    'AgentContext',
    'AgentResult',
    'AgentResource',
    'AgentStrategy',
    'Agent'
]
