# -*- coding: utf-8 -*-
"""
编排层Agent模式数据类（兼容层）

该模块从独立文件re-export AgentContext和AgentResult，保持向后兼容。
新代码应直接从agent_context和agent_result导入。
"""

from src.orchestration.agent.agent_context import AgentContext
from src.orchestration.agent.agent_result import AgentResult

__all__ = ['AgentContext', 'AgentResult']
