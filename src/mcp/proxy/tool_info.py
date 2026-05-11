# AI辅助生成：GLM-5，2026-04-15
"""
MCP代理层工具信息数据类模块

本模块定义了tool信息类ToolInfo。
"""

from dataclasses import dataclass, field
from typing import List

from .tool_method import ToolMethod


@dataclass
class ToolInfo:
    """
    tool信息类
    
    存放tool工具信息，包括tool功能实例名称、描述、提供的方法等。
    
    Attributes:
        name: tool功能实例名称
        description: tool功能实例描述
        methods: tool功能实例提供的方法
    """
    
    name: str
    description: str
    methods: List[ToolMethod] = field(default_factory=list)
    
    def __repr__(self) -> str:
        """返回工具信息对象的字符串表示"""
        return (f"ToolInfo(name='{self.name}', description='{self.description}', "
                f"methods_count={len(self.methods)})")
