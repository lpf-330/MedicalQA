"""
MCP代理层proxy包

该包定义了MCP代理层的核心接口和数据类。
"""

from src.mcp.proxy.data_classes import (
    MethodParam,
    ToolMethod,
    ToolInfo,
    DirectConnectionInfo
)
from src.mcp.proxy.interfaces import (
    MCPTool,
    MCPStandardProxy,
    MCPFakeProxy
)

__all__ = [
    # 数据类
    'MethodParam',
    'ToolMethod',
    'ToolInfo',
    'DirectConnectionInfo',
    # 接口
    'MCPTool',
    'MCPStandardProxy',
    'MCPFakeProxy',
]
