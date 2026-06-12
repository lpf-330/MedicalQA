"""
MCP代理层数据类

该模块定义了MCP代理层所需的数据类，包括方法参数、工具方法、工具信息和直连信息。

兼容性说明：MethodParam、ToolMethod、ToolInfo 已迁移至 tool_info.py，此模块通过 re-export 保持向后兼容。
"""

from dataclasses import dataclass
from typing import Any

from src.mcp.proxy.tool_info import MethodParam  # noqa: F401 — re-export for backward compatibility
from src.mcp.proxy.tool_info import ToolMethod   # noqa: F401 — re-export for backward compatibility
from src.mcp.proxy.tool_info import ToolInfo     # noqa: F401 — re-export for backward compatibility


@dataclass
class DirectConnectionInfo:
    """
    直连信息类

    存放MCP伪代理接口的连接信息，包括直连通信类型标识和端点寻址标识。

    Attributes:
        type: 直连通信类型标识
        endpoint: 直连通信端点寻址标识
    """
    type: str
    endpoint: str

    def __post_init__(self) -> None:
        """初始化后验证参数"""
        if not self.type:
            raise ValueError("直连通信类型标识不能为空")
        if not self.endpoint:
            raise ValueError("直连通信端点寻址标识不能为空")

    def to_dict(self) -> dict:
        """
        将直连信息转换为字典格式

        Returns:
            dict: 直连信息的字典表示
        """
        return {
            "type": self.type,
            "endpoint": self.endpoint
        }

    def __repr__(self) -> str:
        """返回直连信息的字符串表示"""
        return f"DirectConnectionInfo(type='{self.type}', endpoint='{self.endpoint}')"
