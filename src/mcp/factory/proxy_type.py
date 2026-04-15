"""
MCP代理层代理类型枚举模块

本模块定义了代理类型的枚举类ProxyType。
"""

from enum import Enum


class ProxyType(Enum):
    """
    代理类型的枚举类
    
    定义MCP代理tool的代理类型，包括标准MCP协议代理和高效直连代理。
    
    Attributes:
        STANDARD: MCP代理tool的真代理类型，表示支持标准MCP协议
        FAKE: MCP代理tool的伪代理类型，表示高效直连tool功能实例
    """
    
    STANDARD = "STANDARD"
    FAKE = "FAKE"
    
    def __repr__(self) -> str:
        """返回枚举值的字符串表示"""
        return f"ProxyType.{self.name}"
