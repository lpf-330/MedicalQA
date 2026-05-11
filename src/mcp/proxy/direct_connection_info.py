# AI辅助生成：GLM-5，2026-04-15
"""
MCP代理层直连信息数据类模块

本模块定义了MCP伪代理接口的信息类DirectConnectionInfo。
"""

from dataclasses import dataclass


@dataclass
class DirectConnectionInfo:
    """
    MCP伪代理接口的信息类
    
    存放连接信息，包括直连通信类型标识和端点寻址标识。
    
    Attributes:
        type: 直连通信类型标识
        endpoint: 直连通信端点寻址标识
    """
    
    type: str
    endpoint: str
    
    def __repr__(self) -> str:
        """返回直连信息对象的字符串表示"""
        return f"DirectConnectionInfo(type='{self.type}', endpoint='{self.endpoint}')"
