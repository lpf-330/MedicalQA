# AI辅助生成：GLM-5，2026-04-15
"""
MCP代理层工具代理配置类模块

本模块定义了所有MCP代理tool的代理配置类ToolProxyConfig。
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from .proxy_type import ProxyType


@dataclass
class ToolProxyConfig:
    """
    所有MCP代理tool的代理配置类
    
    存放MCP代理tool的代理类型和连接信息。
    
    Attributes:
        proxy_type: MCP代理tool的代理类型，为ProxyType枚举类的类型
        connection_info: MCP代理tool实例的连接信息
    """
    
    proxy_type: ProxyType
    connection_info: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        """返回配置对象的字符串表示"""
        return (f"ToolProxyConfig(proxy_type={self.proxy_type}, "
                f"connection_info_keys={list(self.connection_info.keys())})")
