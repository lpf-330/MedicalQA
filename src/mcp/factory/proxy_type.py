# -*- coding: utf-8 -*-
"""
MCP代理类型枚举模块

定义MCP代理tool的代理类型枚举ProxyType。
"""

from enum import Enum


class ProxyType(Enum):
    """
    代理类型枚举类

    定义MCP代理tool的代理类型。

    Attributes:
        STANDARD: MCP代理tool的真代理类型，表示支持标准MCP协议
        FAKE: MCP代理tool的伪代理类型，表示高效直连tool功能实例
    """
    STANDARD = "STANDARD"
    FAKE = "FAKE"

    def __str__(self) -> str:
        """返回枚举值的字符串表示"""
        return self.value

    def __repr__(self) -> str:
        """返回枚举值的详细字符串表示"""
        return f"ProxyType.{self.name}"
