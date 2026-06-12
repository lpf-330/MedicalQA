"""
MCP代理层枚举和配置类

该模块定义了MCP代理层的枚举类型和配置类，包括代理类型枚举和工具代理配置。

兼容性说明：ProxyType 已迁移至 proxy_type.py，此模块通过 re-export 保持向后兼容。
"""

from dataclasses import dataclass, field
from typing import Any, Dict

from src.mcp.factory.proxy_type import ProxyType  # noqa: F401 — re-export for backward compatibility


@dataclass
class ToolProxyConfig:
    """
    工具代理配置类

    所有MCP代理tool的代理配置类，包含代理类型和连接信息。

    Attributes:
        proxy_type: MCP代理tool的代理类型，为ProxyType枚举类的类型
        connection_info: MCP代理tool实例的连接信息
    """
    proxy_type: ProxyType
    connection_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """初始化后验证参数"""
        if not isinstance(self.proxy_type, ProxyType):
            raise ValueError(f"proxy_type必须是ProxyType枚举类型，当前类型为: {type(self.proxy_type)}")
        if self.connection_info is None:
            self.connection_info = {}

    def to_dict(self) -> dict:
        """
        将工具代理配置转换为字典格式

        Returns:
            dict: 工具代理配置的字典表示
        """
        return {
            "proxy_type": self.proxy_type.value,
            "connection_info": self.connection_info
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ToolProxyConfig':
        """
        从字典创建ToolProxyConfig实例

        Args:
            config_dict: 配置字典

        Returns:
            ToolProxyConfig: 工具代理配置实例

        Raises:
            ValueError: 配置字典格式错误时抛出
        """
        if "proxy_type" not in config_dict:
            raise ValueError("配置字典缺少proxy_type字段")

        proxy_type_str = config_dict["proxy_type"]
        try:
            proxy_type = ProxyType(proxy_type_str)
        except ValueError:
            raise ValueError(f"无效的proxy_type值: {proxy_type_str}")

        connection_info = config_dict.get("connection_info", {})

        return cls(
            proxy_type=proxy_type,
            connection_info=connection_info
        )

    def get_connection_param(self, key: str, default: Any = None) -> Any:
        """
        获取连接信息中的指定参数

        Args:
            key: 参数键
            default: 默认值

        Returns:
            Any: 参数值
        """
        return self.connection_info.get(key, default)

    def set_connection_param(self, key: str, value: Any) -> None:
        """
        设置连接信息中的指定参数

        Args:
            key: 参数键
            value: 参数值
        """
        self.connection_info[key] = value

    def __repr__(self) -> str:
        """返回工具代理配置的字符串表示"""
        return f"ToolProxyConfig(proxy_type={self.proxy_type.name}, connection_info_keys={list(self.connection_info.keys())})"
