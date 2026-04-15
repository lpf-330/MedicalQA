"""
MCP代理层数据类

该模块定义了MCP代理层所需的数据类，包括方法参数、工具方法、工具信息和直连信息。
"""

from dataclasses import dataclass
from typing import Any, List, Type


@dataclass
class MethodParam:
    """
    方法参数类

    存放tool方法参数的参数名、参数描述、参数类型、参数必要性等属性。

    Attributes:
        name: tool方法参数名称
        description: tool方法参数描述
        type: tool方法参数类型
        required: tool方法参数是否为必需
    """
    name: str
    description: str
    type: Type[Any]
    required: bool

    def __post_init__(self) -> None:
        """初始化后验证参数"""
        if not self.name:
            raise ValueError("参数名称不能为空")
        if not self.description:
            raise ValueError("参数描述不能为空")

    def to_dict(self) -> dict:
        """
        将方法参数转换为字典格式

        Returns:
            dict: 方法参数的字典表示
        """
        return {
            "name": self.name,
            "description": self.description,
            "type": str(self.type),
            "required": self.required
        }

    def __repr__(self) -> str:
        """返回方法参数的字符串表示"""
        return f"MethodParam(name='{self.name}', type={self.type.__name__}, required={self.required})"


@dataclass
class ToolMethod:
    """
    工具方法类

    存放tool方法的方法名、方法描述、参数列表、返回类型等属性。

    Attributes:
        name: tool方法名称
        description: tool方法描述
        params: tool方法参数列表
        return_type: tool方法返回类型
    """
    name: str
    description: str
    params: List[MethodParam]
    return_type: Type[Any]

    def __post_init__(self) -> None:
        """初始化后验证参数"""
        if not self.name:
            raise ValueError("方法名称不能为空")
        if not self.description:
            raise ValueError("方法描述不能为空")
        if self.params is None:
            self.params = []

    def to_dict(self) -> dict:
        """
        将工具方法转换为字典格式

        Returns:
            dict: 工具方法的字典表示
        """
        return {
            "name": self.name,
            "description": self.description,
            "params": [param.to_dict() for param in self.params],
            "return_type": str(self.return_type)
        }

    def __repr__(self) -> str:
        """返回工具方法的字符串表示"""
        return f"ToolMethod(name='{self.name}', params_count={len(self.params)})"


@dataclass
class ToolInfo:
    """
    工具信息类

    存放tool功能实例的名称、描述、提供的方法等信息。

    Attributes:
        name: tool功能实例名称
        description: tool功能实例描述
        methods: tool功能实例提供的方法
    """
    name: str
    description: str
    methods: List[ToolMethod]

    def __post_init__(self) -> None:
        """初始化后验证参数"""
        if not self.name:
            raise ValueError("工具名称不能为空")
        if not self.description:
            raise ValueError("工具描述不能为空")
        if self.methods is None:
            self.methods = []

    def to_dict(self) -> dict:
        """
        将工具信息转换为字典格式

        Returns:
            dict: 工具信息的字典表示
        """
        return {
            "name": self.name,
            "description": self.description,
            "methods": [method.to_dict() for method in self.methods]
        }

    def get_method(self, method_name: str) -> ToolMethod:
        """
        根据方法名获取工具方法

        Args:
            method_name: 方法名称

        Returns:
            ToolMethod: 工具方法对象

        Raises:
            ValueError: 方法不存在时抛出
        """
        for method in self.methods:
            if method.name == method_name:
                return method
        raise ValueError(f"方法 '{method_name}' 不存在")

    def has_method(self, method_name: str) -> bool:
        """
        检查是否存在指定方法

        Args:
            method_name: 方法名称

        Returns:
            bool: 是否存在该方法
        """
        return any(method.name == method_name for method in self.methods)

    def __repr__(self) -> str:
        """返回工具信息的字符串表示"""
        return f"ToolInfo(name='{self.name}', methods_count={len(self.methods)})"


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
