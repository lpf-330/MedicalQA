"""
编排层Tool调用处理器接口

该模块定义了ToolCallHandler接口，是tool调用服务的核心抽象。
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from src.mcp.proxy.interfaces import MCPTool

# 定义泛型类型变量
I = TypeVar('I')  # 输入数据类型
O = TypeVar('O')  # 输出数据类型


class ToolCallHandler(ABC, Generic[I, O]):
    """
    ToolCallHandler接口 - Tool调用处理器接口

    每一个在不同业务下定义的tool服务调用类必须实现的接口。
    其实现类为agent策略和chain策略提供tool调用服务。

    职责：
        - 初始化MCP代理tool
        - 调用tool服务
        - 释放tool功能实例

    使用示例：
        >>> class Neo4jToolHandler(ToolCallHandler[Neo4jQuery, Neo4jResult]):
        ...     def __init__(self):
        ...         self._tool: Optional[MCPTool] = None
        ...
        ...     def _init_tool(self, tool: MCPTool) -> None:
        ...         self._tool = tool
        ...         tool._init_tool()
        ...
        ...     def call_tool(self, context: Neo4jQuery) -> Neo4jResult:
        ...         if self._tool is None:
        ...             raise ValueError("tool未初始化")
        ...         result = self._tool.call("query", {"cypher": context.cypher})
        ...         return Neo4jResult(data=result)
        ...
        ...     def release(self) -> None:
        ...         if self._tool is not None:
        ...             self._tool.release_tool(None)
        ...             self._tool = None

    生命周期：
        1. 创建ToolCallHandler实例
        2. 调用_init_tool初始化MCP代理tool
        3. 调用call_tool执行tool调用
        4. 调用release释放tool资源

    泛型参数：
        I: tool调用的输入数据类型
        O: tool调用的输出数据类型
    """

    @abstractmethod
    def _init_tool(self, tool: 'MCPTool') -> None:
        """
        初始化MCP代理tool

        在ToolCallHandler实例创建后调用，用于初始化MCP代理tool实例。
        该方法为私有方法，由agent策略或chain策略内部调用。

        Args:
            tool: MCP代理tool实例

        Raises:
            ParamException: 参数错误时抛出
            ResourceException: 资源初始化失败时抛出

        Example:
            >>> handler._init_tool(mcp_tool)
        """
        pass

    @abstractmethod
    def call_tool(self, context: I) -> O:
        """
        调用tool服务

        通过MCP代理tool调用tool功能实例的方法。
        输入类型和输出类型由实现该接口的类型的泛型决定。

        Args:
            context: tool调用的输入数据

        Returns:
            O: tool调用的输出数据

        Raises:
            ParamException: 参数错误时抛出
            BusinessException: 业务逻辑错误时抛出
            ResourceException: 资源访问错误时抛出

        Example:
            >>> result = handler.call_tool(my_context)
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """
        释放tool功能实例

        在ToolCallHandler使用完毕后调用，用于释放tool功能实例。
        该方法应该是幂等的，即多次调用不会产生副作用。

        Example:
            >>> handler.release()
        """
        pass

    def __repr__(self) -> str:
        """返回ToolCallHandler的字符串表示"""
        return f"{self.__class__.__name__}()"
