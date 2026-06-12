# AI辅助生成：GLM-5，2026-04-15
"""
MCP代理层核心接口

该模块定义了MCP代理层的核心接口，包括MCPTool、MCPStandardProxy和MCPFakeProxy。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from src.mcp.proxy.data_classes import ToolInfo
    from src.tools.tool import Tool


class MCPTool(ABC):
    """
    MCPTool接口 - MCP对外统一接口

    作为MCP代理层的对外统一接口，定义了工具代理的基本行为。
    所有MCP代理tool都必须实现此接口。

    MCPTool接口提供了统一的工具调用接口，屏蔽底层工具实现差异。
    支持标准MCP协议（真MCP代理）和高效直连（伪MCP代理）两种调用方式。

    生命周期：
        1. 创建MCP代理实例
        2. 调用_init_tool()初始化tool功能实例
        3. 使用call()方法调用tool功能
        4. 调用release_tool()释放tool功能实例
    """

    @abstractmethod
    def _init_tool(self) -> None:
        """
        初始化tool功能实例

        在MCP代理实例创建后调用，用于初始化tool功能实例。
        该方法为私有方法，由MCP代理内部或MCPProxyFactory调用。

        Raises:
            ResourceException: 资源初始化失败时抛出
            BusinessException: 业务逻辑错误时抛出
        """
        pass

    @abstractmethod
    def release_tool(self, tool: 'Tool') -> None:
        """
        释放tool功能实例

        在MCP代理使用完毕后调用，用于释放tool功能实例。
        该方法为公共方法，由外部调用者（如ToolCallHandler）调用。

        Args:
            tool: 要释放的tool功能实例

        注意：
            - 该方法应该是幂等的，即多次调用不会产生副作用
            - 即使tool未初始化，调用该方法也不应抛出异常

        Raises:
            ResourceException: 资源释放失败时抛出
        """
        pass

    @abstractmethod
    def call(self, method_name: str, params: Dict[str, Any]) -> Any:
        """
        调用tool功能实例的方法

        通过方法名和参数调用tool功能实例的方法。

        Args:
            method_name: 要调用的方法名称
            params: 方法参数字典，key为参数名，value为参数值

        Returns:
            Any: 方法调用的返回值

        Raises:
            ParamException: 参数错误时抛出
            BusinessException: 业务逻辑错误时抛出
            ResourceException: 资源访问错误时抛出

        Example:
            >>> proxy = MCPProxyFactory.get_tool_proxy_instance("neo4j_tool")
            >>> result = proxy.call("query", {"cypher": "MATCH (n) RETURN n LIMIT 10"})
        """
        pass

    @abstractmethod
    def get_tool_info(self) -> 'ToolInfo':
        """
        获取tool功能实例的信息

        返回tool功能实例的基本信息，包括名称、描述、提供的方法等。

        Returns:
            ToolInfo: tool功能实例的信息对象

        Raises:
            ResourceException: 资源访问错误时抛出
        """
        pass

    def __enter__(self) -> 'MCPTool':
        """
        上下文管理器入口方法

        适配with语法，进入上下文时初始化tool功能实例。

        Returns:
            MCPTool: 当前MCP代理实例
        """
        self._init_tool()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        上下文管理器退出方法

        适配with语法，在上下文执行结束时自动释放tool功能实例。

        Args:
            exc_type: 异常类型
            exc_val: 异常值
            exc_tb: 异常堆栈信息
        """
        # 注意：这里不直接调用release_tool，因为需要tool实例
        # 实际释放逻辑由具体实现类处理
        pass


class MCPStandardProxy(MCPTool):
    """
    MCPStandardProxy接口 - MCP真代理接口

    支持标准MCP协议的代理接口，用于与标准MCP服务进行通信。
    实现了MCPTool接口，并添加了标准MCP协议相关的方法。

    标准MCP代理通过MCP协议与远程工具服务进行通信，支持：
        - 协议版本协商
        - 握手流程
        - 能力协商
        - 运行指标监控

    使用场景：
        - 需要与标准MCP服务进行通信的场景
        - 需要跨语言、跨平台调用的场景
        - 需要完整MCP协议支持的场景
    """

    @abstractmethod
    def get_mcp_protocol_version(self) -> str:
        """
        获取MCP代理支持的协议版本信息

        返回当前MCP代理支持的协议版本字符串。

        Returns:
            str: 协议版本信息，如"1.0.0"

        Raises:
            ResourceException: 资源访问错误时抛出
        """
        pass

    @abstractmethod
    def perform_handshake(self) -> bool:
        """
        执行MCP协议握手流程

        执行MCP协议握手流程，完成初始化与能力协商。
        握手成功后，MCP代理将准备好接收工具调用请求。

        Returns:
            bool: 握手是否成功，True表示成功，False表示失败

        Raises:
            NetworkException: 网络连接错误时抛出
            ResourceException: 资源访问错误时抛出
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        检查MCP代理或目标MCP服务的可用性状态

        检查MCP代理是否可用，以及目标MCP服务是否可访问。

        Returns:
            bool: 是否可用，True表示可用，False表示不可用
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取MCP代理的运行指标与性能数据

        返回MCP代理的运行指标和性能数据，包括：
            - 调用次数
            - 平均响应时间
            - 错误率
            - 资源使用情况

        Returns:
            Dict[str, Any]: 运行指标与性能数据的字典

        Raises:
            ResourceException: 资源访问错误时抛出
        """
        pass


class MCPFakeProxy(MCPTool):
    """
    MCPFakeProxy接口 - MCP伪代理接口

    高效直连tool功能实例的代理接口，不使用标准MCP协议。
    实现了MCPTool接口，并添加了直连相关的方法。

    伪MCP代理直接与本地工具实例进行通信，支持：
        - 直连信息获取
        - 模拟响应设置
        - 高效调用

    使用场景：
        - 需要高性能调用的场景
        - 工具实例在本地的场景
        - 测试和开发环境
        - 不需要完整MCP协议支持的场景

    注意：
        伪代理虽然名为"伪"，但提供了更高的性能，
        适用于对性能要求较高的场景。
    """

    @abstractmethod
    def get_direct_connection_info(self) -> 'DirectConnectionInfo':
        """
        获取与tool功能实例的直连信息

        返回与tool功能实例的直连信息，包括通信类型和端点。

        Returns:
            DirectConnectionInfo: 直连信息对象

        Raises:
            ResourceException: 资源访问错误时抛出
        """
        pass

    @abstractmethod
    def set_mock_response(self, method_name: str, response: Any) -> None:
        """
        设置模拟返回数据

        为指定方法设置模拟返回数据，用于测试和开发环境。
        设置后，调用该方法将直接返回模拟数据，而不实际调用工具。

        Args:
            method_name: 方法名称
            response: 模拟返回数据

        注意：
            - 该方法主要用于测试和开发环境
            - 生产环境应谨慎使用
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        检查MCP代理或目标MCP服务的可用性状态

        检查MCP代理是否可用，以及目标工具实例是否可访问。

        Returns:
            bool: 是否可用，True表示可用，False表示不可用
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取MCP代理的运行指标与性能数据

        返回MCP代理的运行指标和性能数据，包括：
            - 调用次数
            - 平均响应时间
            - 错误率
            - 资源使用情况

        Returns:
            Dict[str, Any]: 运行指标与性能数据的字典

        Raises:
            ResourceException: 资源访问错误时抛出
        """
        pass


# 导入DirectConnectionInfo用于类型提示
from src.mcp.proxy.data_classes import DirectConnectionInfo
