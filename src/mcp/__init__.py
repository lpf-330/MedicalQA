"""
MCP代理层

该层在编排层和工具实现层之间提供统一的工具调用接口，屏蔽底层工具实现差异。

核心职责：
    - 提供统一的工具调用接口，屏蔽底层工具实现差异
    - 支持标准MCP协议（真MCP代理）和高效直连（伪MCP代理）两种调用方式
    - 工具实例的生命周期管理、缓存
    - 协议转换：将编排层的抽象调用转换为具体工具的执行请求

重要说明：
    MCP代理层只代理真正的Tool（如Neo4jMedicalTool、VectorRetrievalTool、ValidationTool），
    不代理模型调用！模型调用由编排层的模型业务服务直接完成。
"""

from src.mcp.proxy import (
    # 数据类
    MethodParam,
    ToolMethod,
    ToolInfo,
    DirectConnectionInfo,
    # 接口
    MCPTool,
    MCPStandardProxy,
    MCPFakeProxy,
)
from src.mcp.factory import (
    # 枚举和配置类
    ProxyType,
    ToolProxyConfig,
    # 工厂类
    MCPProxyFactory,
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
    # 枚举和配置类
    'ProxyType',
    'ToolProxyConfig',
    # 工厂类
    'MCPProxyFactory',
]
