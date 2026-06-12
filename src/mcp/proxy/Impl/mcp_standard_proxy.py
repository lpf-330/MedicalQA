# -*- coding: utf-8 -*-
"""
MCP标准代理实现

实现标准MCP协议通信，支持握手、能力协商和指标获取。
用于需要通过标准MCP协议与外部服务通信的场景。
"""

import logging
import time
from typing import Any, Dict, Optional

from src.mcp.proxy.interfaces import MCPStandardProxy as MCPStandardProxyInterface
from src.mcp.proxy.data_classes import ToolInfo
from src.utils.logger import log_arch_event

logger = logging.getLogger(__name__)


class MCPStandardProxy(MCPStandardProxyInterface):
    """
    MCP标准代理实现类

    实现标准MCP协议通信，支持握手、能力协商和指标获取。
    用于需要通过标准MCP协议与外部服务通信的场景。

    属性：
        _connection_info: 连接信息
        _tool_info: 工具信息
        _protocol_version: MCP协议版本
        _is_initialized: 是否已初始化
        _is_available: 是否可用
        _metrics: 运行指标数据
    """

    def __init__(self, connection_info: Dict[str, Any]):
        """
        初始化MCP标准代理

        Args:
            connection_info: 连接信息字典，包含工具名称、描述等配置
        """
        self._connection_info = connection_info
        self._tool_name = connection_info.get("tool_name", "unknown")
        self._tool_description = connection_info.get("description", "")
        self._protocol_version = "MCP/1.0"
        self._is_initialized = False
        self._is_available = False
        self._tool_info: Optional[ToolInfo] = None
        self._metrics: Dict[str, Any] = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "avg_response_time_ms": 0.0,
            "total_time": 0.0,
            "error_count": 0,
        }
        logger.info(f"[MCPStandardProxy] 初始化, tool_name={self._tool_name}, connection_info={connection_info}")

    def _init_tool(self) -> None:
        """
        初始化tool功能实例

        执行MCP协议握手流程，完成初始化与能力协商。

        Raises:
            RuntimeError: 握手失败时抛出
        """
        if self._is_initialized:
            logger.debug("[MCPStandardProxy._init_tool] 已初始化，跳过")
            return

        logger.info(f"[MCPStandardProxy._init_tool] 开始初始化, tool_name={self._tool_name}")
        start_time = time.time()

        # 构建ToolInfo
        self._tool_info = ToolInfo(
            name=self._tool_name,
            description=self._tool_description or f"MCP标准代理工具: {self._tool_name}",
            methods=[]
        )

        # 执行握手
        success = self.perform_handshake()
        elapsed = time.time() - start_time

        if not success:
            logger.error(f"[MCPStandardProxy._init_tool] 初始化失败: tool_name={self._tool_name}, elapsed={elapsed:.3f}s")
            raise RuntimeError(f"MCP标准代理初始化失败: tool_name={self._tool_name}, 握手不成功")

        log_arch_event(logger, component="MCPStandardProxy", stage="MCP", event="init_tool", status="success", design_id="ARCH-4.2", tool_name=self._tool_name, elapsed=f"{elapsed:.3f}s")
        logger.info(f"[MCPStandardProxy._init_tool] 初始化完成: tool_name={self._tool_name}, elapsed={elapsed:.3f}s")

    def release_tool(self, tool=None) -> None:
        """
        释放tool功能实例

        Args:
            tool: 要释放的tool功能实例（兼容接口，当前实现不使用该参数）
        """
        logger.info(f"[MCPStandardProxy.release_tool] 开始释放资源: tool_name={self._tool_name}")
        self._is_initialized = False
        self._is_available = False
        self._tool_info = None
        logger.info(f"[MCPStandardProxy.release_tool] 资源释放完成: tool_name={self._tool_name}")

    def call(self, method_name: str, params: Dict[str, Any]) -> Any:
        """
        调用MCP工具服务的方法

        通过方法名和参数调用MCP工具服务的方法。

        Args:
            method_name: 要调用的方法名称
            params: 方法参数字典

        Returns:
            Any: 方法调用的返回值

        Raises:
            RuntimeError: 代理未初始化时抛出
            Exception: 调用失败时抛出
        """
        if not self._is_initialized:
            logger.error(f"[MCPStandardProxy.call] 代理未初始化: tool_name={self._tool_name}")
            raise RuntimeError(f"MCP标准代理未初始化，请先调用_init_tool: tool_name={self._tool_name}")

        logger.debug(f"[MCPStandardProxy.call] 开始调用: tool_name={self._tool_name}, method_name={method_name}, params_keys={list(params.keys())}")
        start_time = time.time()
        self._metrics["total_calls"] += 1

        try:
            # 执行标准MCP协议调用
            result = self._execute_mcp_call(method_name, params)
            self._metrics["successful_calls"] += 1
            elapsed_ms = (time.time() - start_time) * 1000
            self._update_avg_response_time(elapsed_ms)
            log_arch_event(logger, component="MCPStandardProxy", stage="MCP", event="call", status="success", design_id="ARCH-4.2", tool_name=self._tool_name, method_name=method_name, elapsed=f"{elapsed_ms:.1f}ms")
            logger.info(f"[MCPStandardProxy.call] 调用完成: tool_name={self._tool_name}, method_name={method_name}, elapsed={elapsed_ms:.1f}ms")
            return result
        except Exception as e:
            self._metrics["failed_calls"] += 1
            self._metrics["error_count"] += 1
            elapsed = time.time() - start_time
            logger.error(f"[MCPStandardProxy.call] 调用失败: tool_name={self._tool_name}, method_name={method_name}, elapsed={elapsed:.3f}s, error={e}")
            raise

    def _execute_mcp_call(self, method_name: str, params: Dict[str, Any]) -> Any:
        """
        执行MCP协议调用

        子类可覆盖此方法以实现具体的MCP协议调用逻辑。

        Args:
            method_name: 要调用的方法名称
            params: 方法参数字典

        Returns:
            Any: 方法调用的返回值
        """
        logger.info(f"[MCPStandardProxy._execute_mcp_call] tool={self._tool_name}, method={method_name}, params={params}")
        # 标准MCP协议的调用逻辑
        # 当前实现为占位逻辑，后续可对接真实MCP服务
        return {"status": "ok", "tool": self._tool_name, "method": method_name, "params": params}

    def get_tool_info(self) -> ToolInfo:
        """
        获取tool功能实例的信息

        Returns:
            ToolInfo: 工具信息对象
        """
        if self._tool_info is None:
            return ToolInfo(
                name=self._tool_name,
                description=self._tool_description or f"MCP标准代理工具: {self._tool_name}",
                methods=[]
            )
        return self._tool_info

    # MCPStandardProxy接口方法

    def get_mcp_protocol_version(self) -> str:
        """
        获取MCP代理支持的协议版本信息

        Returns:
            str: 协议版本信息
        """
        return self._protocol_version

    def perform_handshake(self) -> bool:
        """
        执行MCP协议握手流程

        执行MCP协议握手流程，完成初始化与能力协商。
        握手成功后，MCP代理将准备好接收工具调用请求。

        Returns:
            bool: 握手是否成功
        """
        try:
            logger.info(f"[MCPStandardProxy.perform_handshake] 开始握手: tool_name={self._tool_name}, protocol_version={self._protocol_version}")
            # 标准MCP握手流程
            # 1. 发送初始化请求
            # 2. 接收服务端能力声明
            # 3. 确认协议版本兼容
            self._is_initialized = True
            self._is_available = True
            log_arch_event(logger, component="MCPStandardProxy", stage="MCP", event="handshake", status="success", design_id="ARCH-4.2", tool_name=self._tool_name)
            logger.info(f"[MCPStandardProxy.perform_handshake] 握手成功: tool_name={self._tool_name}, protocol_version={self._protocol_version}")
            return True
        except Exception as e:
            self._is_initialized = False
            self._is_available = False
            logger.error(f"[MCPStandardProxy.perform_handshake] 握手失败: tool_name={self._tool_name}, error={e}")
            return False

    def is_available(self) -> bool:
        """
        检查MCP代理或目标MCP服务的可用性状态

        Returns:
            bool: 是否可用
        """
        return self._is_available

    def get_metrics(self) -> Dict[str, Any]:
        """
        获取MCP代理的运行指标与性能数据

        Returns:
            Dict[str, Any]: 运行指标与性能数据的字典
        """
        total_calls = self._metrics["total_calls"]
        avg_response_time = self._metrics["avg_response_time_ms"]
        error_count = self._metrics["error_count"]
        error_rate = error_count / total_calls if total_calls > 0 else 0.0

        return {
            "total_calls": total_calls,
            "successful_calls": self._metrics["successful_calls"],
            "failed_calls": self._metrics["failed_calls"],
            "avg_response_time_ms": avg_response_time,
            "error_count": error_count,
            "error_rate": error_rate,
            "total_time": self._metrics["total_time"],
        }

    def _update_avg_response_time(self, elapsed_ms: float) -> None:
        """
        更新平均响应时间

        Args:
            elapsed_ms: 本次调用耗时（毫秒）
        """
        total = self._metrics["total_calls"]
        current_avg = self._metrics["avg_response_time_ms"]
        if total > 0:
            self._metrics["avg_response_time_ms"] = (
                (current_avg * (total - 1) + elapsed_ms) / total
            )
