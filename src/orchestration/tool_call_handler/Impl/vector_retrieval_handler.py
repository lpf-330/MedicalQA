# -*- coding: utf-8 -*-

import logging
import time
from typing import Any, Dict, Optional

from src.orchestration.tool_call_handler.tool_call_handler import ToolCallHandler
from src.mcp.proxy.interfaces import MCPTool
from src.mcp.factory.mcp_proxy_factory import MCPProxyFactory

logger = logging.getLogger(__name__)


class VectorRetrievalHandler(ToolCallHandler[Dict, Dict[str, Any]]):

    def __init__(self, tool_proxy_instance_id: str = "vector_retrieval"):
        self._tool: Optional[MCPTool] = None
        self._tool_proxy_instance_id = tool_proxy_instance_id

    def _init_tool(self, tool: MCPTool) -> None:
        logger.info("[VectorRetrievalHandler] _init_tool started")
        logger.info(f"[TOOL_HANDLER_INIT] handler=VectorRetrievalHandler, event=_init_tool, already_initialized={self._tool is not None}")
        start_time = time.time()
        try:
            self._tool = tool
            tool._init_tool()
            elapsed = time.time() - start_time
            logger.info(f"[VectorRetrievalHandler] _init_tool completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VectorRetrievalHandler] _init_tool failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def is_initialized(self) -> bool:
        return self._tool is not None

    def _ensure_initialized(self) -> None:
        if self._tool is not None:
            logger.info("[TOOL_HANDLER_INIT] handler=VectorRetrievalHandler, event=_ensure_initialized, auto_reinit=False, already_initialized=True")
            return

        logger.info("[VectorRetrievalHandler] Tool not initialized, auto-reinitializing...")
        logger.info("[TOOL_HANDLER_INIT] handler=VectorRetrievalHandler, event=_ensure_initialized, auto_reinit=True, already_initialized=False")
        try:
            factory = MCPProxyFactory.get_instance()
            tool = factory.get_tool_proxy_instance(self._tool_proxy_instance_id)
            self._tool = tool
            logger.info(f"[VectorRetrievalHandler] Auto-reinitialization completed using factory, tool_proxy_instance_id={self._tool_proxy_instance_id}")
        except Exception as e:
            logger.error(f"[VectorRetrievalHandler] Auto-reinitialization failed: {str(e)}")
            raise

    def call_tool(self, context: Dict) -> Dict[str, Any]:
        logger.info(f"[TOOL_HANDLER_CALL] {self.__class__.__name__}调用工具")
        logger.debug(f"[VectorRetrievalHandler] call_tool called, context_keys={list(context.keys())}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            auto_init = self._tool is not None
            logger.info(f"[TOOL_CALL] handler=VectorRetrievalHandler, method=hybrid_search, auto_init={auto_init}")

            query_text = context.get("query", "")
            top_k = context.get("top_k", "N/A")
            logger.info(f"[VECTOR_TOOL_CALL] method=hybrid_search, query_length={len(query_text) if isinstance(query_text, str) else 'N/A'}, top_k={top_k}, context_keys={list(context.keys())}")

            result = self._tool.call("hybrid_search", context)

            result_count = len(result) if isinstance(result, list) else "N/A"
            if isinstance(result, dict):
                result_count = len(result.get("results", result.get("entities", [])))
            logger.info(f"[VECTOR_TOOL_RESULT] method=hybrid_search, result_count={result_count}, result_type={type(result).__name__}")

            elapsed = time.time() - start_time
            logger.info(f"[VectorRetrievalHandler] call_tool completed, elapsed={elapsed:.3f}s, result_count={result_count}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VECTOR_TOOL_ERROR] method=hybrid_search, elapsed={elapsed:.3f}s, error_type={type(e).__name__}, error={str(e)}")
            logger.error(f"[VectorRetrievalHandler] call_tool failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def release_tool(self) -> None:
        """释放tool功能实例引用 — 只清除引用，不释放MCP代理实例

        设计依据：2.3.3节 ToolCallHandler生命周期管理规范
        release_tool()语义：将工具引用设置为None，不释放MCP代理实例
        """
        self._tool = None
