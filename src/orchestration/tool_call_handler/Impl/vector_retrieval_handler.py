# -*- coding: utf-8 -*-

import logging
import time
from typing import Any, Dict, Optional

from src.orchestration.tool_call_handler.tool_call_handler import ToolCallHandler
from src.mcp.proxy.interfaces import MCPTool
from src.mcp.factory.mcp_proxy_factory import MCPProxyFactory

logger = logging.getLogger(__name__)


class VectorRetrievalHandler(ToolCallHandler[Dict, Dict[str, Any]]):

    def __init__(self, tool_proxy_instance_id: str = "milvus_medical"):
        self._tool: Optional[MCPTool] = None
        self._tool_proxy_instance_id = tool_proxy_instance_id

    def _init_tool(self, tool: MCPTool) -> None:
        logger.info("[VectorRetrievalHandler] _init_tool started")
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
            return
        
        logger.info("[VectorRetrievalHandler] Tool not initialized, auto-reinitializing...")
        try:
            factory = MCPProxyFactory.get_instance()
            tool = factory.get_tool_proxy_instance(self._tool_proxy_instance_id)
            self._tool = tool
            logger.info(f"[VectorRetrievalHandler] Auto-reinitialization completed using factory, tool_proxy_instance_id={self._tool_proxy_instance_id}")
        except Exception as e:
            logger.error(f"[VectorRetrievalHandler] Auto-reinitialization failed: {str(e)}")
            raise

    def call_tool(self, context: Dict) -> Dict[str, Any]:
        logger.debug(f"[VectorRetrievalHandler] call_tool called, context_keys={list(context.keys())}")
        start_time = time.time()
        try:
            self._ensure_initialized()

            result = self._tool.call("hybrid_search", context)
            elapsed = time.time() - start_time
            logger.info(f"[VectorRetrievalHandler] call_tool completed, elapsed={elapsed:.3f}s, result_count={len(result) if isinstance(result, list) else 'N/A'}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VectorRetrievalHandler] call_tool failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def release(self) -> None:
        logger.info("[VectorRetrievalHandler] release started")
        start_time = time.time()
        try:
            self._tool = None
            elapsed = time.time() - start_time
            logger.info(f"[VectorRetrievalHandler] release completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VectorRetrievalHandler] release failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise
