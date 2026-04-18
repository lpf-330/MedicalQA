# -*- coding: utf-8 -*-

import logging
import time
from typing import Any, Dict, Optional

from src.orchestration.tool_call_handler.tool_call_handler import ToolCallHandler
from src.mcp.proxy.interfaces import MCPTool

logger = logging.getLogger(__name__)


class VectorRetrievalHandler(ToolCallHandler[Dict, Dict[str, Any]]):

    def __init__(self):
        self._tool: Optional[MCPTool] = None

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

    def call_tool(self, context: Dict) -> Dict[str, Any]:
        logger.debug(f"[VectorRetrievalHandler] call_tool called, context_keys={list(context.keys())}")
        start_time = time.time()
        try:
            if self._tool is None:
                raise RuntimeError("Tool not initialized, call _init_tool first")

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
            if self._tool is not None:
                self._tool.release_tool(None)
                self._tool = None
            elapsed = time.time() - start_time
            logger.info(f"[VectorRetrievalHandler] release completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VectorRetrievalHandler] release failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise
