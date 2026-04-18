# -*- coding: utf-8 -*-

import logging
import time
from typing import Any, Dict, Optional

from src.orchestration.tool_call_handler.tool_call_handler import ToolCallHandler
from src.mcp.proxy.interfaces import MCPTool

logger = logging.getLogger(__name__)


class IntentClassificationHandler(ToolCallHandler[Dict, Dict[str, Any]]):

    def __init__(self):
        self._tool: Optional[MCPTool] = None

    def _init_tool(self, tool: MCPTool) -> None:
        logger.info("[IntentClassificationHandler] _init_tool started")
        start_time = time.time()
        try:
            self._tool = tool
            tool._init_tool()
            elapsed = time.time() - start_time
            logger.info(f"[IntentClassificationHandler] _init_tool completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[IntentClassificationHandler] _init_tool failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def call_tool(self, context: Dict) -> Dict[str, Any]:
        method = context.get("method")
        logger.debug(f"[IntentClassificationHandler] call_tool called, method={method}, context_keys={list(context.keys())}")
        start_time = time.time()
        try:
            if self._tool is None:
                raise RuntimeError("Tool not initialized, call _init_tool first")

            if method == "classify_intent":
                result = self._tool.call("classify_intent", context)
            elif method == "extract_entities":
                result = self._tool.call("extract_entities", context)
            else:
                raise ValueError(f"Unknown method: {method}")

            elapsed = time.time() - start_time
            logger.info(f"[IntentClassificationHandler] call_tool completed, method={method}, elapsed={elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[IntentClassificationHandler] call_tool failed, method={method}, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def release(self) -> None:
        logger.info("[IntentClassificationHandler] release started")
        start_time = time.time()
        try:
            if self._tool is not None:
                self._tool.release_tool(None)
                self._tool = None
            elapsed = time.time() - start_time
            logger.info(f"[IntentClassificationHandler] release completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[IntentClassificationHandler] release failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise
