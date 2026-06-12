# -*- coding: utf-8 -*-

import logging
import time
from typing import Any, Dict, Optional

from src.orchestration.tool_call_handler.tool_call_handler import ToolCallHandler
from src.mcp.proxy.interfaces import MCPTool
from src.mcp.factory.mcp_proxy_factory import MCPProxyFactory

logger = logging.getLogger(__name__)


class IntentClassificationHandler(ToolCallHandler[Dict, Dict[str, Any]]):

    def __init__(self, tool_proxy_instance_id: str = "intent_classification"):
        self._tool: Optional[MCPTool] = None
        self._tool_proxy_instance_id = tool_proxy_instance_id

    def _init_tool(self, tool: MCPTool) -> None:
        logger.info(f"[TOOL_HANDLER_INIT] handler=IntentClassificationHandler, event=_init_tool, already_initialized={self._tool is not None}")
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

    def is_initialized(self) -> bool:
        return self._tool is not None

    def _ensure_initialized(self) -> None:
        if self._tool is not None:
            logger.info("[TOOL_HANDLER_INIT] handler=IntentClassificationHandler, event=_ensure_initialized, auto_reinit=False, already_initialized=True")
            return

        logger.info("[IntentClassificationHandler] Tool not initialized, auto-reinitializing...")
        logger.info("[TOOL_HANDLER_INIT] handler=IntentClassificationHandler, event=_ensure_initialized, auto_reinit=True, already_initialized=False")
        try:
            factory = MCPProxyFactory.get_instance()
            tool = factory.get_tool_proxy_instance(self._tool_proxy_instance_id)
            self._tool = tool
            logger.info(f"[IntentClassificationHandler] Auto-reinitialization completed using factory, tool_proxy_instance_id={self._tool_proxy_instance_id}")
        except Exception as e:
            logger.error(f"[IntentClassificationHandler] Auto-reinitialization failed: {str(e)}")
            raise

    def call_tool(self, context: Dict) -> Dict[str, Any]:
        logger.info(f"[TOOL_HANDLER_CALL] {self.__class__.__name__}调用工具")
        method = context.get("method")
        logger.debug(f"[IntentClassificationHandler] call_tool called, method={method}, context_keys={list(context.keys())}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            auto_init = self._tool is not None
            logger.info(f"[TOOL_CALL] handler=IntentClassificationHandler, method={method}, auto_init={auto_init}")

            if method == "classify_intent":
                input_text = context.get("text", context.get("query", ""))
                logger.info(f"[INTENT_CLASSIFY_INPUT] method=classify_intent, text_length={len(input_text) if isinstance(input_text, str) else 'N/A'}, context_keys={list(context.keys())}")

                result = self._tool.call("classify_intent", context)

                intent_label = result.get("intent_label", "N/A") if isinstance(result, dict) else "N/A"
                confidence = result.get("confidence", "N/A") if isinstance(result, dict) else "N/A"
                logger.info(f"[INTENT_CLASSIFY_RESULT] method=classify_intent, intent_label={intent_label}, confidence={confidence}")

            elif method == "extract_entities":
                result = self._tool.call("extract_entities", context)

                entity_count = len(result.get("entities", [])) if isinstance(result, dict) else (len(result) if isinstance(result, list) else "N/A")
                logger.info(f"[ENTITY_EXTRACT] method=extract_entities, entity_count={entity_count}")

            else:
                raise ValueError(f"Unknown method: {method}")

            elapsed = time.time() - start_time
            logger.info(f"[IntentClassificationHandler] call_tool completed, method={method}, elapsed={elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[INTENT_CLASSIFY_ERROR] method={method}, elapsed={elapsed:.3f}s, error_type={type(e).__name__}, error={str(e)}")
            logger.error(f"[IntentClassificationHandler] call_tool failed, method={method}, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def release_tool(self) -> None:
        """释放tool功能实例引用 — 只清除引用，不释放MCP代理实例

        设计依据：2.3.3节 ToolCallHandler生命周期管理规范
        release_tool()语义：将工具引用设置为None，不释放MCP代理实例
        """
        self._tool = None
