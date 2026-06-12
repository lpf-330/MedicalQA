# -*- coding: utf-8 -*-
"""
NER模型Handler

为编排层提供NER实体提取的tool调用服务。
"""

import logging
import time
from typing import Any, Dict, Optional

from src.orchestration.tool_call_handler.tool_call_handler import ToolCallHandler
from src.mcp.proxy.interfaces import MCPTool
from src.mcp.factory.mcp_proxy_factory import MCPProxyFactory

logger = logging.getLogger(__name__)


class NerModelHandler(ToolCallHandler[Dict, Dict[str, Any]]):

    def __init__(self, tool_proxy_instance_id: str = "ner_model"):
        self._tool: Optional[MCPTool] = None
        self._tool_proxy_instance_id = tool_proxy_instance_id

    def _init_tool(self, tool: MCPTool) -> None:
        logger.info(f"[TOOL_HANDLER_INIT] handler=NerModelHandler, event=_init_tool, already_initialized={self._tool is not None}")
        start_time = time.time()
        try:
            self._tool = tool
            tool._init_tool()
            elapsed = time.time() - start_time
            logger.info(f"[NerModelHandler] _init_tool completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NerModelHandler] _init_tool failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def is_initialized(self) -> bool:
        return self._tool is not None

    def _ensure_initialized(self) -> None:
        if self._tool is not None:
            return

        logger.info("[NerModelHandler] Tool not initialized, auto-reinitializing...")
        try:
            factory = MCPProxyFactory.get_instance()
            tool = factory.get_tool_proxy_instance(self._tool_proxy_instance_id)
            self._tool = tool
            logger.info(f"[NerModelHandler] Auto-reinitialization completed, tool_proxy_instance_id={self._tool_proxy_instance_id}")
        except Exception as e:
            logger.error(f"[NerModelHandler] Auto-reinitialization failed: {str(e)}")
            raise

    def call_tool(self, context: Dict) -> Dict[str, Any]:
        method = context.get("method", "extract_entities")
        logger.debug(f"[NerModelHandler] call_tool called, method={method}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            result = self._tool.call(method, context)
            entity_count = len(result) if isinstance(result, list) else (len(result.get("entities", [])) if isinstance(result, dict) else "N/A")
            logger.info(f"[ENTITY_EXTRACT] handler=NerModelHandler, method={method}, entity_count={entity_count}")
            elapsed = time.time() - start_time
            logger.info(f"[NerModelHandler] call_tool completed, method={method}, elapsed={elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NerModelHandler] call_tool failed, method={method}, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def release_tool(self) -> None:
        """释放tool功能实例引用 — 只清除引用，不释放MCP代理实例

        设计依据：2.3.3节 ToolCallHandler生命周期管理规范
        release_tool()语义：将工具引用设置为None，不释放MCP代理实例
        """
        self._tool = None
