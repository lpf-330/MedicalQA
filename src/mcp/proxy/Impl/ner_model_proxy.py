# -*- coding: utf-8 -*-
"""
NER模型MCP伪代理

直连调用NerModelTool，不走标准MCP协议。
"""

import logging
import time
from typing import Any, Dict, Optional

from src.mcp.proxy.interfaces import MCPFakeProxy
from src.mcp.proxy.data_classes import ToolInfo, ToolMethod, DirectConnectionInfo
from src.tools.ner_model_tool import NerModelTool
from src.utils.logger import log_arch_event

logger = logging.getLogger(__name__)


class NerModelProxy(MCPFakeProxy):

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        logger.info("[PROXY_INIT] NerModelProxy初始化: resource_config=managed_by_resource_pool")
        self._tool: Optional[NerModelTool] = None
        self._call_count = 0
        self._total_time = 0.0
        self._error_count = 0
        self._mock_responses: Dict[str, Any] = {}
        self._tool_info = ToolInfo(
            name="ner_model",
            description="医学命名实体识别工具",
            methods=[
                ToolMethod(
                    name="extract_entities",
                    description="医学实体提取",
                    params=[],
                    return_type=list
                )
            ]
        )

    def _init_tool(self) -> None:
        if self._tool is not None:
            logger.debug("[NerModelProxy] _init_tool skipped, tool already initialized")
            return

        logger.info("[NerModelProxy] _init_tool started")
        start_time = time.time()
        tool = None
        try:
            tool = NerModelTool()
            tool._init_resource()
            self._tool = tool
            log_arch_event(logger, component="NerModelProxy", stage="MCP", event="init_tool", status="success", design_id="ARCH-4.3", tool="NerModelTool")
            logger.info("[MCP_PROXY_INIT] type=FAKE, tool=NerModelTool")
            elapsed = time.time() - start_time
            logger.info(f"[NerModelProxy] _init_tool completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            if tool is not None:
                try:
                    tool.release_source()
                except Exception as cleanup_error:
                    logger.warning(f"[NerModelProxy] _init_tool cleanup failed: {cleanup_error}")
            self._tool = None
            elapsed = time.time() - start_time
            logger.error(f"[NerModelProxy] _init_tool failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def release_tool(self, tool=None) -> None:
        logger.info("[NerModelProxy] release_tool started")
        start_time = time.time()
        try:
            if self._tool is not None:
                self._tool.release_source()
                self._tool = None
            log_arch_event(logger, component="NerModelProxy", stage="MCP", event="release_tool", status="success", design_id="ARCH-4.3", tool="NerModelTool")
            elapsed = time.time() - start_time
            logger.info(f"[NerModelProxy] release_tool completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NerModelProxy] release_tool failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def call(self, method_name: str, params: Dict[str, Any]) -> Any:
        if method_name in self._mock_responses:
            logger.debug(f"[NerModelProxy] call using mock response, method_name={method_name}")
            return self._mock_responses[method_name]

        if self._tool is None:
            raise RuntimeError("Tool not initialized, call _init_tool first")

        logger.debug(f"[NerModelProxy] call started, method_name={method_name}, params_keys={list(params.keys())}")
        start_time = time.time()
        try:
            tool_params = {k: v for k, v in params.items() if k != "method"}

            if method_name == "extract_entities":
                result = self._tool.extract_entities(**tool_params)
            else:
                raise AttributeError(f"Method {method_name} not found")

            self._call_count += 1
            elapsed = time.time() - start_time
            log_arch_event(logger, component="NerModelProxy", stage="MCP", event="proxy_call", status="success", design_id="ARCH-4.3", method_name=method_name, elapsed=f"{elapsed:.3f}s")
            logger.info(f"[NerModelProxy] call completed, method_name={method_name}, elapsed={elapsed:.3f}s")
            return result
        except Exception as e:
            self._error_count += 1
            elapsed = time.time() - start_time
            logger.error(f"[NerModelProxy] call failed, method_name={method_name}, elapsed={elapsed:.3f}s, error={str(e)}")
            raise e
        finally:
            self._total_time += time.time() - start_time

    def get_tool_info(self) -> ToolInfo:
        return self._tool_info

    def get_direct_connection_info(self) -> DirectConnectionInfo:
        return DirectConnectionInfo(type="local", endpoint="local_tool_instance")

    def set_mock_response(self, method_name: str, response: Any) -> None:
        self._mock_responses[method_name] = response

    def is_available(self) -> bool:
        return self._tool is not None

    def get_metrics(self) -> Dict[str, Any]:
        avg_time = self._total_time / self._call_count if self._call_count > 0 else 0
        error_rate = self._error_count / self._call_count if self._call_count > 0 else 0
        return {
            "tool_type": "ner_model",
            "available": self.is_available(),
            "call_count": self._call_count,
            "error_count": self._error_count,
            "average_response_time": avg_time,
            "error_rate": error_rate,
            "total_time": self._total_time
        }
