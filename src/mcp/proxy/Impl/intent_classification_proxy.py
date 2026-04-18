# -*- coding: utf-8 -*-

import logging
import time
from typing import Any, Dict, Optional

from src.mcp.proxy.interfaces import MCPFakeProxy
from src.mcp.proxy.data_classes import ToolInfo, ToolMethod, DirectConnectionInfo
from src.tools.intent_classification_tool import IntentClassificationTool

logger = logging.getLogger(__name__)


class IntentClassificationProxy(MCPFakeProxy):

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._tool: Optional[IntentClassificationTool] = None
        self._call_count = 0
        self._total_time = 0.0
        self._error_count = 0
        self._mock_responses: Dict[str, Any] = {}
        self._tool_info = ToolInfo(
            name="intent_classification",
            description="意图分类工具",
            methods=[
                ToolMethod(
                    name="classify_intent",
                    description="意图分类",
                    params=[],
                    return_type=dict
                ),
                ToolMethod(
                    name="extract_entities",
                    description="实体抽取",
                    params=[],
                    return_type=list
                )
            ]
        )

    def _init_tool(self) -> None:
        if self._tool is not None:
            logger.debug("[IntentClassificationProxy] _init_tool skipped, tool already initialized")
            return

        logger.info("[IntentClassificationProxy] _init_tool started")
        start_time = time.time()
        try:
            self._tool = IntentClassificationTool(
                model_path=self._config.get("model_path"),
                device=self._config.get("device", "cpu"),
                max_length=self._config.get("max_length", 128)
            )
            self._tool._init_resource()
            elapsed = time.time() - start_time
            logger.info(f"[IntentClassificationProxy] _init_tool completed, elapsed={elapsed:.3f}s, model_path={self._config.get('model_path')}")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[IntentClassificationProxy] _init_tool failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def release_tool(self, tool=None) -> None:
        logger.info("[IntentClassificationProxy] release_tool started")
        start_time = time.time()
        try:
            if self._tool is not None:
                self._tool.release_source()
                self._tool = None
            elapsed = time.time() - start_time
            logger.info(f"[IntentClassificationProxy] release_tool completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[IntentClassificationProxy] release_tool failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def call(self, method_name: str, params: Dict[str, Any]) -> Any:
        if method_name in self._mock_responses:
            logger.debug(f"[IntentClassificationProxy] call using mock response, method_name={method_name}")
            return self._mock_responses[method_name]

        if self._tool is None:
            raise RuntimeError("Tool not initialized, call _init_tool first")

        logger.debug(f"[IntentClassificationProxy] call started, method_name={method_name}, params_keys={list(params.keys())}")
        start_time = time.time()
        try:
            tool_params = {k: v for k, v in params.items() if k != "method"}
            
            if method_name == "classify_intent":
                result = self._tool.classify_intent(**tool_params)
            elif method_name == "extract_entities":
                result = self._tool.extract_entities(**tool_params)
            else:
                raise AttributeError(f"Method {method_name} not found")

            self._call_count += 1
            elapsed = time.time() - start_time
            logger.info(f"[IntentClassificationProxy] call completed, method_name={method_name}, elapsed={elapsed:.3f}s")
            return result
        except Exception as e:
            self._error_count += 1
            elapsed = time.time() - start_time
            logger.error(f"[IntentClassificationProxy] call failed, method_name={method_name}, elapsed={elapsed:.3f}s, error={str(e)}")
            raise e
        finally:
            self._total_time += time.time() - start_time

    def get_tool_info(self) -> ToolInfo:
        return self._tool_info

    def get_direct_connection_info(self) -> DirectConnectionInfo:
        logger.debug("[IntentClassificationProxy] get_direct_connection_info called")
        return DirectConnectionInfo(
            type="intent_model",
            endpoint=self._config.get("model_path", "local_tool_instance")
        )

    def set_mock_response(self, method_name: str, response: Any) -> None:
        self._mock_responses[method_name] = response

    def is_available(self) -> bool:
        available = self._tool is not None
        logger.debug(f"[IntentClassificationProxy] is_available={available}")
        return available

    def get_metrics(self) -> Dict[str, Any]:
        logger.debug("[IntentClassificationProxy] get_metrics called")
        avg_time = self._total_time / self._call_count if self._call_count > 0 else 0
        error_rate = self._error_count / self._call_count if self._call_count > 0 else 0

        return {
            "tool_type": "intent_classification",
            "available": self.is_available(),
            "call_count": self._call_count,
            "error_count": self._error_count,
            "average_response_time": avg_time,
            "error_rate": error_rate,
            "total_time": self._total_time
        }
