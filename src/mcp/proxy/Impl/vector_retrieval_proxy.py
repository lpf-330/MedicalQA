# -*- coding: utf-8 -*-

import logging
import time
from typing import Any, Dict, Optional

from src.mcp.proxy.interfaces import MCPFakeProxy
from src.mcp.proxy.data_classes import ToolInfo, ToolMethod, DirectConnectionInfo
from src.tools.vector_retrieval_tool import VectorRetrievalTool
from src.utils.logger import log_arch_event

logger = logging.getLogger(__name__)


class VectorRetrievalProxy(MCPFakeProxy):

    def __init__(self, config: Dict[str, Any]):
        self._config = {
            "fusion_threshold": config.get("fusion_threshold", 0.6),
            "entity_weight": config.get("entity_weight", 0.40),
            "attribute_weight": config.get("attribute_weight", 0.30),
            "relation_weight": config.get("relation_weight", 0.30),
        }
        logger.info("[PROXY_INIT] VectorRetrievalProxy初始化: resource_config=managed_by_resource_pool")
        self._tool: Optional[VectorRetrievalTool] = None
        self._call_count = 0
        self._total_time = 0.0
        self._error_count = 0
        self._mock_responses: Dict[str, Any] = {}
        self._tool_info = ToolInfo(
            name="vector_retrieval",
            description="向量增强检索工具",
            methods=[
                ToolMethod(
                    name="hybrid_search",
                    description="三集合混合检索",
                    params=[],
                    return_type=list
                ),
                ToolMethod(
                    name="search_entities",
                    description="单集合检索medical_entity",
                    params=[],
                    return_type=list
                ),
                ToolMethod(
                    name="search_attributes",
                    description="单集合检索entity_attributes",
                    params=[],
                    return_type=list
                ),
                ToolMethod(
                    name="search_relations",
                    description="单集合检索entity_relations",
                    params=[],
                    return_type=list
                )
            ]
        )

    def _init_tool(self) -> None:
        if self._tool is not None:
            logger.debug("[VectorRetrievalProxy] _init_tool skipped, tool already initialized")
            return

        logger.info("[VectorRetrievalProxy] _init_tool started")
        start_time = time.time()
        tool = None
        try:
            tool = VectorRetrievalTool(
                fusion_threshold=self._config.get("fusion_threshold", 0.6),
                entity_weight=self._config.get("entity_weight", 0.40),
                attribute_weight=self._config.get("attribute_weight", 0.30),
                relation_weight=self._config.get("relation_weight", 0.30)
            )
            tool._init_resource()
            self._tool = tool
            log_arch_event(logger, component="VectorRetrievalProxy", stage="MCP", event="init_tool", status="success", design_id="ARCH-4.3", tool="VectorRetrievalTool")
            logger.info("[MCP_PROXY_INIT] type=FAKE, tool=VectorRetrievalTool")
            elapsed = time.time() - start_time
            logger.info(f"[VectorRetrievalProxy] _init_tool completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            if tool is not None:
                try:
                    tool.release_source()
                except Exception as cleanup_error:
                    logger.warning(f"[VectorRetrievalProxy] _init_tool cleanup failed: error_type={type(cleanup_error).__name__}")
            self._tool = None
            elapsed = time.time() - start_time
            logger.error(f"[VectorRetrievalProxy] _init_tool failed, elapsed={elapsed:.3f}s, error_type={type(e).__name__}")
            raise

    def release_tool(self, tool=None) -> None:
        logger.info("[VectorRetrievalProxy] release_tool started")
        start_time = time.time()
        try:
            if self._tool is not None:
                self._tool.release_source()
                self._tool = None
            log_arch_event(logger, component="VectorRetrievalProxy", stage="MCP", event="release_tool", status="success", design_id="ARCH-4.3", tool="VectorRetrievalTool")
            elapsed = time.time() - start_time
            logger.info(f"[VectorRetrievalProxy] release_tool completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VectorRetrievalProxy] release_tool failed, elapsed={elapsed:.3f}s, error_type={type(e).__name__}")
            raise

    def call(self, method_name: str, params: Dict[str, Any]) -> Any:
        if method_name in self._mock_responses:
            logger.debug(f"[VectorRetrievalProxy] call using mock response, method_name={method_name}")
            return self._mock_responses[method_name]

        if self._tool is None:
            raise RuntimeError("Tool not initialized, call _init_tool first")

        logger.debug(f"[VectorRetrievalProxy] call started, method_name={method_name}, params_keys={list(params.keys())}")
        start_time = time.time()
        try:
            tool_params = {k: v for k, v in params.items() if k != "method"}
            
            if method_name == "hybrid_search":
                result = self._tool.hybrid_search(**tool_params)
            elif method_name == "search_entities":
                result = self._tool.search_entities(**tool_params)
            elif method_name == "search_attributes":
                result = self._tool.search_attributes(**tool_params)
            elif method_name == "search_relations":
                result = self._tool.search_relations(**tool_params)
            else:
                raise AttributeError(f"Method {method_name} not found")

            self._call_count += 1
            elapsed = time.time() - start_time
            log_arch_event(logger, component="VectorRetrievalProxy", stage="MCP", event="proxy_call", status="success", design_id="ARCH-4.3", method_name=method_name, elapsed=f"{elapsed:.3f}s")
            logger.info(f"[VectorRetrievalProxy] call completed, method_name={method_name}, elapsed={elapsed:.3f}s, result_count={len(result) if isinstance(result, list) else 'N/A'}")
            return result
        except Exception as e:
            self._error_count += 1
            elapsed = time.time() - start_time
            logger.error(f"[VectorRetrievalProxy] call failed, method_name={method_name}, elapsed={elapsed:.3f}s, error_type={type(e).__name__}")
            raise e
        finally:
            self._total_time += time.time() - start_time

    def get_tool_info(self) -> ToolInfo:
        return self._tool_info

    def get_direct_connection_info(self) -> DirectConnectionInfo:
        logger.debug("[VectorRetrievalProxy] get_direct_connection_info called")
        return DirectConnectionInfo(
            type="local",
            endpoint="local_tool_instance"
        )

    def set_mock_response(self, method_name: str, response: Any) -> None:
        self._mock_responses[method_name] = response

    def is_available(self) -> bool:
        available = self._tool is not None
        logger.debug(f"[VectorRetrievalProxy] is_available={available}")
        return available

    def get_metrics(self) -> Dict[str, Any]:
        logger.debug("[VectorRetrievalProxy] get_metrics called")
        avg_time = self._total_time / self._call_count if self._call_count > 0 else 0
        error_rate = self._error_count / self._call_count if self._call_count > 0 else 0

        return {
            "tool_type": "vector_retrieval",
            "available": self.is_available(),
            "call_count": self._call_count,
            "error_count": self._error_count,
            "average_response_time": avg_time,
            "error_rate": error_rate,
            "total_time": self._total_time
        }
