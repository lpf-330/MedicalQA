# -*- coding: utf-8 -*-

import logging
import time
from typing import Any, Dict, Optional

from src.mcp.proxy.interfaces import MCPFakeProxy
from src.mcp.proxy.data_classes import ToolInfo, ToolMethod, DirectConnectionInfo
from src.tools.vector_retrieval_tool import VectorEnhancedRetrievalTool

logger = logging.getLogger(__name__)


class MilvusMedicalProxy(MCPFakeProxy):

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._tool: Optional[VectorEnhancedRetrievalTool] = None
        self._call_count = 0
        self._total_time = 0.0
        self._error_count = 0
        self._mock_responses: Dict[str, Any] = {}
        self._tool_info = ToolInfo(
            name="milvus_medical",
            description="Milvus向量检索工具",
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
            logger.debug("[MilvusMedicalProxy] _init_tool skipped, tool already initialized")
            return

        logger.info("[MilvusMedicalProxy] _init_tool started")
        start_time = time.time()
        try:
            self._tool = VectorEnhancedRetrievalTool(
                milvus_uri=self._config.get("milvus_uri"),
                milvus_user=self._config.get("milvus_user"),
                milvus_password=self._config.get("milvus_password"),
                milvus_token=self._config.get("milvus_token", ""),
                vector_model_path=self._config.get("vector_model_path", ""),
                vector_device=self._config.get("vector_device", "cpu"),
                vector_dimension=self._config.get("vector_dimension", 1024),
                fusion_threshold=self._config.get("fusion_threshold", 0.6),
                entity_weight=self._config.get("entity_weight", 0.40),
                attribute_weight=self._config.get("attribute_weight", 0.30),
                relation_weight=self._config.get("relation_weight", 0.30)
            )
            self._tool._init_resource()
            elapsed = time.time() - start_time
            logger.info(f"[MilvusMedicalProxy] _init_tool completed, elapsed={elapsed:.3f}s, milvus_uri={self._config.get('milvus_uri')}")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[MilvusMedicalProxy] _init_tool failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def release_tool(self, tool=None) -> None:
        logger.info("[MilvusMedicalProxy] release_tool started")
        start_time = time.time()
        try:
            if self._tool is not None:
                self._tool.release_source()
                self._tool = None
            elapsed = time.time() - start_time
            logger.info(f"[MilvusMedicalProxy] release_tool completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[MilvusMedicalProxy] release_tool failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def call(self, method_name: str, params: Dict[str, Any]) -> Any:
        if method_name in self._mock_responses:
            logger.debug(f"[MilvusMedicalProxy] call using mock response, method_name={method_name}")
            return self._mock_responses[method_name]

        if self._tool is None:
            raise RuntimeError("Tool not initialized, call _init_tool first")

        logger.debug(f"[MilvusMedicalProxy] call started, method_name={method_name}, params_keys={list(params.keys())}")
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
            logger.info(f"[MilvusMedicalProxy] call completed, method_name={method_name}, elapsed={elapsed:.3f}s, result_count={len(result) if isinstance(result, list) else 'N/A'}")
            return result
        except Exception as e:
            self._error_count += 1
            elapsed = time.time() - start_time
            logger.error(f"[MilvusMedicalProxy] call failed, method_name={method_name}, elapsed={elapsed:.3f}s, error={str(e)}")
            raise e
        finally:
            self._total_time += time.time() - start_time

    def get_tool_info(self) -> ToolInfo:
        return self._tool_info

    def get_direct_connection_info(self) -> DirectConnectionInfo:
        logger.debug("[MilvusMedicalProxy] get_direct_connection_info called")
        return DirectConnectionInfo(
            type="milvus",
            endpoint=self._config.get("milvus_uri", "local_tool_instance")
        )

    def set_mock_response(self, method_name: str, response: Any) -> None:
        self._mock_responses[method_name] = response

    def is_available(self) -> bool:
        available = self._tool is not None
        logger.debug(f"[MilvusMedicalProxy] is_available={available}")
        return available

    def get_metrics(self) -> Dict[str, Any]:
        logger.debug("[MilvusMedicalProxy] get_metrics called")
        avg_time = self._total_time / self._call_count if self._call_count > 0 else 0
        error_rate = self._error_count / self._call_count if self._call_count > 0 else 0

        return {
            "tool_type": "milvus_medical",
            "available": self.is_available(),
            "call_count": self._call_count,
            "error_count": self._error_count,
            "average_response_time": avg_time,
            "error_rate": error_rate,
            "total_time": self._total_time
        }
