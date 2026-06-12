# -*- coding: utf-8 -*-

import logging
import time
from typing import Any, Dict, List, Optional

from src.orchestration.tool_call_handler.tool_call_handler import ToolCallHandler
from src.mcp.proxy.interfaces import MCPTool
from src.mcp.factory.mcp_proxy_factory import MCPProxyFactory

logger = logging.getLogger(__name__)


class Neo4jMedicalHandler(ToolCallHandler[Any, Any]):

    def __init__(self, tool_proxy_instance_id: str = "neo4j_medical"):
        self._tool: Optional[MCPTool] = None
        self._tool_proxy_instance_id = tool_proxy_instance_id

    def _init_tool(self, tool: MCPTool) -> None:
        logger.info("[Neo4jMedicalHandler] _init_tool started")
        logger.info(f"[TOOL_HANDLER_INIT] handler=Neo4jMedicalHandler, event=_init_tool, already_initialized={self._tool is not None}")
        start_time = time.time()
        try:
            self._tool = tool
            tool._init_tool()
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] _init_tool completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] _init_tool failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def is_initialized(self) -> bool:
        return self._tool is not None

    def _ensure_initialized(self) -> None:
        if self._tool is not None:
            logger.info("[TOOL_HANDLER_INIT] handler=Neo4jMedicalHandler, event=_ensure_initialized, auto_reinit=False, already_initialized=True")
            return

        logger.info("[Neo4jMedicalHandler] Tool not initialized, auto-reinitializing...")
        logger.info("[TOOL_HANDLER_INIT] handler=Neo4jMedicalHandler, event=_ensure_initialized, auto_reinit=True, already_initialized=False")
        try:
            factory = MCPProxyFactory.get_instance()
            tool = factory.get_tool_proxy_instance(self._tool_proxy_instance_id)
            self._tool = tool
            logger.info(f"[Neo4jMedicalHandler] Auto-reinitialization completed using factory, tool_proxy_instance_id={self._tool_proxy_instance_id}")
        except Exception as e:
            logger.error(f"[Neo4jMedicalHandler] Auto-reinitialization failed: {str(e)}")
            raise

    # 正向查询方法：以疾病名称为参数
    _FORWARD_METHODS = frozenset({
        "get_disease_info", "get_symptoms_by_disease", "get_drugs_by_disease",
        "get_foods_by_disease", "get_checks_by_disease", "get_department_by_disease",
        "get_cure_methods_by_disease", "get_complications_by_disease",
    })

    # 反向查询方法：以node_id为参数（Milvus返回elementId字符串，直接传递）
    _NODE_ID_METHODS = frozenset({
        "get_diseases_by_drug_node_id", "get_diseases_by_food_node_id",
        "get_diseases_by_check_node_id", "get_diseases_by_department_node_id",
        "get_diseases_by_cure_node_id", "get_diseases_by_symptom_node_id",
    })

    def call_tool(self, context: Any) -> Any:
        logger.info(f"[TOOL_HANDLER_CALL] {self.__class__.__name__}调用工具")
        context_keys = list(context.keys()) if isinstance(context, dict) else []
        logger.debug(f"[Neo4jMedicalHandler] call_tool called, context_type={type(context).__name__}, context_keys={context_keys}")
        start_time = time.time()
        method = "get_disease_info"
        params = {"disease_name": context}
        try:
            if isinstance(context, dict):
                method = context.get("method", "get_disease_info")
                if method in self._FORWARD_METHODS:
                    disease_name = context.get("disease_name", context.get("entity_name", ""))
                    params = {"disease_name": disease_name}
                elif method == "search_diseases_by_symptom":
                    params = {"symptom_name": context.get("symptom_name", context.get("entity_name", ""))}
                    if "limit" in context:
                        params["limit"] = context["limit"]
                elif method in self._NODE_ID_METHODS:
                    raw_id = context.get("node_id", "")
                    params = {"node_id": raw_id}
                    if "limit" in context:
                        params["limit"] = context["limit"]
                else:
                    raise ValueError(f"Unknown method: {method}")

            self._ensure_initialized()
            auto_init = self._tool is not None
            logger.info(f"[TOOL_CALL] handler=Neo4jMedicalHandler, method={method}, auto_init={auto_init}")

            result = self._tool.call(method, params)
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] call_tool completed, method={method}, elapsed={elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] call_tool failed, method={method}, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def get_disease_info(self, disease_name: str) -> Optional[Dict[str, Any]]:
        logger.debug(f"[Neo4jMedicalHandler] get_disease_info called, disease_name={disease_name}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            result = self._tool.call("get_disease_info", {"disease_name": disease_name})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_disease_info completed, elapsed={elapsed:.3f}s, disease_name={disease_name}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_disease_info failed, elapsed={elapsed:.3f}s, disease_name={disease_name}, error={str(e)}")
            raise

    def get_symptoms_by_disease(self, disease_name: str) -> List[str]:
        logger.debug(f"[Neo4jMedicalHandler] get_symptoms_by_disease called, disease_name={disease_name}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            result = self._tool.call("get_symptoms_by_disease", {"disease_name": disease_name})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_symptoms_by_disease completed, elapsed={elapsed:.3f}s, disease_name={disease_name}, symptom_count={len(result) if isinstance(result, list) else 'N/A'}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_symptoms_by_disease failed, elapsed={elapsed:.3f}s, disease_name={disease_name}, error={str(e)}")
            raise

    def get_drugs_by_disease(self, disease_name: str) -> Dict[str, List[str]]:
        logger.debug(f"[Neo4jMedicalHandler] get_drugs_by_disease called, disease_name={disease_name}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            result = self._tool.call("get_drugs_by_disease", {"disease_name": disease_name})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_drugs_by_disease completed, elapsed={elapsed:.3f}s, disease_name={disease_name}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_drugs_by_disease failed, elapsed={elapsed:.3f}s, disease_name={disease_name}, error={str(e)}")
            raise

    def get_foods_by_disease(self, disease_name: str) -> Dict[str, List[str]]:
        logger.debug(f"[Neo4jMedicalHandler] get_foods_by_disease called, disease_name={disease_name}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            result = self._tool.call("get_foods_by_disease", {"disease_name": disease_name})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_foods_by_disease completed, elapsed={elapsed:.3f}s, disease_name={disease_name}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_foods_by_disease failed, elapsed={elapsed:.3f}s, disease_name={disease_name}, error={str(e)}")
            raise

    def get_disease_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        logger.debug(f"[Neo4jMedicalHandler] get_disease_by_node_id called, node_id={node_id}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            result = self._tool.call("get_disease_by_node_id", {"node_id": node_id})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_disease_by_node_id completed, elapsed={elapsed:.3f}s, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_disease_by_node_id failed, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise

    def get_symptom_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        logger.debug(f"[Neo4jMedicalHandler] get_symptom_by_node_id called, node_id={node_id}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            result = self._tool.call("get_symptom_by_node_id", {"node_id": node_id})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_symptom_by_node_id completed, elapsed={elapsed:.3f}s, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_symptom_by_node_id failed, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise

    def get_diseases_by_symptom_node_id(self, node_id: str) -> List[str]:
        logger.debug(f"[Neo4jMedicalHandler] get_diseases_by_symptom_node_id called, node_id={node_id}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            result = self._tool.call("get_diseases_by_symptom_node_id", {"node_id": node_id})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_diseases_by_symptom_node_id completed, elapsed={elapsed:.3f}s, node_id={node_id}, disease_count={len(result) if isinstance(result, list) else 'N/A'}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_diseases_by_symptom_node_id failed, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise

    def get_symptoms_by_node_id(self, node_id: str) -> List[str]:
        logger.debug(f"[Neo4jMedicalHandler] get_symptoms_by_node_id called, node_id={node_id}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            result = self._tool.call("get_symptoms_by_node_id", {"node_id": node_id})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_symptoms_by_node_id completed, elapsed={elapsed:.3f}s, node_id={node_id}, symptom_count={len(result) if isinstance(result, list) else 'N/A'}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_symptoms_by_node_id failed, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise

    def get_drugs_by_node_id(self, node_id: str) -> Dict[str, List[str]]:
        logger.debug(f"[Neo4jMedicalHandler] get_drugs_by_node_id called, node_id={node_id}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            result = self._tool.call("get_drugs_by_node_id", {"node_id": node_id})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_drugs_by_node_id completed, elapsed={elapsed:.3f}s, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_drugs_by_node_id failed, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise

    def get_foods_by_node_id(self, node_id: str) -> Dict[str, List[str]]:
        logger.debug(f"[Neo4jMedicalHandler] get_foods_by_node_id called, node_id={node_id}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            result = self._tool.call("get_foods_by_node_id", {"node_id": node_id})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_foods_by_node_id completed, elapsed={elapsed:.3f}s, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_foods_by_node_id failed, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise

    def get_cure_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        logger.debug(f"[Neo4jMedicalHandler] get_cure_by_node_id called, node_id={node_id}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            result = self._tool.call("get_cure_by_node_id", {"node_id": node_id})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_cure_by_node_id completed, elapsed={elapsed:.3f}s, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_cure_by_node_id failed, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise

    def get_diseases_by_cure_node_id(self, node_id: str) -> List[str]:
        logger.debug(f"[Neo4jMedicalHandler] get_diseases_by_cure_node_id called, node_id={node_id}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            result = self._tool.call("get_diseases_by_cure_node_id", {"node_id": node_id})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_diseases_by_cure_node_id completed, elapsed={elapsed:.3f}s, node_id={node_id}, disease_count={len(result) if isinstance(result, list) else 'N/A'}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_diseases_by_cure_node_id failed, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise

    def get_drug_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        logger.debug(f"[Neo4jMedicalHandler] get_drug_by_node_id called, node_id={node_id}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            result = self._tool.call("get_drug_by_node_id", {"node_id": node_id})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_drug_by_node_id completed, elapsed={elapsed:.3f}s, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_drug_by_node_id failed, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise

    def get_diseases_by_drug_node_id(self, node_id: str) -> Dict[str, List[str]]:
        logger.debug(f"[Neo4jMedicalHandler] get_diseases_by_drug_node_id called, node_id={node_id}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            result = self._tool.call("get_diseases_by_drug_node_id", {"node_id": node_id})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_diseases_by_drug_node_id completed, elapsed={elapsed:.3f}s, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_diseases_by_drug_node_id failed, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise

    def get_food_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        logger.debug(f"[Neo4jMedicalHandler] get_food_by_node_id called, node_id={node_id}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            result = self._tool.call("get_food_by_node_id", {"node_id": node_id})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_food_by_node_id completed, elapsed={elapsed:.3f}s, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_food_by_node_id failed, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise

    def get_diseases_by_food_node_id(self, node_id: str) -> Dict[str, List[str]]:
        logger.debug(f"[Neo4jMedicalHandler] get_diseases_by_food_node_id called, node_id={node_id}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            result = self._tool.call("get_diseases_by_food_node_id", {"node_id": node_id})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_diseases_by_food_node_id completed, elapsed={elapsed:.3f}s, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_diseases_by_food_node_id failed, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise

    def get_check_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        logger.debug(f"[Neo4jMedicalHandler] get_check_by_node_id called, node_id={node_id}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            result = self._tool.call("get_check_by_node_id", {"node_id": node_id})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_check_by_node_id completed, elapsed={elapsed:.3f}s, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_check_by_node_id failed, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise

    def get_diseases_by_check_node_id(self, node_id: str) -> List[str]:
        logger.debug(f"[Neo4jMedicalHandler] get_diseases_by_check_node_id called, node_id={node_id}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            result = self._tool.call("get_diseases_by_check_node_id", {"node_id": node_id})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_diseases_by_check_node_id completed, elapsed={elapsed:.3f}s, node_id={node_id}, disease_count={len(result) if isinstance(result, list) else 'N/A'}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_diseases_by_check_node_id failed, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise

    def get_department_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        logger.debug(f"[Neo4jMedicalHandler] get_department_by_node_id called, node_id={node_id}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            result = self._tool.call("get_department_by_node_id", {"node_id": node_id})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_department_by_node_id completed, elapsed={elapsed:.3f}s, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_department_by_node_id failed, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise

    def get_diseases_by_department_node_id(self, node_id: str) -> List[str]:
        logger.debug(f"[Neo4jMedicalHandler] get_diseases_by_department_node_id called, node_id={node_id}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            result = self._tool.call("get_diseases_by_department_node_id", {"node_id": node_id})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_diseases_by_department_node_id completed, elapsed={elapsed:.3f}s, node_id={node_id}, disease_count={len(result) if isinstance(result, list) else 'N/A'}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_diseases_by_department_node_id failed, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise

    def search_diseases_by_symptom(self, symptom_name: str) -> List[str]:
        logger.debug(f"[Neo4jMedicalHandler] search_diseases_by_symptom called, symptom_name={symptom_name}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            result = self._tool.call("search_diseases_by_symptom", {"symptom_name": symptom_name})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] search_diseases_by_symptom completed, elapsed={elapsed:.3f}s, symptom_name={symptom_name}, disease_count={len(result) if isinstance(result, list) else 'N/A'}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] search_diseases_by_symptom failed, elapsed={elapsed:.3f}s, symptom_name={symptom_name}, error={str(e)}")
            raise

    def get_complications_by_disease(self, disease_name: str) -> List[str]:
        logger.debug(f"[Neo4jMedicalHandler] get_complications_by_disease called, disease_name={disease_name}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            result = self._tool.call("get_complications_by_disease", {"disease_name": disease_name})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_complications_by_disease completed, elapsed={elapsed:.3f}s, disease_name={disease_name}, complication_count={len(result) if isinstance(result, list) else 'N/A'}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_complications_by_disease failed, elapsed={elapsed:.3f}s, disease_name={disease_name}, error={str(e)}")
            raise

    def get_cure_methods_by_disease(self, disease_name: str) -> List[str]:
        logger.debug(f"[Neo4jMedicalHandler] get_cure_methods_by_disease called, disease_name={disease_name}")
        start_time = time.time()
        try:
            self._ensure_initialized()
            result = self._tool.call("get_cure_methods_by_disease", {"disease_name": disease_name})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_cure_methods_by_disease completed, elapsed={elapsed:.3f}s, disease_name={disease_name}, method_count={len(result) if isinstance(result, list) else 'N/A'}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_cure_methods_by_disease failed, elapsed={elapsed:.3f}s, disease_name={disease_name}, error={str(e)}")
            raise

    def release_tool(self) -> None:
        """释放tool功能实例引用 — 只清除引用，不释放MCP代理实例

        设计依据：2.3.3节 ToolCallHandler生命周期管理规范
        release_tool()语义：将工具引用设置为None，不释放MCP代理实例
        """
        self._tool = None
