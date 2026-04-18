# -*- coding: utf-8 -*-

import logging
import time
from typing import Any, Dict, List, Optional

from src.orchestration.tool_call_handler.tool_call_handler import ToolCallHandler
from src.mcp.proxy.interfaces import MCPTool

logger = logging.getLogger(__name__)


class Neo4jMedicalHandler(ToolCallHandler[str, Dict[str, Any]]):

    def __init__(self):
        self._tool: Optional[MCPTool] = None

    def _init_tool(self, tool: MCPTool) -> None:
        logger.info("[Neo4jMedicalHandler] _init_tool started")
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

    def call_tool(self, context: str) -> Dict[str, Any]:
        logger.debug(f"[Neo4jMedicalHandler] call_tool called, context_length={len(context) if context else 0}")
        start_time = time.time()
        try:
            if self._tool is None:
                raise RuntimeError("Tool not initialized, call _init_tool first")

            result = self._tool.call("get_disease_info", {"disease_name": context})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] call_tool completed, elapsed={elapsed:.3f}s, disease_name={context}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] call_tool failed, elapsed={elapsed:.3f}s, disease_name={context}, error={str(e)}")
            raise

    def get_disease_info(self, disease_name: str) -> Optional[Dict[str, Any]]:
        logger.debug(f"[Neo4jMedicalHandler] get_disease_info called, disease_name={disease_name}")
        start_time = time.time()
        try:
            if self._tool is None:
                raise RuntimeError("Tool not initialized")
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
            if self._tool is None:
                raise RuntimeError("Tool not initialized")
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
            if self._tool is None:
                raise RuntimeError("Tool not initialized")
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
            if self._tool is None:
                raise RuntimeError("Tool not initialized")
            result = self._tool.call("get_foods_by_disease", {"disease_name": disease_name})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_foods_by_disease completed, elapsed={elapsed:.3f}s, disease_name={disease_name}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_foods_by_disease failed, elapsed={elapsed:.3f}s, disease_name={disease_name}, error={str(e)}")
            raise

    def get_disease_by_node_id(self, node_id: int) -> Optional[Dict[str, Any]]:
        logger.debug(f"[Neo4jMedicalHandler] get_disease_by_node_id called, node_id={node_id}")
        start_time = time.time()
        try:
            if self._tool is None:
                raise RuntimeError("Tool not initialized")
            result = self._tool.call("get_disease_by_node_id", {"node_id": node_id})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_disease_by_node_id completed, elapsed={elapsed:.3f}s, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_disease_by_node_id failed, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise

    def get_symptom_by_node_id(self, node_id: int) -> Optional[Dict[str, Any]]:
        logger.debug(f"[Neo4jMedicalHandler] get_symptom_by_node_id called, node_id={node_id}")
        start_time = time.time()
        try:
            if self._tool is None:
                raise RuntimeError("Tool not initialized")
            result = self._tool.call("get_symptom_by_node_id", {"node_id": node_id})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_symptom_by_node_id completed, elapsed={elapsed:.3f}s, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_symptom_by_node_id failed, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise

    def get_diseases_by_symptom_node_id(self, node_id: int) -> List[str]:
        logger.debug(f"[Neo4jMedicalHandler] get_diseases_by_symptom_node_id called, node_id={node_id}")
        start_time = time.time()
        try:
            if self._tool is None:
                raise RuntimeError("Tool not initialized")
            result = self._tool.call("get_diseases_by_symptom_node_id", {"node_id": node_id})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_diseases_by_symptom_node_id completed, elapsed={elapsed:.3f}s, node_id={node_id}, disease_count={len(result) if isinstance(result, list) else 'N/A'}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_diseases_by_symptom_node_id failed, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise

    def get_symptoms_by_node_id(self, node_id: int) -> List[str]:
        logger.debug(f"[Neo4jMedicalHandler] get_symptoms_by_node_id called, node_id={node_id}")
        start_time = time.time()
        try:
            if self._tool is None:
                raise RuntimeError("Tool not initialized")
            result = self._tool.call("get_symptoms_by_node_id", {"node_id": node_id})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_symptoms_by_node_id completed, elapsed={elapsed:.3f}s, node_id={node_id}, symptom_count={len(result) if isinstance(result, list) else 'N/A'}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_symptoms_by_node_id failed, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise

    def get_drugs_by_node_id(self, node_id: int) -> Dict[str, List[str]]:
        logger.debug(f"[Neo4jMedicalHandler] get_drugs_by_node_id called, node_id={node_id}")
        start_time = time.time()
        try:
            if self._tool is None:
                raise RuntimeError("Tool not initialized")
            result = self._tool.call("get_drugs_by_node_id", {"node_id": node_id})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_drugs_by_node_id completed, elapsed={elapsed:.3f}s, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_drugs_by_node_id failed, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise

    def get_foods_by_node_id(self, node_id: int) -> Dict[str, List[str]]:
        logger.debug(f"[Neo4jMedicalHandler] get_foods_by_node_id called, node_id={node_id}")
        start_time = time.time()
        try:
            if self._tool is None:
                raise RuntimeError("Tool not initialized")
            result = self._tool.call("get_foods_by_node_id", {"node_id": node_id})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] get_foods_by_node_id completed, elapsed={elapsed:.3f}s, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] get_foods_by_node_id failed, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise

    def search_diseases_by_symptom(self, symptom_name: str) -> List[str]:
        logger.debug(f"[Neo4jMedicalHandler] search_diseases_by_symptom called, symptom_name={symptom_name}")
        start_time = time.time()
        try:
            if self._tool is None:
                raise RuntimeError("Tool not initialized")
            result = self._tool.call("search_diseases_by_symptom", {"symptom_name": symptom_name})
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] search_diseases_by_symptom completed, elapsed={elapsed:.3f}s, symptom_name={symptom_name}, disease_count={len(result) if isinstance(result, list) else 'N/A'}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] search_diseases_by_symptom failed, elapsed={elapsed:.3f}s, symptom_name={symptom_name}, error={str(e)}")
            raise

    def release(self) -> None:
        logger.info("[Neo4jMedicalHandler] release started")
        start_time = time.time()
        try:
            if self._tool is not None:
                self._tool.release_tool(None)
                self._tool = None
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalHandler] release completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalHandler] release failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise
