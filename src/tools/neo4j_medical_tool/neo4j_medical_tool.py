# -*- coding: utf-8 -*-
"""
Neo4j医疗知识图谱工具

封装Neo4j医疗知识图谱的查询功能，提供统一的工具接口。
使用资源池管理Neo4j连接。
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from src.tools.neo4j_medical_tool.neo4j_medical_tool_interface import Neo4jMedicalToolInterface
from src.resource_manager.global_resource_manager import GlobalResourceManager
from src.resource_manager.resource_handle import ResourceHandle
from src.resource_manager.neo4j_connection import Neo4jConnectionClient
from src.schemas.resource_type import ResourceType, ConfigId
from src.utils.logger import log_arch_event

logger = logging.getLogger(__name__)


class Neo4jMedicalTool(Neo4jMedicalToolInterface):
    """
    Neo4j医疗知识图谱工具类

    封装Neo4j医疗知识图谱的查询功能，实现Tool接口。
    使用资源池管理Neo4j连接。

    属性：
        _resource_handle: 资源句柄
        _client: Neo4j连接客户端（ResourceClient封装层）
    """

    def __init__(self):
        """初始化Neo4j医疗知识图谱工具"""
        self._resource_handle: Optional[ResourceHandle] = None
        self._client: Optional[Neo4jConnectionClient] = None
        self._lock = threading.Lock()

    def _init_resource(self) -> None:
        """轻量初始化 — 不再acquire资源，资源在业务方法中按需获取

        设计依据：4.2节 资源管理原则 — 推荐使用上下文管理器自动管理资源
        6.5节 资源使用规范 — 方式2（推荐）使用上下文管理器自动管理资源
        Tool只在工作时掌握资源，用完归还资源池
        """
        logger.info("[Neo4jMedicalTool] _init_resource completed (lightweight, no resource acquire)")

    def _acquire_resource(self) -> None:
        """获取资源 — 幂等，已持有则跳过；线程安全"""
        with self._lock:
            if self._client is not None:
                return
            try:
                self._resource_handle = GlobalResourceManager.acquire(ResourceType.NEO4J_CONNECTION, ConfigId.NEO4J_CONFIG)
                logger.info("[TOOL_RESOURCE_INIT] tool=Neo4jMedicalTool, resource_type=neo4j_connection")
                if self._resource_handle is None:
                    raise RuntimeError("Failed to acquire neo4j_connection resource")
                if not self._resource_handle.resource.is_activate():
                    self._resource_handle.resource.activate()
                self._client = self._resource_handle.get_client()
                logger.info("[Neo4jMedicalTool] neo4j_connection resource acquired")
            except Exception as e:
                logger.debug(f"[Neo4jMedicalTool] 资源获取失败: {e}")
                self._resource_handle = None
                self._client = None
                raise

    def _release_resource(self) -> None:
        """归还资源 — 释放资源句柄归还资源池，保持连接；线程安全"""
        with self._lock:
            if self._resource_handle is not None:
                try:
                    GlobalResourceManager.release(self._resource_handle)
                finally:
                    self._resource_handle = None
                    self._client = None
                logger.info("[Neo4jMedicalTool] neo4j_connection resource released")

    def release_source(self) -> None:
        """释放Neo4j连接资源 - 归还资源池，保持连接"""
        logger.info("[Neo4jMedicalTool] release_source started")
        self._release_resource()
        log_arch_event(logger, component="Neo4jMedicalTool", stage="TOOL", event="release_source", status="success", design_id="ARCH-5.1")

    def destroy_source(self) -> None:
        """彻底销毁Neo4j连接资源 - 断开连接"""
        logger.info(f"[TOOL_DESTROY] {self.__class__.__name__}销毁资源")
        logger.info("[Neo4jMedicalTool] destroy_source started")
        start_time = time.time()
        try:
            if self._resource_handle is not None:
                GlobalResourceManager.destroy(self._resource_handle)
                self._resource_handle = None
                self._client = None
                logger.info("[Neo4jMedicalTool] neo4j_connection resource destroyed")

            elapsed = time.time() - start_time
            log_arch_event(logger, component="Neo4jMedicalTool", stage="TOOL", event="destroy_source", status="success", design_id="ARCH-5.1", elapsed=f"{elapsed:.3f}s")
            logger.info(f"[Neo4jMedicalTool] destroy_source completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalTool] destroy_source failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def query_medical_knowledge(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """查询医学知识 — 支持参数化查询，不从MCP代理对外暴露"""
        logger.info(f"[NEO4J_QUERY] method=query_medical_knowledge, query_length={len(query)}, has_params={params is not None}")
        self._acquire_resource()
        try:
            start_time = time.time()
            if params is not None:
                results = self._client.execute_query_with_params(query, params)
            else:
                results = self._client.execute_query(query)
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=query_medical_knowledge, elapsed={elapsed:.3f}s, result_count={len(results)}")
            return results
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=query_medical_knowledge, elapsed={elapsed:.3f}s, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def get_disease_info(self, disease_name: str) -> Optional[Dict[str, Any]]:
        """获取疾病信息"""
        logger.info(f"[NEO4J_QUERY] method=get_disease_info, disease_name={disease_name}")
        self._acquire_resource()
        try:
            start_time = time.time()
            query = """
            MATCH (d:Disease {name: $name})
            RETURN d.name as name, d.desc as desc, d.cause as cause,
                   d.prevent as prevent, d.cure_lasttime as cure_lasttime,
                   d.cured_prob as cured_prob, d.easy_get as easy_get
            """
            results = self.query_with_params(query, {"name": disease_name})
            result = results[0] if results else None
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=get_disease_info, elapsed={elapsed:.3f}s, found={result is not None}, disease_name={disease_name}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=get_disease_info, elapsed={elapsed:.3f}s, disease_name={disease_name}, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def get_symptoms_by_disease(self, disease_name: str) -> List[str]:
        """获取疾病的症状列表"""
        logger.info(f"[NEO4J_QUERY] method=get_symptoms_by_disease, disease_name={disease_name}")
        self._acquire_resource()
        try:
            start_time = time.time()
            query = """
            MATCH (d:Disease {name: $name})-[:has_symptom]->(s:Symptom)
            RETURN s.name as symptom_name
            """
            results = self.query_with_params(query, {"name": disease_name})
            symptoms = [r["symptom_name"] for r in results]
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=get_symptoms_by_disease, elapsed={elapsed:.3f}s, symptom_count={len(symptoms)}, disease_name={disease_name}")
            return symptoms
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=get_symptoms_by_disease, elapsed={elapsed:.3f}s, disease_name={disease_name}, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def get_drugs_by_disease(self, disease_name: str) -> Dict[str, List[str]]:
        """获取疾病的药物信息"""
        logger.info(f"[NEO4J_QUERY] method=get_drugs_by_disease, disease_name={disease_name}")
        self._acquire_resource()
        try:
            start_time = time.time()
            common_query = """
            MATCH (d:Disease {name: $name})-[:common_drug]->(dr:Drug)
            RETURN dr.name as drug_name
            """
            recommand_query = """
            MATCH (d:Disease {name: $name})-[:recommand_drug]->(dr:Drug)
            RETURN dr.name as drug_name
            """
            common_results = self.query_with_params(common_query, {"name": disease_name})
            recommand_results = self.query_with_params(recommand_query, {"name": disease_name})
            result = {
                "common_drugs": [r["drug_name"] for r in common_results],
                "recommand_drugs": [r["drug_name"] for r in recommand_results]
            }
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=get_drugs_by_disease, elapsed={elapsed:.3f}s, common_count={len(result['common_drugs'])}, recommand_count={len(result['recommand_drugs'])}, disease_name={disease_name}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=get_drugs_by_disease, elapsed={elapsed:.3f}s, disease_name={disease_name}, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def get_foods_by_disease(self, disease_name: str) -> Dict[str, List[str]]:
        """获取疾病的饮食建议"""
        logger.info(f"[NEO4J_QUERY] method=get_foods_by_disease, disease_name={disease_name}")
        self._acquire_resource()
        try:
            start_time = time.time()
            do_eat_query = """
            MATCH (d:Disease {name: $name})-[:do_eat]->(f:Food)
            RETURN f.name as food_name
            """
            no_eat_query = """
            MATCH (d:Disease {name: $name})-[:no_eat]->(f:Food)
            RETURN f.name as food_name
            """
            recommand_eat_query = """
            MATCH (d:Disease {name: $name})-[:recommand_eat]->(f:Food)
            RETURN f.name as food_name
            """
            do_eat_results = self.query_with_params(do_eat_query, {"name": disease_name})
            no_eat_results = self.query_with_params(no_eat_query, {"name": disease_name})
            recommand_eat_results = self.query_with_params(recommand_eat_query, {"name": disease_name})
            result = {
                "do_eat": [r["food_name"] for r in do_eat_results],
                "no_eat": [r["food_name"] for r in no_eat_results],
                "recommand_eat": [r["food_name"] for r in recommand_eat_results]
            }
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=get_foods_by_disease, elapsed={elapsed:.3f}s, do_eat={len(result['do_eat'])}, no_eat={len(result['no_eat'])}, recommand_eat={len(result['recommand_eat'])}, disease_name={disease_name}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=get_foods_by_disease, elapsed={elapsed:.3f}s, disease_name={disease_name}, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def get_checks_by_disease(self, disease_name: str) -> List[str]:
        """获取疾病的检查项目"""
        logger.info(f"[NEO4J_QUERY] method=get_checks_by_disease, disease_name={disease_name}")
        self._acquire_resource()
        try:
            start_time = time.time()
            query = """
            MATCH (d:Disease {name: $name})-[:need_check]->(c:Check)
            RETURN c.name as check_name
            """
            results = self.query_with_params(query, {"name": disease_name})
            checks = [r["check_name"] for r in results]
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=get_checks_by_disease, elapsed={elapsed:.3f}s, check_count={len(checks)}, disease_name={disease_name}")
            return checks
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=get_checks_by_disease, elapsed={elapsed:.3f}s, disease_name={disease_name}, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def get_department_by_disease(self, disease_name: str) -> List[str]:
        """获取疾病所属科室"""
        logger.info(f"[NEO4J_QUERY] method=get_department_by_disease, disease_name={disease_name}")
        self._acquire_resource()
        try:
            start_time = time.time()
            query = """
            MATCH (d:Disease {name: $name})-[:belongs_to]->(dep:Department)
            RETURN dep.name as department_name
            """
            results = self.query_with_params(query, {"name": disease_name})
            departments = [r["department_name"] for r in results]
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=get_department_by_disease, elapsed={elapsed:.3f}s, department_count={len(departments)}, disease_name={disease_name}")
            return departments
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=get_department_by_disease, elapsed={elapsed:.3f}s, disease_name={disease_name}, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def get_complications_by_disease(self, disease_name: str) -> List[str]:
        """获取疾病的并发症"""
        logger.info(f"[NEO4J_QUERY] method=get_complications_by_disease, disease_name={disease_name}")
        self._acquire_resource()
        try:
            start_time = time.time()
            query = """
            MATCH (d:Disease {name: $name})-[:acompany_with]->(comp:Disease)
            RETURN comp.name as complication_name
            """
            results = self.query_with_params(query, {"name": disease_name})
            complications = [r["complication_name"] for r in results]
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=get_complications_by_disease, elapsed={elapsed:.3f}s, complication_count={len(complications)}, disease_name={disease_name}")
            return complications
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=get_complications_by_disease, elapsed={elapsed:.3f}s, disease_name={disease_name}, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def get_cure_methods_by_disease(self, disease_name: str) -> List[str]:
        """获取疾病的治疗方法"""
        logger.info(f"[NEO4J_QUERY] method=get_cure_methods_by_disease, disease_name={disease_name}")
        self._acquire_resource()
        try:
            start_time = time.time()
            query = """
            MATCH (d:Disease {name: $name})-[:cure_way]->(c:Cure)
            RETURN c.name as cure_method
            """
            results = self.query_with_params(query, {"name": disease_name})
            cure_methods = [r["cure_method"] for r in results]
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=get_cure_methods_by_disease, elapsed={elapsed:.3f}s, method_count={len(cure_methods)}, disease_name={disease_name}")
            return cure_methods
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=get_cure_methods_by_disease, elapsed={elapsed:.3f}s, disease_name={disease_name}, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def get_disease_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """通过Neo4j节点elementId获取疾病信息"""
        logger.info(f"[NEO4J_QUERY] method=get_disease_by_node_id, node_id={node_id}")
        self._acquire_resource()
        try:
            start_time = time.time()
            query = """
            MATCH (d:Disease)
            WHERE elementId(d) = $node_id
            RETURN d.name as name, d.desc as desc, d.cause as cause,
                   d.prevent as prevent, d.easy_get as easy_get,
                   d.cure_lasttime as cure_lasttime, d.cured_prob as cured_prob
            """
            results = self.query_with_params(query, {"node_id": node_id})
            result = results[0] if results else None
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=get_disease_by_node_id, elapsed={elapsed:.3f}s, found={result is not None}, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=get_disease_by_node_id, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def get_symptom_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """通过Neo4j节点elementId获取症状信息"""
        logger.info(f"[NEO4J_QUERY] method=get_symptom_by_node_id, node_id={node_id}")
        self._acquire_resource()
        try:
            start_time = time.time()
            query = """
            MATCH (s:Symptom)
            WHERE elementId(s) = $node_id
            RETURN s.name as name
            """
            results = self.query_with_params(query, {"node_id": node_id})
            result = results[0] if results else None
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=get_symptom_by_node_id, elapsed={elapsed:.3f}s, found={result is not None}, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=get_symptom_by_node_id, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def get_diseases_by_symptom_node_id(self, node_id: str, limit: int = 50) -> List[str]:
        """通过症状节点elementId查询相关疾病"""
        logger.info(f"[NEO4J_QUERY] method=get_diseases_by_symptom_node_id, node_id={node_id}, limit={limit}")
        self._acquire_resource()
        try:
            start_time = time.time()
            query = """
            MATCH (s:Symptom)
            WHERE elementId(s) = $node_id
            MATCH (d:Disease)-[:has_symptom]->(s)
            RETURN d.name as disease_name
            LIMIT $limit
            """
            results = self.query_with_params(query, {"node_id": node_id, "limit": limit})
            diseases = [r["disease_name"] for r in results]
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=get_diseases_by_symptom_node_id, elapsed={elapsed:.3f}s, disease_count={len(diseases)}, node_id={node_id}")
            return diseases
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=get_diseases_by_symptom_node_id, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def get_symptoms_by_node_id(self, node_id: str) -> List[str]:
        """通过Neo4j节点elementId获取疾病的症状"""
        logger.info(f"[NEO4J_QUERY] method=get_symptoms_by_node_id, node_id={node_id}")
        self._acquire_resource()
        try:
            start_time = time.time()
            query = """
            MATCH (d:Disease)
            WHERE elementId(d) = $node_id
            MATCH (d)-[:has_symptom]->(s:Symptom)
            RETURN s.name as symptom
            """
            results = self.query_with_params(query, {"node_id": node_id})
            symptoms = [r["symptom"] for r in results]
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=get_symptoms_by_node_id, elapsed={elapsed:.3f}s, symptom_count={len(symptoms)}, node_id={node_id}")
            return symptoms
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=get_symptoms_by_node_id, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def get_drugs_by_node_id(self, node_id: str) -> Dict[str, List[str]]:
        """通过Neo4j节点elementId获取疾病的常用药物和推荐药物"""
        logger.info(f"[NEO4J_QUERY] method=get_drugs_by_node_id, node_id={node_id}")
        self._acquire_resource()
        try:
            start_time = time.time()
            common_drug_query = """
            MATCH (d:Disease)
            WHERE elementId(d) = $node_id
            MATCH (d)-[:common_drug]->(dr:Drug)
            RETURN dr.name as drug_name
            """
            recommand_drug_query = """
            MATCH (d:Disease)
            WHERE elementId(d) = $node_id
            MATCH (d)-[:recommand_drug]->(dr:Drug)
            RETURN dr.name as drug_name
            """
            common_drugs = [r["drug_name"] for r in self.query_with_params(common_drug_query, {"node_id": node_id})]
            recommand_drugs = [r["drug_name"] for r in self.query_with_params(recommand_drug_query, {"node_id": node_id})]
            result = {"common_drugs": common_drugs, "recommand_drugs": recommand_drugs}
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=get_drugs_by_node_id, elapsed={elapsed:.3f}s, common_count={len(common_drugs)}, recommand_count={len(recommand_drugs)}, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=get_drugs_by_node_id, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def get_foods_by_node_id(self, node_id: str) -> Dict[str, List[str]]:
        """通过Neo4j节点elementId获取疾病的饮食建议"""
        logger.info(f"[NEO4J_QUERY] method=get_foods_by_node_id, node_id={node_id}")
        self._acquire_resource()
        try:
            start_time = time.time()
            do_eat_query = """
            MATCH (d:Disease)
            WHERE elementId(d) = $node_id
            MATCH (d)-[:do_eat]->(f:Food)
            RETURN f.name as food_name
            """
            no_eat_query = """
            MATCH (d:Disease)
            WHERE elementId(d) = $node_id
            MATCH (d)-[:no_eat]->(f:Food)
            RETURN f.name as food_name
            """
            recommand_eat_query = """
            MATCH (d:Disease)
            WHERE elementId(d) = $node_id
            MATCH (d)-[:recommand_eat]->(f:Food)
            RETURN f.name as food_name
            """
            do_eat = [r["food_name"] for r in self.query_with_params(do_eat_query, {"node_id": node_id})]
            no_eat = [r["food_name"] for r in self.query_with_params(no_eat_query, {"node_id": node_id})]
            recommand_eat = [r["food_name"] for r in self.query_with_params(recommand_eat_query, {"node_id": node_id})]
            result = {"do_eat": do_eat, "no_eat": no_eat, "recommand_eat": recommand_eat}
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=get_foods_by_node_id, elapsed={elapsed:.3f}s, do_eat={len(do_eat)}, no_eat={len(no_eat)}, recommand_eat={len(recommand_eat)}, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=get_foods_by_node_id, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def search_diseases_by_symptom(self, symptom_name: str, limit: int = 50) -> List[str]:
        """根据症状搜索可能的疾病"""
        logger.info(f"[NEO4J_QUERY] method=search_diseases_by_symptom, symptom_name={symptom_name}, limit={limit}")
        self._acquire_resource()
        try:
            start_time = time.time()
            query = """
            MATCH (s:Symptom)-[:has_symptom]-(d:Disease)
            WHERE s.name CONTAINS $symptom
            RETURN DISTINCT d.name as disease_name
            LIMIT $limit
            """
            results = self.query_with_params(query, {"symptom": symptom_name, "limit": limit})
            diseases = [r["disease_name"] for r in results]
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=search_diseases_by_symptom, elapsed={elapsed:.3f}s, disease_count={len(diseases)}, symptom_name={symptom_name}")
            return diseases
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=search_diseases_by_symptom, elapsed={elapsed:.3f}s, symptom_name={symptom_name}, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def get_drug_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """通过Neo4j节点elementId获取药物信息"""
        logger.info(f"[NEO4J_QUERY] method=get_drug_by_node_id, node_id={node_id}")
        self._acquire_resource()
        try:
            start_time = time.time()
            query = """
            MATCH (dr:Drug)
            WHERE elementId(dr) = $node_id
            RETURN dr.name as name
            """
            results = self.query_with_params(query, {"node_id": node_id})
            result = results[0] if results else None
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=get_drug_by_node_id, elapsed={elapsed:.3f}s, found={result is not None}, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=get_drug_by_node_id, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def get_diseases_by_drug_node_id(self, node_id: str, limit: int = 50) -> Dict[str, List[str]]:
        """通过药物节点elementId查询相关疾病"""
        logger.info(f"[NEO4J_QUERY] method=get_diseases_by_drug_node_id, node_id={node_id}, limit={limit}")
        self._acquire_resource()
        try:
            start_time = time.time()
            common_drug_query = """
            MATCH (dr:Drug)
            WHERE elementId(dr) = $node_id
            MATCH (d:Disease)-[:common_drug]->(dr)
            RETURN d.name as disease_name
            LIMIT $limit
            """
            recommand_drug_query = """
            MATCH (dr:Drug)
            WHERE elementId(dr) = $node_id
            MATCH (d:Disease)-[:recommand_drug]->(dr)
            RETURN d.name as disease_name
            LIMIT $limit
            """
            common_diseases = [r["disease_name"] for r in self.query_with_params(common_drug_query, {"node_id": node_id, "limit": limit})]
            recommand_diseases = [r["disease_name"] for r in self.query_with_params(recommand_drug_query, {"node_id": node_id, "limit": limit})]
            result = {"common_drug_diseases": common_diseases, "recommand_drug_diseases": recommand_diseases}
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=get_diseases_by_drug_node_id, elapsed={elapsed:.3f}s, common_count={len(common_diseases)}, recommand_count={len(recommand_diseases)}, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=get_diseases_by_drug_node_id, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def get_food_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """通过Neo4j节点elementId获取食物信息"""
        logger.info(f"[NEO4J_QUERY] method=get_food_by_node_id, node_id={node_id}")
        self._acquire_resource()
        try:
            start_time = time.time()
            query = """
            MATCH (f:Food)
            WHERE elementId(f) = $node_id
            RETURN f.name as name
            """
            results = self.query_with_params(query, {"node_id": node_id})
            result = results[0] if results else None
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=get_food_by_node_id, elapsed={elapsed:.3f}s, found={result is not None}, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=get_food_by_node_id, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def get_diseases_by_food_node_id(self, node_id: str, limit: int = 50) -> Dict[str, List[str]]:
        """通过食物节点elementId查询相关疾病"""
        logger.info(f"[NEO4J_QUERY] method=get_diseases_by_food_node_id, node_id={node_id}, limit={limit}")
        self._acquire_resource()
        try:
            start_time = time.time()
            do_eat_query = """
            MATCH (f:Food)
            WHERE elementId(f) = $node_id
            MATCH (d:Disease)-[:do_eat]->(f)
            RETURN d.name as disease_name
            LIMIT $limit
            """
            no_eat_query = """
            MATCH (f:Food)
            WHERE elementId(f) = $node_id
            MATCH (d:Disease)-[:no_eat]->(f)
            RETURN d.name as disease_name
            LIMIT $limit
            """
            recommand_eat_query = """
            MATCH (f:Food)
            WHERE elementId(f) = $node_id
            MATCH (d:Disease)-[:recommand_eat]->(f)
            RETURN d.name as disease_name
            LIMIT $limit
            """
            do_eat_diseases = [r["disease_name"] for r in self.query_with_params(do_eat_query, {"node_id": node_id, "limit": limit})]
            no_eat_diseases = [r["disease_name"] for r in self.query_with_params(no_eat_query, {"node_id": node_id, "limit": limit})]
            recommand_diseases = [r["disease_name"] for r in self.query_with_params(recommand_eat_query, {"node_id": node_id, "limit": limit})]
            result = {"do_eat_diseases": do_eat_diseases, "no_eat_diseases": no_eat_diseases, "recommand_diseases": recommand_diseases}
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=get_diseases_by_food_node_id, elapsed={elapsed:.3f}s, do_eat={len(do_eat_diseases)}, no_eat={len(no_eat_diseases)}, recommand={len(recommand_diseases)}, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=get_diseases_by_food_node_id, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def get_check_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """通过Neo4j节点elementId获取检查项目信息"""
        logger.info(f"[NEO4J_QUERY] method=get_check_by_node_id, node_id={node_id}")
        self._acquire_resource()
        try:
            start_time = time.time()
            query = """
            MATCH (c:Check)
            WHERE elementId(c) = $node_id
            RETURN c.name as name
            """
            results = self.query_with_params(query, {"node_id": node_id})
            result = results[0] if results else None
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=get_check_by_node_id, elapsed={elapsed:.3f}s, found={result is not None}, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=get_check_by_node_id, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def get_diseases_by_check_node_id(self, node_id: str, limit: int = 50) -> List[str]:
        """通过检查项目节点elementId查询相关疾病"""
        logger.info(f"[NEO4J_QUERY] method=get_diseases_by_check_node_id, node_id={node_id}, limit={limit}")
        self._acquire_resource()
        try:
            start_time = time.time()
            query = """
            MATCH (c:Check)
            WHERE elementId(c) = $node_id
            MATCH (d:Disease)-[:need_check]->(c)
            RETURN d.name as disease_name
            LIMIT $limit
            """
            results = self.query_with_params(query, {"node_id": node_id, "limit": limit})
            diseases = [r["disease_name"] for r in results]
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=get_diseases_by_check_node_id, elapsed={elapsed:.3f}s, disease_count={len(diseases)}, node_id={node_id}")
            return diseases
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=get_diseases_by_check_node_id, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def get_department_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """通过Neo4j节点elementId获取科室信息"""
        logger.info(f"[NEO4J_QUERY] method=get_department_by_node_id, node_id={node_id}")
        self._acquire_resource()
        try:
            start_time = time.time()
            query = """
            MATCH (dep:Department)
            WHERE elementId(dep) = $node_id
            RETURN dep.name as name
            """
            results = self.query_with_params(query, {"node_id": node_id})
            result = results[0] if results else None
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=get_department_by_node_id, elapsed={elapsed:.3f}s, found={result is not None}, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=get_department_by_node_id, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def get_diseases_by_department_node_id(self, node_id: str, limit: int = 50) -> List[str]:
        """通过科室节点elementId查询相关疾病"""
        logger.info(f"[NEO4J_QUERY] method=get_diseases_by_department_node_id, node_id={node_id}, limit={limit}")
        self._acquire_resource()
        try:
            start_time = time.time()
            query = """
            MATCH (dep:Department)
            WHERE elementId(dep) = $node_id
            MATCH (d:Disease)-[:belongs_to]->(dep)
            RETURN d.name as disease_name
            LIMIT $limit
            """
            results = self.query_with_params(query, {"node_id": node_id, "limit": limit})
            diseases = [r["disease_name"] for r in results]
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=get_diseases_by_department_node_id, elapsed={elapsed:.3f}s, disease_count={len(diseases)}, node_id={node_id}")
            return diseases
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=get_diseases_by_department_node_id, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def get_cure_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """通过Neo4j节点elementId获取治疗方法信息"""
        logger.info(f"[NEO4J_QUERY] method=get_cure_by_node_id, node_id={node_id}")
        self._acquire_resource()
        try:
            start_time = time.time()
            query = """
            MATCH (c:Cure)
            WHERE elementId(c) = $node_id
            RETURN c.name as name
            """
            results = self.query_with_params(query, {"node_id": node_id})
            result = results[0] if results else None
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=get_cure_by_node_id, elapsed={elapsed:.3f}s, found={result is not None}, node_id={node_id}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=get_cure_by_node_id, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def get_diseases_by_cure_node_id(self, node_id: str, limit: int = 50) -> List[str]:
        """通过治疗方法节点elementId查询相关疾病"""
        logger.info(f"[NEO4J_QUERY] method=get_diseases_by_cure_node_id, node_id={node_id}, limit={limit}")
        self._acquire_resource()
        try:
            start_time = time.time()
            query = """
            MATCH (c:Cure)
            WHERE elementId(c) = $node_id
            MATCH (d:Disease)-[:cure_way]->(c)
            RETURN d.name as disease_name
            LIMIT $limit
            """
            results = self.query_with_params(query, {"node_id": node_id, "limit": limit})
            diseases = [r["disease_name"] for r in results]
            elapsed = time.time() - start_time
            logger.info(f"[NEO4J_QUERY_RESULT] method=get_diseases_by_cure_node_id, elapsed={elapsed:.3f}s, disease_count={len(diseases)}, node_id={node_id}")
            return diseases
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NEO4J_QUERY_ERROR] method=get_diseases_by_cure_node_id, elapsed={elapsed:.3f}s, node_id={node_id}, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def query_with_params(
        self,
        query: str,
        params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """执行带参数的Cypher查询 — 内部方法，调用者负责acquire/release"""
        logger.debug(f"[NEO4J_CYPHER] query_preview={query[:100]}..., params_keys={list(params.keys())}")
        start_time = time.time()
        results = self._client.execute_query_with_params(query, params)
        elapsed = time.time() - start_time
        logger.debug(f"[NEO4J_CYPHER_RESULT] elapsed={elapsed:.3f}s, result_count={len(results)}")
        return results
