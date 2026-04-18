# -*- coding: utf-8 -*-
"""
Neo4j医疗知识图谱工具

封装Neo4j医疗知识图谱的查询功能，提供统一的工具接口。
使用资源池管理Neo4j连接。
"""

import logging
import time
from typing import Any, Dict, List, Optional

from src.tools.tool import Tool
from src.resource_manager.global_resource_manager import GlobalResourceManager
from src.resource_manager.resource_handle import ResourceHandle
from src.resource_manager.neo4j_connection.neo4j_connection_resource import Neo4jConnectionResource

logger = logging.getLogger(__name__)


class Neo4jMedicalTool(Tool):
    """
    Neo4j医疗知识图谱工具类
    
    封装Neo4j医疗知识图谱的查询功能，实现Tool接口。
    使用资源池管理Neo4j连接。
    
    属性：
        _resource_handle: 资源句柄
        _resource: Neo4j连接资源
        _adapter: Neo4j适配器实例
    """
    
    def __init__(self):
        """初始化Neo4j医疗知识图谱工具"""
        self._resource_handle: Optional[ResourceHandle] = None
        self._resource: Optional[Neo4jConnectionResource] = None
        self._adapter = None
    
    def _init_resource(self) -> None:
        """初始化Neo4j连接资源 - 从资源池获取连接"""
        if self._resource is not None:
            logger.debug("[Neo4jMedicalTool] _init_resource skipped, already initialized")
            return
        
        logger.info("[Neo4jMedicalTool] _init_resource started")
        start_time = time.time()
        try:
            self._resource_handle = GlobalResourceManager.acquire("neo4j_connection", "neo4j_config")
            if self._resource_handle is None:
                raise RuntimeError("Failed to acquire neo4j_connection resource")
            
            self._resource = self._resource_handle.resource
            if not self._resource.is_activate():
                self._resource.activate()
            
            self._adapter = self._resource.get_adapter()
            
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalTool] _init_resource completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalTool] _init_resource failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise
    
    def release_source(self) -> None:
        """释放Neo4j连接资源 - 归还资源池，保持连接"""
        logger.info("[Neo4jMedicalTool] release_source started")
        start_time = time.time()
        try:
            if self._resource_handle is not None:
                GlobalResourceManager.release(self._resource_handle)
                self._resource_handle = None
                self._resource = None
                self._adapter = None
                logger.info("[Neo4jMedicalTool] neo4j_connection resource released")
            
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalTool] release_source completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalTool] release_source failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise
    
    def destroy_source(self) -> None:
        """彻底销毁Neo4j连接资源 - 断开连接"""
        logger.info("[Neo4jMedicalTool] destroy_source started")
        start_time = time.time()
        try:
            if self._resource_handle is not None:
                GlobalResourceManager.destroy(self._resource_handle)
                self._resource_handle = None
                self._resource = None
                self._adapter = None
                logger.info("[Neo4jMedicalTool] neo4j_connection resource destroyed")
            
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalTool] destroy_source completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalTool] destroy_source failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise
    
    def query_medical_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """查询医学知识"""
        if self._adapter is None:
            raise RuntimeError("Tool not initialized, call _init_resource first")
        return self._adapter.execute_query(query)
    
    def get_disease_info(self, disease_name: str) -> Optional[Dict[str, Any]]:
        """获取疾病信息"""
        query = """
        MATCH (d:Disease {name: $name})
        RETURN d.name as name, d.desc as desc, d.cause as cause, 
               d.prevent as prevent, d.cure_lasttime as cure_lasttime,
               d.cured_prob as cured_prob, d.easy_get as easy_get
        """
        results = self.query_with_params(query, {"name": disease_name})
        return results[0] if results else None
    
    def get_symptoms_by_disease(self, disease_name: str) -> List[str]:
        """获取疾病的症状列表"""
        query = """
        MATCH (d:Disease {name: $name})-[:has_symptom]->(s:Symptom)
        RETURN s.name as symptom_name
        """
        results = self.query_with_params(query, {"name": disease_name})
        return [r["symptom_name"] for r in results]
    
    def get_drugs_by_disease(self, disease_name: str) -> Dict[str, List[str]]:
        """获取疾病的药物信息"""
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
        
        return {
            "common_drugs": [r["drug_name"] for r in common_results],
            "recommand_drugs": [r["drug_name"] for r in recommand_results]
        }
    
    def get_foods_by_disease(self, disease_name: str) -> Dict[str, List[str]]:
        """获取疾病的饮食建议"""
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
        
        return {
            "do_eat": [r["food_name"] for r in do_eat_results],
            "no_eat": [r["food_name"] for r in no_eat_results],
            "recommand_eat": [r["food_name"] for r in recommand_eat_results]
        }
    
    def get_checks_by_disease(self, disease_name: str) -> List[str]:
        """获取疾病的检查项目"""
        query = """
        MATCH (d:Disease {name: $name})-[:need_check]->(c:Check)
        RETURN c.name as check_name
        """
        results = self.query_with_params(query, {"name": disease_name})
        return [r["check_name"] for r in results]
    
    def get_department_by_disease(self, disease_name: str) -> List[str]:
        """获取疾病所属科室"""
        query = """
        MATCH (d:Disease {name: $name})-[:belongs_to]->(dep:Department)
        RETURN dep.name as department_name
        """
        results = self.query_with_params(query, {"name": disease_name})
        return [r["department_name"] for r in results]
    
    def get_complications_by_disease(self, disease_name: str) -> List[str]:
        """获取疾病的并发症"""
        query = """
        MATCH (d:Disease {name: $name})-[:acompany_with]->(comp:Disease)
        RETURN comp.name as complication_name
        """
        results = self.query_with_params(query, {"name": disease_name})
        return [r["complication_name"] for r in results]
    
    def get_cure_methods_by_disease(self, disease_name: str) -> List[str]:
        """获取疾病的治疗方法"""
        query = """
        MATCH (d:Disease {name: $name})-[:cure_way]->(c:Cure)
        RETURN c.name as cure_method
        """
        results = self.query_with_params(query, {"name": disease_name})
        return [r["cure_method"] for r in results]
    
    def get_disease_by_node_id(self, node_id: int) -> Optional[Dict[str, Any]]:
        """通过Neo4j节点ID获取疾病信息"""
        query = """
        MATCH (d:Disease)
        WHERE id(d) = $node_id
        RETURN d.name as name, d.desc as desc, d.cause as cause, 
               d.prevent as prevent, d.easy_get as easy_get,
               d.cure_lasttime as cure_lasttime, d.cured_prob as cured_prob
        """
        results = self.query_with_params(query, {"node_id": node_id})
        return results[0] if results else None
    
    def get_symptom_by_node_id(self, node_id: int) -> Optional[Dict[str, Any]]:
        """通过Neo4j节点ID获取症状信息"""
        query = """
        MATCH (s:Symptom)
        WHERE id(s) = $node_id
        RETURN s.name as name
        """
        results = self.query_with_params(query, {"node_id": node_id})
        return results[0] if results else None
    
    def get_diseases_by_symptom_node_id(self, node_id: int) -> List[str]:
        """通过症状节点ID查询相关疾病"""
        query = """
        MATCH (s:Symptom)
        WHERE id(s) = $node_id
        MATCH (d:Disease)-[:has_symptom]->(s)
        RETURN d.name as disease_name
        """
        results = self.query_with_params(query, {"node_id": node_id})
        return [r["disease_name"] for r in results]
    
    def get_symptoms_by_node_id(self, node_id: int) -> List[str]:
        """通过Neo4j节点ID获取疾病的症状"""
        query = """
        MATCH (d:Disease)
        WHERE id(d) = $node_id
        MATCH (d)-[:has_symptom]->(s:Symptom)
        RETURN s.name as symptom
        """
        results = self.query_with_params(query, {"node_id": node_id})
        return [r["symptom"] for r in results]
    
    def get_drugs_by_node_id(self, node_id: int) -> Dict[str, List[str]]:
        """通过Neo4j节点ID获取疾病的常用药物和推荐药物"""
        common_drug_query = """
        MATCH (d:Disease)
        WHERE id(d) = $node_id
        MATCH (d)-[:common_drug]->(dr:Drug)
        RETURN dr.name as drug_name
        """
        recommand_drug_query = """
        MATCH (d:Disease)
        WHERE id(d) = $node_id
        MATCH (d)-[:recommand_drug]->(dr:Drug)
        RETURN dr.name as drug_name
        """
        common_drugs = [r["drug_name"] for r in self.query_with_params(common_drug_query, {"node_id": node_id})]
        recommand_drugs = [r["drug_name"] for r in self.query_with_params(recommand_drug_query, {"node_id": node_id})]
        return {"common_drugs": common_drugs, "recommand_drugs": recommand_drugs}
    
    def get_foods_by_node_id(self, node_id: int) -> Dict[str, List[str]]:
        """通过Neo4j节点ID获取疾病的饮食建议"""
        do_eat_query = """
        MATCH (d:Disease)
        WHERE id(d) = $node_id
        MATCH (d)-[:do_eat]->(f:Food)
        RETURN f.name as food_name
        """
        no_eat_query = """
        MATCH (d:Disease)
        WHERE id(d) = $node_id
        MATCH (d)-[:no_eat]->(f:Food)
        RETURN f.name as food_name
        """
        recommand_eat_query = """
        MATCH (d:Disease)
        WHERE id(d) = $node_id
        MATCH (d)-[:recommand_eat]->(f:Food)
        RETURN f.name as food_name
        """
        do_eat = [r["food_name"] for r in self.query_with_params(do_eat_query, {"node_id": node_id})]
        no_eat = [r["food_name"] for r in self.query_with_params(no_eat_query, {"node_id": node_id})]
        recommand_eat = [r["food_name"] for r in self.query_with_params(recommand_eat_query, {"node_id": node_id})]
        return {"do_eat": do_eat, "no_eat": no_eat, "recommand_eat": recommand_eat}
    
    def search_diseases_by_symptom(self, symptom_name: str) -> List[str]:
        """根据症状搜索可能的疾病"""
        query = """
        MATCH (s:Symptom)-[:has_symptom]-(d:Disease)
        WHERE s.name CONTAINS $symptom
        RETURN DISTINCT d.name as disease_name
        """
        results = self.query_with_params(query, {"symptom": symptom_name})
        return [r["disease_name"] for r in results]
    
    def query_with_params(
        self, 
        query: str, 
        params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """执行带参数的Cypher查询"""
        if self._adapter is None:
            raise RuntimeError("Tool not initialized, call _init_resource first")
        return self._adapter.execute_query_with_params(query, params)
