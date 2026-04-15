# -*- coding: utf-8 -*-
"""
Neo4j医疗知识图谱工具

封装Neo4j医疗知识图谱的查询功能，提供统一的工具接口。
"""

from typing import Any, Dict, List, Optional

from src.tools.tool import Tool
from src.adapters import Neo4jAdapterImpl


class Neo4jMedicalTool(Tool):
    """
    Neo4j医疗知识图谱工具类
    
    封装Neo4j医疗知识图谱的查询功能，实现Tool接口。
    
    属性：
        _adapter: Neo4j适配器实例
        _uri: Neo4j连接URI
        _user: 用户名
        _password: 密码
    """
    
    def __init__(
        self, 
        uri: str, 
        user: str, 
        password: str
    ):
        """
        初始化Neo4j医疗知识图谱工具
        
        Args:
            uri: Neo4j连接URI
            user: 用户名
            password: 密码
        """
        self._uri = uri
        self._user = user
        self._password = password
        self._adapter: Optional[Neo4jAdapterImpl] = None
    
    def _init_resource(self) -> None:
        """初始化Neo4j连接资源"""
        if self._adapter is not None:
            return
        
        self._adapter = Neo4jAdapterImpl(
            uri=self._uri,
            user=self._user,
            password=self._password
        )
        self._adapter.connect()
    
    def release_source(self) -> None:
        """释放Neo4j连接资源"""
        if self._adapter is not None:
            self._adapter.disconnect()
            self._adapter = None
    
    def query_medical_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """
        查询医学知识
        
        Args:
            query: Cypher查询语句
            
        Returns:
            查询结果列表
        """
        if self._adapter is None:
            raise RuntimeError("Tool not initialized, call _init_resource first")
        return self._adapter.execute_query(query)
    
    def get_disease_info(self, disease_name: str) -> Optional[Dict[str, Any]]:
        """
        获取疾病信息
        
        Args:
            disease_name: 疾病名称
            
        Returns:
            疾病信息字典，包含：
            - name: 疾病名称
            - desc: 疾病描述
            - cause: 病因
            - prevent: 预防措施
            - cure_lasttime: 治疗时间
            - cured_prob: 治愈概率
            - easy_get: 易感人群
        """
        query = """
        MATCH (d:Disease {name: $name})
        RETURN d.name as name, d.desc as desc, d.cause as cause, 
               d.prevent as prevent, d.cure_lasttime as cure_lasttime,
               d.cured_prob as cured_prob, d.easy_get as easy_get
        """
        results = self.query_with_params(query, {"name": disease_name})
        return results[0] if results else None
    
    def get_symptoms_by_disease(self, disease_name: str) -> List[str]:
        """
        获取疾病的症状列表
        
        Args:
            disease_name: 疾病名称
            
        Returns:
            症状名称列表
        """
        query = """
        MATCH (d:Disease {name: $name})-[:has_symptom]->(s:Symptom)
        RETURN s.name as symptom_name
        """
        results = self.query_with_params(query, {"name": disease_name})
        return [r["symptom_name"] for r in results]
    
    def get_drugs_by_disease(self, disease_name: str) -> Dict[str, List[str]]:
        """
        获取疾病的药物信息
        
        Args:
            disease_name: 疾病名称
            
        Returns:
            药物信息字典，包含：
            - common_drugs: 常用药物列表
            - recommand_drugs: 推荐药物列表
        """
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
        """
        获取疾病的饮食建议
        
        Args:
            disease_name: 疾病名称
            
        Returns:
            饮食建议字典，包含：
            - do_eat: 宜吃食物列表
            - no_eat: 忌吃食物列表
            - recommand_eat: 推荐食谱列表
        """
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
        """
        获取疾病的检查项目
        
        Args:
            disease_name: 疾病名称
            
        Returns:
            检查项目名称列表
        """
        query = """
        MATCH (d:Disease {name: $name})-[:need_check]->(c:Check)
        RETURN c.name as check_name
        """
        results = self.query_with_params(query, {"name": disease_name})
        return [r["check_name"] for r in results]
    
    def get_department_by_disease(self, disease_name: str) -> List[str]:
        """
        获取疾病所属科室
        
        Args:
            disease_name: 疾病名称
            
        Returns:
            科室名称列表
        """
        query = """
        MATCH (d:Disease {name: $name})-[:belongs_to]->(dep:Department)
        RETURN dep.name as department_name
        """
        results = self.query_with_params(query, {"name": disease_name})
        return [r["department_name"] for r in results]
    
    def get_complications_by_disease(self, disease_name: str) -> List[str]:
        """
        获取疾病的并发症
        
        Args:
            disease_name: 疾病名称
            
        Returns:
            并发症名称列表
        """
        query = """
        MATCH (d:Disease {name: $name})-[:acompany_with]->(comp:Disease)
        RETURN comp.name as complication_name
        """
        results = self.query_with_params(query, {"name": disease_name})
        return [r["complication_name"] for r in results]
    
    def get_cure_methods_by_disease(self, disease_name: str) -> List[str]:
        """
        获取疾病的治疗方法
        
        Args:
            disease_name: 疾病名称
            
        Returns:
            治疗方法名称列表
        """
        query = """
        MATCH (d:Disease {name: $name})-[:cure_way]->(c:Cure)
        RETURN c.name as cure_method
        """
        results = self.query_with_params(query, {"name": disease_name})
        return [r["cure_method"] for r in results]
    
    def search_diseases_by_symptom(self, symptom_name: str) -> List[str]:
        """
        根据症状搜索可能的疾病
        
        Args:
            symptom_name: 症状名称
            
        Returns:
            疾病名称列表
        """
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
        """
        执行带参数的Cypher查询
        
        Args:
            query: Cypher查询语句
            params: 参数字典
            
        Returns:
            查询结果列表
        """
        if self._adapter is None:
            raise RuntimeError("Tool not initialized, call _init_resource first")
        return self._adapter.execute_query_with_params(query, params)
