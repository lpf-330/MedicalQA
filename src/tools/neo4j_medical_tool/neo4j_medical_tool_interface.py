# -*- coding: utf-8 -*-
"""
Neo4j医疗知识图谱工具接口

定义Neo4jMedicalTool的包内抽象接口，继承外部Tool基类。
实现类必须实现此接口，不得直接实现外部Tool。
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional

from src.tools.tool import Tool


class Neo4jMedicalToolInterface(Tool):
    """
    Neo4j医疗知识图谱工具接口

    继承Tool基类，声明Neo4j医疗知识图谱查询的公共方法为抽象方法。
    实现类必须实现此接口而非直接实现Tool。
    """

    @abstractmethod
    def query_medical_knowledge(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """查询医学知识 — 支持参数化查询，不从MCP代理对外暴露"""
        pass

    @abstractmethod
    def get_disease_info(self, disease_name: str) -> Optional[Dict[str, Any]]:
        """获取疾病信息"""
        pass

    @abstractmethod
    def get_symptoms_by_disease(self, disease_name: str) -> List[str]:
        """获取疾病的症状列表"""
        pass

    @abstractmethod
    def get_drugs_by_disease(self, disease_name: str) -> Dict[str, List[str]]:
        """获取疾病的药物信息"""
        pass

    @abstractmethod
    def get_foods_by_disease(self, disease_name: str) -> Dict[str, List[str]]:
        """获取疾病的饮食建议"""
        pass

    @abstractmethod
    def get_checks_by_disease(self, disease_name: str) -> List[str]:
        """获取疾病的检查项目"""
        pass

    @abstractmethod
    def get_department_by_disease(self, disease_name: str) -> List[str]:
        """获取疾病所属科室"""
        pass

    @abstractmethod
    def get_complications_by_disease(self, disease_name: str) -> List[str]:
        """获取疾病的并发症"""
        pass

    @abstractmethod
    def get_cure_methods_by_disease(self, disease_name: str) -> List[str]:
        """获取疾病的治疗方法"""
        pass

    @abstractmethod
    def get_disease_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """通过Neo4j节点elementId获取疾病信息"""
        pass

    @abstractmethod
    def get_symptom_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """通过Neo4j节点elementId获取症状信息"""
        pass

    @abstractmethod
    def get_diseases_by_symptom_node_id(self, node_id: str, limit: int = 50) -> List[str]:
        """通过症状节点elementId查询相关疾病"""
        pass

    @abstractmethod
    def get_symptoms_by_node_id(self, node_id: str) -> List[str]:
        """通过Neo4j节点elementId获取疾病的症状"""
        pass

    @abstractmethod
    def get_drugs_by_node_id(self, node_id: str) -> Dict[str, List[str]]:
        """通过Neo4j节点elementId获取疾病的常用药物和推荐药物"""
        pass

    @abstractmethod
    def get_foods_by_node_id(self, node_id: str) -> Dict[str, List[str]]:
        """通过Neo4j节点elementId获取疾病的饮食建议"""
        pass

    @abstractmethod
    def search_diseases_by_symptom(self, symptom_name: str, limit: int = 50) -> List[str]:
        """根据症状搜索可能的疾病"""
        pass

    @abstractmethod
    def get_drug_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """通过Neo4j节点elementId获取药物信息"""
        pass

    @abstractmethod
    def get_diseases_by_drug_node_id(self, node_id: str, limit: int = 50) -> Dict[str, List[str]]:
        """通过药物节点elementId查询相关疾病"""
        pass

    @abstractmethod
    def get_food_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """通过Neo4j节点elementId获取食物信息"""
        pass

    @abstractmethod
    def get_diseases_by_food_node_id(self, node_id: str, limit: int = 50) -> Dict[str, List[str]]:
        """通过食物节点elementId查询相关疾病"""
        pass

    @abstractmethod
    def get_check_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """通过Neo4j节点elementId获取检查项目信息"""
        pass

    @abstractmethod
    def get_diseases_by_check_node_id(self, node_id: str, limit: int = 50) -> List[str]:
        """通过检查项目节点elementId查询相关疾病"""
        pass

    @abstractmethod
    def get_department_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """通过Neo4j节点elementId获取科室信息"""
        pass

    @abstractmethod
    def get_diseases_by_department_node_id(self, node_id: str, limit: int = 50) -> List[str]:
        """通过科室节点elementId查询相关疾病"""
        pass

    @abstractmethod
    def get_cure_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """通过Neo4j节点elementId获取治疗方法信息"""
        pass

    @abstractmethod
    def get_diseases_by_cure_node_id(self, node_id: str, limit: int = 50) -> List[str]:
        """通过治疗方法节点elementId查询相关疾病"""
        pass

    @abstractmethod
    def query_with_params(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """执行带参数的Cypher查询"""
        pass
