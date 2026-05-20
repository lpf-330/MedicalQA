# -*- coding: utf-8 -*-
"""
维度评估Chain策略

实现健康报告生成业务的维度评估逻辑，支持8个维度的评估。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.orchestration.chain.chain import Chain
from src.orchestration.chain.data_classes import ChainContext, ChainResult
from src.orchestration.tool_call_handler.Impl.vector_retrieval_handler import VectorRetrievalHandler
from src.orchestration.tool_call_handler.Impl.neo4j_medical_handler import Neo4jMedicalHandler

logger = logging.getLogger(__name__)


# 维度ID到维度名称的映射
DIMENSION_NAME_MAP = {
    "1": "疾病风险评估",
    "2": "用药建议",
    "3": "治疗方案",
    "4": "饮食建议",
    "5": "检查建议",
    "6": "并发症预警",
    "7": "预防措施",
    "8": "易感人群"
}


@dataclass
class DimensionEvaluationContextBody:
    """
    维度评估Chain策略专属输入数据类

    Attributes:
        anomalies: 异常指标列表
        risk_factors: 风险因子列表
        medical_entities: 医疗实体列表
        dimension_id: 维度ID（1-8）
    """
    anomalies: List[Dict] = field(default_factory=list)
    risk_factors: List[Dict] = field(default_factory=list)
    medical_entities: List[Dict] = field(default_factory=list)
    dimension_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "anomalies": self.anomalies,
            "risk_factors": self.risk_factors,
            "medical_entities": self.medical_entities,
            "dimension_id": self.dimension_id
        }


@dataclass
class DimensionEvaluationResultData:
    """
    维度评估Chain策略专属输出数据类

    Attributes:
        dimension_id: 维度ID
        dimension_name: 维度名称
        evaluation_result: 评估结果（包含具体建议和依据）
        confidence: 置信度（0.0-1.0）
    """
    dimension_id: str = ""
    dimension_name: str = ""
    evaluation_result: Dict = field(default_factory=dict)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "dimension_id": self.dimension_id,
            "dimension_name": self.dimension_name,
            "evaluation_result": self.evaluation_result,
            "confidence": self.confidence
        }


@dataclass
class DimensionEvaluationResource:
    """
    维度评估Chain策略专属资源类

    Attributes:
        vector_handler: 向量检索Handler（复用健康咨询的Handler）
        neo4j_handler: Neo4j医疗Handler（复用健康咨询的Handler）
        vector_encode_service: 向量编码服务（复用健康咨询的Service）
    """
    vector_handler: Optional[VectorRetrievalHandler] = None
    neo4j_handler: Optional[Neo4jMedicalHandler] = None
    vector_encode_service: Optional[Any] = None


class DimensionEvaluationChain(Chain[ChainContext[DimensionEvaluationContextBody], ChainResult[DimensionEvaluationResultData]]):
    """
    维度评估Chain策略类

    实现8个维度的评估逻辑：
    1. 疾病风险评估：基于异常指标和病史，调用向量检索和图谱查询
    2. 用药建议：基于既往病史和当前用药，查询药物信息
    3. 治疗方案：基于疾病诊断，查询治疗方案
    4. 饮食建议：基于疾病和BMI，查询饮食建议
    5. 检查建议：基于疾病和异常指标，查询检查项目
    6. 并发症预警：基于疾病和病史，查询并发症
    7. 预防措施：基于疾病和风险因子，查询预防措施
    8. 易感人群：基于年龄、性别、病史，评估易感疾病
    """

    # 相关性阈值
    RELEVANCE_THRESHOLD = 0.5
    # 检索结果数量限制
    TOP_K_LIMIT = 10

    def __init__(self, resource: DimensionEvaluationResource):
        """
        初始化维度评估Chain策略

        Args:
            resource: Chain策略专属资源
        """
        self._resource = resource
        self._milvus_degraded = False
        self._neo4j_degraded = False

    def execute(self, chain_context: ChainContext[DimensionEvaluationContextBody]) -> ChainResult[DimensionEvaluationResultData]:
        """
        执行Chain策略

        Args:
            chain_context: Chain输入数据容器

        Returns:
            ChainResult: Chain输出数据容器
        """
        start_time = time.time()
        logger.info(f"[DimensionEvaluationChain] 开始执行Chain: session_id={chain_context.session_id}")

        body = chain_context.body
        if body is None:
            logger.warning(f"[DimensionEvaluationChain] 输入数据为空: session_id={chain_context.session_id}")
            return ChainResult(
                session_id=chain_context.session_id,
                data=DimensionEvaluationResultData()
            )

        # 获取维度ID和名称
        dimension_id = body.dimension_id
        dimension_name = DIMENSION_NAME_MAP.get(dimension_id, "未知维度")
        logger.info(f"[DimensionEvaluationChain] 维度ID={dimension_id}, 维度名称={dimension_name}")

        # 重置降级标志
        self._milvus_degraded = False
        self._neo4j_degraded = False

        # 根据维度ID执行对应的评估逻辑
        try:
            evaluation_result, confidence = self._evaluate_by_dimension(body)
        except Exception as e:
            logger.error(f"[DimensionEvaluationChain] 维度评估失败: dimension_id={dimension_id}, error={str(e)}")
            evaluation_result = {"error": str(e), "suggestions": []}
            confidence = 0.0

        # 构建结果数据
        result_data = DimensionEvaluationResultData(
            dimension_id=dimension_id,
            dimension_name=dimension_name,
            evaluation_result=evaluation_result,
            confidence=confidence
        )

        elapsed = time.time() - start_time
        logger.info(f"[DimensionEvaluationChain] Chain执行完成: session_id={chain_context.session_id}, "
                   f"dimension_id={dimension_id}, confidence={confidence:.2f}, elapsed={elapsed:.2f}s")

        return ChainResult(session_id=chain_context.session_id, data=result_data)

    def _evaluate_by_dimension(self, body: DimensionEvaluationContextBody) -> Tuple[Dict, float]:
        """
        根据维度ID执行对应的评估逻辑

        Args:
            body: Chain策略专属输入数据

        Returns:
            Tuple[评估结果, 置信度]
        """
        dimension_id = body.dimension_id

        # 维度1：疾病风险评估
        if dimension_id == "1":
            return self._evaluate_disease_risk(body)
        # 维度2：用药建议
        elif dimension_id == "2":
            return self._evaluate_medication(body)
        # 维度3：治疗方案
        elif dimension_id == "3":
            return self._evaluate_treatment(body)
        # 维度4：饮食建议
        elif dimension_id == "4":
            return self._evaluate_diet(body)
        # 维度5：检查建议
        elif dimension_id == "5":
            return self._evaluate_examination(body)
        # 维度6：并发症预警
        elif dimension_id == "6":
            return self._evaluate_complication(body)
        # 维度7：预防措施
        elif dimension_id == "7":
            return self._evaluate_prevention(body)
        # 维度8：易感人群
        elif dimension_id == "8":
            return self._evaluate_susceptible_population(body)
        else:
            logger.warning(f"[DimensionEvaluationChain] 未知的维度ID: {dimension_id}")
            return {"error": f"未知的维度ID: {dimension_id}", "suggestions": []}, 0.0

    def _evaluate_disease_risk(self, body: DimensionEvaluationContextBody) -> Tuple[Dict, float]:
        """
        维度1：疾病风险评估

        基于异常指标和病史，调用向量检索和图谱查询

        Args:
            body: Chain策略专属输入数据

        Returns:
            Tuple[评估结果, 置信度]
        """
        logger.info("[DimensionEvaluationChain] 执行疾病风险评估")

        # 构建查询文本
        query_text = self._build_disease_risk_query(body)
        logger.info(f"[DimensionEvaluationChain] 疾病风险评估查询文本: {query_text}")

        # 执行检索
        vector_results, knowledge_results = self._retrieve_knowledge(query_text, body.medical_entities)

        # 整合评估结果
        evaluation_result = {
            "risk_level": self._calculate_risk_level(body.anomalies, body.risk_factors),
            "risk_diseases": self._extract_diseases_from_results(knowledge_results),
            "suggestions": self._generate_risk_suggestions(vector_results, knowledge_results),
            "basis": self._extract_basis_from_results(vector_results, knowledge_results)
        }

        confidence = self._calculate_confidence(vector_results, knowledge_results)
        return evaluation_result, confidence

    def _evaluate_medication(self, body: DimensionEvaluationContextBody) -> Tuple[Dict, float]:
        """
        维度2：用药建议

        基于既往病史和当前用药，查询药物信息

        Args:
            body: Chain策略专属输入数据

        Returns:
            Tuple[评估结果, 置信度]
        """
        logger.info("[DimensionEvaluationChain] 执行用药建议评估")

        # 构建查询文本
        query_text = self._build_medication_query(body)
        logger.info(f"[DimensionEvaluationChain] 用药建议查询文本: {query_text}")

        # 执行检索
        vector_results, knowledge_results = self._retrieve_knowledge(query_text, body.medical_entities)

        # 整合评估结果
        evaluation_result = {
            "current_medication_review": self._review_current_medications(body.medical_entities),
            "recommended_medications": self._extract_drugs_from_results(knowledge_results),
            "precautions": self._extract_medication_precautions(vector_results, knowledge_results),
            "suggestions": self._generate_medication_suggestions(vector_results, knowledge_results)
        }

        confidence = self._calculate_confidence(vector_results, knowledge_results)
        return evaluation_result, confidence

    def _evaluate_treatment(self, body: DimensionEvaluationContextBody) -> Tuple[Dict, float]:
        """
        维度3：治疗方案

        基于疾病诊断，查询治疗方案

        Args:
            body: Chain策略专属输入数据

        Returns:
            Tuple[评估结果, 置信度]
        """
        logger.info("[DimensionEvaluationChain] 执行治疗方案评估")

        # 构建查询文本
        query_text = self._build_treatment_query(body)
        logger.info(f"[DimensionEvaluationChain] 治疗方案查询文本: {query_text}")

        # 执行检索
        vector_results, knowledge_results = self._retrieve_knowledge(query_text, body.medical_entities)

        # 整合评估结果
        evaluation_result = {
            "treatment_options": self._extract_treatments_from_results(knowledge_results),
            "lifestyle_interventions": self._extract_lifestyle_from_results(vector_results),
            "follow_up_plan": self._generate_follow_up_plan(body.anomalies),
            "suggestions": self._generate_treatment_suggestions(vector_results, knowledge_results)
        }

        confidence = self._calculate_confidence(vector_results, knowledge_results)
        return evaluation_result, confidence

    def _evaluate_diet(self, body: DimensionEvaluationContextBody) -> Tuple[Dict, float]:
        """
        维度4：饮食建议

        基于疾病和BMI，查询饮食建议

        Args:
            body: Chain策略专属输入数据

        Returns:
            Tuple[评估结果, 置信度]
        """
        logger.info("[DimensionEvaluationChain] 执行饮食建议评估")

        # 构建查询文本
        query_text = self._build_diet_query(body)
        logger.info(f"[DimensionEvaluationChain] 饮食建议查询文本: {query_text}")

        # 执行检索
        vector_results, knowledge_results = self._retrieve_knowledge(query_text, body.medical_entities)

        # 整合评估结果
        evaluation_result = {
            "recommended_foods": self._extract_recommended_foods(knowledge_results),
            "forbidden_foods": self._extract_forbidden_foods(knowledge_results),
            "dietary_principles": self._extract_dietary_principles(vector_results),
            "suggestions": self._generate_diet_suggestions(vector_results, knowledge_results)
        }

        confidence = self._calculate_confidence(vector_results, knowledge_results)
        return evaluation_result, confidence

    def _evaluate_examination(self, body: DimensionEvaluationContextBody) -> Tuple[Dict, float]:
        """
        维度5：检查建议

        基于疾病和异常指标，查询检查项目

        Args:
            body: Chain策略专属输入数据

        Returns:
            Tuple[评估结果, 置信度]
        """
        logger.info("[DimensionEvaluationChain] 执行检查建议评估")

        # 构建查询文本
        query_text = self._build_examination_query(body)
        logger.info(f"[DimensionEvaluationChain] 检查建议查询文本: {query_text}")

        # 执行检索
        vector_results, knowledge_results = self._retrieve_knowledge(query_text, body.medical_entities)

        # 整合评估结果
        evaluation_result = {
            "recommended_examinations": self._extract_examinations_from_results(knowledge_results),
            "examination_frequency": self._determine_examination_frequency(body.anomalies),
            "key_indicators": self._extract_key_indicators(body.anomalies),
            "suggestions": self._generate_examination_suggestions(vector_results, knowledge_results)
        }

        confidence = self._calculate_confidence(vector_results, knowledge_results)
        return evaluation_result, confidence

    def _evaluate_complication(self, body: DimensionEvaluationContextBody) -> Tuple[Dict, float]:
        """
        维度6：并发症预警

        基于疾病和病史，查询并发症

        Args:
            body: Chain策略专属输入数据

        Returns:
            Tuple[评估结果, 置信度]
        """
        logger.info("[DimensionEvaluationChain] 执行并发症预警评估")

        # 构建查询文本
        query_text = self._build_complication_query(body)
        logger.info(f"[DimensionEvaluationChain] 并发症预警查询文本: {query_text}")

        # 执行检索
        vector_results, knowledge_results = self._retrieve_knowledge(query_text, body.medical_entities)

        # 整合评估结果
        evaluation_result = {
            "potential_complications": self._extract_complications_from_results(knowledge_results),
            "warning_signs": self._extract_warning_signs(vector_results),
            "prevention_measures": self._extract_prevention_measures(vector_results, knowledge_results),
            "suggestions": self._generate_complication_suggestions(vector_results, knowledge_results)
        }

        confidence = self._calculate_confidence(vector_results, knowledge_results)
        return evaluation_result, confidence

    def _evaluate_prevention(self, body: DimensionEvaluationContextBody) -> Tuple[Dict, float]:
        """
        维度7：预防措施

        基于疾病和风险因子，查询预防措施

        Args:
            body: Chain策略专属输入数据

        Returns:
            Tuple[评估结果, 置信度]
        """
        logger.info("[DimensionEvaluationChain] 执行预防措施评估")

        # 构建查询文本
        query_text = self._build_prevention_query(body)
        logger.info(f"[DimensionEvaluationChain] 预防措施查询文本: {query_text}")

        # 执行检索
        vector_results, knowledge_results = self._retrieve_knowledge(query_text, body.medical_entities)

        # 整合评估结果
        evaluation_result = {
            "lifestyle_prevention": self._extract_lifestyle_prevention(vector_results),
            "vaccination_recommendations": self._extract_vaccination_recommendations(knowledge_results),
            "regular_checkups": self._generate_regular_checkup_plan(body.risk_factors),
            "suggestions": self._generate_prevention_suggestions(vector_results, knowledge_results)
        }

        confidence = self._calculate_confidence(vector_results, knowledge_results)
        return evaluation_result, confidence

    def _evaluate_susceptible_population(self, body: DimensionEvaluationContextBody) -> Tuple[Dict, float]:
        """
        维度8：易感人群

        基于年龄、性别、病史，评估易感疾病

        Args:
            body: Chain策略专属输入数据

        Returns:
            Tuple[评估结果, 置信度]
        """
        logger.info("[DimensionEvaluationChain] 执行易感人群评估")

        # 构建查询文本
        query_text = self._build_susceptible_query(body)
        logger.info(f"[DimensionEvaluationChain] 易感人群查询文本: {query_text}")

        # 执行检索
        vector_results, knowledge_results = self._retrieve_knowledge(query_text, body.medical_entities)

        # 整合评估结果
        evaluation_result = {
            "susceptible_diseases": self._extract_susceptible_diseases(knowledge_results),
            "risk_factors_analysis": self._analyze_risk_factors(body.risk_factors),
            "prevention_priorities": self._extract_prevention_priorities(vector_results),
            "suggestions": self._generate_susceptible_suggestions(vector_results, knowledge_results)
        }

        confidence = self._calculate_confidence(vector_results, knowledge_results)
        return evaluation_result, confidence

    def _retrieve_knowledge(self, query_text: str, medical_entities: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        执行知识检索：向量检索 + 图谱查询

        Args:
            query_text: 查询文本
            medical_entities: 医疗实体列表

        Returns:
            Tuple[向量检索结果, 图谱查询结果]
        """
        vector_results: List[Dict] = []
        knowledge_results: List[Dict] = []

        # 步骤1：向量检索
        try:
            vector_results = self._vector_search(query_text)
            logger.info(f"[DimensionEvaluationChain] 向量检索完成: result_count={len(vector_results)}")
        except Exception as e:
            logger.error(f"[DimensionEvaluationChain] 向量检索失败，启用Milvus降级策略: {e}")
            self._milvus_degraded = True

        # 步骤2：图谱查询
        try:
            if medical_entities:
                knowledge_results = self._graph_query(medical_entities, query_text)
                logger.info(f"[DimensionEvaluationChain] 图谱查询完成: result_count={len(knowledge_results)}")
        except Exception as e:
            logger.error(f"[DimensionEvaluationChain] 图谱查询失败，启用Neo4j降级策略: {e}")
            self._neo4j_degraded = True

        # 记录降级状态
        if self._milvus_degraded:
            logger.warning("[DimensionEvaluationChain] 降级模式: Milvus不可用，使用Neo4j模糊匹配替代")
        if self._neo4j_degraded:
            logger.warning("[DimensionEvaluationChain] 降级模式: Neo4j不可用，仅使用向量检索结果")

        return vector_results, knowledge_results

    def _vector_search(self, query_text: str) -> List[Dict]:
        """
        向量检索

        Args:
            query_text: 查询文本

        Returns:
            向量检索结果列表
        """
        if self._resource.vector_handler is None:
            logger.warning("[DimensionEvaluationChain] vector_handler未初始化")
            return []

        # 调用Milvus三集合检索
        search_result = self._resource.vector_handler.call_tool({
            "query": query_text,
            "top_k": self.TOP_K_LIMIT,
            "collections": ["medical_entity", "entity_attributes", "entity_relations"],
            "weights": {"medical_entity": 0.40, "entity_attributes": 0.30, "entity_relations": 0.30}
        })

        results = []
        if search_result:
            if isinstance(search_result, list):
                results_list = search_result
            elif isinstance(search_result, dict):
                results_list = search_result.get("results", [])
                if not results_list:
                    for key, value in search_result.items():
                        if isinstance(value, list):
                            results_list = value
                            break
            else:
                results_list = []

            for item in results_list:
                if isinstance(item, dict):
                    results.append(item)

        logger.info(f"[DimensionEvaluationChain] 向量检索: query='{query_text}', total={len(results)}")
        return results

    def _graph_query(self, medical_entities: List[Dict], query: str) -> List[Dict]:
        """
        图谱查询
        
        支持两种数据格式：
        1. 带neo4j_node_id的格式（来自IntentClassificationHandler）
        2. 不带neo4j_node_id的格式（来自规则引擎降级）
        
        Args:
            medical_entities: 医疗实体列表
            query: 查询文本
        
        Returns:
            图谱查询结果列表
        """
        if self._resource.neo4j_handler is None:
            logger.warning("[DimensionEvaluationChain] neo4j_handler未初始化")
            return []
        
        knowledge_results: List[Dict] = []
        seen_node_ids = set()
        seen_entity_names = set()
        
        for entity in medical_entities:
            entity_data = entity.get("entity", {})
            
            if not entity_data:
                entity_data = entity
            
            entity_name = entity_data.get("name") or entity_data.get("entity_name")
            entity_type = entity_data.get("entity_type", "Disease")
            
            if not entity_name:
                continue
            
            neo4j_node_id_str = entity_data.get("neo4j_node_id")
            if neo4j_node_id_str:
                try:
                    neo4j_node_id = int(neo4j_node_id_str)
                    if neo4j_node_id in seen_node_ids:
                        continue
                    seen_node_ids.add(neo4j_node_id)
                    
                    knowledge_item = self._query_knowledge_by_node_id(neo4j_node_id, entity_type, entity)
                    if knowledge_item:
                        knowledge_results.append(knowledge_item)
                    continue
                except (ValueError, TypeError):
                    pass
            
            if entity_name in seen_entity_names:
                continue
            seen_entity_names.add(entity_name)
            
            knowledge_item = self._query_knowledge_by_name(entity_name, entity_type, entity)
            if knowledge_item:
                knowledge_results.append(knowledge_item)
        
        return knowledge_results
    
    def _query_knowledge_by_node_id(self, node_id: int, entity_type: str, entity: Dict) -> Optional[Dict]:
        """
        通过node_id查询知识
        
        Args:
            node_id: Neo4j节点ID
            entity_type: 实体类型
            entity: 实体数据
        
        Returns:
            知识字典
        """
        if entity_type == "Disease":
            return self._query_disease_knowledge(node_id, entity)
        elif entity_type == "Symptom":
            return self._query_symptom_knowledge(node_id, entity)
        return None
    
    def _query_knowledge_by_name(self, entity_name: str, entity_type: str, entity: Dict) -> Optional[Dict]:
        """
        通过实体名称查询知识（降级方案）
        
        Args:
            entity_name: 实体名称
            entity_type: 实体类型
            entity: 实体数据
        
        Returns:
            知识字典
        """
        if self._resource.neo4j_handler is None:
            return None
        
        try:
            if entity_type == "Disease":
                disease_info = self._resource.neo4j_handler.get_disease_info(entity_name)
                if not disease_info:
                    return None
                
                logger.info(f"[DimensionEvaluationChain] 通过名称查询疾病知识: disease={entity_name}")
                
                symptoms = self._resource.neo4j_handler.get_symptoms_by_disease(entity_name)
                drugs = self._resource.neo4j_handler.get_drugs_by_disease(entity_name)
                foods = self._resource.neo4j_handler.get_foods_by_disease(entity_name)
                
                return {
                    "source": "neo4j",
                    "type": "disease",
                    "entity": entity_name,
                    "data": {
                        "name": entity_name,
                        "desc": disease_info.get("desc", ""),
                        "cause": disease_info.get("cause", ""),
                        "prevent": disease_info.get("prevent", ""),
                        "symptoms": symptoms,
                        "drugs": drugs,
                        "foods": foods
                    },
                    "score": entity.get("confidence", 0.0)
                }
            elif entity_type == "Symptom":
                diseases = self._resource.neo4j_handler.search_diseases_by_symptom(entity_name)
                
                logger.info(f"[DimensionEvaluationChain] 通过名称查询症状知识: symptom={entity_name}")
                
                return {
                    "source": "neo4j",
                    "type": "symptom",
                    "entity": entity_name,
                    "data": {
                        "name": entity_name,
                        "related_diseases": diseases
                    },
                    "score": entity.get("confidence", 0.0)
                }
        except Exception as e:
            logger.error(f"[DimensionEvaluationChain] 通过名称查询知识失败: entity_name={entity_name}, error={str(e)}")
        
        return None

    def _query_disease_knowledge(self, node_id: int, entity: Dict) -> Optional[Dict]:
        """
        查询疾病知识

        Args:
            node_id: Neo4j节点ID
            entity: 实体数据

        Returns:
            疾病知识字典
        """
        if self._resource.neo4j_handler is None:
            return None

        # 查询疾病信息
        disease_info = self._resource.neo4j_handler.get_disease_by_node_id(node_id)
        if not disease_info:
            return None

        disease_name = disease_info.get("name", "")
        logger.info(f"[DimensionEvaluationChain] 查询疾病知识: disease={disease_name}")

        # 查询症状
        symptoms = self._resource.neo4j_handler.get_symptoms_by_node_id(node_id)

        # 查询药物
        drugs = self._resource.neo4j_handler.get_drugs_by_node_id(node_id)

        # 查询饮食建议
        foods = self._resource.neo4j_handler.get_foods_by_node_id(node_id)

        return {
            "source": "neo4j",
            "type": "disease",
            "entity": disease_name,
            "data": {
                "name": disease_name,
                "desc": disease_info.get("desc", ""),
                "cause": disease_info.get("cause", ""),
                "prevent": disease_info.get("prevent", ""),
                "symptoms": symptoms,
                "drugs": drugs,
                "foods": foods
            },
            "score": entity.get("score", 0.0)
        }

    def _query_symptom_knowledge(self, node_id: int, entity: Dict) -> Optional[Dict]:
        """
        查询症状知识

        Args:
            node_id: Neo4j节点ID
            entity: 实体数据

        Returns:
            症状知识字典
        """
        if self._resource.neo4j_handler is None:
            return None

        # 查询症状信息
        symptom_info = self._resource.neo4j_handler.get_symptom_by_node_id(node_id)
        if not symptom_info:
            return None

        symptom_name = symptom_info.get("name", "")
        logger.info(f"[DimensionEvaluationChain] 查询症状知识: symptom={symptom_name}")

        # 查询相关疾病
        diseases = self._resource.neo4j_handler.get_diseases_by_symptom_node_id(node_id)

        return {
            "source": "neo4j",
            "type": "symptom",
            "entity": symptom_name,
            "data": {
                "name": symptom_name,
                "related_diseases": diseases
            },
            "score": entity.get("score", 0.0)
        }

    # ========== 查询文本构建方法 ==========

    def _build_disease_risk_query(self, body: DimensionEvaluationContextBody) -> str:
        """构建疾病风险评估查询文本"""
        parts = []
        if body.anomalies:
            anomaly_names = [a.get("name", "") for a in body.anomalies if a.get("name")]
            if anomaly_names:
                parts.append(f"异常指标: {', '.join(anomaly_names)}")
        if body.risk_factors:
            factor_names = [f.get("name", "") for f in body.risk_factors if f.get("name")]
            if factor_names:
                parts.append(f"风险因子: {', '.join(factor_names)}")
        return " ".join(parts) if parts else "疾病风险评估"

    def _build_medication_query(self, body: DimensionEvaluationContextBody) -> str:
        """构建用药建议查询文本"""
        parts = []
        if body.medical_entities:
            disease_names = [e.get("entity", {}).get("name", "") for e in body.medical_entities
                           if e.get("entity", {}).get("entity_type") == "Disease"]
            if disease_names:
                parts.append(f"疾病: {', '.join(disease_names)}")
        return " ".join(parts) if parts else "用药建议"

    def _build_treatment_query(self, body: DimensionEvaluationContextBody) -> str:
        """构建治疗方案查询文本"""
        parts = []
        if body.medical_entities:
            disease_names = [e.get("entity", {}).get("name", "") for e in body.medical_entities
                           if e.get("entity", {}).get("entity_type") == "Disease"]
            if disease_names:
                parts.append(f"疾病: {', '.join(disease_names)}")
        return " ".join(parts) if parts else "治疗方案"

    def _build_diet_query(self, body: DimensionEvaluationContextBody) -> str:
        """构建饮食建议查询文本"""
        parts = []
        if body.medical_entities:
            disease_names = [e.get("entity", {}).get("name", "") for e in body.medical_entities
                           if e.get("entity", {}).get("entity_type") == "Disease"]
            if disease_names:
                parts.append(f"疾病: {', '.join(disease_names)}")
        return " ".join(parts) if parts else "饮食建议"

    def _build_examination_query(self, body: DimensionEvaluationContextBody) -> str:
        """构建检查建议查询文本"""
        parts = []
        if body.anomalies:
            anomaly_names = [a.get("name", "") for a in body.anomalies if a.get("name")]
            if anomaly_names:
                parts.append(f"异常指标: {', '.join(anomaly_names)}")
        if body.medical_entities:
            disease_names = [e.get("entity", {}).get("name", "") for e in body.medical_entities
                           if e.get("entity", {}).get("entity_type") == "Disease"]
            if disease_names:
                parts.append(f"疾病: {', '.join(disease_names)}")
        return " ".join(parts) if parts else "检查建议"

    def _build_complication_query(self, body: DimensionEvaluationContextBody) -> str:
        """构建并发症预警查询文本"""
        parts = []
        if body.medical_entities:
            disease_names = [e.get("entity", {}).get("name", "") for e in body.medical_entities
                           if e.get("entity", {}).get("entity_type") == "Disease"]
            if disease_names:
                parts.append(f"疾病: {', '.join(disease_names)}")
        return " ".join(parts) if parts else "并发症预警"

    def _build_prevention_query(self, body: DimensionEvaluationContextBody) -> str:
        """构建预防措施查询文本"""
        parts = []
        if body.medical_entities:
            disease_names = [e.get("entity", {}).get("name", "") for e in body.medical_entities
                           if e.get("entity", {}).get("entity_type") == "Disease"]
            if disease_names:
                parts.append(f"疾病: {', '.join(disease_names)}")
        if body.risk_factors:
            factor_names = [f.get("name", "") for f in body.risk_factors if f.get("name")]
            if factor_names:
                parts.append(f"风险因子: {', '.join(factor_names)}")
        return " ".join(parts) if parts else "预防措施"

    def _build_susceptible_query(self, body: DimensionEvaluationContextBody) -> str:
        """构建易感人群查询文本"""
        parts = []
        if body.risk_factors:
            factor_names = [f.get("name", "") for f in body.risk_factors if f.get("name")]
            if factor_names:
                parts.append(f"风险因子: {', '.join(factor_names)}")
        return " ".join(parts) if parts else "易感人群"

    # ========== 结果提取方法 ==========

    def _calculate_risk_level(self, anomalies: List[Dict], risk_factors: List[Dict]) -> str:
        """计算风险等级"""
        risk_score = 0
        if anomalies:
            risk_score += len(anomalies) * 10
        if risk_factors:
            risk_score += len(risk_factors) * 15

        if risk_score >= 50:
            return "高风险"
        elif risk_score >= 30:
            return "中风险"
        else:
            return "低风险"

    def _extract_diseases_from_results(self, knowledge_results: List[Dict]) -> List[str]:
        """从结果中提取疾病列表"""
        diseases = []
        for item in knowledge_results:
            if item.get("type") == "disease":
                disease_name = item.get("entity", "")
                if disease_name and disease_name not in diseases:
                    diseases.append(disease_name)
        return diseases[:5]  # 最多返回5个

    def _extract_drugs_from_results(self, knowledge_results: List[Dict]) -> Dict[str, List[str]]:
        """从结果中提取药物信息"""
        drugs = {"common_drugs": [], "recommand_drugs": []}
        for item in knowledge_results:
            if item.get("type") == "disease":
                data = item.get("data", {})
                item_drugs = data.get("drugs", {})
                if isinstance(item_drugs, dict):
                    common = item_drugs.get("common_drugs", [])
                    recommand = item_drugs.get("recommand_drugs", [])
                    drugs["common_drugs"].extend(common)
                    drugs["recommand_drugs"].extend(recommand)
        # 去重
        drugs["common_drugs"] = list(set(drugs["common_drugs"]))
        drugs["recommand_drugs"] = list(set(drugs["recommand_drugs"]))
        return drugs

    def _extract_treatments_from_results(self, knowledge_results: List[Dict]) -> List[str]:
        """从结果中提取治疗方案"""
        treatments = []
        for item in knowledge_results:
            if item.get("type") == "disease":
                data = item.get("data", {})
                desc = data.get("desc", "")
                if desc:
                    treatments.append(desc)
        return treatments[:3]

    def _extract_recommended_foods(self, knowledge_results: List[Dict]) -> List[str]:
        """从结果中提取推荐食物"""
        foods = []
        for item in knowledge_results:
            if item.get("type") == "disease":
                data = item.get("data", {})
                item_foods = data.get("foods", {})
                if isinstance(item_foods, dict):
                    do_eat = item_foods.get("do_eat", [])
                    recommand_eat = item_foods.get("recommand_eat", [])
                    foods.extend(do_eat)
                    foods.extend(recommand_eat)
        return list(set(foods))[:10]

    def _extract_forbidden_foods(self, knowledge_results: List[Dict]) -> List[str]:
        """从结果中提取禁忌食物"""
        foods = []
        for item in knowledge_results:
            if item.get("type") == "disease":
                data = item.get("data", {})
                item_foods = data.get("foods", {})
                if isinstance(item_foods, dict):
                    no_eat = item_foods.get("no_eat", [])
                    foods.extend(no_eat)
        return list(set(foods))[:10]

    def _extract_examinations_from_results(self, knowledge_results: List[Dict]) -> List[str]:
        """从结果中提取检查项目"""
        examinations = []
        for item in knowledge_results:
            if item.get("type") == "disease":
                data = item.get("data", {})
                symptoms = data.get("symptoms", [])
                examinations.extend(symptoms)  # 简化处理，实际应查询检查项目
        return list(set(examinations))[:5]

    def _extract_complications_from_results(self, knowledge_results: List[Dict]) -> List[str]:
        """从结果中提取并发症"""
        complications = []
        for item in knowledge_results:
            if item.get("type") == "disease":
                data = item.get("data", {})
                cause = data.get("cause", "")
                if cause:
                    complications.append(cause)
        return complications[:3]

    def _extract_susceptible_diseases(self, knowledge_results: List[Dict]) -> List[str]:
        """从结果中提取易感疾病"""
        diseases = []
        for item in knowledge_results:
            if item.get("type") == "disease":
                disease_name = item.get("entity", "")
                if disease_name:
                    diseases.append(disease_name)
            elif item.get("type") == "symptom":
                data = item.get("data", {})
                related = data.get("related_diseases", [])
                diseases.extend(related)
        return list(set(diseases))[:5]

    def _extract_basis_from_results(self, vector_results: List[Dict], knowledge_results: List[Dict]) -> List[str]:
        """从结果中提取依据"""
        basis = []
        for item in knowledge_results:
            entity = item.get("entity", "")
            item_type = item.get("type", "")
            if entity and item_type:
                basis.append(f"{item_type}: {entity}")
        return basis[:5]

    def _generate_risk_suggestions(self, vector_results: List[Dict], knowledge_results: List[Dict]) -> List[str]:
        """生成风险评估建议"""
        suggestions = []
        for item in knowledge_results:
            data = item.get("data", {})
            prevent = data.get("prevent", "")
            if prevent:
                suggestions.append(prevent)
        return suggestions[:3]

    def _review_current_medications(self, medical_entities: List[Dict]) -> List[str]:
        """审查当前用药"""
        medications = []
        for entity in medical_entities:
            entity_data = entity.get("entity", {})
            if entity_data.get("entity_type") == "Drug":
                drug_name = entity_data.get("name", "")
                if drug_name:
                    medications.append(drug_name)
        return medications

    def _extract_medication_precautions(self, vector_results: List[Dict], knowledge_results: List[Dict]) -> List[str]:
        """提取用药注意事项"""
        precautions = []
        for item in knowledge_results:
            data = item.get("data", {})
            desc = data.get("desc", "")
            if desc:
                precautions.append(desc)
        return precautions[:3]

    def _generate_medication_suggestions(self, vector_results: List[Dict], knowledge_results: List[Dict]) -> List[str]:
        """生成用药建议"""
        suggestions = []
        for item in knowledge_results:
            data = item.get("data", {})
            drugs = data.get("drugs", {})
            if isinstance(drugs, dict):
                recommand = drugs.get("recommand_drugs", [])
                if recommand:
                    suggestions.append(f"推荐药物: {', '.join(recommand[:3])}")
        return suggestions[:3]

    def _extract_lifestyle_from_results(self, vector_results: List[Dict]) -> List[str]:
        """提取生活方式干预"""
        return ["保持规律作息", "适量运动", "戒烟限酒"]

    def _generate_follow_up_plan(self, anomalies: List[Dict]) -> str:
        """生成随访计划"""
        if anomalies:
            return "建议每月复查异常指标"
        return "建议定期体检"

    def _generate_treatment_suggestions(self, vector_results: List[Dict], knowledge_results: List[Dict]) -> List[str]:
        """生成治疗建议"""
        suggestions = []
        for item in knowledge_results:
            data = item.get("data", {})
            prevent = data.get("prevent", "")
            if prevent:
                suggestions.append(prevent)
        return suggestions[:3]

    def _extract_dietary_principles(self, vector_results: List[Dict]) -> List[str]:
        """提取饮食原则"""
        return ["清淡饮食", "均衡营养", "少食多餐"]

    def _generate_diet_suggestions(self, vector_results: List[Dict], knowledge_results: List[Dict]) -> List[str]:
        """生成饮食建议"""
        suggestions = []
        for item in knowledge_results:
            data = item.get("data", {})
            foods = data.get("foods", {})
            if isinstance(foods, dict):
                recommand_eat = foods.get("recommand_eat", [])
                if recommand_eat:
                    suggestions.append(f"推荐食用: {', '.join(recommand_eat[:3])}")
        return suggestions[:3]

    def _determine_examination_frequency(self, anomalies: List[Dict]) -> str:
        """确定检查频率"""
        if anomalies:
            return "建议每月检查一次"
        return "建议每季度检查一次"

    def _extract_key_indicators(self, anomalies: List[Dict]) -> List[str]:
        """提取关键指标"""
        indicators = []
        for anomaly in anomalies:
            name = anomaly.get("name", "")
            if name:
                indicators.append(name)
        return indicators[:5]

    def _generate_examination_suggestions(self, vector_results: List[Dict], knowledge_results: List[Dict]) -> List[str]:
        """生成检查建议"""
        suggestions = []
        for item in knowledge_results:
            entity = item.get("entity", "")
            if entity:
                suggestions.append(f"建议进行{entity}相关检查")
        return suggestions[:3]

    def _extract_warning_signs(self, vector_results: List[Dict]) -> List[str]:
        """提取预警信号"""
        return ["持续发热", "体重骤降", "剧烈疼痛"]

    def _extract_prevention_measures(self, vector_results: List[Dict], knowledge_results: List[Dict]) -> List[str]:
        """提取预防措施"""
        measures = []
        for item in knowledge_results:
            data = item.get("data", {})
            prevent = data.get("prevent", "")
            if prevent:
                measures.append(prevent)
        return measures[:3]

    def _generate_complication_suggestions(self, vector_results: List[Dict], knowledge_results: List[Dict]) -> List[str]:
        """生成并发症预防建议"""
        suggestions = ["定期复查", "遵医嘱用药", "保持健康生活方式"]
        return suggestions

    def _extract_lifestyle_prevention(self, vector_results: List[Dict]) -> List[str]:
        """提取生活方式预防"""
        return ["规律作息", "健康饮食", "适量运动", "戒烟限酒"]

    def _extract_vaccination_recommendations(self, knowledge_results: List[Dict]) -> List[str]:
        """提取疫苗接种建议"""
        return ["流感疫苗", "肺炎疫苗"]

    def _generate_regular_checkup_plan(self, risk_factors: List[Dict]) -> str:
        """生成定期体检计划"""
        if risk_factors:
            return "建议每半年进行一次全面体检"
        return "建议每年进行一次全面体检"

    def _generate_prevention_suggestions(self, vector_results: List[Dict], knowledge_results: List[Dict]) -> List[str]:
        """生成预防建议"""
        suggestions = []
        for item in knowledge_results:
            data = item.get("data", {})
            prevent = data.get("prevent", "")
            if prevent:
                suggestions.append(prevent)
        return suggestions[:3]

    def _analyze_risk_factors(self, risk_factors: List[Dict]) -> List[str]:
        """分析风险因子"""
        factors = []
        for factor in risk_factors:
            name = factor.get("name", "")
            if name:
                factors.append(name)
        return factors[:5]

    def _extract_prevention_priorities(self, vector_results: List[Dict]) -> List[str]:
        """提取预防重点"""
        return ["控制体重", "管理血压", "调节血糖"]

    def _generate_susceptible_suggestions(self, vector_results: List[Dict], knowledge_results: List[Dict]) -> List[str]:
        """生成易感人群建议"""
        suggestions = ["加强锻炼", "定期体检", "注意饮食"]
        return suggestions

    def _calculate_confidence(self, vector_results: List[Dict], knowledge_results: List[Dict]) -> float:
        """
        计算置信度

        Args:
            vector_results: 向量检索结果
            knowledge_results: 图谱查询结果

        Returns:
            置信度（0.0-1.0）
        """
        # 基础置信度
        confidence = 0.0

        # 向量检索贡献
        if vector_results:
            avg_score = sum(item.get("score", 0.0) for item in vector_results) / len(vector_results)
            confidence += avg_score * 0.4

        # 图谱查询贡献
        if knowledge_results:
            confidence += 0.6

        # 降级惩罚
        if self._milvus_degraded:
            confidence *= 0.7
        if self._neo4j_degraded:
            confidence *= 0.8

        return min(max(confidence, 0.0), 1.0)
