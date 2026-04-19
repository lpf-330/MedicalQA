# -*- coding: utf-8 -*-
"""
报告知识检索Chain策略

实现健康报告生成业务的知识检索Chain策略。
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


@dataclass
class ReportKnowledgeRetrievalContextBody:
    """
    报告知识检索Chain策略专属输入数据类

    Attributes:
        anomalies: 异常指标列表
        medical_entities: 医疗实体列表
        risk_diseases: 风险疾病列表
    """
    anomalies: List[Dict] = field(default_factory=list)
    medical_entities: List[Dict] = field(default_factory=list)
    risk_diseases: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "anomalies": self.anomalies,
            "medical_entities": self.medical_entities,
            "risk_diseases": self.risk_diseases
        }


@dataclass
class ReportKnowledgeRetrievalResultData:
    """
    报告知识检索Chain策略专属输出数据类

    Attributes:
        vector_results: 向量检索原始结果
        knowledge_results: 图谱查询增强结果
        merged_results: 合并去重后的最终知识素材
    """
    vector_results: List[Dict] = field(default_factory=list)
    knowledge_results: List[Dict] = field(default_factory=list)
    merged_results: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "vector_results": self.vector_results,
            "knowledge_results": self.knowledge_results,
            "merged_results": self.merged_results
        }


@dataclass
class ReportKnowledgeRetrievalResource:
    """
    报告知识检索Chain策略专属资源类

    Attributes:
        vector_handler: 向量检索Handler（复用健康咨询的Handler）
        neo4j_handler: Neo4j医疗Handler（复用健康咨询的Handler）
        vector_encode_service: 向量编码服务（复用健康咨询的Service）
    """
    vector_handler: Optional[VectorRetrievalHandler] = None
    neo4j_handler: Optional[Neo4jMedicalHandler] = None
    vector_encode_service: Optional[Any] = None


class ReportKnowledgeRetrievalChain(Chain[ChainContext[ReportKnowledgeRetrievalContextBody], ChainResult[ReportKnowledgeRetrievalResultData]]):
    """
    报告知识检索Chain策略类

    实现健康报告生成业务的知识检索固定流程：
    1. 向量检索锚定实体（基于异常指标和医疗实体）
    2. 图谱查询结构化推理增强
    3. 知识整合去重排序
    """

    RELEVANCE_THRESHOLD = 0.5

    def __init__(self, resource: ReportKnowledgeRetrievalResource):
        """
        初始化报告知识检索Chain策略

        Args:
            resource: Chain策略专属资源
        """
        self._resource = resource
        self._milvus_degraded = False
        self._neo4j_degraded = False

    def execute(self, chain_context: ChainContext[ReportKnowledgeRetrievalContextBody]) -> ChainResult[ReportKnowledgeRetrievalResultData]:
        """
        执行Chain策略

        Args:
            chain_context: Chain输入数据容器

        Returns:
            ChainResult: Chain输出数据容器
        """
        start_time = time.time()
        logger.info(f"[ReportKnowledgeRetrievalChain] 开始执行Chain: session_id={chain_context.session_id}")

        body = chain_context.body
        if body is None:
            logger.warning(f"[ReportKnowledgeRetrievalChain] 输入数据为空: session_id={chain_context.session_id}")
            return ChainResult(
                session_id=chain_context.session_id,
                data=ReportKnowledgeRetrievalResultData()
            )

        vector_results: List[Dict] = []
        knowledge_results: List[Dict] = []

        self._milvus_degraded = False
        self._neo4j_degraded = False

        # Step1：向量检索锚定实体
        try:
            vector_results = self._vector_search_step(body)
            logger.info(f"[ReportKnowledgeRetrievalChain] 向量检索完成: vector_results={len(vector_results)}")
        except Exception as e:
            logger.error(f"[ReportKnowledgeRetrievalChain] 向量检索失败，启用Milvus降级策略: {e}")
            self._milvus_degraded = True

        # Step2：图查询结构化推理增强
        try:
            if vector_results:
                knowledge_results = self._graph_query_step(vector_results, body)
                logger.info(f"[ReportKnowledgeRetrievalChain] 图谱查询完成: knowledge_results={len(knowledge_results)}")
        except Exception as e:
            logger.error(f"[ReportKnowledgeRetrievalChain] 图谱查询失败，启用Neo4j降级策略: {e}")
            self._neo4j_degraded = True

        # 记录降级状态
        if self._milvus_degraded:
            logger.warning("[ReportKnowledgeRetrievalChain] 降级模式: Milvus不可用，使用Neo4j模糊匹配替代")
        if self._neo4j_degraded:
            logger.warning("[ReportKnowledgeRetrievalChain] 降级模式: Neo4j不可用，仅使用向量检索结果")

        # 知识整合
        merged_results = self._integrate_knowledge(vector_results, knowledge_results)
        logger.info(f"[ReportKnowledgeRetrievalChain] 知识整合完成: merged_results={len(merged_results)}")

        result_data = ReportKnowledgeRetrievalResultData(
            vector_results=vector_results,
            knowledge_results=knowledge_results,
            merged_results=merged_results
        )

        elapsed = time.time() - start_time
        logger.info(f"[ReportKnowledgeRetrievalChain] Chain执行完成: session_id={chain_context.session_id}, "
                   f"merged_results={len(merged_results)}, elapsed={elapsed:.2f}s")

        return ChainResult(session_id=chain_context.session_id, data=result_data)

    def _build_query_text(self, context_body: ReportKnowledgeRetrievalContextBody) -> str:
        """
        构建查询文本

        基于异常指标和医疗实体构建查询文本

        Args:
            context_body: Chain策略专属输入数据

        Returns:
            查询文本
        """
        query_parts = []

        # 添加异常指标
        for anomaly in context_body.anomalies:
            if isinstance(anomaly, dict):
                indicator_name = anomaly.get("name", "")
                if indicator_name:
                    query_parts.append(indicator_name)

        # 添加医疗实体
        for entity in context_body.medical_entities:
            if isinstance(entity, dict):
                entity_name = entity.get("name", "")
                entity_type = entity.get("type", "")
                if entity_name:
                    if entity_type:
                        query_parts.append(f"{entity_name} {entity_type}")
                    else:
                        query_parts.append(entity_name)

        # 添加风险疾病
        for disease in context_body.risk_diseases:
            if isinstance(disease, dict):
                disease_name = disease.get("name", "")
                if disease_name:
                    query_parts.append(disease_name)

        query_text = " ".join(query_parts)
        logger.info(f"[ReportKnowledgeRetrievalChain] 构建查询文本: query_text={query_text}")

        return query_text

    def _vector_search_step(self, context_body: ReportKnowledgeRetrievalContextBody) -> List[Dict]:
        """
        步骤1：向量检索锚定实体

        基于异常指标和医疗实体构建查询文本，调用Milvus三集合检索

        Args:
            context_body: Chain策略专属输入数据

        Returns:
            向量检索结果列表
        """
        if self._resource.vector_handler is None:
            logger.warning("[ReportKnowledgeRetrievalChain] vector_handler未初始化")
            return []

        # 构建查询文本
        query_text = self._build_query_text(context_body)

        if not query_text:
            logger.warning("[ReportKnowledgeRetrievalChain] 查询文本为空，跳过向量检索")
            return []

        # 调用Milvus三集合检索（medical_entity、entity_attributes、entity_relations）
        # 实现加权融合逻辑（entity: 0.40, attributes: 0.30, relations: 0.30）
        search_result = self._resource.vector_handler.call_tool({
            "query": query_text,
            "top_k": 30,
            "collections": ["medical_entity", "entity_attributes", "entity_relations"],
            "weights": {"medical_entity": 0.40, "entity_attributes": 0.30, "entity_relations": 0.30}
        })

        vector_results = []

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
                    vector_results.append(item)

        logger.info(f"[ReportKnowledgeRetrievalChain] 向量检索: total={len(vector_results)}, "
                   f"collections=['medical_entity', 'entity_attributes', 'entity_relations'], "
                   f"weights={{'medical_entity': 0.40, 'entity_attributes': 0.30, 'entity_relations': 0.30}}")

        return vector_results

    def _graph_query_step(self, vector_results: List[Dict], context_body: ReportKnowledgeRetrievalContextBody) -> List[Dict]:
        """
        步骤2：基于向量检索结果查询图谱进行结构化推理增强

        使用neo4j_node_id直接查询Neo4j节点

        Args:
            vector_results: 向量检索结果
            context_body: Chain策略专属输入数据

        Returns:
            图谱查询结果列表
        """
        if self._resource.neo4j_handler is None:
            logger.warning("[ReportKnowledgeRetrievalChain] neo4j_handler未初始化")
            return []

        knowledge_results: List[Dict] = []
        seen_node_ids = set()

        # 从向量检索结果中提取实体
        for item in vector_results:
            entity_data = item.get("entity", {})
            neo4j_node_id_str = entity_data.get("neo4j_node_id")
            entity_type = entity_data.get("entity_type", "Disease")

            if not neo4j_node_id_str:
                continue

            # 将neo4j_node_id转换为整数类型
            try:
                neo4j_node_id = int(neo4j_node_id_str)
            except (ValueError, TypeError) as e:
                logger.warning(f"[ReportKnowledgeRetrievalChain] neo4j_node_id转换失败: neo4j_node_id={neo4j_node_id_str}, error={str(e)}")
                continue

            # 避免重复查询
            if neo4j_node_id in seen_node_ids:
                continue
            seen_node_ids.add(neo4j_node_id)

            try:
                # 根据entity_type判断节点类型
                if entity_type == "Disease":
                    # 查询疾病信息
                    disease_info = self._resource.neo4j_handler.get_disease_by_node_id(neo4j_node_id)
                    if not disease_info:
                        logger.warning(f"[ReportKnowledgeRetrievalChain] 未找到node_id={neo4j_node_id}对应的疾病")
                        continue

                    disease_name = disease_info.get("name", "")
                    logger.info(f"[ReportKnowledgeRetrievalChain] 通过node_id={neo4j_node_id}查询到疾病: {disease_name}")

                    # 查询症状
                    symptoms = self._resource.neo4j_handler.get_symptoms_by_node_id(neo4j_node_id)

                    # 查询药物
                    drugs = self._resource.neo4j_handler.get_drugs_by_node_id(neo4j_node_id)

                    # 查询饮食建议
                    foods = self._resource.neo4j_handler.get_foods_by_node_id(neo4j_node_id)

                    # 构建知识结果
                    knowledge_item = {
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
                        "score": item.get("score", 0.0)
                    }

                    knowledge_results.append(knowledge_item)
                    logger.info(f"[ReportKnowledgeRetrievalChain] 查询疾病知识完成: disease={disease_name}, "
                               f"symptoms={len(symptoms)}, "
                               f"drugs={len(drugs.get('common_drugs', [])) + len(drugs.get('recommand_drugs', []))}, "
                               f"foods={len(foods.get('do_eat', [])) + len(foods.get('no_eat', [])) + len(foods.get('recommand_eat', []))}")

                elif entity_type == "Symptom":
                    symptom_info = self._resource.neo4j_handler.get_symptom_by_node_id(neo4j_node_id)
                    if not symptom_info:
                        logger.warning(f"[ReportKnowledgeRetrievalChain] 未找到node_id={neo4j_node_id}对应的症状")
                        continue

                    symptom_name = symptom_info.get("name", "")
                    logger.info(f"[ReportKnowledgeRetrievalChain] 通过node_id={neo4j_node_id}查询到症状: {symptom_name}")

                    diseases = self._resource.neo4j_handler.get_diseases_by_symptom_node_id(neo4j_node_id)

                    knowledge_item = {
                        "source": "neo4j",
                        "type": "symptom",
                        "entity": symptom_name,
                        "data": {
                            "name": symptom_name,
                            "related_diseases": diseases
                        },
                        "score": item.get("score", 0.0)
                    }

                    knowledge_results.append(knowledge_item)
                    logger.info(f"[ReportKnowledgeRetrievalChain] 查询症状知识完成: symptom={symptom_name}, related_diseases={len(diseases)}")

                elif entity_type == "Drug":
                    drug_info = self._resource.neo4j_handler.get_drug_by_node_id(neo4j_node_id)
                    if not drug_info:
                        logger.warning(f"[ReportKnowledgeRetrievalChain] 未找到node_id={neo4j_node_id}对应的药物")
                        continue

                    drug_name = drug_info.get("name", "")
                    logger.info(f"[ReportKnowledgeRetrievalChain] 通过node_id={neo4j_node_id}查询到药物: {drug_name}")

                    diseases = self._resource.neo4j_handler.get_diseases_by_drug_node_id(neo4j_node_id)

                    knowledge_item = {
                        "source": "neo4j",
                        "type": "drug",
                        "entity": drug_name,
                        "data": {
                            "name": drug_name,
                            "related_diseases": diseases
                        },
                        "score": item.get("score", 0.0)
                    }

                    knowledge_results.append(knowledge_item)
                    logger.info(f"[ReportKnowledgeRetrievalChain] 查询药物知识完成: drug={drug_name}, common_drug_diseases={len(diseases.get('common_drug_diseases', []))}, recommand_drug_diseases={len(diseases.get('recommand_drug_diseases', []))}")

                elif entity_type == "Food":
                    food_info = self._resource.neo4j_handler.get_food_by_node_id(neo4j_node_id)
                    if not food_info:
                        logger.warning(f"[ReportKnowledgeRetrievalChain] 未找到node_id={neo4j_node_id}对应的食物")
                        continue

                    food_name = food_info.get("name", "")
                    logger.info(f"[ReportKnowledgeRetrievalChain] 通过node_id={neo4j_node_id}查询到食物: {food_name}")

                    diseases = self._resource.neo4j_handler.get_diseases_by_food_node_id(neo4j_node_id)

                    knowledge_item = {
                        "source": "neo4j",
                        "type": "food",
                        "entity": food_name,
                        "data": {
                            "name": food_name,
                            "recommendations": diseases
                        },
                        "score": item.get("score", 0.0)
                    }

                    knowledge_results.append(knowledge_item)
                    logger.info(f"[ReportKnowledgeRetrievalChain] 查询食物知识完成: food={food_name}, do_eat_diseases={len(diseases.get('do_eat_diseases', []))}, no_eat_diseases={len(diseases.get('no_eat_diseases', []))}, recommand_diseases={len(diseases.get('recommand_diseases', []))}")

                elif entity_type == "Check":
                    check_info = self._resource.neo4j_handler.get_check_by_node_id(neo4j_node_id)
                    if not check_info:
                        logger.warning(f"[ReportKnowledgeRetrievalChain] 未找到node_id={neo4j_node_id}对应的检查项目")
                        continue

                    check_name = check_info.get("name", "")
                    logger.info(f"[ReportKnowledgeRetrievalChain] 通过node_id={neo4j_node_id}查询到检查项目: {check_name}")

                    diseases = self._resource.neo4j_handler.get_diseases_by_check_node_id(neo4j_node_id)

                    knowledge_item = {
                        "source": "neo4j",
                        "type": "check",
                        "entity": check_name,
                        "data": {
                            "name": check_name,
                            "related_diseases": diseases
                        },
                        "score": item.get("score", 0.0)
                    }

                    knowledge_results.append(knowledge_item)
                    logger.info(f"[ReportKnowledgeRetrievalChain] 查询检查项目知识完成: check={check_name}, related_diseases={len(diseases)}")

                elif entity_type == "Department":
                    department_info = self._resource.neo4j_handler.get_department_by_node_id(neo4j_node_id)
                    if not department_info:
                        logger.warning(f"[ReportKnowledgeRetrievalChain] 未找到node_id={neo4j_node_id}对应的科室")
                        continue

                    department_name = department_info.get("name", "")
                    logger.info(f"[ReportKnowledgeRetrievalChain] 通过node_id={neo4j_node_id}查询到科室: {department_name}")

                    diseases = self._resource.neo4j_handler.get_diseases_by_department_node_id(neo4j_node_id)

                    knowledge_item = {
                        "source": "neo4j",
                        "type": "department",
                        "entity": department_name,
                        "data": {
                            "name": department_name,
                            "related_diseases": diseases
                        },
                        "score": item.get("score", 0.0)
                    }

                    knowledge_results.append(knowledge_item)
                    logger.info(f"[ReportKnowledgeRetrievalChain] 查询科室知识完成: department={department_name}, related_diseases={len(diseases)}")

                elif entity_type == "Cure":
                    cure_info = self._resource.neo4j_handler.get_cure_by_node_id(neo4j_node_id)
                    if not cure_info:
                        logger.warning(f"[ReportKnowledgeRetrievalChain] 未找到node_id={neo4j_node_id}对应的治疗方法")
                        continue

                    cure_name = cure_info.get("name", "")
                    logger.info(f"[ReportKnowledgeRetrievalChain] 通过node_id={neo4j_node_id}查询到治疗方法: {cure_name}")

                    diseases = self._resource.neo4j_handler.get_diseases_by_cure_node_id(neo4j_node_id)

                    knowledge_item = {
                        "source": "neo4j",
                        "type": "cure",
                        "entity": cure_name,
                        "data": {
                            "name": cure_name,
                            "related_diseases": diseases
                        },
                        "score": item.get("score", 0.0)
                    }

                    knowledge_results.append(knowledge_item)
                    logger.info(f"[ReportKnowledgeRetrievalChain] 查询治疗方法知识完成: cure={cure_name}, related_diseases={len(diseases)}")

                elif entity_type == "Producer":
                    logger.debug(f"[ReportKnowledgeRetrievalChain] 跳过Producer类型实体: node_id={neo4j_node_id}")
                    continue

                else:
                    logger.warning(f"[ReportKnowledgeRetrievalChain] 未知的entity_type={entity_type}, node_id={neo4j_node_id}")
                    continue

            except Exception as e:
                logger.error(f"[ReportKnowledgeRetrievalChain] 查询node_id={neo4j_node_id}失败: {str(e)}")
                continue

        # 如果向量检索没有锚定实体，尝试基于风险疾病查询
        if not vector_results and context_body.risk_diseases:
            for disease in context_body.risk_diseases:
                if isinstance(disease, dict):
                    disease_name = disease.get("name", "")
                    if disease_name:
                        try:
                            disease_info = self._resource.neo4j_handler.get_disease_info(disease_name)
                            if disease_info:
                                knowledge_results.append({
                                    "source": "neo4j",
                                    "type": "risk_disease",
                                    "entity": disease_name,
                                    "data": disease_info,
                                    "score": disease.get("score", 0.0)
                                })
                                logger.info(f"[ReportKnowledgeRetrievalChain] 查询风险疾病知识完成: disease={disease_name}")
                        except Exception as e:
                            logger.error(f"[ReportKnowledgeRetrievalChain] 查询风险疾病失败: disease={disease_name}, error={str(e)}")

        return knowledge_results

    def _integrate_knowledge(self, vector_results: List[Dict], knowledge_results: List[Dict]) -> List[Dict]:
        """
        知识整合：去重、排序、过滤、Top-30限制

        正常流程：优先使用neo4j返回的数据作为LLM的知识
        降级流程：neo4j不可用时，使用向量检索的数据

        Args:
            vector_results: 向量检索结果
            knowledge_results: 图谱查询结果

        Returns:
            整合后的知识列表（Top-30）
        """
        merged: List[Dict] = []
        seen_ids = set()

        # 优先使用neo4j返回的数据（正常流程）
        if knowledge_results:
            logger.info(f"[ReportKnowledgeRetrievalChain] 使用Neo4j知识（正常流程）: knowledge_count={len(knowledge_results)}")
            for item in knowledge_results:
                item_id = item.get("entity", "") + "_" + item.get("type", "")
                if not item_id or item_id == "_":
                    item_id = str(id(item))
                if item_id not in seen_ids:
                    seen_ids.add(item_id)
                    merged.append({
                        "source": item.get("source", "neo4j"),
                        "type": item.get("type", ""),
                        "entity": item.get("entity", ""),
                        "data": item.get("data", item),
                        "score": item.get("score", 0.0)
                    })
        else:
            # Neo4j无数据，使用向量检索数据（降级流程）
            logger.warning(f"[ReportKnowledgeRetrievalChain] Neo4j无数据，使用向量检索数据（降级流程）: vector_count={len(vector_results)}")
            for item in vector_results:
                item_id = item.get("id", str(id(item)))
                if item_id not in seen_ids:
                    seen_ids.add(item_id)
                    # 精简数据，只保留entity和collection字段
                    simplified_data = {
                        "collection": item.get("collection", ""),
                        "entity": item.get("entity", {})
                    }
                    merged.append({
                        "source": "vector_degraded",
                        "data": simplified_data,
                        "score": item.get("score", 0.0)
                    })

        # 按相关性得分排序（降序）
        merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        # 过滤低于阈值的结果（neo4j数据不过滤）
        merged = [item for item in merged if item.get("score", 0.0) >= self.RELEVANCE_THRESHOLD or item.get("source") in ["neo4j", "vector_degraded"]]

        # 限制为Top-30结果
        merged = merged[:30]

        logger.info(f"[ReportKnowledgeRetrievalChain] 知识整合: total_results={len(merged)}, top_k_limit=30")

        return merged
