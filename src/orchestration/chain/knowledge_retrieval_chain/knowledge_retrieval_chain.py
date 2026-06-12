# -*- coding: utf-8 -*-
"""
知识检索Chain策略

实现向量检索与图谱查询的顺序检索Chain策略。
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from src.orchestration.chain.chain import Chain
from src.utils.logger import log_arch_event
from src.orchestration.chain.data_classes import ChainContext, ChainResult
from src.orchestration.chain.knowledge_retrieval_chain.knowledge_retrieval_context import KnowledgeRetrievalContextBody
from src.orchestration.chain.knowledge_retrieval_chain.knowledge_retrieval_result import KnowledgeRetrievalResultData
from src.orchestration.chain.knowledge_retrieval_chain.knowledge_retrieval_resource import KnowledgeRetrievalResource
from src.orchestration.tool_call_handler.Impl.vector_retrieval_handler import VectorRetrievalHandler
from src.orchestration.tool_call_handler.Impl.neo4j_medical_handler import Neo4jMedicalHandler
from src.config.business.consult_service_config import get_runtime_config

logger = logging.getLogger(__name__)

# 从业务配置读取参数，替代硬编码
# 使用惰性获取模式：模块级常量在import时求值，此时ConfigManager可能尚未初始化，
# 因此改为在函数内部通过get_runtime_config()获取运行期配置值。
def _get_relevance_threshold() -> float:
    return get_runtime_config().knowledge_fusion_threshold

def _get_vector_entity_weight() -> float:
    return get_runtime_config().vector_entity_weight

def _get_vector_attribute_weight() -> float:
    return get_runtime_config().vector_attribute_weight

def _get_vector_relation_weight() -> float:
    return get_runtime_config().vector_relation_weight

def _get_knowledge_retrieval_top_k() -> int:
    return get_runtime_config().knowledge_retrieval_top_k

def _get_knowledge_merge_limit() -> int:
    return get_runtime_config().knowledge_merge_limit

def _get_knowledge_sufficiency_min_count() -> int:
    return get_runtime_config().knowledge_sufficiency_min_count

class KnowledgeRetrievalChain(Chain[ChainContext[KnowledgeRetrievalContextBody], ChainResult[KnowledgeRetrievalResultData]]):
    """
    知识检索Chain策略类

    实现向量检索与图谱查询的顺序检索固定流程：
    1. 向量检索锚定实体
    2. 图谱查询结构化推理增强
    3. 知识整合去重排序
    """

    def __init__(self, resource: KnowledgeRetrievalResource):
        """
        初始化知识检索Chain策略

        Args:
            resource: Chain策略专属资源
        """
        self._resource = resource
        self._milvus_degraded = False
        self._neo4j_degraded = False

    def execute(self, chain_context: ChainContext[KnowledgeRetrievalContextBody]) -> ChainResult[KnowledgeRetrievalResultData]:
        """
        执行Chain策略

        Args:
            chain_context: Chain输入数据容器

        Returns:
            ChainResult: Chain输出数据容器
        """
        start_time = time.time()
        logger.info(f"[KnowledgeRetrievalChain] 开始执行Chain: session_id={chain_context.session_id}")

        body = chain_context.body
        if body is None:
            logger.warning(f"[KnowledgeRetrievalChain] 输入数据为空: session_id={chain_context.session_id}")
            return ChainResult(
                session_id=chain_context.session_id,
                data=KnowledgeRetrievalResultData()
            )

        vector_results: List[Dict] = []
        anchored_entities: List[Dict] = []
        anchored_relations: List[Dict] = []
        knowledge_results: List[Dict] = []

        self._milvus_degraded = False
        self._neo4j_degraded = False

        try:
            vector_results, anchored_entities, anchored_relations = self._vector_search_step(body)
            logger.info(f"[KnowledgeRetrievalChain] 向量检索完成: vector_results={len(vector_results)}, "
                       f"anchored_entities={len(anchored_entities)}, anchored_relations={len(anchored_relations)}")
        except Exception as e:
            logger.error(f"[KnowledgeRetrievalChain] 向量检索失败，启用Milvus降级策略: {e}")
            self._milvus_degraded = True

        try:
            if anchored_entities:
                knowledge_results = self._graph_query_step(anchored_entities, anchored_relations, body.query_text)
                logger.info(f"[KnowledgeRetrievalChain] 图谱查询完成: knowledge_results={len(knowledge_results)}")
        except Exception as e:
            logger.error(f"[KnowledgeRetrievalChain] 图谱查询失败，启用Neo4j降级策略: {e}")
            self._neo4j_degraded = True

        if self._milvus_degraded:
            logger.warning("[KnowledgeRetrievalChain] 降级模式: Milvus不可用，使用Neo4j模糊匹配替代")
        if self._neo4j_degraded:
            logger.warning("[KnowledgeRetrievalChain] 降级模式: Neo4j不可用，仅使用向量检索结果")

        merged_results = self._integrate_knowledge(vector_results, knowledge_results)
        logger.info(f"[KnowledgeRetrievalChain] 知识整合完成: merged_results={len(merged_results)}")

        result_data = KnowledgeRetrievalResultData(
            vector_results=vector_results,
            knowledge_results=knowledge_results,
            merged_results=merged_results,
            anchored_entities=anchored_entities,
            anchored_relations=anchored_relations
        )

        elapsed = time.time() - start_time
        logger.info(f"[KnowledgeRetrievalChain] Chain执行完成: session_id={chain_context.session_id}, "
                   f"merged_results={len(merged_results)}, elapsed={elapsed:.2f}s")

        return ChainResult(session_id=chain_context.session_id, data=result_data)

    def _vector_search_step(self, context_body: KnowledgeRetrievalContextBody) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        步骤1：向量检索锚定实体

        Args:
            context_body: Chain策略专属输入数据

        Returns:
            Tuple[向量检索结果, 锚定实体列表, 锚定关系列表]
        """
        if self._resource.vector_handler is None:
            logger.warning("[KnowledgeRetrievalChain] vector_handler未初始化")
            return [], [], []

        log_arch_event(
            logger,
            component="KnowledgeRetrievalChain",
            stage="CHAIN",
            event="vector_search_step",
            status="start",
            design_id="BIZ-4.1",
        )

        # 调用Milvus三集合检索（medical_entity、entity_attributes、entity_relations）
        # 实现加权融合逻辑
        top_k = _get_knowledge_retrieval_top_k()
        collections = ["medical_entity", "entity_attributes", "entity_relations"]
        weights = {"medical_entity": _get_vector_entity_weight(), "entity_attributes": _get_vector_attribute_weight(), "entity_relations": _get_vector_relation_weight()}

        logger.debug(f"[RETRIEVAL_PARAMS] query={context_body.query_text}, top_k={top_k}, collections={collections}, weights={weights}")

        search_result = self._resource.vector_handler.call_tool({
            "query": context_body.query_text,
            "top_k": top_k,
            "collections": collections,
            "weights": weights
        })

        vector_results = []
        anchored_entities = []
        anchored_relations = []
        seen_entity_node_ids = set()

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
                if not isinstance(item, dict):
                    continue

                vector_results.append(item)

                collection = item.get("collection", item.get("source", ""))
                if collection == "medical_entity":
                    anchored_entities.append(item)
                elif collection == "entity_attributes":
                    # entity_attributes集合的数据也包含neo4j_node_id，可以用来查询Neo4j
                    # 避免重复：同一个疾病的多个属性只查询一次
                    entity_data = item.get("entity", {})
                    neo4j_node_id = entity_data.get("neo4j_node_id")
                    if neo4j_node_id and neo4j_node_id not in seen_entity_node_ids:
                        seen_entity_node_ids.add(neo4j_node_id)
                        anchored_entities.append(item)
                elif collection == "entity_relations":
                    anchored_relations.append(item)

        logger.info(f"[KnowledgeRetrievalChain] 向量检索: total={len(vector_results)}, "
                   f"entities={len(anchored_entities)}, relations={len(anchored_relations)}, "
                   f"collections=['medical_entity', 'entity_attributes', 'entity_relations'], "
                   f"weights={{'medical_entity': {_get_vector_entity_weight()}, 'entity_attributes': {_get_vector_attribute_weight()}, 'entity_relations': {_get_vector_relation_weight()}}}")

        # 结果分数分布日志
        if vector_results:
            scores = [r.get("score", 0.0) for r in vector_results if isinstance(r, dict)]
            if scores:
                avg_score = sum(scores) / len(scores)
                max_score = max(scores)
                min_score = min(scores)
                high_count = sum(1 for s in scores if s >= 0.8)
                mid_count = sum(1 for s in scores if 0.6 <= s < 0.8)
                low_count = sum(1 for s in scores if s < 0.6)
                logger.debug(f"[RESULT_DISTRIBUTION] total={len(scores)}, avg_score={avg_score:.4f}, "
                           f"max_score={max_score:.4f}, min_score={min_score:.4f}, "
                           f"high(>=0.8)={high_count}, mid(0.6-0.8)={mid_count}, low(<0.6)={low_count}")
                # 逐条结果的相关性评分详情
                for idx, item in enumerate(vector_results):
                    if isinstance(item, dict):
                        score = item.get("score", 0.0)
                        collection = item.get("collection", item.get("source", ""))
                        entity = item.get("entity", {})
                        entity_name = entity.get("name", entity.get("entity_name", "")) if isinstance(entity, dict) else str(entity)
                        logger.debug(f"[RELEVANCE_SCORE] idx={idx}, vector={score:.4f}, "
                                   f"collection={collection}, entity_name={entity_name}")
        else:
            logger.debug("[RESULT_DISTRIBUTION] total=0, no results returned from vector search")

        return vector_results, anchored_entities, anchored_relations

    def _graph_query_step(self, anchored_entities: List[Dict], anchored_relations: List[Dict], query: str) -> List[Dict]:
        """
        步骤2：基于锚定实体查询图谱进行结构化推理增强
        
        使用neo4j_node_id直接查询Neo4j节点，避免实体名称不匹配的问题

        Args:
            anchored_entities: 锚定实体列表
            anchored_relations: 锚定关系列表
            query: 查询文本

        Returns:
            图谱查询结果列表
        """
        if self._resource.neo4j_handler is None:
            logger.warning("[KnowledgeRetrievalChain] neo4j_handler未初始化")
            return []

        log_arch_event(
            logger,
            component="KnowledgeRetrievalChain",
            stage="CHAIN",
            event="graph_query_step",
            status="start",
            design_id="BIZ-4.2",
        )

        knowledge_results: List[Dict] = []
        seen_node_ids = set()

        for entity in anchored_entities:
            # 从entity字典中获取neo4j_node_id和entity_type
            entity_data = entity.get("entity", {})
            neo4j_node_id_str = entity_data.get("neo4j_node_id")
            entity_type = entity_data.get("entity_type", "Disease")
            
            if not neo4j_node_id_str:
                logger.warning(f"[KnowledgeRetrievalChain] 实体缺少neo4j_node_id: entity_data={entity_data}")
                continue

            # neo4j_node_id为elementId字符串，直接传递给Neo4j查询
            neo4j_node_id = neo4j_node_id_str
            
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
                        logger.warning(f"[KnowledgeRetrievalChain] 未找到node_id={neo4j_node_id}对应的疾病")
                        continue
                    
                    disease_name = disease_info.get("name", "")
                    logger.info(f"[KnowledgeRetrievalChain] 通过node_id={neo4j_node_id}查询到疾病: {disease_name}")
                    
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
                        "score": entity.get("score", 0.0)
                    }
                    
                    knowledge_results.append(knowledge_item)
                    logger.info(f"[KnowledgeRetrievalChain] 查询疾病知识完成: disease={disease_name}, symptoms={len(symptoms)}, drugs={len(drugs.get('common_drugs', [])) + len(drugs.get('recommand_drugs', []))}, foods={len(foods.get('do_eat', [])) + len(foods.get('no_eat', [])) + len(foods.get('recommand_eat', []))}")
                
                elif entity_type == "Symptom":
                    symptom_info = self._resource.neo4j_handler.get_symptom_by_node_id(neo4j_node_id)
                    if not symptom_info:
                        logger.warning(f"[KnowledgeRetrievalChain] 未找到node_id={neo4j_node_id}对应的症状")
                        continue
                    
                    symptom_name = symptom_info.get("name", "")
                    logger.info(f"[KnowledgeRetrievalChain] 通过node_id={neo4j_node_id}查询到症状: {symptom_name}")
                    
                    diseases = self._resource.neo4j_handler.get_diseases_by_symptom_node_id(neo4j_node_id)
                    
                    knowledge_item = {
                        "source": "neo4j",
                        "type": "symptom",
                        "entity": symptom_name,
                        "data": {
                            "name": symptom_name,
                            "related_diseases": diseases
                        },
                        "score": entity.get("score", 0.0)
                    }
                    
                    knowledge_results.append(knowledge_item)
                    logger.info(f"[KnowledgeRetrievalChain] 查询症状知识完成: symptom={symptom_name}, related_diseases={len(diseases)}")
                
                elif entity_type == "Drug":
                    drug_info = self._resource.neo4j_handler.get_drug_by_node_id(neo4j_node_id)
                    if not drug_info:
                        logger.warning(f"[KnowledgeRetrievalChain] 未找到node_id={neo4j_node_id}对应的药物")
                        continue
                    
                    drug_name = drug_info.get("name", "")
                    logger.info(f"[KnowledgeRetrievalChain] 通过node_id={neo4j_node_id}查询到药物: {drug_name}")
                    
                    diseases = self._resource.neo4j_handler.get_diseases_by_drug_node_id(neo4j_node_id)
                    
                    knowledge_item = {
                        "source": "neo4j",
                        "type": "drug",
                        "entity": drug_name,
                        "data": {
                            "name": drug_name,
                            "related_diseases": diseases
                        },
                        "score": entity.get("score", 0.0)
                    }
                    
                    knowledge_results.append(knowledge_item)
                    logger.info(f"[KnowledgeRetrievalChain] 查询药物知识完成: drug={drug_name}, common_drug_diseases={len(diseases.get('common_drug_diseases', []))}, recommand_drug_diseases={len(diseases.get('recommand_drug_diseases', []))}")
                
                elif entity_type == "Food":
                    food_info = self._resource.neo4j_handler.get_food_by_node_id(neo4j_node_id)
                    if not food_info:
                        logger.warning(f"[KnowledgeRetrievalChain] 未找到node_id={neo4j_node_id}对应的食物")
                        continue
                    
                    food_name = food_info.get("name", "")
                    logger.info(f"[KnowledgeRetrievalChain] 通过node_id={neo4j_node_id}查询到食物: {food_name}")
                    
                    diseases = self._resource.neo4j_handler.get_diseases_by_food_node_id(neo4j_node_id)
                    
                    knowledge_item = {
                        "source": "neo4j",
                        "type": "food",
                        "entity": food_name,
                        "data": {
                            "name": food_name,
                            "recommendations": diseases
                        },
                        "score": entity.get("score", 0.0)
                    }
                    
                    knowledge_results.append(knowledge_item)
                    logger.info(f"[KnowledgeRetrievalChain] 查询食物知识完成: food={food_name}, do_eat_diseases={len(diseases.get('do_eat_diseases', []))}, no_eat_diseases={len(diseases.get('no_eat_diseases', []))}, recommand_diseases={len(diseases.get('recommand_diseases', []))}")
                
                elif entity_type == "Check":
                    check_info = self._resource.neo4j_handler.get_check_by_node_id(neo4j_node_id)
                    if not check_info:
                        logger.warning(f"[KnowledgeRetrievalChain] 未找到node_id={neo4j_node_id}对应的检查项目")
                        continue
                    
                    check_name = check_info.get("name", "")
                    logger.info(f"[KnowledgeRetrievalChain] 通过node_id={neo4j_node_id}查询到检查项目: {check_name}")
                    
                    diseases = self._resource.neo4j_handler.get_diseases_by_check_node_id(neo4j_node_id)
                    
                    knowledge_item = {
                        "source": "neo4j",
                        "type": "check",
                        "entity": check_name,
                        "data": {
                            "name": check_name,
                            "related_diseases": diseases
                        },
                        "score": entity.get("score", 0.0)
                    }
                    
                    knowledge_results.append(knowledge_item)
                    logger.info(f"[KnowledgeRetrievalChain] 查询检查项目知识完成: check={check_name}, related_diseases={len(diseases)}")
                
                elif entity_type == "Department":
                    department_info = self._resource.neo4j_handler.get_department_by_node_id(neo4j_node_id)
                    if not department_info:
                        logger.warning(f"[KnowledgeRetrievalChain] 未找到node_id={neo4j_node_id}对应的科室")
                        continue
                    
                    department_name = department_info.get("name", "")
                    logger.info(f"[KnowledgeRetrievalChain] 通过node_id={neo4j_node_id}查询到科室: {department_name}")
                    
                    diseases = self._resource.neo4j_handler.get_diseases_by_department_node_id(neo4j_node_id)
                    
                    knowledge_item = {
                        "source": "neo4j",
                        "type": "department",
                        "entity": department_name,
                        "data": {
                            "name": department_name,
                            "related_diseases": diseases
                        },
                        "score": entity.get("score", 0.0)
                    }
                    
                    knowledge_results.append(knowledge_item)
                    logger.info(f"[KnowledgeRetrievalChain] 查询科室知识完成: department={department_name}, related_diseases={len(diseases)}")
                
                elif entity_type == "Cure":
                    cure_info = self._resource.neo4j_handler.get_cure_by_node_id(neo4j_node_id)
                    if not cure_info:
                        logger.warning(f"[KnowledgeRetrievalChain] 未找到node_id={neo4j_node_id}对应的治疗方法")
                        continue
                    
                    cure_name = cure_info.get("name", "")
                    logger.info(f"[KnowledgeRetrievalChain] 通过node_id={neo4j_node_id}查询到治疗方法: {cure_name}")
                    
                    diseases = self._resource.neo4j_handler.get_diseases_by_cure_node_id(neo4j_node_id)
                    
                    knowledge_item = {
                        "source": "neo4j",
                        "type": "cure",
                        "entity": cure_name,
                        "data": {
                            "name": cure_name,
                            "related_diseases": diseases
                        },
                        "score": entity.get("score", 0.0)
                    }
                    
                    knowledge_results.append(knowledge_item)
                    logger.info(f"[KnowledgeRetrievalChain] 查询治疗方法知识完成: cure={cure_name}, related_diseases={len(diseases)}")
                
                elif entity_type == "Producer":
                    logger.debug(f"[KnowledgeRetrievalChain] 跳过Producer类型实体: node_id={neo4j_node_id}")
                    continue
                
                else:
                    logger.warning(f"[KnowledgeRetrievalChain] 未知的entity_type={entity_type}, node_id={neo4j_node_id}")
                    continue
                
            except Exception as e:
                logger.error(f"[KnowledgeRetrievalChain] 查询node_id={neo4j_node_id}失败: {str(e)}")
                continue

        if not anchored_entities:
            try:
                diseases = self._resource.neo4j_handler.search_diseases_by_symptom(query)
                if diseases:
                    knowledge_results.append({
                        "source": "neo4j",
                        "type": "possible_diseases",
                        "entity": query,
                        "data": diseases,
                        "score": 0.0
                    })
            except Exception as e:
                logger.error(f"[KnowledgeRetrievalChain] 症状搜索疾病失败: query={query}, error={e}")

        return knowledge_results

    def _integrate_knowledge(self, vector_results: List[Dict], knowledge_results: List[Dict]) -> List[Dict]:
        """
        知识整合：去重、排序、过滤、Top-15限制
        
        正常流程：优先使用neo4j返回的数据作为LLM的知识
        降级流程：neo4j不可用时，使用向量检索的数据，但要精简数据，只保留'entity'和'collection'

        Args:
            vector_results: 向量检索结果
            knowledge_results: 图谱查询结果

        Returns:
            整合后的知识列表（Top-15）
        """
        merged: List[Dict] = []
        seen_ids = set()
        
        # 优先使用neo4j返回的数据（正常流程）
        if knowledge_results:
            logger.info(f"[KnowledgeRetrievalChain] 使用Neo4j知识（正常流程）: knowledge_count={len(knowledge_results)}")
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
            logger.warning(f"[KnowledgeRetrievalChain] Neo4j无数据，使用向量检索数据（降级流程）: vector_count={len(vector_results)}")
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
        before_filter_count = len(merged)
        merged = [item for item in merged if item.get("score", 0.0) >= _get_relevance_threshold() or item.get("source") in ["neo4j", "vector_degraded"]]
        after_filter_count = len(merged)
        filtered_out_count = before_filter_count - after_filter_count

        # 充分性判断日志
        is_sufficient = after_filter_count >= _get_knowledge_sufficiency_min_count()
        gaps = []
        if not is_sufficient:
            gaps.append(f"有效结果不足(仅{after_filter_count}条,需要>={_get_knowledge_sufficiency_min_count()})")
        if after_filter_count > 0:
            scores = [item.get("score", 0.0) for item in merged]
            avg_score = sum(scores) / len(scores)
            if avg_score < _get_relevance_threshold():
                gaps.append(f"平均相关性得分偏低(avg={avg_score:.4f}, threshold={_get_relevance_threshold()})")
        else:
            gaps.append("无有效结果")
        logger.debug(f"[SUFFICIENCY] is_sufficient={is_sufficient}, confidence={after_filter_count}, "
                   f"gaps={gaps}, filtered_out={filtered_out_count}, threshold={_get_relevance_threshold()}")

        # 限制为Top结果
        merged = merged[:_get_knowledge_merge_limit()]

        logger.info(f"[KnowledgeRetrievalChain] 知识整合: total_results={len(merged)}, top_k_limit={_get_knowledge_merge_limit()}")

        return merged
