# -*- coding: utf-8 -*-
"""
检索执行模块

按PlanRetrieval选择的路径执行Neo4j图查询，返回完整结构化知识。

设计原则：
- 向量预扫描提供entity→neo4j_node_id映射，图查询通过node_id获取完整内容
- 向量检索结果不直接使用（信息不完整），仅作为候选发现
- 每条路径的查询方法名与Neo4jMedicalHandler方法名一一对应
- 单条查询失败不影响其他路径，记录warning继续执行
"""

import logging
from typing import Any, Dict, List, Optional, Set

from src.orchestration.agent.comprehensive_health_analysis_strategy.path_registry import (
    get_path_info,
)
from src.orchestration.agent.comprehensive_health_analysis_strategy.retrieval_planner import RetrievalPlan
from src.orchestration.agent.comprehensive_health_analysis_strategy.vector_prescan import VectorPrescanResult
from src.config.business.report_service_config import get_runtime_config

logger = logging.getLogger(__name__)


class RetrievalExecutor:
    """
    检索执行器

    按RetrievalPlan选择的路径执行图查询，合并结果。
    支持正向查询（Disease→X）和反向查询（X→Disease）。
    """

    def execute_plan(
        self,
        plan: Dict[str, RetrievalPlan],
        prescan_results: Dict[str, VectorPrescanResult],
        neo4j_tool: Any,
        context_entities: Dict[str, List[str]],
        candidate_limit: Optional[int] = None,
    ) -> Dict[str, List[Dict]]:
        """
        执行检索规划

        Args:
            plan: 维度名→检索规划（来自RetrievalPlanner）
            prescan_results: 实体名→向量预扫描结果（提供neo4j_node_id）
            neo4j_tool: Neo4jMedicalHandler实例
            context_entities: NER实体分类，如 {"disease_names": [...], ...}
            candidate_limit: 候选检索上限（默认从配置获取，不做最终截断）

        Returns:
            Dict[str, List[Dict]]: 维度名→候选知识条目列表
        """
        if candidate_limit is None:
            candidate_limit = get_runtime_config().agent_candidate_retrieve_limit
        results: Dict[str, List[Dict]] = {}

        for dim_name, dim_plan in plan.items():
            dim_knowledge = self._execute_dimension(
                dim_name, dim_plan, prescan_results, neo4j_tool, context_entities, candidate_limit
            )
            results[dim_name] = dim_knowledge

        total_items = sum(len(v) for v in results.values())
        logger.info(
            f"[RetrievalExecutor] 执行完成: 维度数={len(results)}, "
            f"总知识条目={total_items}, candidate_limit={candidate_limit}"
        )
        return results

    def execute_supplement(
        self,
        supplement_paths: Dict[str, List[str]],
        prescan_results: Dict[str, VectorPrescanResult],
        existing_knowledge: Dict[str, List[Dict]],
        neo4j_tool: Any,
        context_entities: Dict[str, List[str]],
    ) -> Dict[str, List[Dict]]:
        """
        执行补充检索（EvaluateSufficiency指定supplement路径）

        Args:
            supplement_paths: 维度名→需要补充的路径名列表
            prescan_results: 实体名→向量预扫描结果
            existing_knowledge: 维度名→已有知识条目（用于去重）
            neo4j_tool: Neo4jMedicalHandler实例
            context_entities: NER实体分类

        Returns:
            Dict[str, List[Dict]]: 维度名→补充知识条目（不含已有知识）
        """
        supplement_results: Dict[str, List[Dict]] = {}

        for dim_name, paths in supplement_paths.items():
            dim_plan = RetrievalPlan(paths=paths, entities=[])
            new_knowledge = self._execute_dimension(
                dim_name, dim_plan, prescan_results, neo4j_tool, context_entities
            )

            existing = existing_knowledge.get(dim_name, [])
            existing_targets = self._collect_targets(existing)
            deduped = [
                item for item in new_knowledge
                if item.get("target_entity", item.get("entity_name", "")) not in existing_targets
            ]

            supplement_results[dim_name] = deduped
            logger.info(
                f"[RetrievalExecutor] 维度{dim_name}补充检索: "
                f"新获取={len(new_knowledge)}, 去重后={len(deduped)}"
            )

        return supplement_results

    def _execute_dimension(
        self,
        dim_name: str,
        dim_plan: RetrievalPlan,
        prescan_results: Dict[str, VectorPrescanResult],
        neo4j_tool: Any,
        context_entities: Dict[str, List[str]],
        limit: int = 50,
        blacklist: Optional[Set[str]] = None,
    ) -> List[Dict]:
        """执行单维度的所有检索路径，返回候选知识（不做最终截断）"""
        blacklist = blacklist or set()
        all_knowledge: List[Dict] = []
        seen_targets: Set[str] = set()

        for path_name in dim_plan.paths:
            path_info = get_path_info(path_name)
            if not path_info:
                logger.warning(f"[RetrievalExecutor] 路径{path_name}不在注册表中，跳过")
                continue

            path_knowledge = self._execute_path(
                path_name, path_info, dim_plan, prescan_results,
                neo4j_tool, context_entities, dim_name, limit,
            )

            for item in path_knowledge:
                target = item.get("target_entity", item.get("entity_name", ""))
                if target and target not in seen_targets and target not in blacklist:
                    seen_targets.add(target)
                    all_knowledge.append(item)

            if len(all_knowledge) >= limit:
                break

        if len(all_knowledge) > limit:
            logger.info(
                f"[RetrievalExecutor] 维度{dim_name}候选截取: "
                f"{len(all_knowledge)} -> {limit}"
            )
            all_knowledge = all_knowledge[:limit]

        logger.info(
            f"[RetrievalExecutor] 维度{dim_name}: 路径数={len(dim_plan.paths)}, "
            f"候选知识条目={len(all_knowledge)}"
        )
        return all_knowledge

    def _execute_path(
        self,
        path_name: str,
        path_info: Dict,
        dim_plan: RetrievalPlan,
        prescan_results: Dict[str, VectorPrescanResult],
        neo4j_tool: Any,
        context_entities: Dict[str, List[str]],
        dim_name: str,
        limit: int = 50,
    ) -> List[Dict]:
        """执行单条检索路径"""
        query_method = path_info["query_method"]
        direction = path_info["direction"]
        source_type = path_info["source_type"]

        try:
            if direction == "forward" and source_type == "Disease":
                return self._execute_forward_disease(
                    query_method, dim_plan, neo4j_tool, context_entities, dim_name, path_name, limit,
                )
            elif direction == "reverse":
                return self._execute_reverse(
                    query_method, source_type, dim_plan, prescan_results,
                    neo4j_tool, context_entities, dim_name, path_name, limit,
                )
            elif direction == "self" and source_type == "Disease":
                return self._execute_disease_attributes(
                    dim_plan, neo4j_tool, context_entities, dim_name, path_name,
                )
            else:
                logger.warning(
                    f"[RetrievalExecutor] 路径{path_name}方向{direction}+源{source_type}无处理器"
                )
                return []

        except Exception as e:
            logger.warning(
                f"[RetrievalExecutor] 路径{path_name}执行失败: "
                f"{type(e).__name__}: {str(e)}"
            )
            return []

    def _execute_forward_disease(
        self,
        query_method: str,
        dim_plan: RetrievalPlan,
        neo4j_tool: Any,
        context_entities: Dict[str, List[str]],
        dim_name: str,
        path_name: str,
        limit: int = 50,
    ) -> List[Dict]:
        """执行正向查询：Disease → X"""
        disease_names = context_entities.get("disease_names", dim_plan.entities)
        if not disease_names:
            return []

        results = []
        for disease_name in disease_names[:3]:
            try:
                call_params = {
                    "method": query_method,
                    "entity_name": disease_name,
                }
                raw_result = neo4j_tool.call_tool(call_params)
                normalized = self._normalize_forward_result(
                    raw_result, disease_name, path_name, dim_name
                )
                results.extend(normalized)
            except Exception as e:
                logger.debug(
                    f"[RetrievalExecutor] 正向查询{query_method}失败: "
                    f"disease={disease_name}, error={type(e).__name__}"
                )

        return results

    def _execute_reverse(
        self,
        query_method: str,
        source_type: str,
        dim_plan: RetrievalPlan,
        prescan_results: Dict[str, VectorPrescanResult],
        neo4j_tool: Any,
        context_entities: Dict[str, List[str]],
        dim_name: str,
        path_name: str,
        limit: int = 50,
    ) -> List[Dict]:
        """执行反向查询：X → Disease

        symptom_to_diseases 特殊处理：search_diseases_by_symptom 用 symptom_name 参数，
        其他反向路径用 node_id 参数。
        """
        entity_names = self._find_entities_by_type(
            source_type, prescan_results, context_entities, dim_plan
        )
        if not entity_names:
            return []

        results = []
        for entity_name in entity_names[:3]:
            try:
                if query_method == "search_diseases_by_symptom":
                    raw_result = neo4j_tool.call_tool({
                        "method": query_method,
                        "symptom_name": entity_name,
                        "limit": limit,
                    })
                else:
                    prescan = prescan_results.get(entity_name)
                    if not prescan or not prescan.neo4j_node_id:
                        continue
                    raw_result = neo4j_tool.call_tool({
                        "method": query_method,
                        "node_id": prescan.neo4j_node_id,
                        "limit": limit,
                    })

                normalized = self._normalize_reverse_result(
                    raw_result, entity_name, source_type, path_name, dim_name
                )
                results.extend(normalized)
            except Exception as e:
                logger.debug(
                    f"[RetrievalExecutor] 反向查询{query_method}失败: "
                    f"entity={entity_name}, error={type(e).__name__}"
                )

        return results

    def _execute_disease_attributes(
        self,
        dim_plan: RetrievalPlan,
        neo4j_tool: Any,
        context_entities: Dict[str, List[str]],
        dim_name: str,
        path_name: str,
    ) -> List[Dict]:
        """执行疾病属性查询"""
        disease_names = context_entities.get("disease_names", dim_plan.entities)
        if not disease_names:
            return []

        results = []
        for disease_name in disease_names[:3]:
            try:
                raw_result = neo4j_tool.call_tool({
                    "method": "get_disease_info",
                    "entity_name": disease_name,
                })
                normalized = self._normalize_disease_info(
                    raw_result, disease_name, dim_name, path_name
                )
                if normalized:
                    results.append(normalized)
            except Exception as e:
                logger.debug(
                    f"[RetrievalExecutor] 疾病属性查询失败: "
                    f"disease={disease_name}, error={type(e).__name__}"
                )

        return results

    def _find_entities_by_type(
        self,
        entity_type: str,
        prescan_results: Dict[str, VectorPrescanResult],
        context_entities: Dict[str, List[str]],
        dim_plan: RetrievalPlan,
    ) -> List[str]:
        """从预扫描结果中查找指定类型的实体名"""
        matched = []
        for name, prescan in prescan_results.items():
            if prescan.entity_type == entity_type:
                matched.append(name)

        type_key_map = {
            "Symptom": "symptom_names",
            "Drug": "medication_names",
            "Food": "food_names",
            "Check": "check_names",
            "Department": "department_names",
        }
        ner_key = type_key_map.get(entity_type, "")
        if ner_key:
            for name in context_entities.get(ner_key, []):
                if name not in matched:
                    matched.append(name)

        for name in dim_plan.entities:
            if name not in matched:
                matched.append(name)

        return matched

    def _normalize_forward_result(
        self,
        raw_result: Any,
        source_entity: str,
        path_name: str,
        dim_name: str,
    ) -> List[Dict]:
        """规范化正向查询结果"""
        if not raw_result:
            return []

        items = []
        if isinstance(raw_result, dict):
            for key, values in raw_result.items():
                if isinstance(values, list):
                    for val in values:
                        if isinstance(val, str) and val:
                            items.append(self._make_knowledge_item(
                                source_entity=source_entity,
                                relation_type=key,
                                target_entity=val,
                                content=val,
                                path_name=path_name,
                                dim_name=dim_name,
                            ))
                elif isinstance(values, str) and values:
                    items.append(self._make_knowledge_item(
                        source_entity=source_entity,
                        relation_type=key,
                        target_entity=values,
                        content=values,
                        path_name=path_name,
                        dim_name=dim_name,
                    ))
        elif isinstance(raw_result, list):
            for val in raw_result:
                if isinstance(val, str) and val:
                    items.append(self._make_knowledge_item(
                        source_entity=source_entity,
                        relation_type=path_name,
                        target_entity=val,
                        content=val,
                        path_name=path_name,
                        dim_name=dim_name,
                    ))
                elif isinstance(val, dict):
                    items.append(self._normalize_dict_item(val, source_entity, path_name, dim_name))

        return items

    def _normalize_reverse_result(
        self,
        raw_result: Any,
        source_entity: str,
        source_type: str,
        path_name: str,
        dim_name: str,
    ) -> List[Dict]:
        """规范化反向查询结果"""
        if not raw_result:
            return []

        items = []
        if isinstance(raw_result, list):
            for val in raw_result:
                if isinstance(val, str) and val:
                    items.append(self._make_knowledge_item(
                        source_entity=source_entity,
                        relation_type=path_name,
                        target_entity=val,
                        content=f"{source_type}：{source_entity} → 疾病：{val}",
                        path_name=path_name,
                        dim_name=dim_name,
                    ))
                elif isinstance(val, dict):
                    items.append(self._normalize_dict_item(val, source_entity, path_name, dim_name))
        elif isinstance(raw_result, dict):
            for key, values in raw_result.items():
                if isinstance(values, list):
                    for val in values:
                        if isinstance(val, str) and val:
                            items.append(self._make_knowledge_item(
                                source_entity=source_entity,
                                relation_type=f"{path_name}.{key}",
                                target_entity=val,
                                content=val,
                                path_name=path_name,
                                dim_name=dim_name,
                            ))

        return items

    def _normalize_disease_info(
        self,
        raw_result: Any,
        disease_name: str,
        dim_name: str,
        path_name: str,
    ) -> Optional[Dict]:
        """规范化疾病属性查询结果"""
        if not raw_result or not isinstance(raw_result, dict):
            return None

        content_parts = []
        for key in ("desc", "cause", "prevent", "easy_get", "cure_lasttime", "cured_prob"):
            val = raw_result.get(key, "")
            if val and isinstance(val, str):
                content_parts.append(f"{key}: {val}")

        content = "; ".join(content_parts)
        if not content:
            return None

        return self._make_knowledge_item(
            source_entity=disease_name,
            relation_type="disease_attributes",
            target_entity=disease_name,
            content=content,
            path_name=path_name,
            dim_name=dim_name,
        )

    def _normalize_dict_item(
        self,
        item: Dict,
        source_entity: str,
        path_name: str,
        dim_name: str,
    ) -> Dict:
        """规范化字典形式的查询结果"""
        return self._make_knowledge_item(
            source_entity=item.get("source_entity", source_entity),
            relation_type=item.get("relation_type", path_name),
            target_entity=item.get("target_entity", item.get("name", "")),
            content=item.get("content", item.get("description", "")),
            path_name=path_name,
            dim_name=dim_name,
        )

    @staticmethod
    def _make_knowledge_item(
        source_entity: str,
        relation_type: str,
        target_entity: str,
        content: str,
        path_name: str,
        dim_name: str,
    ) -> Dict:
        """构建标准化的知识条目"""
        return {
            "source_entity": source_entity,
            "relation_type": relation_type,
            "target_entity": target_entity,
            "content": content,
            "_source": "graph",
            "_dimension": dim_name,
            "_path_name": path_name,
        }

    @staticmethod
    def _collect_targets(knowledge_items: List[Dict]) -> Set[str]:
        """收集知识条目中的target_entity集合，用于去重"""
        targets: Set[str] = set()
        for item in knowledge_items:
            target = item.get("target_entity", item.get("entity_name", ""))
            if target:
                targets.add(target)
        return targets
