# -*- coding: utf-8 -*-
"""
向量预扫描模块

NER实体名称 → Milvus向量检索 → entity_name + entity_type + neo4j_node_id 映射。
为后续图查询提供锚定ID。

向量预扫描只做"候选发现"：发现NER实体在向量库中对应的entity_type和neo4j_node_id，
不做知识内容的直接利用。图查询通过neo4j_node_id获取完整内容。
"""

import logging
from typing import Any, Dict, List, Optional

from src.config.business.report_service_config import get_runtime_config

logger = logging.getLogger(__name__)


class VectorPrescanResult:
    """单实体的向量预扫描结果"""

    __slots__ = ("entity_name", "entity_type", "neo4j_node_id", "score")

    def __init__(
        self,
        entity_name: str,
        entity_type: str,
        neo4j_node_id: str,
        score: float,
    ):
        self.entity_name = entity_name
        self.entity_type = entity_type
        self.neo4j_node_id = neo4j_node_id
        self.score = score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_name": self.entity_name,
            "entity_type": self.entity_type,
            "neo4j_node_id": self.neo4j_node_id,
            "score": self.score,
        }


class VectorPrescan:
    """
    向量预扫描器

    对NER识别的实体名称做Milvus检索，获取：
    - entity_type: 实体在知识图谱中的类型（Disease/Drug/Symptom/...）
    - neo4j_node_id: 实体在Neo4j中的节点ID，用于后续图查询锚定

    使用方式：通过call_tool调用VectorRetrievalTool的hybrid_search方法。
    """

    def __init__(self) -> None:
        self._config = get_runtime_config()

    def prescan_entities(
        self,
        entity_names: List[str],
        vector_tool: Any,
        top_k: int = 0,
        score_threshold: float = 0.0,
    ) -> Dict[str, VectorPrescanResult]:
        """
        对一批实体名称做向量预扫描

        Args:
            entity_names: NER识别的实体名称列表
            vector_tool: VectorRetrievalTool实例（通过resource.get_tool_handler获取）
            top_k: 每个实体检索的top_k数量，0则使用配置默认值
            score_threshold: 分数阈值，0则使用配置默认值

        Returns:
            Dict[str, VectorPrescanResult]: entity_name → 预扫描结果映射
        """
        if not entity_names or not vector_tool:
            logger.info("[VectorPrescan] 实体列表为空或向量工具不可用，跳过预扫描")
            return {}

        effective_top_k = top_k if top_k > 0 else self._config.agent_vector_prescan_top_k
        effective_threshold = score_threshold if score_threshold > 0 else self._config.agent_vector_prescan_threshold

        results: Dict[str, VectorPrescanResult] = {}

        for entity_name in entity_names:
            if not entity_name:
                continue
            prescan = self._prescan_single(entity_name, vector_tool, effective_top_k, effective_threshold)
            if prescan is not None:
                results[entity_name] = prescan

        logger.info(
            f"[VectorPrescan] 预扫描完成: 输入实体数={len(entity_names)}, "
            f"匹配数={len(results)}, 命中率={len(results)/max(len(entity_names),1):.1%}"
        )
        return results

    def _prescan_single(
        self,
        entity_name: str,
        vector_tool: Any,
        top_k: int,
        score_threshold: float,
    ) -> Optional[VectorPrescanResult]:
        """对单个实体名称做向量检索，返回最佳匹配的预扫描结果"""
        try:
            search_results = vector_tool.call_tool({
                "query": entity_name,
                "top_k": top_k,
                "collections": ["medical_entity"],
                "weights": {"medical_entity": 1.0},
            })

            if not isinstance(search_results, list) or not search_results:
                return None

            best_match = search_results[0]
            match_name = best_match.get("entity_name", best_match.get("name", ""))
            match_score = float(best_match.get("score", 0.0))
            match_type = best_match.get("entity_type", "")
            match_node_id = str(best_match.get("neo4j_node_id", ""))

            if match_score < score_threshold:
                logger.debug(
                    f"[VectorPrescan] 实体'{entity_name}'最佳匹配分数{match_score:.3f}"
                    f"低于阈值{score_threshold}，跳过"
                )
                return None

            if not match_type or not match_node_id:
                logger.debug(
                    f"[VectorPrescan] 实体'{entity_name}'匹配结果缺少"
                    f"entity_type或neo4j_node_id，跳过"
                )
                return None

            return VectorPrescanResult(
                entity_name=match_name,
                entity_type=match_type,
                neo4j_node_id=match_node_id,
                score=match_score,
            )

        except Exception as e:
            logger.warning(
                f"[VectorPrescan] 实体'{entity_name}'预扫描失败: "
                f"{type(e).__name__}: {str(e)}"
            )
            return None

    def prescan_by_dimension_queries(
        self,
        dimension_queries: Dict[str, str],
        vector_tool: Any,
        top_k: int = 0,
        score_threshold: float = 0.0,
    ) -> Dict[str, List[VectorPrescanResult]]:
        """
        对8维度查询文本做向量预扫描，为每个维度发现候选实体

        Args:
            dimension_queries: 维度名→查询文本映射
            vector_tool: VectorRetrievalTool实例
            top_k: 每维度检索数量
            score_threshold: 分数阈值

        Returns:
            Dict[str, List[VectorPrescanResult]]: 维度名→候选实体列表
        """
        if not dimension_queries or not vector_tool:
            return {}

        effective_top_k = top_k if top_k > 0 else self._config.agent_vector_prescan_top_k
        effective_threshold = score_threshold if score_threshold > 0 else self._config.agent_vector_prescan_threshold

        results: Dict[str, List[VectorPrescanResult]] = {}

        for dim_name, query in dimension_queries.items():
            if not query:
                continue
            dim_candidates = self._prescan_dimension_query(
                query, vector_tool, effective_top_k, effective_threshold
            )
            if dim_candidates:
                results[dim_name] = dim_candidates

        logger.info(
            f"[VectorPrescan] 维度预扫描完成: 维度数={len(dimension_queries)}, "
            f"有候选维度数={len(results)}"
        )
        return results

    def _prescan_dimension_query(
        self,
        query: str,
        vector_tool: Any,
        top_k: int,
        score_threshold: float,
    ) -> List[VectorPrescanResult]:
        """对单个维度查询文本做向量检索"""
        try:
            search_results = vector_tool.call_tool({
                "query": query,
                "top_k": top_k,
                "collections": ["medical_entity"],
                "weights": {"medical_entity": 1.0},
            })

            if not isinstance(search_results, list):
                return []

            candidates = []
            for item in search_results:
                match_name = item.get("entity_name", item.get("name", ""))
                match_score = float(item.get("score", 0.0))
                match_type = item.get("entity_type", "")
                match_node_id = str(item.get("neo4j_node_id", ""))

                if not match_name or match_score < score_threshold:
                    continue
                if not match_type or not match_node_id:
                    continue

                candidates.append(VectorPrescanResult(
                    entity_name=match_name,
                    entity_type=match_type,
                    neo4j_node_id=match_node_id,
                    score=match_score,
                ))

            return candidates

        except Exception as e:
            logger.warning(
                f"[VectorPrescan] 维度查询预扫描失败: "
                f"{type(e).__name__}: {str(e)}"
            )
            return []
