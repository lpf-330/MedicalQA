# -*- coding: utf-8 -*-

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from pymilvus import MilvusClient

from .milvus_adapter import MilvusAdapter

logger = logging.getLogger(__name__)


class MilvusAdapterImpl(MilvusAdapter):

    def __init__(self):
        self._uri: Optional[str] = None
        self._user: Optional[str] = None
        self._password: Optional[str] = None
        self._token: Optional[str] = None
        self._client: Optional[Any] = None
        logger.debug("[MilvusAdapter] 初始化Milvus适配器")

    def connect(self, uri: str, user: str, password: str, token: str) -> None:
        if self._client is not None:
            logger.debug("[MilvusAdapter] 已连接，跳过")
            return

        logger.info(f"[MilvusAdapter] 开始连接Milvus: uri={uri}")
        start_time = time.time()

        self._uri = uri
        self._user = user
        self._password = password
        self._token = token

        self._client = MilvusClient(
            uri=uri,
            user=user,
            password=password,
            token=token
        )

        elapsed = time.time() - start_time
        logger.info(f"[MilvusAdapter] Milvus连接成功: elapsed={elapsed:.2f}s")

    def disconnect(self) -> None:
        logger.info("[MilvusAdapter] 开始断开Milvus连接")

        if self._client is not None:
            self._client.close()
            self._client = None
            logger.debug("[MilvusAdapter] Client已关闭")

        logger.info("[MilvusAdapter] Milvus连接已断开")

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int,
        **kwargs
    ) -> List[Dict]:
        if self._client is None:
            logger.error("[MilvusAdapter] 搜索失败，未连接Milvus")
            raise RuntimeError("Not connected to Milvus")

        logger.debug(f"[MilvusAdapter] 执行搜索: collection={collection_name}, top_k={top_k}")
        start_time = time.time()

        search_params = kwargs.pop("search_params", None)
        output_fields = kwargs.pop("output_fields", ["*"])

        search_kwargs: Dict[str, Any] = {
            "collection_name": collection_name,
            "data": [query_vector],
            "limit": top_k,
            "output_fields": output_fields,
        }
        if search_params is not None:
            search_kwargs["search_params"] = search_params
        search_kwargs.update(kwargs)

        results = self._client.search(**search_kwargs)

        formatted_results: List[Dict] = []
        if results and len(results) > 0:
            for hit in results[0]:
                entity_data = hit.get("entity", {})
                # 不返回vector字段，减少数据传输
                if "vector" in entity_data:
                    del entity_data["vector"]
                item = {
                    "id": hit.get("id"),
                    "distance": hit.get("distance"),
                    "entity": entity_data,
                }
                formatted_results.append(item)

        elapsed = time.time() - start_time
        logger.info(f"[MilvusAdapter] 搜索完成: result_count={len(formatted_results)}, elapsed={elapsed:.3f}s")
        return formatted_results

    def hybrid_search(
        self,
        query_vector: List[float],
        collections: List[str],
        top_k: int,
        weights: Dict[str, float]
    ) -> List[Dict]:
        if self._client is None:
            logger.error("[MilvusAdapter] 混合搜索失败，未连接Milvus")
            raise RuntimeError("Not connected to Milvus")

        logger.debug(f"[MilvusAdapter] 执行混合搜索: collections={collections}, top_k={top_k}, weights={weights}")
        start_time = time.time()

        all_collection_results: Dict[str, List[Dict]] = {}

        def _search_collection(collection_name: str) -> tuple:
            results = self.search(
                collection_name=collection_name,
                query_vector=query_vector,
                top_k=top_k
            )
            return collection_name, results

        with ThreadPoolExecutor(max_workers=len(collections)) as executor:
            futures = {
                executor.submit(_search_collection, col): col
                for col in collections
            }
            for future in as_completed(futures):
                collection_name, results = future.result()
                all_collection_results[collection_name] = results

        merged_results: List[Dict] = []
        for collection_name, results in all_collection_results.items():
            normalized = self._normalize_scores(results)
            weighted = self._weighted_fusion(normalized, {collection_name: weights.get(collection_name, 1.0)})
            for item in weighted:
                item["collection"] = collection_name
            merged_results.extend(weighted)

        deduplicated = self._deduplicate(merged_results)
        threshold = weights.get("threshold", 0.0)
        filtered = self._filter_by_threshold(deduplicated, threshold)
        filtered.sort(key=lambda x: x.get("score", 0), reverse=True)

        final_results = filtered[:top_k]

        elapsed = time.time() - start_time
        logger.info(f"[MilvusAdapter] 混合搜索完成: result_count={len(final_results)}, elapsed={elapsed:.3f}s")
        return final_results

    def insert(
        self,
        collection_name: str,
        data: List[Dict]
    ) -> List[int]:
        if self._client is None:
            logger.error("[MilvusAdapter] 插入失败，未连接Milvus")
            raise RuntimeError("Not connected to Milvus")

        logger.debug(f"[MilvusAdapter] 执行插入: collection={collection_name}, data_count={len(data)}")
        start_time = time.time()

        result = self._client.insert(
            collection_name=collection_name,
            data=data
        )

        ids = result.get("ids", []) if isinstance(result, dict) else []

        elapsed = time.time() - start_time
        logger.info(f"[MilvusAdapter] 插入完成: inserted_count={len(ids)}, elapsed={elapsed:.3f}s")
        return ids

    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        **kwargs
    ) -> None:
        if self._client is None:
            logger.error("[MilvusAdapter] 创建集合失败，未连接Milvus")
            raise RuntimeError("Not connected to Milvus")

        logger.info(f"[MilvusAdapter] 创建集合: collection={collection_name}, dimension={dimension}")
        start_time = time.time()

        self._client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            **kwargs
        )

        elapsed = time.time() - start_time
        logger.info(f"[MilvusAdapter] 集合创建完成: elapsed={elapsed:.2f}s")

    def drop_collection(self, collection_name: str) -> None:
        if self._client is None:
            logger.error("[MilvusAdapter] 删除集合失败，未连接Milvus")
            raise RuntimeError("Not connected to Milvus")

        logger.info(f"[MilvusAdapter] 删除集合: collection={collection_name}")
        start_time = time.time()

        self._client.drop_collection(collection_name=collection_name)

        elapsed = time.time() - start_time
        logger.info(f"[MilvusAdapter] 集合删除完成: elapsed={elapsed:.2f}s")

    def is_connected(self) -> bool:
        return self._client is not None

    def _normalize_scores(self, results: List[Dict]) -> List[Dict]:
        if not results:
            return []

        distances = [r.get("distance", 0.0) for r in results]
        min_dist = min(distances)
        max_dist = max(distances)

        normalized: List[Dict] = []
        for r in results:
            distance = r.get("distance", 0.0)
            if max_dist == min_dist:
                norm_score = 1.0
            else:
                norm_score = (distance - min_dist) / (max_dist - min_dist)
            item = dict(r)
            item["score"] = norm_score
            normalized.append(item)

        return normalized

    def _weighted_fusion(self, results: List[Dict], weights: Dict[str, float]) -> List[Dict]:
        fused: List[Dict] = []
        for r in results:
            item = dict(r)
            current_score = item.get("score", 0.0)
            total_weight = sum(weights.values()) if weights else 1.0
            weighted_score = current_score * (sum(weights.values()) / total_weight) if total_weight > 0 else 0.0
            item["score"] = weighted_score
            fused.append(item)
        return fused

    def _deduplicate(self, results: List[Dict]) -> List[Dict]:
        seen_ids: set = set()
        deduplicated: List[Dict] = []
        for r in results:
            entity = r.get("entity", {})
            dedup_key = entity.get("neo4j_node_id") or entity.get("neo4j_relation_id") or r.get("id")
            if dedup_key is not None and dedup_key in seen_ids:
                continue
            if dedup_key is not None:
                seen_ids.add(dedup_key)
            deduplicated.append(r)
        return deduplicated

    def _filter_by_threshold(self, results: List[Dict], threshold: float) -> List[Dict]:
        return [r for r in results if r.get("score", 0.0) >= threshold]

    def __enter__(self) -> 'MilvusAdapterImpl':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
