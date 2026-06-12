# -*- coding: utf-8 -*-

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from src.tools.vector_retrieval_tool.vector_retrieval_tool_interface import VectorRetrievalToolInterface
from src.resource_manager.global_resource_manager import GlobalResourceManager
from src.resource_manager.milvus_connection import MilvusConnectionClient
from src.resource_manager.vector_model import VectorModelClient
from src.resource_manager.resource_handle import ResourceHandle
from src.schemas.resource_type import ResourceType, ConfigId
from src.config.business.consult_service_config import get_runtime_config
from src.utils.logger import log_arch_event

logger = logging.getLogger(__name__)


class VectorRetrievalTool(VectorRetrievalToolInterface):

    def __init__(
        self,
        fusion_threshold: float = None,
        entity_weight: float = None,
        attribute_weight: float = None,
        relation_weight: float = None
    ):
        consult_config = get_runtime_config()
        self._fusion_threshold = fusion_threshold if fusion_threshold is not None else consult_config.knowledge_fusion_threshold
        self._entity_weight = entity_weight if entity_weight is not None else consult_config.vector_entity_weight
        self._attribute_weight = attribute_weight if attribute_weight is not None else consult_config.vector_attribute_weight
        self._relation_weight = relation_weight if relation_weight is not None else consult_config.vector_relation_weight
        self._milvus_client: Optional[MilvusConnectionClient] = None
        self._vector_client: Optional[VectorModelClient] = None
        self._milvus_handle: Optional[ResourceHandle] = None
        self._vector_handle: Optional[ResourceHandle] = None
        self._lock = threading.Lock()

    def _init_resource(self) -> None:
        """轻量初始化 — 不再acquire资源，资源在业务方法中按需获取"""
        logger.info("[VectorRetrievalTool] _init_resource completed (lightweight, no resource acquire)")

    def _acquire_resource(self) -> None:
        """获取资源 — 幂等，已持有则跳过；同时获取milvus连接和向量模型；线程安全"""
        with self._lock:
            if self._milvus_handle is not None:
                return
            try:
                self._milvus_handle = GlobalResourceManager.acquire(ResourceType.MILVUS_CONNECTION, ConfigId.MILVUS_CONFIG)
                logger.info("[TOOL_RESOURCE_INIT] tool=VectorRetrievalTool, resource_type=milvus_connection")
                if self._milvus_handle is None:
                    raise RuntimeError("Failed to acquire milvus_connection resource")
                if not self._milvus_handle.resource.is_activate():
                    self._milvus_handle.resource.activate()
                self._milvus_client = self._milvus_handle.get_client()
                logger.info("[VectorRetrievalTool] milvus_connection resource acquired")

                self._vector_handle = GlobalResourceManager.acquire(ResourceType.VECTOR_MODEL, ConfigId.VECTOR_MODEL_CONFIG)
                logger.info("[TOOL_RESOURCE_INIT] tool=VectorRetrievalTool, resource_type=vector_model")
                if self._vector_handle is None:
                    raise RuntimeError("Failed to acquire vector_model resource")
                if not self._vector_handle.resource.is_activate():
                    self._vector_handle.resource.activate()
                self._vector_client = self._vector_handle.get_client()
                logger.info("[VectorRetrievalTool] vector_model resource acquired")
            except Exception as e:
                logger.debug(f"[VectorRetrievalTool] 资源获取失败: {e}")
                if self._milvus_handle is not None:
                    try:
                        GlobalResourceManager.release(self._milvus_handle)
                    except Exception as e:
                        logger.debug(f"[VectorRetrievalTool] 释放milvus资源失败: {e}")
                self._milvus_handle = None
                self._milvus_client = None
                self._vector_handle = None
                self._vector_client = None
                raise

    def _release_resource(self) -> None:
        """归还资源 — 释放资源句柄归还资源池，保持连接；线程安全"""
        with self._lock:
            if self._milvus_handle is not None:
                try:
                    GlobalResourceManager.release(self._milvus_handle)
                finally:
                    self._milvus_handle = None
                    self._milvus_client = None
                logger.info("[VectorRetrievalTool] milvus_connection resource released")
            if self._vector_handle is not None:
                try:
                    GlobalResourceManager.release(self._vector_handle)
                finally:
                    self._vector_handle = None
                    self._vector_client = None
                logger.info("[VectorRetrievalTool] vector_model resource released")

    def release_source(self) -> None:
        """释放资源 - 归还资源池，保持连接"""
        logger.info("[VectorRetrievalTool] release_source started")
        self._release_resource()
        log_arch_event(logger, component="VectorRetrievalTool", stage="TOOL", event="release_source", status="success", design_id="ARCH-5.1")

    def destroy_source(self) -> None:
        """彻底销毁Milvus连接和向量模型资源 - 断开连接"""
        logger.info(f"[TOOL_DESTROY] {self.__class__.__name__}销毁资源")
        logger.info("[VectorRetrievalTool] destroy_source started")
        start_time = time.time()
        try:
            if self._milvus_handle is not None:
                GlobalResourceManager.destroy(self._milvus_handle)
                self._milvus_handle = None
                self._milvus_client = None
                logger.info("[VectorRetrievalTool] milvus_connection resource destroyed")

            if self._vector_handle is not None:
                GlobalResourceManager.destroy(self._vector_handle)
                self._vector_handle = None
                self._vector_client = None
                logger.info("[VectorRetrievalTool] vector_model resource destroyed")

            elapsed = time.time() - start_time
            log_arch_event(logger, component="VectorRetrievalTool", stage="TOOL", event="destroy_source", status="success", design_id="ARCH-5.1", elapsed=f"{elapsed:.3f}s")
            logger.info(f"[VectorRetrievalTool] destroy_source completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VectorRetrievalTool] destroy_source failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def hybrid_search(
        self,
        query: str,
        top_k: int = 20,
        collections: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        logger.debug(f"[VectorRetrievalTool] hybrid_search called, query_length={len(query)}, top_k={top_k}")
        self._acquire_resource()
        try:
            start_time = time.time()
            if collections is None:
                collections = ["medical_entity", "entity_attributes", "entity_relations"]
            if weights is None:
                weights = {
                    "medical_entity": self._entity_weight,
                    "entity_attributes": self._attribute_weight,
                    "entity_relations": self._relation_weight
                }

            query_vector = self._encode_query(query)
            results = self._milvus_client.hybrid_search(
                query_vector=query_vector,
                collections=collections,
                top_k=top_k,
                weights=weights
            )
            filtered = self._filter_by_threshold(results, self._fusion_threshold)
            elapsed = time.time() - start_time

            # 检索参数日志
            logger.debug(f"[RETRIEVAL_PARAMS] query_length={len(query)}, top_k={top_k}, "
                        f"collections={collections}, weights={weights}, fusion_threshold={self._fusion_threshold}")

            # 结果分布日志
            if results:
                scores = [r.get("score", 0.0) for r in results if isinstance(r, dict)]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    max_score = max(scores)
                    min_score = min(scores)
                    logger.debug(f"[RESULT_DISTRIBUTION] total={len(results)}, filtered={len(filtered)}, "
                               f"avg_score={avg_score:.4f}, max_score={max_score:.4f}, min_score={min_score:.4f}, "
                               f"threshold={self._fusion_threshold}")

            logger.info(f"[VectorRetrievalTool] hybrid_search completed, elapsed={elapsed:.3f}s, total_results={len(results)}, filtered_results={len(filtered)}")
            return filtered
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VectorRetrievalTool] hybrid_search failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def search_entities(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        logger.debug(f"[VectorRetrievalTool] search_entities called, query_length={len(query)}, top_k={top_k}")
        self._acquire_resource()
        try:
            start_time = time.time()
            query_vector = self._encode_query(query)
            result = self._milvus_client.search(
                collection_name="medical_entity",
                query_vector=query_vector,
                top_k=top_k
            )
            elapsed = time.time() - start_time
            logger.info(f"[VectorRetrievalTool] search_entities completed, elapsed={elapsed:.3f}s, result_count={len(result)}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VectorRetrievalTool] search_entities failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def search_attributes(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        logger.debug(f"[VectorRetrievalTool] search_attributes called, query_length={len(query)}, top_k={top_k}")
        self._acquire_resource()
        try:
            start_time = time.time()
            query_vector = self._encode_query(query)
            result = self._milvus_client.search(
                collection_name="entity_attributes",
                query_vector=query_vector,
                top_k=top_k
            )
            elapsed = time.time() - start_time
            logger.info(f"[VectorRetrievalTool] search_attributes completed, elapsed={elapsed:.3f}s, result_count={len(result)}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VectorRetrievalTool] search_attributes failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def search_relations(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        logger.debug(f"[VectorRetrievalTool] search_relations called, query_length={len(query)}, top_k={top_k}")
        self._acquire_resource()
        try:
            start_time = time.time()
            query_vector = self._encode_query(query)
            result = self._milvus_client.search(
                collection_name="entity_relations",
                query_vector=query_vector,
                top_k=top_k
            )
            elapsed = time.time() - start_time
            logger.info(f"[VectorRetrievalTool] search_relations completed, elapsed={elapsed:.3f}s, result_count={len(result)}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VectorRetrievalTool] search_relations failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise
        finally:
            self._release_resource()

    def _encode_query(self, query: str) -> List[float]:
        logger.debug(f"[VectorRetrievalTool] _encode_query called, query_length={len(query)}")
        start_time = time.time()
        try:
            result = self._vector_client.encode(text=query)
            elapsed = time.time() - start_time
            logger.debug(f"[VectorRetrievalTool] _encode_query completed, elapsed={elapsed:.3f}s, vector_dim={len(result)}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VectorRetrievalTool] _encode_query failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def _filter_by_threshold(self, results: List[Dict], threshold: float) -> List[Dict]:
        return [r for r in results if r.get("score", 0) >= threshold]
