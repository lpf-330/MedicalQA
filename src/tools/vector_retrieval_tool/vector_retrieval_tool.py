# -*- coding: utf-8 -*-

import logging
import time
from typing import Any, Dict, List, Optional

from src.tools.tool import Tool
from src.resource_manager.global_resource_manager import GlobalResourceManager
from src.resource_manager.milvus_connection.milvus_connection_resource import MilvusConnectionResource
from src.resource_manager.vector_model.vector_model_resource import VectorModelResource

logger = logging.getLogger(__name__)


class VectorEnhancedRetrievalTool(Tool):

    def __init__(
        self,
        milvus_uri: str,
        milvus_user: str,
        milvus_password: str,
        milvus_token: str = "",
        vector_model_path: str = "",
        vector_device: str = "cpu",
        vector_dimension: int = 1024,
        fusion_threshold: float = 0.6,
        entity_weight: float = 0.40,
        attribute_weight: float = 0.30,
        relation_weight: float = 0.30
    ):
        self._milvus_uri = milvus_uri
        self._milvus_user = milvus_user
        self._milvus_password = milvus_password
        self._milvus_token = milvus_token
        self._vector_model_path = vector_model_path
        self._vector_device = vector_device
        self._vector_dimension = vector_dimension
        self._fusion_threshold = fusion_threshold
        self._entity_weight = entity_weight
        self._attribute_weight = attribute_weight
        self._relation_weight = relation_weight
        self._milvus_resource: Optional[MilvusConnectionResource] = None
        self._vector_resource: Optional[VectorModelResource] = None
        self._milvus_handle = None
        self._vector_handle = None

    def _init_resource(self) -> None:
        if self._milvus_resource is not None:
            logger.debug("[VectorEnhancedRetrievalTool] _init_resource skipped, already initialized")
            return

        logger.info("[VectorEnhancedRetrievalTool] _init_resource started")
        start_time = time.time()
        try:
            self._milvus_handle = GlobalResourceManager.acquire("milvus_connection", "milvus_config")
            if self._milvus_handle is not None:
                self._milvus_resource = self._milvus_handle.resource
                if not self._milvus_resource.is_activate():
                    self._milvus_resource.activate()
                logger.info("[VectorEnhancedRetrievalTool] milvus_connection resource acquired")
            else:
                logger.warning("[VectorEnhancedRetrievalTool] failed to acquire milvus_connection resource")

            self._vector_handle = GlobalResourceManager.acquire("vector_model", "vector_model_config")
            if self._vector_handle is not None:
                self._vector_resource = self._vector_handle.resource
                if not self._vector_resource.is_activate():
                    self._vector_resource.activate()
                logger.info("[VectorEnhancedRetrievalTool] vector_model resource acquired")
            else:
                logger.warning("[VectorEnhancedRetrievalTool] failed to acquire vector_model resource")

            elapsed = time.time() - start_time
            logger.info(f"[VectorEnhancedRetrievalTool] _init_resource completed, elapsed={elapsed:.3f}s, milvus_ready={self._milvus_resource is not None}, vector_ready={self._vector_resource is not None}")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VectorEnhancedRetrievalTool] _init_resource failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def release_source(self) -> None:
        logger.info("[VectorEnhancedRetrievalTool] release_source started")
        start_time = time.time()
        try:
            if self._milvus_handle is not None:
                GlobalResourceManager.release(self._milvus_handle)
                self._milvus_handle = None
                self._milvus_resource = None
                logger.info("[VectorEnhancedRetrievalTool] milvus_connection resource released")

            if self._vector_handle is not None:
                GlobalResourceManager.release(self._vector_handle)
                self._vector_handle = None
                self._vector_resource = None
                logger.info("[VectorEnhancedRetrievalTool] vector_model resource released")

            elapsed = time.time() - start_time
            logger.info(f"[VectorEnhancedRetrievalTool] release_source completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VectorEnhancedRetrievalTool] release_source failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def hybrid_search(
        self,
        query: str,
        top_k: int = 20,
        collections: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        logger.debug(f"[VectorEnhancedRetrievalTool] hybrid_search called, query_length={len(query)}, top_k={top_k}")
        start_time = time.time()
        try:
            if self._milvus_resource is None or self._vector_resource is None:
                raise RuntimeError("Tool not initialized, call _init_resource first")

            if collections is None:
                collections = ["medical_entity", "entity_attributes", "entity_relations"]
            if weights is None:
                weights = {
                    "medical_entity": self._entity_weight,
                    "entity_attributes": self._attribute_weight,
                    "entity_relations": self._relation_weight
                }

            query_vector = self._encode_query(query)
            milvus_adapter = self._milvus_resource.get_adapter()
            if milvus_adapter is None:
                raise RuntimeError("Milvus adapter not initialized")
            results = milvus_adapter.hybrid_search(
                query_vector=query_vector,
                collections=collections,
                top_k=top_k,
                weights=weights
            )
            filtered = self._filter_by_threshold(results, self._fusion_threshold)
            elapsed = time.time() - start_time
            logger.info(f"[VectorEnhancedRetrievalTool] hybrid_search completed, elapsed={elapsed:.3f}s, total_results={len(results)}, filtered_results={len(filtered)}")
            return filtered
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VectorEnhancedRetrievalTool] hybrid_search failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def search_entities(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        logger.debug(f"[VectorEnhancedRetrievalTool] search_entities called, query_length={len(query)}, top_k={top_k}")
        start_time = time.time()
        try:
            if self._milvus_resource is None or self._vector_resource is None:
                raise RuntimeError("Tool not initialized, call _init_resource first")

            query_vector = self._encode_query(query)
            milvus_adapter = self._milvus_resource.get_adapter()
            if milvus_adapter is None:
                raise RuntimeError("Milvus adapter not initialized")
            result = milvus_adapter.search(
                collection_name="medical_entity",
                query_vector=query_vector,
                top_k=top_k
            )
            elapsed = time.time() - start_time
            logger.info(f"[VectorEnhancedRetrievalTool] search_entities completed, elapsed={elapsed:.3f}s, result_count={len(result)}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VectorEnhancedRetrievalTool] search_entities failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def search_attributes(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        logger.debug(f"[VectorEnhancedRetrievalTool] search_attributes called, query_length={len(query)}, top_k={top_k}")
        start_time = time.time()
        try:
            if self._milvus_resource is None or self._vector_resource is None:
                raise RuntimeError("Tool not initialized, call _init_resource first")

            query_vector = self._encode_query(query)
            milvus_adapter = self._milvus_resource.get_adapter()
            if milvus_adapter is None:
                raise RuntimeError("Milvus adapter not initialized")
            result = milvus_adapter.search(
                collection_name="entity_attributes",
                query_vector=query_vector,
                top_k=top_k
            )
            elapsed = time.time() - start_time
            logger.info(f"[VectorEnhancedRetrievalTool] search_attributes completed, elapsed={elapsed:.3f}s, result_count={len(result)}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VectorEnhancedRetrievalTool] search_attributes failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def search_relations(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        logger.debug(f"[VectorEnhancedRetrievalTool] search_relations called, query_length={len(query)}, top_k={top_k}")
        start_time = time.time()
        try:
            if self._milvus_resource is None or self._vector_resource is None:
                raise RuntimeError("Tool not initialized, call _init_resource first")

            query_vector = self._encode_query(query)
            milvus_adapter = self._milvus_resource.get_adapter()
            if milvus_adapter is None:
                raise RuntimeError("Milvus adapter not initialized")
            result = milvus_adapter.search(
                collection_name="entity_relations",
                query_vector=query_vector,
                top_k=top_k
            )
            elapsed = time.time() - start_time
            logger.info(f"[VectorEnhancedRetrievalTool] search_relations completed, elapsed={elapsed:.3f}s, result_count={len(result)}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VectorEnhancedRetrievalTool] search_relations failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def _encode_query(self, query: str) -> List[float]:
        logger.debug(f"[VectorEnhancedRetrievalTool] _encode_query called, query_length={len(query)}")
        start_time = time.time()
        try:
            if self._vector_resource is None:
                raise RuntimeError("Vector resource not initialized")
            vector_adapter = self._vector_resource.get_adapter()
            if vector_adapter is None:
                raise RuntimeError("Vector adapter not initialized")
            result = vector_adapter.encode(text=query)
            elapsed = time.time() - start_time
            logger.debug(f"[VectorEnhancedRetrievalTool] _encode_query completed, elapsed={elapsed:.3f}s, vector_dim={len(result)}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VectorEnhancedRetrievalTool] _encode_query failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def _filter_by_threshold(self, results: List[Dict], threshold: float) -> List[Dict]:
        return [r for r in results if r.get("score", 0) >= threshold]
