# -*- coding: utf-8 -*-

"""Milvus连接资源客户端类。"""

import logging
import time
from typing import Dict, List

from src.resource_manager.resource import Resource
from src.resource_manager.resource_client import ResourceClient
from src.resource_manager.milvus_connection.milvus_connection_resource import MilvusConnectionResource

logger = logging.getLogger(__name__)


class MilvusConnectionClient(ResourceClient):

    def __init__(self, resource: MilvusConnectionResource):
        self._resource = resource

    def get_resource_type(self) -> str:
        return self._resource.get_type()

    def get_raw_resource(self) -> Resource:
        """获取原始资源实例"""
        return self._resource

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int
    ) -> List[Dict]:
        logger.debug(f"[MilvusConnectionClient] search called, collection_name={collection_name}, top_k={top_k}, vector_dim={len(query_vector)}")
        start_time = time.time()
        try:
            adapter = self._resource.get_adapter()
            if adapter is None:
                raise RuntimeError("Milvus adapter not initialized")
            result = adapter.search(
                collection_name=collection_name,
                query_vector=query_vector,
                top_k=top_k
            )
            elapsed = time.time() - start_time
            logger.info(f"[MilvusConnectionClient] search completed, elapsed={elapsed:.3f}s, collection_name={collection_name}, result_count={len(result)}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[MilvusConnectionClient] search failed, elapsed={elapsed:.3f}s, collection_name={collection_name}, error={str(e)}")
            raise

    def hybrid_search(
        self,
        query_vector: List[float],
        collections: List[str],
        top_k: int,
        weights: Dict[str, float]
    ) -> List[Dict]:
        logger.debug(f"[MilvusConnectionClient] hybrid_search called, collections={collections}, top_k={top_k}, weights={weights}, vector_dim={len(query_vector)}")
        start_time = time.time()
        try:
            adapter = self._resource.get_adapter()
            if adapter is None:
                raise RuntimeError("Milvus adapter not initialized")
            result = adapter.hybrid_search(
                query_vector=query_vector,
                collections=collections,
                top_k=top_k,
                weights=weights
            )
            elapsed = time.time() - start_time
            logger.info(f"[MilvusConnectionClient] hybrid_search completed, elapsed={elapsed:.3f}s, collections={collections}, result_count={len(result)}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[MilvusConnectionClient] hybrid_search failed, elapsed={elapsed:.3f}s, collections={collections}, error={str(e)}")
            raise

