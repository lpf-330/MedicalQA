# -*- coding: utf-8 -*-
"""向量模型资源客户端类。"""

import logging
import time
from typing import List

from src.resource_manager.resource import Resource
from src.resource_manager.resource_client import ResourceClient
from src.resource_manager.vector_model.vector_model_resource import VectorModelResource

logger = logging.getLogger(__name__)


class VectorModelClient(ResourceClient):

    def __init__(self, resource: VectorModelResource):
        self._resource = resource

    def get_resource_type(self) -> str:
        return self._resource.get_type()

    def get_raw_resource(self) -> Resource:
        """获取原始资源实例"""
        return self._resource

    def encode(self, text: str) -> List[float]:
        logger.debug(f"[VectorModelClient] encode called, text_length={len(text)}")
        start_time = time.time()
        try:
            adapter = self._resource.get_adapter()
            if adapter is None:
                raise RuntimeError("Transformers adapter not initialized")
            result = adapter.encode(text=text)
            elapsed = time.time() - start_time
            logger.info(f"[VectorModelClient] encode completed, elapsed={elapsed:.3f}s, vector_dim={len(result)}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VectorModelClient] encode failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        logger.debug(f"[VectorModelClient] encode_batch called, batch_size={len(texts)}")
        start_time = time.time()
        try:
            adapter = self._resource.get_adapter()
            if adapter is None:
                raise RuntimeError("Transformers adapter not initialized")
            result = adapter.encode_batch(texts=texts)
            elapsed = time.time() - start_time
            logger.info(f"[VectorModelClient] encode_batch completed, elapsed={elapsed:.3f}s, batch_size={len(texts)}, result_count={len(result)}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VectorModelClient] encode_batch failed, elapsed={elapsed:.3f}s, batch_size={len(texts)}, error={str(e)}")
            raise

