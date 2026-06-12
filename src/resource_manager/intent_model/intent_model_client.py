# -*- coding: utf-8 -*-
"""
意图模型资源客户端类

提供意图模型资源的客户端访问接口，封装意图分类业务操作。
实体提取由NerModelClient（nlp_raner模型）负责。
"""

import logging
import time
from typing import Dict

from src.resource_manager.resource import Resource
from src.resource_manager.resource_client import ResourceClient
from src.resource_manager.intent_model.intent_model_resource import IntentModelResource

logger = logging.getLogger(__name__)


class IntentModelClient(ResourceClient):

    def __init__(self, resource: IntentModelResource):
        self._resource = resource

    def get_resource_type(self) -> str:
        return self._resource.get_type()

    def get_raw_resource(self) -> Resource:
        """获取原始资源实例"""
        return self._resource

    def classify_intent(self, text: str) -> Dict:
        logger.debug(f"[IntentModelClient] classify_intent called, text_length={len(text)}")
        start_time = time.time()
        try:
            adapter = self._resource.get_adapter()
            if adapter is None:
                raise RuntimeError("Transformers adapter not initialized")
            result = adapter.predict(text=text)
            intent_result = {
                "intent_label": result.get("label", ""),
                "confidence": result.get("confidence", 0.0)
            }
            elapsed = time.time() - start_time
            logger.info(f"[IntentModelClient] classify_intent completed, elapsed={elapsed:.3f}s, intent_label={intent_result['intent_label']}, confidence={intent_result['confidence']:.4f}")
            return intent_result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[IntentModelClient] classify_intent failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

