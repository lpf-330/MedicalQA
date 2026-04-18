# -*- coding: utf-8 -*-

import logging
import time
from typing import Any, Dict, List, Optional

from src.tools.tool import Tool
from src.resource_manager.global_resource_manager import GlobalResourceManager
from src.resource_manager.intent_model.intent_model_resource import IntentModelResource

logger = logging.getLogger(__name__)


class IntentClassificationTool(Tool):

    LABEL_MAPPING = {
        "LABEL_0": "health_consultation",
        "LABEL_1": "chat",
        "LABEL_2": "other",
        "health": "health_consultation",
        "chat": "chat",
        "other": "other",
        "non_health": "other"
    }

    HEALTH_KEYWORDS = [
        "症状", "治疗", "药物", "疾病", "医院", "医生", "诊断", "检查",
        "糖尿病", "高血压", "感冒", "发烧", "咳嗽", "头痛", "胃痛",
        "吃什么", "怎么治", "怎么办", "怎么调理", "注意事项",
        "副作用", "禁忌", "用量", "用法", "疗程",
        "预防", "保健", "养生", "营养", "饮食",
        "疼痛", "不适", "不舒服", "难受",
        "药", "片", "胶囊", "注射", "输液"
    ]

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        max_length: int = 128
    ):
        self._model_path = model_path
        self._device = device
        self._max_length = max_length
        self._intent_resource: Optional[IntentModelResource] = None
        self._intent_handle = None

    def _init_resource(self) -> None:
        if self._intent_resource is not None:
            logger.debug("[IntentClassificationTool] _init_resource skipped, already initialized")
            return

        logger.info("[IntentClassificationTool] _init_resource started")
        start_time = time.time()
        try:
            self._intent_handle = GlobalResourceManager.acquire("intent_model", "intent_model_config")
            if self._intent_handle is not None:
                self._intent_resource = self._intent_handle.resource
                if not self._intent_resource.is_activate():
                    self._intent_resource.activate()
                logger.info("[IntentClassificationTool] intent_model resource acquired")
            else:
                logger.warning("[IntentClassificationTool] failed to acquire intent_model resource")

            elapsed = time.time() - start_time
            logger.info(f"[IntentClassificationTool] _init_resource completed, elapsed={elapsed:.3f}s, client_ready={self._intent_resource is not None}")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[IntentClassificationTool] _init_resource failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def release_source(self) -> None:
        logger.info("[IntentClassificationTool] release_source started")
        start_time = time.time()
        try:
            if self._intent_handle is not None:
                GlobalResourceManager.release(self._intent_handle)
                self._intent_handle = None
                self._intent_resource = None
                logger.info("[IntentClassificationTool] intent_model resource released")

            elapsed = time.time() - start_time
            logger.info(f"[IntentClassificationTool] release_source completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[IntentClassificationTool] release_source failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def classify_intent(self, text: str) -> Dict[str, Any]:
        logger.debug(f"[IntentClassificationTool] classify_intent called, text_length={len(text)}")
        start_time = time.time()
        try:
            if self._intent_resource is None:
                raise RuntimeError("Tool not initialized, call _init_resource first")
            adapter = self._intent_resource.get_adapter()
            if adapter is None:
                raise RuntimeError("Adapter not initialized")
            result = adapter.predict(text=text)
            raw_label = result.get("label", "")
            intent_label = self.LABEL_MAPPING.get(raw_label, raw_label)
            confidence = result.get("confidence", 0.0)
            
            matched_keywords = []
            for keyword in self.HEALTH_KEYWORDS:
                if keyword in text:
                    matched_keywords.append(keyword)
            
            if matched_keywords:
                original_label = intent_label
                intent_label = "health_consultation"
                confidence = max(0.8, confidence)
                logger.info(f"[IntentClassificationTool] 关键词强制修正: original_label={original_label}, new_label={intent_label}, matched_keywords={matched_keywords}, confidence={confidence:.4f}")
            
            intent_result = {
                "intent_label": intent_label,
                "confidence": confidence
            }
            elapsed = time.time() - start_time
            logger.info(f"[IntentClassificationTool] classify_intent completed, elapsed={elapsed:.3f}s, intent_label={intent_result['intent_label']}, confidence={intent_result['confidence']:.4f}")
            return intent_result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[IntentClassificationTool] classify_intent failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        logger.debug(f"[IntentClassificationTool] extract_entities called, text_length={len(text)}")
        start_time = time.time()
        try:
            if self._intent_resource is None:
                raise RuntimeError("Tool not initialized, call _init_resource first")
            adapter = self._intent_resource.get_adapter()
            if adapter is None:
                raise RuntimeError("Adapter not initialized")
            result = adapter.predict(text=text)
            entities = []
            label = result.get("label", "")
            confidence = result.get("confidence", 0.0)
            import re
            if confidence > 0.3:
                keywords = re.findall(r'[\u4e00-\u9fff]+', text)
                for keyword in keywords:
                    if len(keyword) >= 2:
                        entities.append({
                            "entity_name": keyword,
                            "entity_type": "medical_term"
                        })
            elapsed = time.time() - start_time
            logger.info(f"[IntentClassificationTool] extract_entities completed, elapsed={elapsed:.3f}s, entity_count={len(entities)}, confidence={confidence:.4f}")
            return entities
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[IntentClassificationTool] extract_entities failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise
