# -*- coding: utf-8 -*-

import logging
import threading
import time
from typing import Any, Dict, Optional

from src.tools.intent_classification_tool.intent_classification_tool_interface import IntentClassificationToolInterface
from src.resource_manager.global_resource_manager import GlobalResourceManager
from src.resource_manager.intent_model import IntentModelClient
from src.schemas.resource_type import ResourceType, ConfigId
from src.config.business.consult_service_config import get_runtime_config
from src.utils.logger import log_arch_event

logger = logging.getLogger(__name__)


class IntentClassificationTool(IntentClassificationToolInterface):

    LABEL_MAPPING = {
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

    def __init__(self):
        self._intent_client: Optional[IntentModelClient] = None
        self._intent_handle = None
        self._lock = threading.Lock()

    def _init_resource(self) -> None:
        """轻量初始化 — 不再acquire资源，资源在业务方法中按需获取"""
        logger.info("[IntentClassificationTool] _init_resource completed (lightweight, no resource acquire)")

    def _acquire_resource(self) -> None:
        """获取资源（幂等）— acquire-on-use 模式；线程安全"""
        with self._lock:
            if self._intent_client is not None:
                return
            try:
                self._intent_handle = GlobalResourceManager.acquire(ResourceType.INTENT_MODEL, ConfigId.INTENT_MODEL_CONFIG)
                logger.info("[TOOL_RESOURCE_ACQUIRE] tool=IntentClassificationTool, resource_type=intent_model")
                if self._intent_handle is None:
                    raise RuntimeError("Failed to acquire intent_model resource")
                if not self._intent_handle.resource.is_activate():
                    self._intent_handle.resource.activate()
                self._intent_client = self._intent_handle.get_client()
                logger.info("[IntentClassificationTool] intent_model resource acquired")
            except Exception as e:
                logger.debug(f"[IntentClassificationTool] 资源获取失败: {e}")
                self._intent_handle = None
                self._intent_client = None
                raise

    def _release_resource(self) -> None:
        """释放资源 — release-after-use 模式；线程安全"""
        with self._lock:
            if self._intent_handle is not None:
                try:
                    GlobalResourceManager.release(self._intent_handle)
                finally:
                    self._intent_handle = None
                    self._intent_client = None
                logger.info("[IntentClassificationTool] intent_model resource released")

    def release_source(self) -> None:
        """释放资源 — 委托给 _release_resource"""
        logger.info(f"[TOOL_RELEASE] {self.__class__.__name__}释放资源")
        self._release_resource()

    def destroy_source(self) -> None:
        """彻底销毁意图识别模型资源 - 断开连接"""
        logger.info(f"[TOOL_DESTROY] {self.__class__.__name__}销毁资源")
        logger.info("[IntentClassificationTool] destroy_source started")
        start_time = time.time()
        try:
            if self._intent_handle is not None:
                GlobalResourceManager.destroy(self._intent_handle)
                self._intent_handle = None
                self._intent_client = None
                logger.info("[IntentClassificationTool] intent_model resource destroyed")

            elapsed = time.time() - start_time
            log_arch_event(logger, component="IntentClassificationTool", stage="TOOL", event="destroy_source", status="success", design_id="ARCH-5.1", elapsed=f"{elapsed:.3f}s")
            logger.info(f"[IntentClassificationTool] destroy_source completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[IntentClassificationTool] destroy_source failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def classify_intent(self, text: str) -> Dict[str, Any]:
        logger.debug(f"[IntentClassificationTool] classify_intent called, text_length={len(text)}")
        start_time = time.time()
        self._acquire_resource()
        try:
            result = self._intent_client.classify_intent(text)
            raw_label = result.get("intent_label", "")
            intent_label = self.LABEL_MAPPING.get(raw_label, raw_label)
            confidence = result.get("confidence", 0.0)

            matched_keywords = []
            for keyword in self.HEALTH_KEYWORDS:
                if keyword in text:
                    matched_keywords.append(keyword)

            if matched_keywords:
                original_label = intent_label
                intent_label = "health_consultation"
                confidence = max(get_runtime_config().confidence_high, confidence)
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
        finally:
            self._release_resource()
