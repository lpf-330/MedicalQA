# -*- coding: utf-8 -*-
"""
NER模型工具实现

通过GlobalResourceManager获取ner_model资源，提供医学实体提取能力。
资源采用 acquire-on-use / release-after-use 模式：仅在工作时持有资源，用完归还资源池。
"""

import logging
import threading
import time
from typing import Dict, List, Optional

from src.tools.ner_model_tool.ner_model_tool_interface import NerModelToolInterface
from src.resource_manager.global_resource_manager import GlobalResourceManager
from src.resource_manager.ner_model import NerModelClient
from src.schemas.resource_type import ResourceType, ConfigId
from src.utils.logger import log_arch_event

logger = logging.getLogger(__name__)


class NerModelTool(NerModelToolInterface):

    def __init__(self):
        self._ner_client: Optional[NerModelClient] = None
        self._ner_handle = None
        self._lock = threading.Lock()

    def _init_resource(self) -> None:
        """轻量初始化 — 不再acquire资源，资源在业务方法中按需获取"""
        logger.info("[NerModelTool] _init_resource completed (lightweight, no resource acquire)")

    def _acquire_resource(self) -> None:
        """获取资源（幂等）— acquire-on-use 模式；线程安全"""
        with self._lock:
            if self._ner_client is not None:
                return
            try:
                self._ner_handle = GlobalResourceManager.acquire(ResourceType.NER_MODEL, ConfigId.NER_MODEL_CONFIG)
                logger.info("[TOOL_RESOURCE_ACQUIRE] tool=NerModelTool, resource_type=ner_model")
                if self._ner_handle is None:
                    raise RuntimeError("Failed to acquire ner_model resource")
                if not self._ner_handle.resource.is_activate():
                    self._ner_handle.resource.activate()
                self._ner_client = self._ner_handle.get_client()
                logger.info("[NerModelTool] ner_model resource acquired")
            except Exception as e:
                logger.debug(f"[NerModelTool] 资源获取失败: {e}")
                self._ner_handle = None
                self._ner_client = None
                raise

    def _release_resource(self) -> None:
        """释放资源 — release-after-use 模式；线程安全"""
        with self._lock:
            if self._ner_handle is not None:
                try:
                    GlobalResourceManager.release(self._ner_handle)
                finally:
                    self._ner_handle = None
                    self._ner_client = None
                logger.info("[NerModelTool] ner_model resource released")

    def release_source(self) -> None:
        """释放资源 — 委托给 _release_resource"""
        logger.info(f"[TOOL_RELEASE] {self.__class__.__name__}释放资源")
        self._release_resource()

    def destroy_source(self) -> None:
        """彻底销毁NER模型资源 - 断开连接"""
        logger.info("[NerModelTool] destroy_source started")
        start_time = time.time()
        try:
            if self._ner_handle is not None:
                GlobalResourceManager.destroy(self._ner_handle)
                self._ner_handle = None
                self._ner_client = None
                logger.info("[NerModelTool] ner_model resource destroyed")

            elapsed = time.time() - start_time
            log_arch_event(logger, component="NerModelTool", stage="TOOL", event="destroy_source", status="success", design_id="ARCH-5.1", elapsed=f"{elapsed:.3f}s")
            logger.info(f"[NerModelTool] destroy_source completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NerModelTool] destroy_source failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def extract_entities(self, text: str) -> List[Dict]:
        logger.debug(f"[NerModelTool] extract_entities called, text_length={len(text)}")
        start_time = time.time()
        self._acquire_resource()
        try:
            entities = self._ner_client.extract_entities(text)
            elapsed = time.time() - start_time
            logger.info(f"[NerModelTool] extract_entities completed, elapsed={elapsed:.3f}s, entity_count={len(entities)}")
            return entities
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NerModelTool] extract_entities failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise
        finally:
            self._release_resource()
