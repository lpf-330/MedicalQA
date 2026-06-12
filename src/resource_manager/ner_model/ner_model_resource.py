# -*- coding: utf-8 -*-
"""
NER模型资源封装类

管理NER命名实体识别模型资源的生命周期，包括激活（加载模型）、停用（标记空闲）、
销毁（释放模型）等操作。通过TransformersAdapter访问底层模型推理能力。
"""

import logging
import time
from typing import Optional

from src.resource_manager.resource import Resource
from src.resource_manager.ner_model.ner_model_config import NerModelConfig
from src.adapters.transformers.transformers_adapter import TransformersAdapter
from src.adapters.transformers.transformers_adapter_impl import TransformersAdapterImpl

logger = logging.getLogger(__name__)


class NerModelResource(Resource):

    def __init__(self, config: 'NerModelConfig'):
        self._config = config
        self._adapter: Optional[TransformersAdapter] = None
        self._last_used_time = int(time.time() * 1000)
        self._is_active = False

    def get_type(self) -> str:
        return "ner_model"

    def get_last_used_time(self) -> int:
        return self._last_used_time

    def is_activate(self) -> bool:
        return self._is_active

    def activate(self) -> None:
        if self._is_active:
            logger.debug("[NerModelResource] activate skipped, already active")
            return

        logger.info("[NerModelResource] activate started")
        start_time = time.time()
        try:
            config_protocol = self._config.config_protocol
            self._adapter = TransformersAdapterImpl()
            self._adapter.load_model(
                model_path=config_protocol["model_path"],
                device=config_protocol["device"],
                model_type="ner"
            )
            self._is_active = True
            self._last_used_time = int(time.time() * 1000)
            elapsed = time.time() - start_time
            logger.info(f"[NerModelResource] activate completed, elapsed={elapsed:.3f}s, model_path={config_protocol['model_path']}, device={config_protocol['device']}")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NerModelResource] activate failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def deactivate(self) -> None:
        if not self._is_active:
            logger.debug("[NerModelResource] deactivate skipped, not active")
            return

        logger.debug("[NerModelResource] deactivate: 保持连接，标记为空闲")
        self._is_active = False

    def destroy(self) -> None:
        logger.info("[NerModelResource] destroy started")
        start_time = time.time()
        try:
            if self._adapter is not None:
                self._adapter.unload_model()
            self._adapter = None
            self._is_active = False
            elapsed = time.time() - start_time
            logger.info(f"[NerModelResource] destroy completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NerModelResource] destroy failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def get_adapter(self) -> Optional[TransformersAdapter]:
        return self._adapter