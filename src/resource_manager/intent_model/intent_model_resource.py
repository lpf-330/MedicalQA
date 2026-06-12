# -*- coding: utf-8 -*-
"""
意图模型资源封装类

管理意图分类模型资源的生命周期，包括激活（加载模型）、停用（标记空闲）、
销毁（释放模型）等操作。通过TransformersAdapter访问底层模型推理能力。
"""

import logging
import time
from typing import Optional

from src.resource_manager.resource import Resource
from src.resource_manager.intent_model.intent_model_config import IntentModelConfig
from src.adapters.transformers.transformers_adapter import TransformersAdapter
from src.adapters.transformers.transformers_adapter_impl import TransformersAdapterImpl

logger = logging.getLogger(__name__)


class IntentModelResource(Resource):

    def __init__(self, config: 'IntentModelConfig'):
        self._config = config
        self._adapter: Optional[TransformersAdapter] = None
        self._last_used_time = int(time.time() * 1000)
        self._is_active = False

    def get_type(self) -> str:
        return "intent_model"

    def get_last_used_time(self) -> int:
        return self._last_used_time

    def is_activate(self) -> bool:
        return self._is_active

    def activate(self) -> None:
        if self._is_active:
            logger.debug("[IntentModelResource] activate skipped, already active")
            return

        logger.info("[IntentModelResource] activate started")
        start_time = time.time()
        try:
            config_protocol = self._config.config_protocol
            self._adapter = TransformersAdapterImpl()
            self._adapter.load_model(
                model_path=config_protocol["model_path"],
                device=config_protocol["device"],
                model_type="intent_classification"
            )
            self._is_active = True
            self._last_used_time = int(time.time() * 1000)
            elapsed = time.time() - start_time
            logger.info(f"[IntentModelResource] activate completed, elapsed={elapsed:.3f}s, model_path={config_protocol['model_path']}, device={config_protocol['device']}")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[IntentModelResource] activate failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def deactivate(self) -> None:
        """
        停用资源（释放回池，保持连接）

        语义：资源从活跃状态变为空闲状态，归还到资源池
        行为：仅标记状态，不断开连接
        场景：资源使用完毕，释放回资源池复用
        """
        if not self._is_active:
            logger.debug("[IntentModelResource] deactivate skipped, not active")
            return

        logger.debug("[IntentModelResource] deactivate: 保持连接，标记为空闲")
        self._is_active = False

    def destroy(self) -> None:
        """
        销毁资源（彻底释放）

        语义：资源彻底销毁，从资源池移除
        行为：断开连接，释放所有资源
        场景：资源池关闭、资源过期、资源异常需销毁
        """
        logger.info("[IntentModelResource] destroy started")
        start_time = time.time()
        try:
            if self._adapter is not None:
                self._adapter.unload_model()
            self._adapter = None
            self._is_active = False
            elapsed = time.time() - start_time
            logger.info(f"[IntentModelResource] destroy completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[IntentModelResource] destroy failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def get_adapter(self) -> Optional[TransformersAdapter]:
        return self._adapter
