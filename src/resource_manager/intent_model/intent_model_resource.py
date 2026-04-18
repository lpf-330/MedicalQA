# -*- coding: utf-8 -*-

import logging
import re
import time
from typing import Any, Dict, List, Optional

from src.resource_manager.resource import Resource
from src.resource_manager.resource_config import ResourceConfig
from src.resource_manager.resource_factory import ResourceFactory
from src.resource_manager.resource_client import ResourceClient
from src.adapters import TransformersAdapterImpl
from src.adapters.transformers.transformers_adapter import TransformersAdapter

logger = logging.getLogger(__name__)


class IntentModelResource(Resource):

    def __init__(self, config: 'IntentModelConfig'):
        self._config = config
        self._adapter: Optional[TransformersAdapterImpl] = None
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
                model_type="classification"
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

    def get_adapter(self) -> Optional[TransformersAdapterImpl]:
        return self._adapter


class IntentModelConfig(ResourceConfig[Dict[str, Any]]):

    def __init__(
        self,
        model_path: str,
        model_name: str = "intent-classification",
        device: str = "cpu",
        max_length: int = 128,
        resource_name: str = "意图分类模型"
    ):
        self._resource_type = "intent_model"
        self._resource_name = resource_name
        self._config_protocol: Dict[str, Any] = {
            "model_path": model_path,
            "model_name": model_name,
            "device": device,
            "max_length": max_length
        }

    @property
    def resource_type(self) -> str:
        return self._resource_type

    @property
    def resource_name(self) -> str:
        return self._resource_name

    @property
    def config_protocol(self) -> Dict[str, Any]:
        return self._config_protocol

    def to_dict(self) -> dict:
        return {
            "resource_type": self._resource_type,
            "resource_name": self._resource_name,
            "config_protocol": self._config_protocol
        }

    def validate(self) -> bool:
        if not self._config_protocol.get("model_path"):
            return False
        return True


class IntentModelFactory(ResourceFactory):

    def create(self, config: ResourceConfig) -> Resource:
        if not isinstance(config, IntentModelConfig):
            raise TypeError(f"Expected IntentModelConfig, got {type(config)}")

        return IntentModelResource(config)

    def destroy(self, resource: Resource) -> None:
        if not isinstance(resource, IntentModelResource):
            raise TypeError(f"Expected IntentModelResource, got {type(resource)}")

        resource.destroy()


class IntentModelClient(ResourceClient):

    def __init__(self, resource: IntentModelResource):
        self._resource = resource
        self._adapter: Optional[TransformersAdapter] = resource.get_adapter()

    def get_resource_type(self) -> str:
        return self._resource.get_type()

    def get_raw_resource(self) -> Resource:
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

    def extract_entities(self, text: str) -> List[Dict]:
        logger.debug(f"[IntentModelClient] extract_entities called, text_length={len(text)}")
        start_time = time.time()
        try:
            adapter = self._resource.get_adapter()
            if adapter is None:
                raise RuntimeError("Transformers adapter not initialized")
            result = adapter.predict(text=text)
            entities = []
            label = result.get("label", "")
            confidence = result.get("confidence", 0.0)
            if confidence > 0.3:
                keywords = re.findall(r'[\u4e00-\u9fff]+', text)
                for keyword in keywords:
                    if len(keyword) >= 2:
                        entities.append({
                            "entity_name": keyword,
                            "entity_type": "medical_term"
                        })
            elapsed = time.time() - start_time
            logger.info(f"[IntentModelClient] extract_entities completed, elapsed={elapsed:.3f}s, entity_count={len(entities)}, confidence={confidence:.4f}")
            return entities
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[IntentModelClient] extract_entities failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def get_adapter(self) -> TransformersAdapter:
        logger.debug("[IntentModelClient] get_adapter called")
        return self._resource.get_adapter()
