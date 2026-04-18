# -*- coding: utf-8 -*-

import logging
import time
from typing import Any, Dict, List, Optional

from src.resource_manager.resource import Resource
from src.resource_manager.resource_config import ResourceConfig
from src.resource_manager.resource_factory import ResourceFactory
from src.resource_manager.resource_client import ResourceClient
from src.adapters import MilvusAdapterImpl
from src.adapters.milvus.milvus_adapter import MilvusAdapter

logger = logging.getLogger(__name__)


class MilvusConnectionResource(Resource):

    def __init__(self, config: 'MilvusConnectionConfig'):
        self._config = config
        self._adapter: Optional[MilvusAdapterImpl] = None
        self._last_used_time = int(time.time() * 1000)
        self._is_active = False

    def get_type(self) -> str:
        return "milvus_connection"

    def get_last_used_time(self) -> int:
        return self._last_used_time

    def is_activate(self) -> bool:
        return self._is_active

    def activate(self) -> None:
        if self._is_active:
            logger.debug("[MilvusConnectionResource] activate skipped, already active")
            return

        logger.info("[MilvusConnectionResource] activate started")
        start_time = time.time()
        try:
            config_protocol = self._config.config_protocol
            self._adapter = MilvusAdapterImpl()
            self._adapter.connect(
                uri=config_protocol["uri"],
                user=config_protocol["user"],
                password=config_protocol["password"],
                token=config_protocol["token"]
            )
            self._is_active = True
            self._last_used_time = int(time.time() * 1000)
            elapsed = time.time() - start_time
            logger.info(f"[MilvusConnectionResource] activate completed, elapsed={elapsed:.3f}s, uri={config_protocol['uri']}")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[MilvusConnectionResource] activate failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def deactivate(self) -> None:
        """
        停用资源（释放回池，保持连接）
        
        语义：资源从活跃状态变为空闲状态，归还到资源池
        行为：仅标记状态，不断开连接
        场景：资源使用完毕，释放回资源池复用
        """
        if not self._is_active:
            logger.debug("[MilvusConnectionResource] deactivate skipped, not active")
            return

        logger.debug("[MilvusConnectionResource] deactivate: 保持连接，标记为空闲")
        self._is_active = False

    def destroy(self) -> None:
        """
        销毁资源（彻底释放）
        
        语义：资源彻底销毁，从资源池移除
        行为：断开连接，释放所有资源
        场景：资源池关闭、资源过期、资源异常需销毁
        """
        logger.info("[MilvusConnectionResource] destroy started")
        start_time = time.time()
        try:
            if self._adapter is not None:
                self._adapter.disconnect()
            self._adapter = None
            self._is_active = False
            elapsed = time.time() - start_time
            logger.info(f"[MilvusConnectionResource] destroy completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[MilvusConnectionResource] destroy failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def get_adapter(self) -> Optional[MilvusAdapterImpl]:
        return self._adapter


class MilvusConnectionConfig(ResourceConfig[Dict[str, str]]):

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        token: str = "",
        resource_name: str = "Milvus连接"
    ):
        self._resource_type = "milvus_connection"
        self._resource_name = resource_name
        self._config_protocol: Dict[str, str] = {
            "uri": uri,
            "user": user,
            "password": password,
            "token": token
        }

    @property
    def resource_type(self) -> str:
        return self._resource_type

    @property
    def resource_name(self) -> str:
        return self._resource_name

    @property
    def config_protocol(self) -> Dict[str, str]:
        return self._config_protocol

    def to_dict(self) -> dict:
        return {
            "resource_type": self._resource_type,
            "resource_name": self._resource_name,
            "config_protocol": self._config_protocol
        }

    def validate(self) -> bool:
        if not self._config_protocol.get("uri"):
            return False
        if not self._config_protocol.get("user"):
            return False
        if not self._config_protocol.get("password"):
            return False
        return True


class MilvusConnectionFactory(ResourceFactory):

    def create(self, config: ResourceConfig) -> Resource:
        if not isinstance(config, MilvusConnectionConfig):
            raise TypeError(f"Expected MilvusConnectionConfig, got {type(config)}")

        return MilvusConnectionResource(config)

    def destroy(self, resource: Resource) -> None:
        if not isinstance(resource, MilvusConnectionResource):
            raise TypeError(f"Expected MilvusConnectionResource, got {type(resource)}")

        resource.destroy()


class MilvusConnectionClient(ResourceClient):

    def __init__(self, resource: MilvusConnectionResource):
        self._resource = resource
        self._adapter: Optional[MilvusAdapter] = resource.get_adapter()

    def get_resource_type(self) -> str:
        return self._resource.get_type()

    def get_raw_resource(self) -> Resource:
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

    def get_adapter(self) -> MilvusAdapter:
        logger.debug("[MilvusConnectionClient] get_adapter called")
        return self._resource.get_adapter()
