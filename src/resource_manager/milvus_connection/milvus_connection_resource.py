# -*- coding: utf-8 -*-

"""Milvus连接资源封装类。"""

import logging
import time
from typing import Optional

from src.resource_manager.resource import Resource
from src.resource_manager.milvus_connection.milvus_connection_config import MilvusConnectionConfig
from src.adapters.milvus.milvus_adapter import MilvusAdapter
from src.adapters.milvus.milvus_adapter_impl import MilvusAdapterImpl

logger = logging.getLogger(__name__)


class MilvusConnectionResource(Resource):

    def __init__(self, config: 'MilvusConnectionConfig'):
        self._config = config
        self._adapter: Optional[MilvusAdapter] = None
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
            if self._adapter is None:
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
            logger.info(f"[MilvusConnectionResource] activate completed, elapsed={elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[MilvusConnectionResource] activate failed, elapsed={elapsed:.3f}s, error_type={type(e).__name__}")
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
            logger.error(f"[MilvusConnectionResource] destroy failed, elapsed={elapsed:.3f}s, error_type={type(e).__name__}")
            raise

    def get_adapter(self) -> Optional[MilvusAdapter]:
        return self._adapter
