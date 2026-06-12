# AI辅助生成：GLM-5，2026-04-15
# -*- coding: utf-8 -*-

from abc import abstractmethod
from typing import Dict, List

from src.adapters.base_adapter import BaseAdapter


class MilvusAdapter(BaseAdapter):

    @abstractmethod
    def is_initialized(self) -> bool:
        """
        检查适配器是否已初始化
        
        Returns:
            bool: 是否已初始化（已连接）
        """
        pass

    @abstractmethod
    def connect(self, uri: str, user: str, password: str, token: str) -> None:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int,
        **kwargs
    ) -> List[Dict]:
        pass

    @abstractmethod
    def hybrid_search(
        self,
        query_vector: List[float],
        collections: List[str],
        top_k: int,
        weights: Dict[str, float],
        threshold: float = 0.6
    ) -> List[Dict]:
        pass

    @abstractmethod
    def insert(
        self,
        collection_name: str,
        data: List[Dict]
    ) -> List[int]:
        pass

    @abstractmethod
    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        **kwargs
    ) -> None:
        pass

    @abstractmethod
    def drop_collection(self, collection_name: str) -> None:
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        pass
