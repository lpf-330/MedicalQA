# AI辅助生成：GLM-5，2026-04-15
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class MilvusAdapter(ABC):

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
        weights: Dict[str, float]
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
