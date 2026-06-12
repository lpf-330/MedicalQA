# -*- coding: utf-8 -*-
"""
向量检索工具接口

定义VectorRetrievalTool的包内抽象接口，继承外部Tool基类。
实现类必须实现此接口，不得直接实现外部Tool。
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional

from src.tools.tool import Tool


class VectorRetrievalToolInterface(Tool):
    """
    向量检索工具接口

    继承Tool基类，声明向量检索的公共方法为抽象方法。
    实现类必须实现此接口而非直接实现Tool。
    """

    @abstractmethod
    def hybrid_search(
        self,
        query: str,
        top_k: int = 20,
        collections: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """混合检索：并行查询多个集合并融合结果"""
        pass

    @abstractmethod
    def search_entities(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """实体名称检索：在medical_entity集合中检索"""
        pass

    @abstractmethod
    def search_attributes(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """实体属性检索：在entity_attributes集合中检索"""
        pass

    @abstractmethod
    def search_relations(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """实体关系检索：在entity_relations集合中检索"""
        pass
