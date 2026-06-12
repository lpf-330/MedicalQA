# -*- coding: utf-8 -*-
"""
NER模型工具接口

定义NerModelTool的包内抽象接口，继承外部Tool基类。
"""

from abc import abstractmethod
from typing import Dict, List

from src.tools.tool import Tool


class NerModelToolInterface(Tool):
    """
    NER模型工具接口

    继承Tool基类，声明医学实体提取的抽象方法。
    """

    @abstractmethod
    def extract_entities(self, text: str) -> List[Dict]:
        """使用NER模型从文本中提取医学实体"""
        pass
