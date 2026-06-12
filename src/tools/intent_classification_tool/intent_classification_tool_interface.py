# -*- coding: utf-8 -*-
"""
意图分类工具接口

定义IntentClassificationTool的包内抽象接口，继承外部Tool基类。
实现类必须实现此接口，不得直接实现外部Tool。
"""

from abc import abstractmethod
from typing import Any, Dict

from src.tools.tool import Tool


class IntentClassificationToolInterface(Tool):
    """
    意图分类工具接口

    继承Tool基类，声明意图分类的公共方法为抽象方法。
    实体提取由NerModelTool（nlp_raner模型）负责，不在此接口定义。
    """

    @abstractmethod
    def classify_intent(self, text: str) -> Dict[str, Any]:
        """对用户输入进行意图分类"""
        pass
