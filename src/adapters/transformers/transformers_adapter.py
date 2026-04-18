# -*- coding: utf-8 -*-
"""
Transformers适配器接口

为项目各层级、各类提供统一的Transformers模型操作接口。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class TransformersAdapter(ABC):
    """
    Transformers适配器接口

    定义Transformers模型操作的标准接口，为项目各层级提供统一的访问方式。

    使用示例：
        adapter = TransformersAdapterImpl()
        adapter.load_model(model_path="/path/to/model", device="cuda", model_type="classification")
        result = adapter.predict(text="患者出现发热症状")
        adapter.unload_model()
    """

    @abstractmethod
    def load_model(
        self,
        model_path: str,
        device: str,
        **kwargs
    ) -> None:
        """
        加载模型

        Args:
            model_path: 模型路径
            device: 运行设备（如"cpu"、"cuda"）
            **kwargs: 其他参数（如model_type）

        Raises:
            ModelLoadError: 当模型加载失败时抛出
        """
        pass

    @abstractmethod
    def predict(
        self,
        text: str,
        **kwargs
    ) -> Dict:
        """
        单条文本预测

        Args:
            text: 输入文本
            **kwargs: 其他预测参数

        Returns:
            分类结果: {"label": str, "confidence": float}
            NER结果: [{"entity": str, "type": str, "start": int, "end": int}]

        Raises:
            RuntimeError: 当模型未加载时抛出
        """
        pass

    @abstractmethod
    def predict_batch(
        self,
        texts: List[str],
        **kwargs
    ) -> List[Dict]:
        """
        批量文本预测

        Args:
            texts: 输入文本列表
            **kwargs: 其他预测参数

        Returns:
            预测结果列表

        Raises:
            RuntimeError: 当模型未加载时抛出
        """
        pass

    @abstractmethod
    def encode(
        self,
        text: str,
        **kwargs
    ) -> List[float]:
        """
        单条文本编码为向量

        Args:
            text: 输入文本
            **kwargs: 其他编码参数

        Returns:
            文本向量（浮点数列表）

        Raises:
            RuntimeError: 当模型未加载时抛出
        """
        pass

    @abstractmethod
    def encode_batch(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """
        批量文本编码为向量

        Args:
            texts: 输入文本列表
            **kwargs: 其他编码参数

        Returns:
            文本向量列表

        Raises:
            RuntimeError: 当模型未加载时抛出
        """
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """
        卸载模型，释放资源
        """
        pass

    @abstractmethod
    def is_model_loaded(self) -> bool:
        """
        检查模型是否已加载

        Returns:
            bool: 模型是否已加载
        """
        pass
