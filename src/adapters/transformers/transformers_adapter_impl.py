# -*- coding: utf-8 -*-
"""
Transformers适配器实现类

转接适配Transformers库，为项目各层级提供统一的模型操作接口。
"""

import logging
import time
from typing import Any, Dict, List, Optional

from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer

from .transformers_adapter import TransformersAdapter

logger = logging.getLogger(__name__)


class TransformersAdapterImpl(TransformersAdapter):
    """
    Transformers适配器实现类

    封装transformers和sentence-transformers库，为项目提供统一的模型操作接口。

    属性：
        _model_path: 模型路径
        _device: 运行设备
        _model: transformers模型实例
        _tokenizer: transformers分词器实例
        _pipeline: transformers pipeline实例
        _embedding_model: sentence-transformers模型实例
        _model_type: 模型类型标识（classification/ner/embedding）
    """

    def __init__(self):
        """初始化Transformers适配器"""
        self._model_path: Optional[str] = None
        self._device: Optional[str] = None
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._pipeline: Optional[Any] = None
        self._embedding_model: Optional[Any] = None
        self._model_type: Optional[str] = None
        logger.debug("[TransformersAdapter] 初始化Transformers适配器")

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
            device: 运行设备
            **kwargs: 其他参数（model_type为必传参数）
        """
        model_type = kwargs.get("model_type", "classification")
        self._model_type = model_type
        self._model_path = model_path
        self._device = device

        logger.info(f"[TransformersAdapter] 开始加载模型: model_path={model_path}, device={device}, model_type={model_type}")
        start_time = time.time()

        if model_type == "classification":
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self._model.to(device)
            self._pipeline = pipeline(
                "text-classification",
                model=self._model,
                tokenizer=self._tokenizer,
                device=0 if device == "cuda" else -1
            )
        elif model_type == "ner":
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModelForTokenClassification.from_pretrained(model_path)
            self._model.to(device)
            self._pipeline = pipeline(
                "ner",
                model=self._model,
                tokenizer=self._tokenizer,
                device=0 if device == "cuda" else -1
            )
        elif model_type == "embedding":
            self._embedding_model = SentenceTransformer(model_path, device=device)
        else:
            logger.error(f"[TransformersAdapter] 不支持的模型类型: model_type={model_type}")
            raise ValueError(f"Unsupported model_type: {model_type}")

        elapsed = time.time() - start_time
        logger.info(f"[TransformersAdapter] 模型加载完成: model_path={model_path}, model_type={model_type}, elapsed={elapsed:.2f}s")

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
            分类结果或NER结果
        """
        if not self.is_model_loaded():
            logger.error("[TransformersAdapter] 预测失败，模型未加载")
            raise RuntimeError("Model not loaded")

        logger.debug(f"[TransformersAdapter] 开始预测: text_len={len(text)}, model_type={self._model_type}")
        start_time = time.time()

        if self._model_type == "classification":
            result = self._pipeline(text, **kwargs)
            prediction = {
                "label": result[0]["label"],
                "confidence": result[0]["score"]
            }
        elif self._model_type == "ner":
            raw_results = self._pipeline(text, **kwargs)
            prediction = [
                {
                    "entity": item["word"],
                    "type": item["entity"],
                    "start": item["start"],
                    "end": item["end"]
                }
                for item in raw_results
            ]
        else:
            logger.error(f"[TransformersAdapter] 预测失败，当前模型类型不支持predict: model_type={self._model_type}")
            raise RuntimeError(f"predict not supported for model_type: {self._model_type}")

        elapsed = time.time() - start_time
        logger.info(f"[TransformersAdapter] 预测完成: model_type={self._model_type}, elapsed={elapsed:.3f}s")
        return prediction

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
        """
        if not self.is_model_loaded():
            logger.error("[TransformersAdapter] 批量预测失败，模型未加载")
            raise RuntimeError("Model not loaded")

        logger.debug(f"[TransformersAdapter] 开始批量预测: batch_size={len(texts)}, model_type={self._model_type}")
        start_time = time.time()

        if self._model_type == "classification":
            raw_results = self._pipeline(texts, **kwargs)
            results = [
                {
                    "label": item[0]["label"],
                    "confidence": item[0]["score"]
                }
                for item in raw_results
            ]
        elif self._model_type == "ner":
            raw_results = self._pipeline(texts, **kwargs)
            results = [
                [
                    {
                        "entity": ent["word"],
                        "type": ent["entity"],
                        "start": ent["start"],
                        "end": ent["end"]
                    }
                    for ent in item
                ]
                for item in raw_results
            ]
        else:
            logger.error(f"[TransformersAdapter] 批量预测失败，当前模型类型不支持predict_batch: model_type={self._model_type}")
            raise RuntimeError(f"predict_batch not supported for model_type: {self._model_type}")

        elapsed = time.time() - start_time
        logger.info(f"[TransformersAdapter] 批量预测完成: batch_size={len(results)}, elapsed={elapsed:.3f}s")
        return results

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
        """
        if self._embedding_model is None:
            logger.error("[TransformersAdapter] 编码失败，嵌入模型未加载")
            raise RuntimeError("Embedding model not loaded")

        logger.debug(f"[TransformersAdapter] 开始编码: text_len={len(text)}")
        start_time = time.time()

        embedding = self._embedding_model.encode(text, **kwargs)
        result = embedding.tolist()

        elapsed = time.time() - start_time
        logger.info(f"[TransformersAdapter] 编码完成: dim={len(result)}, elapsed={elapsed:.3f}s")
        return result

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
        """
        if self._embedding_model is None:
            logger.error("[TransformersAdapter] 批量编码失败，嵌入模型未加载")
            raise RuntimeError("Embedding model not loaded")

        logger.debug(f"[TransformersAdapter] 开始批量编码: batch_size={len(texts)}")
        start_time = time.time()

        embeddings = self._embedding_model.encode(texts, **kwargs)
        results = embeddings.tolist()

        elapsed = time.time() - start_time
        logger.info(f"[TransformersAdapter] 批量编码完成: batch_size={len(results)}, elapsed={elapsed:.3f}s")
        return results

    def unload_model(self) -> None:
        """卸载模型，释放资源"""
        logger.info(f"[TransformersAdapter] 开始卸载模型: model_path={self._model_path}, model_type={self._model_type}")

        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            logger.debug("[TransformersAdapter] Pipeline已释放")

        if self._model is not None:
            del self._model
            self._model = None
            logger.debug("[TransformersAdapter] Model已释放")

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
            logger.debug("[TransformersAdapter] Tokenizer已释放")

        if self._embedding_model is not None:
            del self._embedding_model
            self._embedding_model = None
            logger.debug("[TransformersAdapter] EmbeddingModel已释放")

        self._model_path = None
        self._device = None
        self._model_type = None
        logger.info("[TransformersAdapter] 模型卸载完成")

    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._model is not None or self._embedding_model is not None

    def __enter__(self) -> 'TransformersAdapterImpl':
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器退出"""
        self.unload_model()
