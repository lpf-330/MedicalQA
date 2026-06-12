# -*- coding: utf-8 -*-
"""
Transformers适配器实现类

转接适配Transformers库，为项目各层级提供统一的模型操作接口。
"""

import logging
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, BertConfig, BertModel, pipeline
from sentence_transformers import SentenceTransformer

from .transformers_adapter import TransformersAdapter
from src.utils.logger import log_arch_event, truncate_for_log

logger = logging.getLogger(__name__)


class CRF(nn.Module):
    """条件随机场解码层，用于NER序列标注的标签转移约束"""

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

    def decode(self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Viterbi解码，寻找全局最优标签序列"""
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2]).bool()
        return self._viterbi_decode(emissions, mask)

    def _viterbi_decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        history: list = []

        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)
            history.append(indices)

        score = score + self.end_transitions
        _, best_last_tag = score.max(dim=1)
        best_tags = [best_last_tag.unsqueeze(1)]

        for hist in reversed(history):
            best_last_tag = hist.gather(1, best_last_tag.unsqueeze(1)).squeeze(1)
            best_tags.insert(0, best_last_tag.unsqueeze(1))

        return torch.cat(best_tags, dim=1)


class RANERModel(nn.Module):
    """RANER模型：Transformer编码器 + 线性投影 + CRF解码

    用于加载包含CRF层的NER模型（如iic/nlp_raner系列），
    其权重键名为 encoder.*/linear.*/crf.* 而非标准 bert.*/classifier.*。
    """

    def __init__(self, bert_model: BertModel, num_labels: int):
        super().__init__()
        self.encoder = bert_model
        self.linear = nn.Linear(bert_model.config.hidden_size, num_labels)
        self.crf = CRF(num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> tuple:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emissions = self.linear(outputs.last_hidden_state)
        mask = attention_mask.bool() if attention_mask is not None else None
        tag_ids = self.crf.decode(emissions, mask)
        return tag_ids, emissions


def _is_raner_model(model_path: str) -> bool:
    """检测模型是否为RANER架构（包含CRF层）"""
    import os
    import json

    config_path = os.path.join(model_path, "configuration.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            model_type = cfg.get("model", {}).get("type", "")
            if "crf" in model_type.lower():
                return True
        except Exception as e:
            logger.debug(f"[TransformersAdapter] 读取模型配置文件失败: {e}")

    ckpt_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            has_crf = any(k.startswith("crf.") for k in ckpt.keys())
            has_encoder = any(k.startswith("encoder.") for k in ckpt.keys())
            if has_crf and has_encoder:
                return True
        except Exception as e:
            logger.debug(f"[TransformersAdapter] 加载模型权重文件失败: {e}")

    return False


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

    # 意图分类参考文本（ernie-health-zh Embedding相似度分类）
    _INTENT_REF_TEXTS = {
        "health": [
            "头痛怎么治疗",
            "感冒吃什么药",
            "高血压注意事项",
            "糖尿病预防方法",
            "胃痛怎么调理",
            "咳嗽怎么办",
            "发烧怎么退烧",
            "腰痛怎么缓解",
        ],
        "chat": [
            "今天天气真好",
            "给我讲个故事",
            "你是谁",
            "聊天吧",
            "你好呀",
            "推荐一部电影",
        ],
        "other": [
            "数学计算题",
            "物理原理是什么",
            "编程怎么写",
            "历史事件介绍",
        ],
    }

    def __init__(self):
        """初始化Transformers适配器"""
        super().__init__()
        self._model_path: Optional[str] = None
        self._device: Optional[str] = None
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._pipeline: Optional[Any] = None
        self._embedding_model: Optional[Any] = None
        self._model_type: Optional[str] = None
        self._ner_mode: Optional[str] = None
        self._id2label: Optional[Dict[str, str]] = None
        self._ref_embeddings: Optional[Dict[str, torch.Tensor]] = None
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
            if _is_raner_model(model_path):
                self._ner_mode = "raner"
                self._tokenizer = AutoTokenizer.from_pretrained(model_path)
                bert_config = BertConfig.from_pretrained(model_path)
                bert_model = BertModel(bert_config)
                num_labels = bert_config.num_labels
                raner = RANERModel(bert_model, num_labels)
                ckpt = torch.load(
                    f"{model_path}/pytorch_model.bin",
                    map_location="cpu",
                    weights_only=True,
                )
                raner.load_state_dict(ckpt, strict=False)
                raner.to(device)
                raner.eval()
                self._model = raner
                with open(f"{model_path}/config.json", "r", encoding="utf-8") as f:
                    import json as _json
                    self._id2label = _json.load(f).get("id2label", {})
                logger.info(f"[TransformersAdapter] RANER模型加载完成(CRF+Viterbi), num_labels={num_labels}")
            else:
                self._ner_mode = "pipeline"
                from transformers import AutoModelForTokenClassification
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
        elif model_type == "intent_classification":
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModel.from_pretrained(model_path)
            self._model.to(device)
            self._model.eval()
            self._init_intent_ref_embeddings()
            logger.info("[TransformersAdapter] 意图分类模型加载完成(Embedding相似度), ref_labels=%s", list(self._ref_embeddings.keys()))
        else:
            logger.error(f"[TransformersAdapter] 不支持的模型类型: model_type={model_type}")
            raise ValueError(f"Unsupported model_type: {model_type}")

        elapsed = time.time() - start_time
        self._set_initialized(True)
        log_arch_event(logger, component="TransformersAdapter", stage="ADAPTER", event="load_model", status="success", design_id="ARCH-7.6", model_type=model_type, elapsed=f"{elapsed:.2f}s")
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
        logger.debug(f"[TransformersAdapter] request: text={truncate_for_log(repr(text), 400)}, model_type={self._model_type}")
        start_time = time.time()

        if self._model_type == "classification":
            result = self._pipeline(text, **kwargs)
            prediction = {
                "label": result[0]["label"],
                "confidence": result[0]["score"]
            }
        elif self._model_type == "intent_classification":
            prediction = self._predict_intent(text)
        elif self._model_type == "ner":
            if self._ner_mode == "raner":
                prediction = self._predict_raner(text)
            else:
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
        logger.debug(f"[TransformersAdapter] response: {truncate_for_log(repr(prediction), 500)}")
        log_arch_event(logger, component="TransformersAdapter", stage="ADAPTER", event="predict", status="success", design_id="ARCH-7.6", model_type=self._model_type, elapsed=f"{elapsed:.3f}s")
        logger.info(f"[TransformersAdapter] 预测完成: model_type={self._model_type}, elapsed={elapsed:.3f}s")
        return prediction

    def _init_intent_ref_embeddings(self) -> None:
        """预计算意图分类参考文本的CLS embedding"""
        self._ref_embeddings = {}
        for label, texts in self._INTENT_REF_TEXTS.items():
            embs = []
            for t in texts:
                emb = self._get_cls_embedding(t)
                embs.append(emb)
            self._ref_embeddings[label] = torch.cat(embs, dim=0)

    def _get_cls_embedding(self, text: str) -> torch.Tensor:
        """获取文本的CLS token embedding"""
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        return outputs.last_hidden_state[:, 0, :].detach()

    def _predict_intent(self, text: str) -> Dict:
        """基于Embedding余弦相似度的意图分类"""
        query_emb = self._get_cls_embedding(text)

        scores = {}
        for label, ref_emb in self._ref_embeddings.items():
            sim = torch.nn.functional.cosine_similarity(query_emb, ref_emb, dim=-1)
            scores[label] = sim.max().item()

        best_label = max(scores, key=scores.get)
        best_confidence = scores[best_label]
        return {
            "label": best_label,
            "confidence": best_confidence,
            "all_scores": scores,
        }

    def _predict_raner(self, text: str) -> List[Dict]:
        """使用RANER模型（CRF+Viterbi）进行NER推理

        Args:
            text: 输入文本

        Returns:
            NER结果列表，格式与pipeline输出一致
        """
        device = torch.device(self._device) if self._device else torch.device("cpu")
        inputs = self._tokenizer(
            text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=512
        )
        offset_mapping = inputs.pop("offset_mapping")[0]
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            tag_ids, _ = self._model(**inputs)

        tokens = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        tags = [self._id2label.get(str(t.item()), "O") for t in tag_ids[0]]

        prediction = []
        for token, tag, (start, end) in zip(tokens, tags, offset_mapping):
            if token in ("[CLS]", "[SEP]", "[PAD]"):
                continue
            if start == 0 and end == 0:
                continue
            clean = token.replace("##", "")
            prediction.append({
                "entity": clean,
                "type": tag,
                "start": start.item(),
                "end": end.item(),
            })

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
            if self._ner_mode == "raner":
                results = [self._predict_raner(t) for t in texts]
            else:
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
        logger.debug(f"[TransformersAdapter] request: text={truncate_for_log(repr(text), 400)}")
        start_time = time.time()

        embedding = self._embedding_model.encode(text, **kwargs)
        result = embedding.tolist()

        elapsed = time.time() - start_time
        logger.debug(f"[TransformersAdapter] response: dim={len(result)}, preview={truncate_for_log(repr(result[:5]), 200)}")
        log_arch_event(logger, component="TransformersAdapter", stage="ADAPTER", event="encode", status="success", design_id="ARCH-7.6", dim=len(result), elapsed=f"{elapsed:.3f}s")
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

        if self._ref_embeddings is not None:
            del self._ref_embeddings
            self._ref_embeddings = None
            logger.debug("[TransformersAdapter] RefEmbeddings已释放")

        self._model_path = None
        self._device = None
        self._model_type = None
        self._ner_mode = None
        self._id2label = None
        self._set_initialized(False)
        log_arch_event(logger, component="TransformersAdapter", stage="ADAPTER", event="unload_model", status="success", design_id="ARCH-7.6")
        logger.info("[TransformersAdapter] 模型卸载完成")

    def is_initialized(self) -> bool:
        return self._model is not None or self._embedding_model is not None or self._ref_embeddings is not None

    def is_model_loaded(self) -> bool:
        return self._model is not None or self._embedding_model is not None or self._ref_embeddings is not None
    
    def __enter__(self) -> 'TransformersAdapterImpl':
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器退出"""
        self.unload_model()
