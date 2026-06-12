# -*- coding: utf-8 -*-
"""
NER模型资源客户端类

提供NER模型资源的客户端访问接口，封装医学实体提取等业务操作。
"""

import logging
import time
from typing import Dict, List

from src.resource_manager.resource import Resource
from src.resource_manager.resource_client import ResourceClient
from src.resource_manager.ner_model.ner_model_resource import NerModelResource

logger = logging.getLogger(__name__)

# CMeEE标签到统一实体类型的映射
CMEEE_TYPE_MAP = {
    "dis": "disease",
    "sym": "symptom",
    "dru": "medication",
    "pro": "procedure",
    "equ": "examination",
    "bod": "body_part",
    "ite": "medical_item",
    "dep": "department",
    "mic": "microorganism",
    "sur": "surgery",
}


class NerModelClient(ResourceClient):

    def __init__(self, resource: NerModelResource):
        self._resource = resource

    def get_resource_type(self) -> str:
        return self._resource.get_type()

    def get_raw_resource(self) -> Resource:
        return self._resource

    def extract_entities(self, text: str) -> List[Dict]:
        """
        使用NER模型提取医学实体

        将BIO标签序列合并为完整实体，并映射为统一的entity_type。

        Args:
            text: 待提取的文本

        Returns:
            实体列表，每个实体包含 entity_name、entity_type、start、end
        """
        logger.debug(f"[NerModelClient] extract_entities called, text_length={len(text)}")
        start_time = time.time()
        try:
            adapter = self._resource.get_adapter()
            if adapter is None:
                raise RuntimeError("Transformers adapter not initialized")
            result = adapter.predict(text=text)

            raw_entities = []
            for item in result:
                raw_entities.append({
                    "word": item.get("entity", ""),
                    "tag": item.get("type", ""),
                    "start": item.get("start", 0),
                    "end": item.get("end", 0),
                    "score": item.get("score", 0.0),
                })

            merged = self._merge_bio_entities(text, raw_entities)

            elapsed = time.time() - start_time
            logger.info(f"[NerModelClient] extract_entities completed, elapsed={elapsed:.3f}s, "
                       f"raw_tokens={len(raw_entities)}, merged_entities={len(merged)}")
            return merged
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[NerModelClient] extract_entities failed, elapsed={elapsed:.3f}s, error={str(e)}")
            raise

    def _merge_bio_entities(self, text: str, raw_entities: List[Dict]) -> List[Dict]:
        """
        合并BIO标签序列为完整实体

        支持BIO/BIOES标注方案，将连续的同类型BIO token合并为单个实体。
        标签格式: B-DIS, I-DIS, E-DIS, S-DIS 等

        Args:
            text: 原始文本
            raw_entities: 字符级NER结果列表

        Returns:
            合并后的实体列表
        """
        merged = []
        current_entity = None

        for item in raw_entities:
            tag = item["tag"]
            word = item["word"]
            start = item["start"]
            end = item["end"]

            if not tag or "-" not in tag:
                if current_entity:
                    merged.append(current_entity)
                    current_entity = None
                continue

            bio_prefix, entity_sub = tag.split("-", 1)
            bio_prefix = bio_prefix.upper()
            entity_type = CMEEE_TYPE_MAP.get(entity_sub.lower(), entity_sub.lower())

            if bio_prefix in ("B", "S"):
                if current_entity:
                    merged.append(current_entity)
                current_entity = {
                    "entity_name": word,
                    "entity_type": entity_type,
                    "start": start,
                    "end": end,
                }
                if bio_prefix == "S":
                    merged.append(current_entity)
                    current_entity = None
            elif bio_prefix in ("I", "E") and current_entity:
                if entity_type == current_entity["entity_type"]:
                    current_entity["entity_name"] += word
                    current_entity["end"] = end
                    if bio_prefix == "E":
                        merged.append(current_entity)
                        current_entity = None
                else:
                    merged.append(current_entity)
                    current_entity = {
                        "entity_name": word,
                        "entity_type": entity_type,
                        "start": start,
                        "end": end,
                    }
            else:
                if current_entity:
                    merged.append(current_entity)
                    current_entity = None

        if current_entity:
            merged.append(current_entity)

        # 过滤空实体和纯标点实体
        result = []
        for e in merged:
            name = e["entity_name"].strip()
            if len(name) >= 2:
                e["entity_name"] = name
                result.append(e)

        return result
