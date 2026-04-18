# -*- coding: utf-8 -*-
"""
意图解析Chain策略

实现用户意图识别与实体提取的Chain策略。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.orchestration.chain.chain import Chain
from src.orchestration.chain.data_classes import ChainContext, ChainResult
from src.orchestration.tool_call_handler.Impl.intent_classification_handler import IntentClassificationHandler

logger = logging.getLogger(__name__)


@dataclass
class IntentParseContextBody:
    """
    意图解析Chain策略专属输入数据类

    Attributes:
        query_text: 当前用户查询文本
        chat_history: 对话历史
    """
    query_text: str
    chat_history: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "query_text": self.query_text,
            "chat_history": self.chat_history
        }


@dataclass
class IntentParseResultData:
    """
    意图解析Chain策略专属输出数据类

    Attributes:
        intent_label: 意图标签
        confidence: 意图识别置信度
        extracted_entities: 提取的医疗实体
        rewritten_query: 改写后的查询文本
        is_health_consultation: 是否为健康咨询意图
    """
    intent_label: str
    confidence: float
    extracted_entities: List[Dict] = field(default_factory=list)
    rewritten_query: str = ""
    is_health_consultation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "intent_label": self.intent_label,
            "confidence": self.confidence,
            "extracted_entities": self.extracted_entities,
            "rewritten_query": self.rewritten_query,
            "is_health_consultation": self.is_health_consultation
        }


@dataclass
class IntentParseResource:
    """
    意图解析Chain策略专属资源类

    Attributes:
        intent_handler: 意图分类Handler
    """
    intent_handler: Optional[IntentClassificationHandler] = None


class IntentParseChain(Chain[IntentParseContextBody, IntentParseResultData]):
    """
    意图解析Chain策略类

    实现意图识别与实体提取的固定流程：
    1. 调用意图分类Handler识别用户意图
    2. 调用意图分类Handler提取医疗实体
    3. 判断是否为健康咨询意图
    4. 改写查询文本
    """

    def __init__(self, resource: IntentParseResource):
        """
        初始化意图解析Chain策略

        Args:
            resource: Chain策略专属资源
        """
        self._resource = resource

    def execute(self, chain_context: ChainContext[IntentParseContextBody]) -> ChainResult[IntentParseResultData]:
        """
        执行Chain策略

        Args:
            chain_context: Chain输入数据容器

        Returns:
            ChainResult: Chain输出数据容器
        """
        start_time = time.time()
        logger.info(f"[IntentParseChain] 开始执行Chain: session_id={chain_context.session_id}")

        body = chain_context.body
        if body is None:
            logger.warning(f"[IntentParseChain] 输入数据为空: session_id={chain_context.session_id}")
            return ChainResult(
                session_id=chain_context.session_id,
                data=IntentParseResultData(
                    intent_label="error",
                    confidence=0.0,
                    is_health_consultation=False
                )
            )

        try:
            query_text = body.query_text
            logger.info(f"[IntentParseChain] 开始意图分类: query={query_text[:50]}...")

            classify_result = self._resource.intent_handler.call_tool(
                {"method": "classify_intent", "text": query_text}
            )
            intent_label = classify_result.get("intent_label", "chat")
            confidence = classify_result.get("confidence", 0.0)
            logger.info(f"[IntentParseChain] 意图分类完成: intent_label={intent_label}, confidence={confidence}")

            logger.info(f"[IntentParseChain] 开始实体提取: query={query_text[:50]}...")
            extract_result = self._resource.intent_handler.call_tool(
                {"method": "extract_entities", "text": query_text}
            )
            if isinstance(extract_result, list):
                extracted_entities = extract_result
            else:
                extracted_entities = extract_result.get("entities", [])
            logger.info(f"[IntentParseChain] 实体提取完成: entity_count={len(extracted_entities)}")

            is_health_consultation = intent_label == "health_consultation" and confidence >= 0.5
            logger.info(f"[IntentParseChain] 健康咨询判断: is_health_consultation={is_health_consultation}")

            if extracted_entities:
                entity_names = [e.get("entity_name", "") for e in extracted_entities if e.get("entity_name")]
                rewritten_query = " ".join(entity_names) if entity_names else query_text
            else:
                rewritten_query = query_text
            logger.info(f"[IntentParseChain] 查询改写: rewritten_query={rewritten_query[:50]}...")

            result_data = IntentParseResultData(
                intent_label=intent_label,
                confidence=confidence,
                extracted_entities=extracted_entities,
                rewritten_query=rewritten_query,
                is_health_consultation=is_health_consultation
            )

            elapsed = time.time() - start_time
            logger.info(f"[IntentParseChain] Chain执行完成: session_id={chain_context.session_id}, "
                       f"intent_label={intent_label}, is_health_consultation={is_health_consultation}, "
                       f"elapsed={elapsed:.2f}s")

            return ChainResult(session_id=chain_context.session_id, data=result_data)

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[IntentParseChain] Chain执行异常: session_id={chain_context.session_id}, "
                        f"error={str(e)}, elapsed={elapsed:.2f}s")
            return ChainResult(
                session_id=chain_context.session_id,
                data=IntentParseResultData(
                    intent_label="error",
                    confidence=0.0,
                    is_health_consultation=False,
                    rewritten_query=body.query_text if body else "",
                    extracted_entities=[]
                )
            )
