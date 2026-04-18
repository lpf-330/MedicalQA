import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import MagicMock, patch

from src.orchestration.chain.intent_parse_chain.intent_parse_chain import (
    IntentParseChain,
    IntentParseContextBody,
    IntentParseResultData,
    IntentParseResource,
)
from src.orchestration.chain.data_classes import ChainContext, ChainResult


class TestIntentParseChain:

    def _create_chain(self, handler=None):
        if handler is None:
            handler = MagicMock()
        resource = IntentParseResource(intent_handler=handler, model_service=MagicMock())
        return IntentParseChain(resource)

    def test_execute_health_consultation(self):
        handler = MagicMock()
        handler.call_tool.side_effect = [
            {"intent_label": "health_consultation", "confidence": 0.92},
            {"entities": [{"entity_name": "糖尿病", "entity_type": "Disease"}]},
        ]
        chain = self._create_chain(handler)
        context = ChainContext(
            session_id="test_session",
            body=IntentParseContextBody(query_text="糖尿病有什么症状？"),
        )
        result = chain.execute(context)
        assert result.data.is_health_consultation is True
        assert result.data.intent_label == "health_consultation"
        assert result.data.confidence == 0.92

    def test_execute_non_health(self):
        handler = MagicMock()
        handler.call_tool.side_effect = [
            {"intent_label": "chat", "confidence": 0.88},
            {"entities": []},
        ]
        chain = self._create_chain(handler)
        context = ChainContext(
            session_id="test_session",
            body=IntentParseContextBody(query_text="今天天气怎么样？"),
        )
        result = chain.execute(context)
        assert result.data.is_health_consultation is False
        assert result.data.intent_label == "chat"

    def test_execute_low_confidence(self):
        handler = MagicMock()
        handler.call_tool.side_effect = [
            {"intent_label": "health_consultation", "confidence": 0.4},
            {"entities": []},
        ]
        chain = self._create_chain(handler)
        context = ChainContext(
            session_id="test_session",
            body=IntentParseContextBody(query_text="可能有关健康的问题"),
        )
        result = chain.execute(context)
        assert result.data.is_health_consultation is False
        assert result.data.confidence == 0.4

    def test_execute_error_handling(self):
        handler = MagicMock()
        handler.call_tool.side_effect = RuntimeError("handler error")
        chain = self._create_chain(handler)
        context = ChainContext(
            session_id="test_session",
            body=IntentParseContextBody(query_text="糖尿病有什么症状？"),
        )
        result = chain.execute(context)
        assert result.data.intent_label == "error"
        assert result.data.is_health_consultation is False
        assert result.data.confidence == 0.0

    def test_execute_extracts_entities(self):
        handler = MagicMock()
        handler.call_tool.side_effect = [
            {"intent_label": "health_consultation", "confidence": 0.9},
            {"entities": [
                {"entity_name": "糖尿病", "entity_type": "Disease"},
                {"entity_name": "头痛", "entity_type": "Symptom"},
            ]},
        ]
        chain = self._create_chain(handler)
        context = ChainContext(
            session_id="test_session",
            body=IntentParseContextBody(query_text="糖尿病有什么症状？"),
        )
        result = chain.execute(context)
        assert len(result.data.extracted_entities) == 2
        assert result.data.extracted_entities[0]["entity_name"] == "糖尿病"
        assert result.data.extracted_entities[1]["entity_name"] == "头痛"

    def test_execute_rewrites_query(self):
        handler = MagicMock()
        handler.call_tool.side_effect = [
            {"intent_label": "health_consultation", "confidence": 0.9},
            {"entities": [
                {"entity_name": "糖尿病", "entity_type": "Disease"},
                {"entity_name": "头痛", "entity_type": "Symptom"},
            ]},
        ]
        chain = self._create_chain(handler)
        context = ChainContext(
            session_id="test_session",
            body=IntentParseContextBody(query_text="糖尿病有什么症状？"),
        )
        result = chain.execute(context)
        assert "糖尿病" in result.data.rewritten_query
        assert "头痛" in result.data.rewritten_query
