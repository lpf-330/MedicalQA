import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import MagicMock


class TestIntentClassificationHandler:

    def setup_method(self):
        from src.orchestration.tool_call_handler.Impl.intent_classification_handler import IntentClassificationHandler
        self.handler = IntentClassificationHandler()

    def test_init_tool(self):
        mock_tool = MagicMock()
        self.handler._init_tool(mock_tool)

        assert self.handler._tool is mock_tool
        mock_tool._init_tool.assert_called_once()

    def test_call_tool_classify(self):
        mock_tool = MagicMock()
        mock_tool.call.return_value = {
            "intent_label": "health_consultation",
            "confidence": 0.92
        }
        self.handler._tool = mock_tool

        context = {"method": "classify_intent", "text": "糖尿病有什么症状"}
        result = self.handler.call_tool(context)

        mock_tool.call.assert_called_once_with("classify_intent", context)
        assert result["intent_label"] == "health_consultation"

    def test_call_tool_extract(self):
        mock_tool = MagicMock()
        mock_tool.call.return_value = [
            {"entity_name": "糖尿病", "entity_type": "Disease"}
        ]
        self.handler._tool = mock_tool

        context = {"method": "extract_entities", "text": "糖尿病有什么症状"}
        result = self.handler.call_tool(context)

        mock_tool.call.assert_called_once_with("extract_entities", context)
        assert len(result) == 1

    def test_call_tool_unknown(self):
        mock_tool = MagicMock()
        self.handler._tool = mock_tool

        context = {"method": "unknown_method", "text": "test"}
        with pytest.raises(ValueError, match="Unknown method: unknown_method"):
            self.handler.call_tool(context)

    def test_release(self):
        mock_tool = MagicMock()
        self.handler._tool = mock_tool

        self.handler.release()

        mock_tool.release_tool.assert_called_once_with(None)
        assert self.handler._tool is None
