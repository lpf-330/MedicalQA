import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import MagicMock, patch


class TestIntentClassificationProxy:

    def setup_method(self):
        with patch('src.mcp.proxy.Impl.intent_classification_proxy.IntentClassificationTool'):
            from src.mcp.proxy.Impl.intent_classification_proxy import IntentClassificationProxy
            self.config = {
                "model_path": "/fake/model/path",
                "device": "cpu",
                "max_length": 128
            }
            self.proxy = IntentClassificationProxy(self.config)

    def test_init(self):
        tool_info = self.proxy.get_tool_info()
        assert tool_info.name == "intent_classification"
        assert tool_info.description == "意图分类工具"
        method_names = [m.name for m in tool_info.methods]
        assert "classify_intent" in method_names
        assert "extract_entities" in method_names

    def test_call_classify_intent(self):
        mock_tool = MagicMock()
        mock_tool.classify_intent.return_value = {
            "intent_label": "health_consultation",
            "confidence": 0.92
        }
        self.proxy._tool = mock_tool

        params = {"text": "糖尿病有什么症状"}
        result = self.proxy.call("classify_intent", params)

        mock_tool.classify_intent.assert_called_once_with(**params)
        assert result["intent_label"] == "health_consultation"

    def test_call_extract_entities(self):
        mock_tool = MagicMock()
        mock_tool.extract_entities.return_value = [
            {"entity_name": "糖尿病", "entity_type": "Disease"}
        ]
        self.proxy._tool = mock_tool

        params = {"text": "糖尿病有什么症状"}
        result = self.proxy.call("extract_entities", params)

        mock_tool.extract_entities.assert_called_once_with(**params)
        assert len(result) == 1

    def test_get_tool_info(self):
        from src.mcp.proxy.data_classes import ToolInfo
        tool_info = self.proxy.get_tool_info()
        assert isinstance(tool_info, ToolInfo)
        assert tool_info.name == "intent_classification"
        assert len(tool_info.methods) == 2
