import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import MagicMock, patch


class TestIntentClassificationTool:

    def setup_method(self):
        with patch('src.tools.intent_classification_tool.intent_classification_tool.GlobalResourceManager'):
            from src.tools.intent_classification_tool.intent_classification_tool import IntentClassificationTool
            self.tool = IntentClassificationTool(
                model_path="/fake/model/path",
                device="cpu",
                max_length=128
            )

    def test_classify_intent(self):
        mock_intent_client = MagicMock()
        mock_intent_client.classify_intent.return_value = {
            "intent_label": "health_consultation",
            "confidence": 0.92
        }
        self.tool._intent_client = mock_intent_client

        result = self.tool.classify_intent(text="糖尿病有什么症状")

        mock_intent_client.classify_intent.assert_called_once_with(text="糖尿病有什么症状")
        assert result["intent_label"] == "health_consultation"
        assert result["confidence"] == 0.92

    def test_extract_entities(self):
        mock_intent_client = MagicMock()
        mock_intent_client.extract_entities.return_value = [
            {"entity_name": "糖尿病", "entity_type": "Disease"},
            {"entity_name": "头痛", "entity_type": "Symptom"}
        ]
        self.tool._intent_client = mock_intent_client

        result = self.tool.extract_entities(text="糖尿病引起头痛怎么办")

        mock_intent_client.extract_entities.assert_called_once_with(text="糖尿病引起头痛怎么办")
        assert len(result) == 2
        assert result[0]["entity_name"] == "糖尿病"

    def test_not_initialized(self):
        from src.tools.intent_classification_tool.intent_classification_tool import IntentClassificationTool
        with patch('src.tools.intent_classification_tool.intent_classification_tool.GlobalResourceManager'):
            tool = IntentClassificationTool(
                model_path="/fake/model/path",
                device="cpu"
            )

        with pytest.raises(RuntimeError, match="Tool not initialized"):
            tool.classify_intent(text="test")

        with pytest.raises(RuntimeError, match="Tool not initialized"):
            tool.extract_entities(text="test")
