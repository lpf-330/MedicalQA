import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from src.resource_manager.intent_model.intent_model_resource import (
    IntentModelResource,
    IntentModelConfig,
    IntentModelFactory,
    IntentModelClient,
)


class TestIntentModelResource:

    @patch('src.resource_manager.intent_model.intent_model_resource.TransformersAdapterImpl')
    def test_intent_model_resource_activate(self, MockAdapterImpl):
        mock_adapter = MagicMock()
        MockAdapterImpl.return_value = mock_adapter

        config = IntentModelConfig(
            model_path="/fake/intent_model",
            device="cpu"
        )
        resource = IntentModelResource(config)
        resource.activate()

        MockAdapterImpl.assert_called_once()
        mock_adapter.load_model.assert_called_once_with(
            model_path="/fake/intent_model",
            device="cpu",
            model_type="classification"
        )
        assert resource.is_activate() is True
        assert resource.get_adapter() is mock_adapter

    def test_intent_model_resource_get_type(self):
        config = IntentModelConfig(
            model_path="/fake/intent_model",
            device="cpu"
        )
        resource = IntentModelResource(config)
        assert resource.get_type() == "intent_model"


class TestIntentModelClient:

    @patch('src.resource_manager.intent_model.intent_model_resource.TransformersAdapterImpl')
    def test_intent_model_client_classify_intent(self, MockAdapterImpl):
        mock_adapter = MagicMock()
        mock_adapter.predict.return_value = {
            "label": "health_consultation",
            "confidence": 0.92
        }
        MockAdapterImpl.return_value = mock_adapter

        config = IntentModelConfig(
            model_path="/fake/intent_model",
            device="cpu"
        )
        resource = IntentModelResource(config)
        resource.activate()

        client = IntentModelClient(resource)
        result = client.classify_intent(text="糖尿病有什么症状？")

        mock_adapter.predict.assert_called_once_with(text="糖尿病有什么症状？")
        assert result["intent_label"] == "health_consultation"
        assert result["confidence"] == 0.92

    @patch('src.resource_manager.intent_model.intent_model_resource.TransformersAdapterImpl')
    def test_intent_model_client_extract_entities(self, MockAdapterImpl):
        mock_adapter = MagicMock()
        mock_adapter.predict.return_value = {
            "label": "health_consultation",
            "confidence": 0.92
        }
        MockAdapterImpl.return_value = mock_adapter

        config = IntentModelConfig(
            model_path="/fake/intent_model",
            device="cpu"
        )
        resource = IntentModelResource(config)
        resource.activate()

        client = IntentModelClient(resource)
        result = client.extract_entities(text="糖尿病,头痛怎么办？")

        mock_adapter.predict.assert_called_once_with(text="糖尿病,头痛怎么办？")
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0]["entity_name"] == "糖尿病"
        assert result[0]["entity_type"] == "medical_term"
