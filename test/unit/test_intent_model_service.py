import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import MagicMock, patch


class TestIntentModelService:

    def setup_method(self):
        from src.orchestration.model_business_service.Impl.intent_model_service import IntentModelService
        self.service = IntentModelService()

    @patch('src.orchestration.model_business_service.Impl.intent_model_service.GlobalResourceManager')
    @patch('src.orchestration.model_business_service.Impl.intent_model_service.IntentModelClient')
    def test_init_model(self, MockIntentClient, MockGRM):
        mock_handle = MagicMock()
        MockGRM.acquire.return_value = mock_handle
        mock_handle.resource = MagicMock()

        self.service._init_model()

        MockGRM.acquire.assert_called_once_with("intent_model")
        MockIntentClient.assert_called_once_with(mock_handle.resource)
        assert self.service._model_handle is mock_handle

    @patch('src.orchestration.model_business_service.Impl.intent_model_service.GlobalResourceManager')
    @patch('src.orchestration.model_business_service.Impl.intent_model_service.IntentModelClient')
    def test_call_model_classify(self, MockIntentClient, MockGRM):
        mock_handle = MagicMock()
        MockGRM.acquire.return_value = mock_handle
        mock_client = MagicMock()
        mock_client.classify_intent.return_value = {
            "intent_label": "health_consultation",
            "confidence": 0.92
        }
        MockIntentClient.return_value = mock_client

        self.service._init_model()

        messages = {"method": "classify_intent", "text": "糖尿病有什么症状"}
        result = self.service.call_model(messages)

        mock_client.classify_intent.assert_called_once_with("糖尿病有什么症状")
        assert result["intent_label"] == "health_consultation"

    @patch('src.orchestration.model_business_service.Impl.intent_model_service.GlobalResourceManager')
    @patch('src.orchestration.model_business_service.Impl.intent_model_service.IntentModelClient')
    def test_call_model_extract(self, MockIntentClient, MockGRM):
        mock_handle = MagicMock()
        MockGRM.acquire.return_value = mock_handle
        mock_client = MagicMock()
        mock_client.extract_entities.return_value = [
            {"entity_name": "糖尿病", "entity_type": "Disease"}
        ]
        MockIntentClient.return_value = mock_client

        self.service._init_model()

        messages = {"method": "extract_entities", "text": "糖尿病有什么症状"}
        result = self.service.call_model(messages)

        mock_client.extract_entities.assert_called_once_with("糖尿病有什么症状")
        assert len(result) == 1

    @patch('src.orchestration.model_business_service.Impl.intent_model_service.GlobalResourceManager')
    @patch('src.orchestration.model_business_service.Impl.intent_model_service.IntentModelClient')
    def test_call_model_unknown(self, MockIntentClient, MockGRM):
        mock_handle = MagicMock()
        MockGRM.acquire.return_value = mock_handle
        mock_client = MagicMock()
        MockIntentClient.return_value = mock_client

        self.service._init_model()

        messages = {"method": "unknown_method", "text": "test"}
        with pytest.raises(ValueError, match="Unknown method: unknown_method"):
            self.service.call_model(messages)

    @patch('src.orchestration.model_business_service.Impl.intent_model_service.GlobalResourceManager')
    def test_release(self, MockGRM):
        mock_handle = MagicMock()
        self.service._model_handle = mock_handle
        self.service._model_client = MagicMock()

        self.service.release()

        MockGRM.release.assert_called_once_with(mock_handle)
        assert self.service._model_handle is None
        assert self.service._model_client is None
