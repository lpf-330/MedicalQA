import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import MagicMock, patch


class TestConsultModelService:

    def setup_method(self):
        from src.orchestration.model_business_service.Impl.consult_model_service import ConsultModelService
        self.service = ConsultModelService(
            model_path="/fake/model/path",
            system_prompt="你是一个专业的健康咨询助手"
        )

    @patch('src.orchestration.model_business_service.Impl.consult_model_service.GlobalResourceManager')
    @patch('src.orchestration.model_business_service.Impl.consult_model_service.VLLMModelClient')
    def test_init_model(self, MockVLLMClient, MockGRM):
        mock_handle = MagicMock()
        MockGRM.acquire.return_value = mock_handle
        mock_handle.resource = MagicMock()

        self.service._init_model()

        MockGRM.acquire.assert_called_once_with("vllm_model")
        MockVLLMClient.assert_called_once_with(mock_handle.resource)
        assert self.service._model_handle is mock_handle

    @patch('src.orchestration.model_business_service.Impl.consult_model_service.GlobalResourceManager')
    @patch('src.orchestration.model_business_service.Impl.consult_model_service.VLLMModelClient')
    def test_call_model(self, MockVLLMClient, MockGRM):
        mock_handle = MagicMock()
        MockGRM.acquire.return_value = mock_handle
        mock_client = MagicMock()
        mock_client.generate.return_value = "糖尿病是一种慢性代谢性疾病"
        MockVLLMClient.return_value = mock_client

        self.service._init_model()

        messages = [
            {"role": "user", "content": "糖尿病有什么症状"}
        ]
        result = self.service.call_model(messages)

        mock_client.generate.assert_called_once()
        assert result == "糖尿病是一种慢性代谢性疾病"

    @patch('src.orchestration.model_business_service.Impl.consult_model_service.GlobalResourceManager')
    @patch('src.orchestration.model_business_service.Impl.consult_model_service.VLLMModelClient')
    def test_generate_with_context(self, MockVLLMClient, MockGRM):
        mock_handle = MagicMock()
        MockGRM.acquire.return_value = mock_handle
        mock_client = MagicMock()
        mock_client.generate.return_value = "糖尿病的主要症状包括多饮多尿"
        MockVLLMClient.return_value = mock_client

        self.service._init_model()

        result = self.service.generate_with_context(
            user_query="糖尿病有什么症状",
            knowledge_context="糖尿病是一种慢性代谢性疾病"
        )

        mock_client.generate.assert_called_once()
        assert "糖尿病" in result

    @patch('src.orchestration.model_business_service.Impl.consult_model_service.GlobalResourceManager')
    @patch('src.orchestration.model_business_service.Impl.consult_model_service.VLLMModelClient')
    def test_stream_generate_with_context(self, MockVLLMClient, MockGRM):
        mock_handle = MagicMock()
        MockGRM.acquire.return_value = mock_handle
        mock_client = MagicMock()
        mock_client.stream_generate.return_value = iter(["糖", "尿", "病"])
        MockVLLMClient.return_value = mock_client

        self.service._init_model()

        result = self.service.stream_generate_with_context(
            user_query="糖尿病有什么症状",
            knowledge_context="糖尿病是一种慢性代谢性疾病"
        )

        assert hasattr(result, '__iter__')
        tokens = list(result)
        assert tokens == ["糖", "尿", "病"]

    @patch('src.orchestration.model_business_service.Impl.consult_model_service.GlobalResourceManager')
    def test_release(self, MockGRM):
        mock_handle = MagicMock()
        self.service._model_handle = mock_handle
        self.service._model_client = MagicMock()

        self.service.release()

        MockGRM.release.assert_called_once_with(mock_handle)
        assert self.service._model_handle is None
        assert self.service._model_client is None

    def test_call_model_not_initialized(self):
        with pytest.raises(RuntimeError, match="Model not initialized"):
            self.service.call_model([{"role": "user", "content": "test"}])
