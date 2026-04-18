import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import MagicMock, patch


class TestVectorEncodeService:

    def setup_method(self):
        from src.orchestration.model_business_service.Impl.vector_encode_service import VectorEncodeService
        self.service = VectorEncodeService()

    @patch('src.orchestration.model_business_service.Impl.vector_encode_service.GlobalResourceManager')
    @patch('src.orchestration.model_business_service.Impl.vector_encode_service.VectorModelClient')
    def test_init_model(self, MockVectorClient, MockGRM):
        mock_handle = MagicMock()
        MockGRM.acquire.return_value = mock_handle
        mock_handle.resource = MagicMock()

        self.service._init_model()

        MockGRM.acquire.assert_called_once_with("vector_model")
        MockVectorClient.assert_called_once_with(mock_handle.resource)
        assert self.service._model_handle is mock_handle

    @patch('src.orchestration.model_business_service.Impl.vector_encode_service.GlobalResourceManager')
    @patch('src.orchestration.model_business_service.Impl.vector_encode_service.VectorModelClient')
    def test_call_model(self, MockVectorClient, MockGRM):
        mock_handle = MagicMock()
        MockGRM.acquire.return_value = mock_handle
        mock_client = MagicMock()
        mock_client.encode.return_value = [0.1] * 1024
        MockVectorClient.return_value = mock_client

        self.service._init_model()

        result = self.service.call_model("糖尿病有什么症状")

        mock_client.encode.assert_called_once_with("糖尿病有什么症状")
        assert len(result) == 1024

    def test_call_model_not_initialized(self):
        with pytest.raises(RuntimeError, match="Model not initialized"):
            self.service.call_model("test")

    @patch('src.orchestration.model_business_service.Impl.vector_encode_service.GlobalResourceManager')
    def test_release(self, MockGRM):
        mock_handle = MagicMock()
        self.service._model_handle = mock_handle
        self.service._model_client = MagicMock()

        self.service.release()

        MockGRM.release.assert_called_once_with(mock_handle)
        assert self.service._model_handle is None
        assert self.service._model_client is None
