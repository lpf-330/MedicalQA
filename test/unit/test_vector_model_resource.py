import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from src.resource_manager.vector_model.vector_model_resource import (
    VectorModelResource,
    VectorModelConfig,
    VectorModelFactory,
    VectorModelClient,
)


class TestVectorModelResource:

    @patch('src.resource_manager.vector_model.vector_model_resource.TransformersAdapterImpl')
    def test_vector_model_resource_activate(self, MockAdapterImpl):
        mock_adapter = MagicMock()
        MockAdapterImpl.return_value = mock_adapter

        config = VectorModelConfig(
            model_path="/fake/embedding_model",
            device="cpu"
        )
        resource = VectorModelResource(config)
        resource.activate()

        MockAdapterImpl.assert_called_once()
        mock_adapter.load_model.assert_called_once_with(
            model_path="/fake/embedding_model",
            device="cpu",
            model_type="embedding"
        )
        assert resource.is_activate() is True
        assert resource.get_adapter() is mock_adapter

    def test_vector_model_resource_get_type(self):
        config = VectorModelConfig(
            model_path="/fake/embedding_model",
            device="cpu"
        )
        resource = VectorModelResource(config)
        assert resource.get_type() == "vector_model"


class TestVectorModelClient:

    @patch('src.resource_manager.vector_model.vector_model_resource.TransformersAdapterImpl')
    def test_vector_model_client_encode(self, MockAdapterImpl):
        mock_adapter = MagicMock()
        mock_adapter.encode.return_value = [0.1] * 1024
        MockAdapterImpl.return_value = mock_adapter

        config = VectorModelConfig(
            model_path="/fake/embedding_model",
            device="cpu"
        )
        resource = VectorModelResource(config)
        resource.activate()

        client = VectorModelClient(resource)
        result = client.encode(text="糖尿病有什么症状？")

        mock_adapter.encode.assert_called_once_with(text="糖尿病有什么症状？")
        assert isinstance(result, list)
        assert len(result) == 1024

    @patch('src.resource_manager.vector_model.vector_model_resource.TransformersAdapterImpl')
    def test_vector_model_client_encode_batch(self, MockAdapterImpl):
        mock_adapter = MagicMock()
        mock_adapter.encode_batch.return_value = [[0.1] * 1024, [0.2] * 1024]
        MockAdapterImpl.return_value = mock_adapter

        config = VectorModelConfig(
            model_path="/fake/embedding_model",
            device="cpu"
        )
        resource = VectorModelResource(config)
        resource.activate()

        client = VectorModelClient(resource)
        result = client.encode_batch(texts=["糖尿病有什么症状？", "高血压怎么治疗？"])

        mock_adapter.encode_batch.assert_called_once_with(texts=["糖尿病有什么症状？", "高血压怎么治疗？"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert len(result[0]) == 1024
        assert len(result[1]) == 1024
