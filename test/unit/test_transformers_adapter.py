import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np

from src.adapters.transformers.transformers_adapter import TransformersAdapter
from src.adapters.transformers.transformers_adapter_impl import TransformersAdapterImpl


class TestTransformersAdapter:

    def test_transformers_adapter_is_abstract(self):
        with pytest.raises(TypeError):
            TransformersAdapter()


class TestTransformersAdapterImpl:

    @patch('src.adapters.transformers.transformers_adapter_impl.SentenceTransformer')
    @patch('src.adapters.transformers.transformers_adapter_impl.pipeline')
    @patch('src.adapters.transformers.transformers_adapter_impl.AutoModelForSequenceClassification')
    @patch('src.adapters.transformers.transformers_adapter_impl.AutoTokenizer')
    def test_load_classification_model(self, MockTokenizer, MockModel, MockPipeline, MockSentenceTransformer):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        MockTokenizer.from_pretrained.return_value = mock_tokenizer
        MockModel.from_pretrained.return_value = mock_model
        MockPipeline.return_value = MagicMock()

        adapter = TransformersAdapterImpl()
        adapter.load_model(model_path="/fake/model", device="cpu", model_type="classification")

        MockTokenizer.from_pretrained.assert_called_once_with("/fake/model")
        MockModel.from_pretrained.assert_called_once_with("/fake/model")
        mock_model.to.assert_called_once_with("cpu")
        MockPipeline.assert_called_once()
        assert adapter._model_type == "classification"
        assert adapter.is_model_loaded() is True

    @patch('src.adapters.transformers.transformers_adapter_impl.SentenceTransformer')
    def test_load_embedding_model(self, MockSentenceTransformer):
        mock_st = MagicMock()
        MockSentenceTransformer.return_value = mock_st

        adapter = TransformersAdapterImpl()
        adapter.load_model(model_path="/fake/embedding", device="cpu", model_type="embedding")

        MockSentenceTransformer.assert_called_once_with("/fake/embedding", device="cpu")
        assert adapter._embedding_model is mock_st
        assert adapter._model_type == "embedding"
        assert adapter.is_model_loaded() is True

    @patch('src.adapters.transformers.transformers_adapter_impl.SentenceTransformer')
    @patch('src.adapters.transformers.transformers_adapter_impl.pipeline')
    @patch('src.adapters.transformers.transformers_adapter_impl.AutoModelForSequenceClassification')
    @patch('src.adapters.transformers.transformers_adapter_impl.AutoTokenizer')
    def test_predict_classification(self, MockTokenizer, MockModel, MockPipeline, MockSentenceTransformer):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        MockTokenizer.from_pretrained.return_value = mock_tokenizer
        MockModel.from_pretrained.return_value = mock_model

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = [{"label": "health_consultation", "score": 0.92}]
        MockPipeline.return_value = mock_pipeline_instance

        adapter = TransformersAdapterImpl()
        adapter.load_model(model_path="/fake/model", device="cpu", model_type="classification")

        result = adapter.predict(text="糖尿病有什么症状？")

        assert isinstance(result, dict)
        assert "label" in result
        assert "confidence" in result
        assert result["label"] == "health_consultation"
        assert result["confidence"] == 0.92

    @patch('src.adapters.transformers.transformers_adapter_impl.SentenceTransformer')
    def test_encode(self, MockSentenceTransformer):
        mock_st = MagicMock()
        fake_embedding = np.array([0.1] * 1024, dtype=np.float32)
        mock_st.encode.return_value = fake_embedding
        MockSentenceTransformer.return_value = mock_st

        adapter = TransformersAdapterImpl()
        adapter.load_model(model_path="/fake/embedding", device="cpu", model_type="embedding")

        result = adapter.encode(text="糖尿病有什么症状？")

        assert isinstance(result, list)
        assert len(result) == 1024
        assert all(isinstance(v, float) for v in result)

    @patch('src.adapters.transformers.transformers_adapter_impl.SentenceTransformer')
    def test_encode_batch(self, MockSentenceTransformer):
        mock_st = MagicMock()
        fake_embeddings = np.array([[0.1] * 1024, [0.2] * 1024], dtype=np.float32)
        mock_st.encode.return_value = fake_embeddings
        MockSentenceTransformer.return_value = mock_st

        adapter = TransformersAdapterImpl()
        adapter.load_model(model_path="/fake/embedding", device="cpu", model_type="embedding")

        result = adapter.encode_batch(texts=["糖尿病有什么症状？", "高血压怎么治疗？"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(len(vec) == 1024 for vec in result)

    @patch('src.adapters.transformers.transformers_adapter_impl.SentenceTransformer')
    @patch('src.adapters.transformers.transformers_adapter_impl.pipeline')
    @patch('src.adapters.transformers.transformers_adapter_impl.AutoModelForSequenceClassification')
    @patch('src.adapters.transformers.transformers_adapter_impl.AutoTokenizer')
    def test_unload_model(self, MockTokenizer, MockModel, MockPipeline, MockSentenceTransformer):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        MockTokenizer.from_pretrained.return_value = mock_tokenizer
        MockModel.from_pretrained.return_value = mock_model
        MockPipeline.return_value = MagicMock()

        adapter = TransformersAdapterImpl()
        adapter.load_model(model_path="/fake/model", device="cpu", model_type="classification")
        assert adapter.is_model_loaded() is True

        adapter.unload_model()

        assert adapter._model is None
        assert adapter._tokenizer is None
        assert adapter._pipeline is None
        assert adapter._embedding_model is None
        assert adapter._model_path is None
        assert adapter._device is None
        assert adapter._model_type is None
        assert adapter.is_model_loaded() is False

    @patch('src.adapters.transformers.transformers_adapter_impl.SentenceTransformer')
    def test_is_model_loaded(self, MockSentenceTransformer):
        adapter = TransformersAdapterImpl()
        assert adapter.is_model_loaded() is False

        mock_st = MagicMock()
        MockSentenceTransformer.return_value = mock_st
        adapter.load_model(model_path="/fake/embedding", device="cpu", model_type="embedding")
        assert adapter.is_model_loaded() is True

        adapter.unload_model()
        assert adapter.is_model_loaded() is False

    @patch('src.adapters.transformers.transformers_adapter_impl.SentenceTransformer')
    @patch('src.adapters.transformers.transformers_adapter_impl.pipeline')
    @patch('src.adapters.transformers.transformers_adapter_impl.AutoModelForSequenceClassification')
    @patch('src.adapters.transformers.transformers_adapter_impl.AutoTokenizer')
    def test_predict_not_loaded(self, MockTokenizer, MockModel, MockPipeline, MockSentenceTransformer):
        adapter = TransformersAdapterImpl()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.predict(text="糖尿病有什么症状？")

    @patch('src.adapters.transformers.transformers_adapter_impl.SentenceTransformer')
    def test_encode_not_loaded(self, MockSentenceTransformer):
        adapter = TransformersAdapterImpl()
        with pytest.raises(RuntimeError, match="Embedding model not loaded"):
            adapter.encode(text="糖尿病有什么症状？")
