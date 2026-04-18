import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import MagicMock, patch


class TestVectorEnhancedRetrievalTool:

    def setup_method(self):
        with patch('src.tools.vector_retrieval_tool.vector_retrieval_tool.GlobalResourceManager'):
            from src.tools.vector_retrieval_tool.vector_retrieval_tool import VectorEnhancedRetrievalTool
            self.tool = VectorEnhancedRetrievalTool(
                milvus_uri="http://localhost:19530",
                milvus_user="root",
                milvus_password="password"
            )

    def test_hybrid_search_default(self):
        mock_milvus_client = MagicMock()
        mock_vector_client = MagicMock()
        mock_vector_client.encode.return_value = [0.1] * 1024
        mock_milvus_client.hybrid_search.return_value = [
            {"id": "1", "score": 0.9, "name": "糖尿病"},
            {"id": "2", "score": 0.5, "name": "感冒"},
        ]
        self.tool._milvus_client = mock_milvus_client
        self.tool._vector_client = mock_vector_client

        result = self.tool.hybrid_search(query="糖尿病有什么症状")

        mock_vector_client.encode.assert_called_once_with(text="糖尿病有什么症状")
        mock_milvus_client.hybrid_search.assert_called_once()
        call_kwargs = mock_milvus_client.hybrid_search.call_args
        assert call_kwargs[1]["collections"] == ["medical_entity", "entity_attributes", "entity_relations"]
        assert call_kwargs[1]["weights"] == {
            "medical_entity": 0.40,
            "entity_attributes": 0.30,
            "entity_relations": 0.30
        }
        assert len(result) == 1
        assert result[0]["name"] == "糖尿病"

    def test_hybrid_search_custom(self):
        mock_milvus_client = MagicMock()
        mock_vector_client = MagicMock()
        mock_vector_client.encode.return_value = [0.1] * 1024
        mock_milvus_client.hybrid_search.return_value = [
            {"id": "1", "score": 0.9, "name": "糖尿病"},
        ]
        self.tool._milvus_client = mock_milvus_client
        self.tool._vector_client = mock_vector_client

        custom_collections = ["medical_entity"]
        custom_weights = {"medical_entity": 1.0}
        result = self.tool.hybrid_search(
            query="糖尿病",
            collections=custom_collections,
            weights=custom_weights
        )

        call_kwargs = mock_milvus_client.hybrid_search.call_args
        assert call_kwargs[1]["collections"] == custom_collections
        assert call_kwargs[1]["weights"] == custom_weights
        assert len(result) == 1

    def test_search_entities(self):
        mock_milvus_client = MagicMock()
        mock_vector_client = MagicMock()
        mock_vector_client.encode.return_value = [0.1] * 1024
        mock_milvus_client.search.return_value = [
            {"id": "1", "name": "糖尿病", "type": "Disease"}
        ]
        self.tool._milvus_client = mock_milvus_client
        self.tool._vector_client = mock_vector_client

        result = self.tool.search_entities(query="糖尿病", top_k=10)

        mock_milvus_client.search.assert_called_once_with(
            collection_name="medical_entity",
            query_vector=[0.1] * 1024,
            top_k=10
        )
        assert len(result) == 1

    def test_search_attributes(self):
        mock_milvus_client = MagicMock()
        mock_vector_client = MagicMock()
        mock_vector_client.encode.return_value = [0.1] * 1024
        mock_milvus_client.search.return_value = [
            {"id": "1", "name": "糖尿病症状", "type": "Symptom"}
        ]
        self.tool._milvus_client = mock_milvus_client
        self.tool._vector_client = mock_vector_client

        result = self.tool.search_attributes(query="糖尿病症状", top_k=10)

        mock_milvus_client.search.assert_called_once_with(
            collection_name="entity_attributes",
            query_vector=[0.1] * 1024,
            top_k=10
        )
        assert len(result) == 1

    def test_search_relations(self):
        mock_milvus_client = MagicMock()
        mock_vector_client = MagicMock()
        mock_vector_client.encode.return_value = [0.1] * 1024
        mock_milvus_client.search.return_value = [
            {"id": "1", "name": "糖尿病-二甲双胍", "type": "DrugRelation"}
        ]
        self.tool._milvus_client = mock_milvus_client
        self.tool._vector_client = mock_vector_client

        result = self.tool.search_relations(query="糖尿病药物", top_k=10)

        mock_milvus_client.search.assert_called_once_with(
            collection_name="entity_relations",
            query_vector=[0.1] * 1024,
            top_k=10
        )
        assert len(result) == 1

    def test_not_initialized(self):
        from src.tools.vector_retrieval_tool.vector_retrieval_tool import VectorEnhancedRetrievalTool
        with patch('src.tools.vector_retrieval_tool.vector_retrieval_tool.GlobalResourceManager'):
            tool = VectorEnhancedRetrievalTool(
                milvus_uri="http://localhost:19530",
                milvus_user="root",
                milvus_password="password"
            )

        with pytest.raises(RuntimeError, match="Tool not initialized"):
            tool.hybrid_search(query="test")

        with pytest.raises(RuntimeError, match="Tool not initialized"):
            tool.search_entities(query="test")

        with pytest.raises(RuntimeError, match="Tool not initialized"):
            tool.search_attributes(query="test")

        with pytest.raises(RuntimeError, match="Tool not initialized"):
            tool.search_relations(query="test")
