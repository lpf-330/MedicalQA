import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import MagicMock, patch


class TestMilvusMedicalProxy:

    def setup_method(self):
        with patch('src.mcp.proxy.Impl.milvus_medical_proxy.VectorEnhancedRetrievalTool'):
            from src.mcp.proxy.Impl.milvus_medical_proxy import MilvusMedicalProxy
            self.config = {
                "milvus_uri": "http://localhost:19530",
                "milvus_user": "root",
                "milvus_password": "password",
                "milvus_token": "",
                "vector_model_path": "/fake/model",
                "vector_device": "cpu",
                "vector_dimension": 1024,
                "fusion_threshold": 0.6,
                "entity_weight": 0.40,
                "attribute_weight": 0.30,
                "relation_weight": 0.30
            }
            self.proxy = MilvusMedicalProxy(self.config)

    def test_init(self):
        tool_info = self.proxy.get_tool_info()
        assert tool_info.name == "milvus_medical"
        assert tool_info.description == "Milvus向量检索工具"
        method_names = [m.name for m in tool_info.methods]
        assert "hybrid_search" in method_names
        assert "search_entities" in method_names
        assert "search_attributes" in method_names
        assert "search_relations" in method_names

    def test_call_hybrid_search(self):
        mock_tool = MagicMock()
        mock_tool.hybrid_search.return_value = [
            {"id": "1", "score": 0.9, "name": "糖尿病"}
        ]
        self.proxy._tool = mock_tool

        params = {"query": "糖尿病有什么症状", "top_k": 20}
        result = self.proxy.call("hybrid_search", params)

        mock_tool.hybrid_search.assert_called_once_with(**params)
        assert len(result) == 1

    def test_call_search_entities(self):
        mock_tool = MagicMock()
        mock_tool.search_entities.return_value = [
            {"id": "1", "name": "糖尿病", "type": "Disease"}
        ]
        self.proxy._tool = mock_tool

        params = {"query": "糖尿病", "top_k": 10}
        result = self.proxy.call("search_entities", params)

        mock_tool.search_entities.assert_called_once_with(**params)
        assert len(result) == 1

    def test_get_tool_info(self):
        from src.mcp.proxy.data_classes import ToolInfo
        tool_info = self.proxy.get_tool_info()
        assert isinstance(tool_info, ToolInfo)
        assert tool_info.name == "milvus_medical"
        assert len(tool_info.methods) == 4
        method_names = [m.name for m in tool_info.methods]
        assert "hybrid_search" in method_names
        assert "search_entities" in method_names
        assert "search_attributes" in method_names
        assert "search_relations" in method_names

    def test_is_available_false(self):
        assert self.proxy.is_available() is False

    def test_call_unknown_method(self):
        mock_tool = MagicMock()
        self.proxy._tool = mock_tool

        with pytest.raises(AttributeError, match="Method unknown_method not found"):
            self.proxy.call("unknown_method", {})
