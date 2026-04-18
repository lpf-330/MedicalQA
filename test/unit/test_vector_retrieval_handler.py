import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import MagicMock


class TestVectorRetrievalHandler:

    def setup_method(self):
        from src.orchestration.tool_call_handler.Impl.vector_retrieval_handler import VectorRetrievalHandler
        self.handler = VectorRetrievalHandler()

    def test_init_tool(self):
        mock_tool = MagicMock()
        self.handler._init_tool(mock_tool)

        assert self.handler._tool is mock_tool
        mock_tool._init_tool.assert_called_once()

    def test_call_tool(self):
        mock_tool = MagicMock()
        mock_tool.call.return_value = [
            {"id": "1", "score": 0.9, "name": "糖尿病"}
        ]
        self.handler._tool = mock_tool

        context = {"query": "糖尿病有什么症状", "top_k": 20}
        result = self.handler.call_tool(context)

        mock_tool.call.assert_called_once_with("hybrid_search", context)
        assert len(result) == 1

    def test_call_tool_not_initialized(self):
        with pytest.raises(RuntimeError, match="Tool not initialized"):
            self.handler.call_tool({"query": "test"})

    def test_release(self):
        mock_tool = MagicMock()
        self.handler._tool = mock_tool

        self.handler.release()

        mock_tool.release_tool.assert_called_once_with(None)
        assert self.handler._tool is None
