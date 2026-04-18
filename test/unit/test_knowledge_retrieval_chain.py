import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import MagicMock, patch

from src.orchestration.chain.knowledge_retrieval_chain.knowledge_retrieval_chain import (
    KnowledgeRetrievalChain,
    KnowledgeRetrievalContextBody,
    KnowledgeRetrievalResultData,
    KnowledgeRetrievalResource,
)
from src.orchestration.chain.data_classes import ChainContext, ChainResult


class TestKnowledgeRetrievalChain:

    def _make_vector_handler(self, results=None):
        handler = MagicMock()
        if results is None:
            results = {
                "results": [
                    {"id": "v1", "name": "糖尿病", "score": 0.9, "collection": "medical_entity", "entity_type": "Disease"},
                    {"id": "v2", "name": "高血压", "score": 0.8, "collection": "medical_entity", "entity_type": "Disease"},
                ]
            }
        handler.call_tool.return_value = results
        return handler

    def _make_neo4j_handler(self):
        handler = MagicMock()
        handler.get_disease_info.return_value = {
            "id": "disease_001", "name": "糖尿病", "description": "慢性代谢性疾病"
        }
        handler.get_symptoms_by_disease.return_value = [
            {"name": "多饮"}, {"name": "多尿"}
        ]
        handler.get_drugs_by_disease.return_value = [
            {"name": "二甲双胍"}
        ]
        handler.get_foods_by_disease.return_value = [
            {"name": "苦瓜"}
        ]
        return handler

    def test_execute_vector_and_graph(self):
        vector_handler = self._make_vector_handler()
        neo4j_handler = self._make_neo4j_handler()
        resource = KnowledgeRetrievalResource(
            vector_handler=vector_handler,
            neo4j_handler=neo4j_handler,
            vector_encode_service=MagicMock(),
        )
        chain = KnowledgeRetrievalChain(resource)
        context = ChainContext(
            session_id="test_session",
            body=KnowledgeRetrievalContextBody(
                query_text="糖尿病有什么症状？",
                extracted_entities=[{"entity_name": "糖尿病", "entity_type": "Disease"}],
            ),
        )
        result = chain.execute(context)
        assert len(result.data.vector_results) > 0
        assert len(result.data.knowledge_results) > 0
        assert len(result.data.merged_results) > 0

    def test_execute_vector_only(self):
        vector_handler = self._make_vector_handler()
        neo4j_handler = MagicMock()
        neo4j_handler.get_disease_info.side_effect = Exception("Neo4j connection failed")
        neo4j_handler.get_symptoms_by_disease.side_effect = Exception("Neo4j connection failed")
        neo4j_handler.get_drugs_by_disease.side_effect = Exception("Neo4j connection failed")
        neo4j_handler.get_foods_by_disease.side_effect = Exception("Neo4j connection failed")
        resource = KnowledgeRetrievalResource(
            vector_handler=vector_handler,
            neo4j_handler=neo4j_handler,
            vector_encode_service=MagicMock(),
        )
        chain = KnowledgeRetrievalChain(resource)
        context = ChainContext(
            session_id="test_session",
            body=KnowledgeRetrievalContextBody(
                query_text="糖尿病有什么症状？",
                extracted_entities=[{"entity_name": "糖尿病", "entity_type": "Disease"}],
            ),
        )
        result = chain.execute(context)
        assert len(result.data.vector_results) > 0
        assert len(result.data.merged_results) > 0

    def test_execute_graph_only(self):
        vector_handler = MagicMock()
        vector_handler.call_tool.side_effect = Exception("Milvus unavailable")
        neo4j_handler = self._make_neo4j_handler()
        neo4j_handler.search_diseases_by_symptom.return_value = [
            {"name": "糖尿病", "type": "Disease"}
        ]
        resource = KnowledgeRetrievalResource(
            vector_handler=vector_handler,
            neo4j_handler=neo4j_handler,
            vector_encode_service=MagicMock(),
        )
        chain = KnowledgeRetrievalChain(resource)
        context = ChainContext(
            session_id="test_session",
            body=KnowledgeRetrievalContextBody(
                query_text="糖尿病有什么症状？",
                extracted_entities=[],
            ),
        )
        result = chain.execute(context)
        assert len(result.data.vector_results) == 0

    def test_integrate_knowledge_dedup(self):
        resource = KnowledgeRetrievalResource(
            vector_handler=MagicMock(),
            neo4j_handler=MagicMock(),
            vector_encode_service=MagicMock(),
        )
        chain = KnowledgeRetrievalChain(resource)
        vector_results = [
            {"id": "dup1", "name": "糖尿病", "score": 0.9},
            {"id": "dup1", "name": "糖尿病", "score": 0.8},
        ]
        knowledge_results = [
            {"source": "neo4j", "type": "disease_info", "entity": "糖尿病", "data": {"name": "糖尿病"}, "score": 0.7},
            {"source": "neo4j", "type": "disease_info", "entity": "糖尿病", "data": {"name": "糖尿病"}, "score": 0.6},
        ]
        merged = chain._integrate_knowledge(vector_results, knowledge_results)
        ids = [item.get("data", {}).get("id", item.get("entity", "")) for item in merged]
        assert len(merged) < len(vector_results) + len(knowledge_results)

    def test_integrate_knowledge_sort(self):
        resource = KnowledgeRetrievalResource(
            vector_handler=MagicMock(),
            neo4j_handler=MagicMock(),
            vector_encode_service=MagicMock(),
        )
        chain = KnowledgeRetrievalChain(resource)
        vector_results = [
            {"id": "low", "name": "低分", "score": 0.3},
            {"id": "high", "name": "高分", "score": 0.9},
        ]
        knowledge_results = [
            {"source": "neo4j", "type": "disease_info", "entity": "中", "data": {"name": "中"}, "score": 0.6},
        ]
        merged = chain._integrate_knowledge(vector_results, knowledge_results)
        scores = [item.get("score", 0.0) for item in merged]
        assert scores == sorted(scores, reverse=True)

    def test_execute_empty_query(self):
        vector_handler = self._make_vector_handler(results={"results": []})
        neo4j_handler = self._make_neo4j_handler()
        resource = KnowledgeRetrievalResource(
            vector_handler=vector_handler,
            neo4j_handler=neo4j_handler,
            vector_encode_service=MagicMock(),
        )
        chain = KnowledgeRetrievalChain(resource)
        context = ChainContext(
            session_id="test_session",
            body=KnowledgeRetrievalContextBody(query_text=""),
        )
        result = chain.execute(context)
        assert result.data is not None
