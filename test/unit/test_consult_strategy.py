import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import MagicMock, patch

from src.orchestration.agent.consult_strategy.consult_strategy import (
    ConsultStrategy,
    ConsultContextBody,
    ConsultResultData,
)
from src.orchestration.agent.data_classes import AgentContext, AgentResult
from src.orchestration.agent.agent_resource import AgentResource
from src.orchestration.chain.data_classes import ChainContext, ChainResult
from src.orchestration.chain.intent_parse_chain.intent_parse_chain import IntentParseResultData
from src.orchestration.chain.knowledge_retrieval_chain.knowledge_retrieval_chain import KnowledgeRetrievalResultData
from src.orchestration.chain.answer_generation_chain.answer_generation_chain import AnswerGenerationResultData


class TestConsultStrategy:

    def _make_intent_chain_mock(self, is_health=True, confidence=0.9):
        chain = MagicMock()
        intent_label = "health_consultation" if is_health else "chat"
        chain.execute.return_value = ChainResult(
            session_id="test_session",
            data=IntentParseResultData(
                intent_label=intent_label,
                confidence=confidence,
                extracted_entities=[{"entity_name": "糖尿病", "entity_type": "Disease"}] if is_health else [],
                rewritten_query="糖尿病" if is_health else "test",
                is_health_consultation=is_health and confidence >= 0.6,
            ),
        )
        return chain

    def _make_knowledge_chain_mock(self):
        chain = MagicMock()
        chain.execute.return_value = ChainResult(
            session_id="test_session",
            data=KnowledgeRetrievalResultData(
                vector_results=[{"id": "v1", "score": 0.9}],
                knowledge_results=[{"source": "neo4j", "type": "disease_info", "entity": "糖尿病", "data": {"name": "糖尿病"}, "score": 0.8}],
                merged_results=[
                    {"source": "vector", "data": {"id": "v1", "score": 0.9}, "score": 0.9},
                    {"source": "neo4j", "type": "disease_info", "entity": "糖尿病", "data": {"name": "糖尿病"}, "score": 0.8},
                ],
                anchored_entities=[{"name": "糖尿病", "entity_type": "Disease"}],
                anchored_relations=[],
            ),
        )
        return chain

    def _make_answer_chain_mock(self):
        chain = MagicMock()
        chain.execute_stream.return_value = iter(["糖", "尿", "病", "是"])
        return chain

    def _make_resource(self, intent_chain=None, knowledge_chain=None, answer_chain=None):
        resource = AgentResource()
        if intent_chain is None:
            intent_chain = self._make_intent_chain_mock()
        if knowledge_chain is None:
            knowledge_chain = self._make_knowledge_chain_mock()
        if answer_chain is None:
            answer_chain = self._make_answer_chain_mock()
        resource.register_chain("intent_parse_chain", intent_chain)
        resource.register_chain("knowledge_retrieval_chain", knowledge_chain)
        resource.register_chain("answer_generation_chain", answer_chain)
        return resource

    def test_fsm_initial_state(self):
        body = ConsultContextBody(question="糖尿病有什么症状？", session_id="test_session")
        assert body.current_state == "INITIAL"

    def test_initial_to_query_parse(self):
        strategy = ConsultStrategy()
        body = ConsultContextBody(question="糖尿病有什么症状？", session_id="test_session")
        next_state = strategy._handle_initial(body, self._make_resource())
        assert next_state == "QUERY_PARSE"

    def test_query_parse_to_knowledge_retrieval(self):
        strategy = ConsultStrategy()
        body = ConsultContextBody(question="糖尿病有什么症状？", session_id="test_session")
        resource = self._make_resource(intent_chain=self._make_intent_chain_mock(is_health=True))
        next_state = strategy._handle_query_parse(body, resource)
        assert next_state == "KNOWLEDGE_RETRIEVAL"
        assert body.is_health_consultation is True

    def test_query_parse_to_finished(self):
        strategy = ConsultStrategy()
        body = ConsultContextBody(question="今天天气怎么样？", session_id="test_session")
        resource = self._make_resource(intent_chain=self._make_intent_chain_mock(is_health=False))
        next_state = strategy._handle_query_parse(body, resource)
        assert next_state == "FINISHED"
        assert body.is_health_consultation is False

    def test_knowledge_retrieval_to_integration(self):
        strategy = ConsultStrategy()
        body = ConsultContextBody(
            question="糖尿病有什么症状？",
            session_id="test_session",
            rewritten_query="糖尿病",
            extracted_entities=[{"entity_name": "糖尿病", "entity_type": "Disease"}],
        )
        resource = self._make_resource()
        next_state = strategy._handle_knowledge_retrieval(body, resource)
        assert next_state == "KNOWLEDGE_INTEGRATION"

    def test_answer_generation_to_finished(self):
        strategy = ConsultStrategy()
        body = ConsultContextBody(
            question="糖尿病有什么症状？",
            session_id="test_session",
            knowledge_context="糖尿病相关知识",
        )
        resource = self._make_resource()
        next_state = strategy._handle_answer_generation(body, resource)
        assert next_state == "FINISHED"
        assert body.stream_generator is not None
        assert body.is_streaming is True

    def test_error_handling(self):
        strategy = ConsultStrategy()
        body = ConsultContextBody(question="测试", session_id="test_session")
        next_state = strategy._handle_error(body, RuntimeError("Milvus connection failed"))
        assert next_state == "ERROR"
        assert body.error_code == 1001

    def test_context_body_has_stream_generator(self):
        body = ConsultContextBody(question="测试", session_id="test_session")
        assert hasattr(body, "stream_generator")
        assert body.stream_generator is None

    def test_build_result(self):
        strategy = ConsultStrategy()
        body = ConsultContextBody(
            question="糖尿病有什么症状？",
            session_id="test_session",
            answer_text="糖尿病是一种慢性代谢性疾病",
            is_health_consultation=True,
            knowledge_results=[
                {"source": "neo4j", "type": "disease_info", "entity": "糖尿病", "data": {"name": "糖尿病"}},
            ],
        )
        result = strategy._build_result(body)
        assert isinstance(result, ConsultResultData)
        assert result.answer == "糖尿病是一种慢性代谢性疾病"
        assert result.session_id == "test_session"
        assert result.is_health_consultation is True
        assert result.confidence == 0.8

    def test_execute_full_flow(self):
        strategy = ConsultStrategy()
        resource = self._make_resource()
        body = ConsultContextBody(
            question="糖尿病有什么症状？",
            session_id="test_session",
        )
        context = AgentContext(session_id="test_session", current_state="INITIAL", body=body)
        result = strategy.execute(context, resource)
        assert result.data is not None
        assert isinstance(result.data, ConsultResultData)
        assert result.data.is_health_consultation is True
