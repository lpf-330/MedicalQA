import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
import json
from unittest.mock import MagicMock, patch

from src.controller.consult_controller import ConsultController
from src.service.consult_service import ConsultService
from src.orchestration.agent.agent import Agent
from src.orchestration.agent.data_classes import AgentContext, AgentResult
from src.orchestration.agent.agent_resource import AgentResource
from src.orchestration.agent.consult_strategy.consult_strategy import (
    ConsultStrategy,
    ConsultContextBody,
    ConsultResultData,
)
from src.orchestration.chain.data_classes import ChainContext, ChainResult
from src.orchestration.chain.intent_parse_chain.intent_parse_chain import IntentParseResultData
from src.orchestration.chain.knowledge_retrieval_chain.knowledge_retrieval_chain import KnowledgeRetrievalResultData
from src.orchestration.chain.answer_generation_chain.answer_generation_chain import AnswerGenerationResultData, DISCLAIMER
from src.schemas.consult_request import ConsultRequest, ConsultRequestBody, ChatMessage
from starlette.responses import StreamingResponse


class TestFullHealthConsultationFlow:

    def _make_intent_chain(self, is_health=True, confidence=0.92):
        chain = MagicMock()
        chain.execute.return_value = ChainResult(
            session_id="test_session",
            data=IntentParseResultData(
                intent_label="health_consultation" if is_health else "chat",
                confidence=confidence,
                extracted_entities=[{"entity_name": "糖尿病", "entity_type": "Disease"}] if is_health else [],
                rewritten_query="糖尿病" if is_health else "test",
                is_health_consultation=is_health and confidence >= 0.6,
            ),
        )
        chain.execute_stream.return_value = iter(["糖", "尿", "病"])
        return chain

    def _make_knowledge_chain(self):
        chain = MagicMock()
        chain.execute.return_value = ChainResult(
            session_id="test_session",
            data=KnowledgeRetrievalResultData(
                vector_results=[{"id": "v1", "score": 0.9, "name": "糖尿病"}],
                knowledge_results=[{
                    "source": "neo4j", "type": "disease_info", "entity": "糖尿病",
                    "data": {"name": "糖尿病", "description": "慢性代谢性疾病"}, "score": 0.8
                }],
                merged_results=[
                    {"source": "vector", "data": {"id": "v1", "score": 0.9}, "score": 0.9},
                    {"source": "neo4j", "type": "disease_info", "entity": "糖尿病",
                     "data": {"name": "糖尿病", "description": "慢性代谢性疾病"}, "score": 0.8},
                ],
                anchored_entities=[{"name": "糖尿病", "entity_type": "Disease"}],
                anchored_relations=[],
            ),
        )
        return chain

    def _make_answer_chain(self):
        chain = MagicMock()
        chain.execute_stream.return_value = iter([
            "糖尿病是一种慢性代谢性疾病。",
            "\n\n" + DISCLAIMER,
        ])
        return chain

    def _make_resource(self, intent_chain=None, knowledge_chain=None, answer_chain=None):
        resource = AgentResource()
        resource.register_chain("intent_parse_chain", intent_chain or self._make_intent_chain())
        resource.register_chain("knowledge_retrieval_chain", knowledge_chain or self._make_knowledge_chain())
        resource.register_chain("answer_generation_chain", answer_chain or self._make_answer_chain())
        return resource

    def test_full_health_consultation_flow(self):
        resource = self._make_resource()
        strategy = ConsultStrategy()
        agent = Agent(strategy=strategy, resources=resource)
        service = ConsultService(agent)
        controller = ConsultController(service)

        request = ConsultRequest(
            request_id="req_001",
            body=ConsultRequestBody(
                task_id="task_001",
                chat_history=[ChatMessage(role="user", content="糖尿病有什么症状？")],
                question="糖尿病有什么症状？",
                session_id="session_001",
            ),
        )

        response = controller.consult(request)
        assert isinstance(response, StreamingResponse)

        import asyncio

        async def collect_events():
            events = []
            async for chunk in response.body_iterator:
                events.append(chunk)
            return events

        events = asyncio.get_event_loop().run_until_complete(collect_events())
        assert len(events) > 0

    def test_non_health_consultation_flow(self):
        intent_chain = self._make_intent_chain(is_health=False)
        resource = self._make_resource(intent_chain=intent_chain)
        strategy = ConsultStrategy()
        agent = Agent(strategy=strategy, resources=resource)
        service = ConsultService(agent)
        controller = ConsultController(service)

        request = ConsultRequest(
            request_id="req_002",
            body=ConsultRequestBody(
                task_id="task_002",
                chat_history=[ChatMessage(role="user", content="今天天气怎么样？")],
                question="今天天气怎么样？",
                session_id="session_002",
            ),
        )

        response = controller.consult(request)
        assert isinstance(response, StreamingResponse)

    def test_consult_with_milvus_failure(self):
        intent_chain = self._make_intent_chain(is_health=True)
        knowledge_chain = MagicMock()
        knowledge_chain.execute.return_value = ChainResult(
            session_id="test_session",
            data=KnowledgeRetrievalResultData(
                vector_results=[],
                knowledge_results=[{
                    "source": "neo4j", "type": "disease_info", "entity": "糖尿病",
                    "data": {"name": "糖尿病", "description": "慢性代谢性疾病"}, "score": 0.8
                }],
                merged_results=[{
                    "source": "neo4j", "type": "disease_info", "entity": "糖尿病",
                    "data": {"name": "糖尿病", "description": "慢性代谢性疾病"}, "score": 0.8
                }],
                anchored_entities=[{"name": "糖尿病", "entity_type": "Disease"}],
                anchored_relations=[],
            ),
        )
        resource = self._make_resource(intent_chain=intent_chain, knowledge_chain=knowledge_chain)
        strategy = ConsultStrategy()
        agent = Agent(strategy=strategy, resources=resource)
        service = ConsultService(agent)
        controller = ConsultController(service)

        request = ConsultRequest(
            request_id="req_003",
            body=ConsultRequestBody(
                task_id="task_003",
                chat_history=[ChatMessage(role="user", content="糖尿病有什么症状？")],
                question="糖尿病有什么症状？",
                session_id="session_003",
            ),
        )

        response = controller.consult(request)
        assert isinstance(response, StreamingResponse)
