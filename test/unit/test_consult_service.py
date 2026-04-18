import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
import json
from unittest.mock import MagicMock, patch
from typing import Generator

from src.service.consult_service import ConsultService
from src.orchestration.agent.agent import Agent
from src.orchestration.agent.data_classes import AgentContext, AgentResult
from src.orchestration.agent.agent_resource import AgentResource
from src.orchestration.agent.consult_strategy.consult_strategy import ConsultContextBody, ConsultResultData
from src.schemas.consult_request import ConsultRequest, ConsultRequestBody, ChatMessage


class TestConsultService:

    def _make_agent(self):
        strategy = MagicMock()
        body = ConsultContextBody(
            question="糖尿病有什么症状？",
            session_id="test_session",
            is_health_consultation=True,
            stream_generator=iter(["糖", "尿", "病"]),
        )
        result_data = ConsultResultData(
            answer="糖尿病是一种慢性代谢性疾病",
            session_id="test_session",
            is_health_consultation=True,
            confidence=0.8,
        )
        strategy.execute.return_value = AgentResult(session_id="test_session", data=result_data)
        resource = AgentResource()
        agent = Agent(strategy=strategy, resources=resource)
        return agent, body

    def test_process_consult_stream(self):
        agent, body = self._make_agent()
        agent.strategy.execute.return_value = AgentResult(
            session_id="test_session",
            data=ConsultResultData(answer="test", session_id="test_session"),
        )
        service = ConsultService(agent)
        context = AgentContext(session_id="test_session", current_state="INITIAL", body=body)
        result = service.process_consult_stream(context)
        assert isinstance(result, Generator)
        events = list(result)
        assert len(events) > 0

    def test_build_agent_context(self):
        agent, _ = self._make_agent()
        service = ConsultService(agent)
        request = ConsultRequest(
            request_id="test_req",
            body=ConsultRequestBody(
                task_id="task_001",
                chat_history=[ChatMessage(role="user", content="糖尿病有什么症状？")],
                question="糖尿病有什么症状？",
                session_id="session_001",
            ),
        )
        context = service._build_agent_context(request)
        assert isinstance(context, AgentContext)
        assert context.session_id == "session_001"
        assert context.body.question == "糖尿病有什么症状？"
        assert context.body.current_state == "INITIAL"

    def test_assemble_agent_resource(self):
        agent, _ = self._make_agent()
        service = ConsultService(agent)
        with patch.object(service, '_register_handlers'), \
             patch.object(service, '_register_model_services'), \
             patch.object(service, '_register_chains'), \
             patch.object(service, '_register_state_machine'):
            resource = service._assemble_agent_resource()
            assert isinstance(resource, AgentResource)
