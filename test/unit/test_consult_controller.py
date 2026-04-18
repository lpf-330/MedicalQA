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
from src.orchestration.agent.consult_strategy.consult_strategy import ConsultContextBody, ConsultResultData
from src.schemas.consult_request import ConsultRequest, ConsultRequestBody, ChatMessage
from src.utils.exception_handler import MedicalQAException, ErrorCode
from starlette.responses import StreamingResponse


class TestConsultController:

    def _make_controller(self):
        strategy = MagicMock()
        resource = AgentResource()
        agent = Agent(strategy=strategy, resources=resource)
        service = ConsultService(agent)
        controller = ConsultController(service)
        return controller, service, agent

    def _make_request(self, question="糖尿病有什么症状？"):
        return ConsultRequest(
            request_id="test_req",
            body=ConsultRequestBody(
                task_id="task_001",
                chat_history=[ChatMessage(role="user", content=question)],
                question=question,
                session_id="session_001",
            ),
        )

    def test_consult_returns_streaming_response(self):
        controller, service, agent = self._make_controller()
        request = self._make_request()
        with patch.object(service, '_build_agent_context') as mock_build, \
             patch.object(service, 'process_consult_stream') as mock_stream:
            mock_build.return_value = AgentContext(
                session_id="session_001",
                current_state="INITIAL",
                body=ConsultContextBody(question="糖尿病有什么症状？", session_id="session_001"),
            )
            mock_stream.return_value = iter([
                'event: message\ndata: {"content": "糖尿病"}\n\n',
                'event: end\ndata: {"session_id": "session_001"}\n\n',
            ])
            response = controller.consult(request)
            assert isinstance(response, StreamingResponse)

    def test_validate_request_empty_question(self):
        controller, service, agent = self._make_controller()
        request = ConsultRequest(
            request_id="test_req",
            body=ConsultRequestBody(
                task_id="task_001",
                chat_history=[ChatMessage(role="user", content="")],
                question="",
                session_id="session_001",
            ),
        )
        with pytest.raises(MedicalQAException) as exc_info:
            controller._validate_request(request)
        assert "question" in str(exc_info.value.message) or "不能为空" in str(exc_info.value.message)

    def test_consult_error_handling(self):
        controller, service, agent = self._make_controller()
        request = self._make_request()
        with patch.object(service, '_build_agent_context') as mock_build:
            mock_build.side_effect = RuntimeError("unexpected error")
            response = controller.consult(request)
            assert isinstance(response, StreamingResponse)
            import asyncio

            async def collect_body():
                chunks = []
                async for chunk in response.body_iterator:
                    chunks.append(chunk)
                return "".join(chunks)

            content = asyncio.get_event_loop().run_until_complete(collect_body())
            assert "error" in content
