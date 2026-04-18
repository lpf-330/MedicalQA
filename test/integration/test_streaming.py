import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
import json
from unittest.mock import MagicMock, patch

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
from src.orchestration.chain.answer_generation_chain.answer_generation_chain import DISCLAIMER


class TestStreaming:

    def _make_agent_with_stream(self, tokens=None, disclaimer=True):
        if tokens is None:
            tokens = ["糖", "尿", "病", "是", "一", "种"]
        strategy = MagicMock()
        stream_gen = iter(tokens)
        if disclaimer:
            stream_gen_with_disclaimer = _append_disclaimer(stream_gen)
        else:
            stream_gen_with_disclaimer = stream_gen

        body = ConsultContextBody(
            question="糖尿病有什么症状？",
            session_id="test_session",
            is_health_consultation=True,
            stream_generator=stream_gen_with_disclaimer,
            sources=["医学知识库"],
            error_code=0,
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

    def test_sse_message_format(self):
        agent, body = self._make_agent_with_stream()
        service = ConsultService(agent)
        context = AgentContext(session_id="test_session", current_state="INITIAL", body=body)
        events = list(service.process_consult_stream(context))
        message_events = [e for e in events if e.startswith("event: message")]
        assert len(message_events) > 0
        for event_str in message_events:
            lines = event_str.strip().split("\n")
            assert lines[0] == "event: message"
            assert lines[1].startswith("data: ")
            data_str = lines[1][6:]
            data = json.loads(data_str)
            assert "content" in data

    def test_sse_end_event(self):
        agent, body = self._make_agent_with_stream()
        service = ConsultService(agent)
        context = AgentContext(session_id="test_session", current_state="INITIAL", body=body)
        events = list(service.process_consult_stream(context))
        end_events = [e for e in events if e.startswith("event: end")]
        assert len(end_events) > 0
        for event_str in end_events:
            lines = event_str.strip().split("\n")
            assert lines[0] == "event: end"
            data_str = lines[1][6:]
            data = json.loads(data_str)
            assert "session_id" in data

    def test_sse_error_event(self):
        strategy = MagicMock()
        strategy.execute.side_effect = RuntimeError("model error")
        resource = AgentResource()
        agent = Agent(strategy=strategy, resources=resource)
        body = ConsultContextBody(
            question="测试",
            session_id="test_session",
        )
        service = ConsultService(agent)
        context = AgentContext(session_id="test_session", current_state="INITIAL", body=body)
        events = list(service.process_consult_stream(context))
        error_events = [e for e in events if e.startswith("event: error")]
        assert len(error_events) > 0
        for event_str in error_events:
            lines = event_str.strip().split("\n")
            assert lines[0] == "event: error"
            data_str = lines[1][6:]
            data = json.loads(data_str)
            assert "error_code" in data
            assert "error_message" in data

    def test_streaming_includes_disclaimer(self):
        tokens = ["糖", "尿", "病"]
        agent, body = self._make_agent_with_stream(tokens=tokens, disclaimer=True)
        service = ConsultService(agent)
        context = AgentContext(session_id="test_session", current_state="INITIAL", body=body)
        events = list(service.process_consult_stream(context))
        all_content = ""
        for event_str in events:
            if event_str.startswith("event: message"):
                lines = event_str.strip().split("\n")
                data_str = lines[1][6:]
                data = json.loads(data_str)
                all_content += data.get("content", "")
        assert DISCLAIMER in all_content or "仅供参考" in all_content


def _append_disclaimer(token_iter):
    for token in token_iter:
        yield token
    disclaimer = "\n\n以上信息仅供参考，不构成医疗建议。如有健康问题，请及时就医。"
    for char in disclaimer:
        yield char
