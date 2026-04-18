import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import MagicMock, patch

from src.schemas.consult_request import ChatMessage, ConsultRequestBody, ConsultRequest
from src.schemas.consult_response import ConsultResponseData, ConsultResponse


class TestChatMessage:

    def test_chat_message(self):
        msg = ChatMessage(role="user", content="我最近总是头痛")
        assert msg.role == "user"
        assert msg.content == "我最近总是头痛"


class TestConsultRequestBody:

    def test_consult_request_body(self):
        body = ConsultRequestBody(
            task_id="task_001",
            chat_history=[ChatMessage(role="user", content="糖尿病有什么症状？")],
            question="糖尿病有什么症状？",
        )
        assert body.task_id == "task_001"
        assert len(body.chat_history) == 1
        assert body.chat_history[0].role == "user"


class TestConsultResponseData:

    def test_consult_response_data(self):
        data = ConsultResponseData(
            result="糖尿病是一种慢性代谢性疾病",
            event_type="message",
            sources=["医学知识库"],
        )
        assert data.result == "糖尿病是一种慢性代谢性疾病"
        assert data.event_type == "message"
        assert data.sources == ["医学知识库"]


class TestConsultRequestSerialization:

    def test_consult_request_serialization(self):
        body = ConsultRequestBody(
            task_id="task_001",
            chat_history=[ChatMessage(role="user", content="测试")],
            question="测试问题",
            session_id="session_001",
        )
        request = ConsultRequest(
            request_id="req_001",
            body=body,
        )
        result = request.to_dict()
        assert isinstance(result, dict)
        assert result["request_id"] == "req_001"
        assert result["body"]["task_id"] == "task_001"
        assert result["body"]["question"] == "测试问题"


class TestConsultResponseSerialization:

    def test_consult_response_serialization(self):
        data = ConsultResponseData(
            result="测试回答",
            event_type="message",
            sources=["来源1"],
        )
        response = ConsultResponse(
            status_code=200,
            message="成功",
            data=data,
        )
        result = response.to_dict()
        assert isinstance(result, dict)
        assert result["status_code"] == 200
        assert result["data"]["result"] == "测试回答"
