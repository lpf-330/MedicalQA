import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import MagicMock, patch

from src.orchestration.chain.answer_generation_chain.answer_generation_chain import (
    AnswerGenerationChain,
    AnswerGenerationContextBody,
    AnswerGenerationResultData,
    AnswerGenerationResource,
    DISCLAIMER,
)
from src.orchestration.chain.data_classes import ChainContext, ChainResult


class TestAnswerGenerationChain:

    def _create_chain(self, model_service=None):
        if model_service is None:
            model_service = MagicMock()
            model_service.call_model.return_value = (
                "糖尿病是一种慢性代谢性疾病，主要特征是血糖水平持续升高。"
                "常见症状包括多饮、多尿、多食和体重减轻。"
                "治疗方面，常用的药物包括二甲双胍和胰岛素，同时需要配合饮食控制和适量运动。"
                "\n\n以上信息仅供参考，不构成医疗建议。如有健康问题，请及时就医。"
            )
        resource = AnswerGenerationResource(model_service=model_service)
        return AnswerGenerationChain(resource)

    def test_execute(self):
        chain = self._create_chain()
        context = ChainContext(
            session_id="test_session",
            body=AnswerGenerationContextBody(
                query_text="糖尿病有什么症状？",
                knowledge_context="糖尿病是一种慢性代谢性疾病，常用药物包括二甲双胍。",
            ),
        )
        result = chain.execute(context)
        assert result.data.answer_text != ""
        assert DISCLAIMER in result.data.answer_text
        assert result.data.has_disclaimer is True

    def test_execute_stream(self):
        model_service = MagicMock()
        inner_model = MagicMock()
        stream_gen = iter(["糖", "尿", "病", "是"])
        inner_model.stream_generate_with_context.return_value = stream_gen
        model_service.get_model_result.return_value = inner_model
        resource = AnswerGenerationResource(model_service=model_service)
        chain = AnswerGenerationChain(resource)
        context = ChainContext(
            session_id="test_session",
            body=AnswerGenerationContextBody(
                query_text="糖尿病有什么症状？",
                knowledge_context="糖尿病相关知识",
            ),
        )
        tokens = list(chain.execute_stream(context))
        assert len(tokens) > 0
        assert "糖" in tokens

    def test_build_prompt_contains_knowledge(self):
        chain = self._create_chain()
        body = AnswerGenerationContextBody(
            query_text="糖尿病有什么症状？",
            knowledge_context="糖尿病是一种慢性代谢性疾病",
        )
        prompt = chain._build_prompt(body)
        assert "糖尿病是一种慢性代谢性疾病" in prompt["system_message"]

    def test_build_prompt_contains_disclaimer_rule(self):
        chain = self._create_chain()
        body = AnswerGenerationContextBody(
            query_text="糖尿病有什么症状？",
            knowledge_context="",
        )
        prompt = chain._build_prompt(body)
        assert DISCLAIMER in prompt["user_message"]

    def test_format_answer_adds_disclaimer(self):
        chain = self._create_chain()
        raw_answer = "这是一个测试回答，不包含免责声明。"
        formatted = chain._format_answer(raw_answer, [])
        assert DISCLAIMER in formatted

    def test_format_answer_no_duplicate_disclaimer(self):
        chain = self._create_chain()
        raw_answer = "这是一个测试回答。\n\n" + DISCLAIMER
        formatted = chain._format_answer(raw_answer, [])
        assert formatted.count(DISCLAIMER) == 1

    def test_check_quality_pass(self):
        chain = self._create_chain()
        answer = "A" * 300 + "\n\n" + DISCLAIMER
        assert chain._check_quality(answer) is True

    def test_check_quality_fail_length(self):
        chain = self._create_chain()
        answer = "太短了" + "\n\n" + DISCLAIMER
        assert chain._check_quality(answer) is False

    def test_check_quality_fail_no_disclaimer(self):
        chain = self._create_chain()
        answer = "A" * 300
        assert chain._check_quality(answer) is False
