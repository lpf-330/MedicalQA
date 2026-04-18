import sys
import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

for mod_name in ['vllm', 'pymilvus', 'sentence_transformers']:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

if 'vllm' not in sys.modules or not isinstance(sys.modules.get('vllm'), MagicMock):
    pass
else:
    sys.modules['vllm'].LLM = MagicMock
    sys.modules['vllm'].SamplingParams = MagicMock

if 'transformers' not in sys.modules:
    transformers_mock = MagicMock()
    transformers_mock.AutoModelForSequenceClassification = MagicMock
    transformers_mock.AutoModelForTokenClassification = MagicMock
    transformers_mock.AutoTokenizer = MagicMock
    transformers_mock.pipeline = MagicMock
    sys.modules['transformers'] = transformers_mock
    sys.modules['transformers.models'] = MagicMock()
    sys.modules['transformers.models.auto'] = MagicMock()

from test.fixtures.mock_data import *


@pytest.fixture
def mock_milvus_adapter():
    adapter = MagicMock()
    adapter.is_connected.return_value = True
    adapter.search.return_value = SAMPLE_VECTOR_RESULTS
    adapter.hybrid_search.return_value = SAMPLE_HYBRID_SEARCH_RESULTS
    adapter.connect.return_value = None
    adapter.disconnect.return_value = None
    return adapter


@pytest.fixture
def mock_transformers_adapter():
    adapter = MagicMock()
    adapter.is_model_loaded.return_value = True
    adapter.predict.return_value = SAMPLE_INTENT_RESULT
    adapter.predict_batch.return_value = [SAMPLE_INTENT_RESULT]
    adapter.encode.return_value = SAMPLE_QUERY_VECTOR
    adapter.encode_batch.return_value = [SAMPLE_QUERY_VECTOR]
    adapter.load_model.return_value = None
    adapter.unload_model.return_value = None
    return adapter


@pytest.fixture
def mock_vllm_adapter():
    adapter = MagicMock()
    adapter.is_model_loaded.return_value = True
    adapter.generate.return_value = SAMPLE_LLM_RESPONSE
    adapter.stream_generate.return_value = iter(["糖", "尿", "病", "是"])
    adapter.load_model.return_value = None
    adapter.unload_model.return_value = None
    return adapter


@pytest.fixture
def mock_milvus_resource():
    resource = MagicMock()
    resource.get_type.return_value = "milvus_connection"
    resource.is_activate.return_value = True
    resource.get_adapter.return_value = mock_milvus_adapter()
    return resource


@pytest.fixture
def mock_intent_model_resource():
    resource = MagicMock()
    resource.get_type.return_value = "intent_model"
    resource.is_activate.return_value = True
    return resource


@pytest.fixture
def mock_vector_model_resource():
    resource = MagicMock()
    resource.get_type.return_value = "vector_model"
    resource.is_activate.return_value = True
    return resource


@pytest.fixture
def mock_resource_handle():
    handle = MagicMock()
    handle.resource = MagicMock()
    handle.resource.get_adapter.return_value = MagicMock()
    return handle


@pytest.fixture
def mock_global_resource_manager():
    with patch('src.resource_manager.global_resource_manager.GlobalResourceManager') as mock_grm:
        instance = mock_grm.INSTANCE
        instance.acquire.return_value = mock_resource_handle()
        instance.release.return_value = None
        yield instance


@pytest.fixture
def sample_consult_request():
    from src.schemas.consult_request import ConsultRequest, ConsultRequestBody, ChatMessage
    body = ConsultRequestBody(
        task_id="test_task_001",
        chat_history=[ChatMessage(role="user", content="糖尿病有什么症状？")],
        question="糖尿病有什么症状？",
        session_id="test_session_001"
    )
    return ConsultRequest(
        request_id="test_req_001",
        timestamp="2026-04-15T00:00:00",
        body=body
    )


@pytest.fixture
def sample_intent_parse_context_body():
    from src.orchestration.chain.intent_parse_chain.intent_parse_chain import IntentParseContextBody
    return IntentParseContextBody(
        query_text="糖尿病有什么症状？",
        chat_history=[{"role": "user", "content": "糖尿病有什么症状？"}]
    )


@pytest.fixture
def sample_knowledge_retrieval_context_body():
    from src.orchestration.chain.knowledge_retrieval_chain.knowledge_retrieval_chain import KnowledgeRetrievalContextBody
    return KnowledgeRetrievalContextBody(
        query_text="糖尿病有什么症状？",
        extracted_entities=SAMPLE_EXTRACTED_ENTITIES,
        intent_label="health_consultation"
    )


@pytest.fixture
def sample_answer_generation_context_body():
    from src.orchestration.chain.answer_generation_chain.answer_generation_chain import AnswerGenerationContextBody
    return AnswerGenerationContextBody(
        query_text="糖尿病有什么症状？",
        knowledge_context="糖尿病是一种慢性代谢性疾病，常用药物包括二甲双胍。",
        intent_label="health_consultation",
        chat_history=[{"role": "user", "content": "糖尿病有什么症状？"}]
    )
