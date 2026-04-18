import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from typing import List, Dict, Any

SAMPLE_VECTOR_RESULTS = [
    {"id": "entity_1", "distance": 0.85, "entity": {"name": "糖尿病", "type": "Disease", "neo4j_node_id": "disease_001"}, "collection": "medical_entity"},
    {"id": "entity_2", "distance": 0.78, "entity": {"name": "高血压", "type": "Disease", "neo4j_node_id": "disease_002"}, "collection": "medical_entity"},
    {"id": "attr_1", "distance": 0.72, "entity": {"name": "糖尿病症状", "type": "Symptom", "neo4j_node_id": "attr_001", "attribute_of": "disease_001"}, "collection": "entity_attributes"},
    {"id": "rel_1", "distance": 0.68, "entity": {"name": "糖尿病-二甲双胍", "type": "DrugRelation", "neo4j_relation_id": "rel_001", "source": "disease_001", "target": "drug_001"}, "collection": "entity_relations"},
]

SAMPLE_INTENT_RESULT = {
    "intent_label": "health_consultation",
    "confidence": 0.92,
}

SAMPLE_EXTRACTED_ENTITIES = [
    {"entity_name": "糖尿病", "entity_type": "Disease"},
    {"entity_name": "头痛", "entity_type": "Symptom"},
]

SAMPLE_NEO4J_DISEASE_INFO = {
    "name": "糖尿病",
    "description": "糖尿病是一种慢性代谢性疾病",
    "department": "内分泌科",
    "cure_method": "药物治疗、饮食控制、运动疗法",
}

SAMPLE_NEO4J_SYMPTOMS = [
    {"name": "多饮", "type": "Symptom"},
    {"name": "多尿", "type": "Symptom"},
    {"name": "多食", "type": "Symptom"},
]

SAMPLE_NEO4J_DRUGS = [
    {"name": "二甲双胍", "type": "Drug"},
    {"name": "胰岛素", "type": "Drug"},
]

SAMPLE_NEO4J_FOODS = [
    {"name": "苦瓜", "type": "Food", "recommendation": "适宜"},
    {"name": "燕麦", "type": "Food", "recommendation": "适宜"},
]

SAMPLE_LLM_RESPONSE = "糖尿病是一种慢性代谢性疾病，主要特征是血糖水平持续升高。常见症状包括多饮、多尿、多食和体重减轻。治疗方面，常用的药物包括二甲双胍和胰岛素，同时需要配合饮食控制和适量运动。\n\n以上信息仅供参考，不构成医疗建议。如有健康问题，请及时就医。"

SAMPLE_CONSULT_REQUEST_BODY = {
    "task_id": "test_task_001",
    "chat_history": [
        {"role": "user", "content": "糖尿病有什么症状？"}
    ],
    "question": "糖尿病有什么症状？",
    "session_id": "test_session_001",
}

SAMPLE_QUERY_VECTOR = [0.1] * 1024

SAMPLE_HYBRID_SEARCH_RESULTS = [
    {"id": "entity_1", "score": 0.92, "name": "糖尿病", "type": "Disease", "neo4j_node_id": "disease_001"},
    {"id": "entity_2", "score": 0.85, "name": "高血压", "type": "Disease", "neo4j_node_id": "disease_002"},
    {"id": "rel_1", "score": 0.78, "name": "糖尿病-二甲双胍", "type": "DrugRelation", "neo4j_relation_id": "rel_001"},
]

DISCLAIMER_TEXT = "以上信息仅供参考，不构成医疗建议。如有健康问题，请及时就医。"
