# -*- coding: utf-8 -*-
"""
检索路径注册表

声明式定义所有可用的图谱查询路径和维度推荐路径。
PATH_REGISTRY：每条路径的元数据（查询方法、方向、实体类型、内容完整性）
DIMENSION_RECOMMENDATIONS：每个维度的推荐路径和补充路径

设计原则：
- 路径名作为唯一标识，Qwen3决策和程序执行都通过路径名引用
- 查询方法名与Neo4jMedicalHandler方法名一一对应
- 补充检索路径来自DIMENSION_RECOMMENDATIONS.supplement_paths
"""

from typing import Dict, List

# ============================================================================
# PATH_REGISTRY - 所有可用的图谱查询路径
# ============================================================================

PATH_REGISTRY: Dict[str, Dict] = {
    # ---- 正向查询：Disease → X ----
    "disease_to_symptoms": {
        "query_method": "get_symptoms_by_disease",
        "direction": "forward",
        "source_type": "Disease",
        "target_type": "Symptom",
        "content_completeness": "full",
        "description": "疾病→症状列表",
    },
    "disease_to_drugs_common": {
        "query_method": "get_drugs_by_disease",
        "direction": "forward",
        "source_type": "Disease",
        "target_type": "Drug",
        "content_completeness": "full",
        "description": "疾病→常用药品",
    },
    "disease_to_foods": {
        "query_method": "get_foods_by_disease",
        "direction": "forward",
        "source_type": "Disease",
        "target_type": "Food",
        "content_completeness": "full",
        "description": "疾病→饮食建议(宜吃/忌吃/推荐)",
    },
    "disease_to_checks": {
        "query_method": "get_checks_by_disease",
        "direction": "forward",
        "source_type": "Disease",
        "target_type": "Check",
        "content_completeness": "full",
        "description": "疾病→诊断检查",
    },
    "disease_to_departments": {
        "query_method": "get_department_by_disease",
        "direction": "forward",
        "source_type": "Disease",
        "target_type": "Department",
        "content_completeness": "full",
        "description": "疾病→所属科室",
    },
    "disease_to_cures": {
        "query_method": "get_cure_methods_by_disease",
        "direction": "forward",
        "source_type": "Disease",
        "target_type": "Cure",
        "content_completeness": "full",
        "description": "疾病→治疗方法",
    },
    "disease_to_complications": {
        "query_method": "get_complications_by_disease",
        "direction": "forward",
        "source_type": "Disease",
        "target_type": "Disease",
        "content_completeness": "full",
        "description": "疾病→并发症",
    },
    "disease_attributes": {
        "query_method": "get_disease_info",
        "direction": "self",
        "source_type": "Disease",
        "target_type": "Disease",
        "content_completeness": "full",
        "description": "疾病完整属性(desc/cause/prevent/easy_get/cure_lasttime/cured_prob)",
    },

    # ---- 反向查询：X → Disease ----
    "symptom_to_diseases": {
        "query_method": "search_diseases_by_symptom",
        "direction": "reverse",
        "source_type": "Symptom",
        "target_type": "Disease",
        "content_completeness": "full",
        "description": "症状→相关疾病",
    },
    "drug_to_diseases": {
        "query_method": "get_diseases_by_drug_node_id",
        "direction": "reverse",
        "source_type": "Drug",
        "target_type": "Disease",
        "content_completeness": "full",
        "description": "药品→治疗疾病",
    },
    "food_to_diseases": {
        "query_method": "get_diseases_by_food_node_id",
        "direction": "reverse",
        "source_type": "Food",
        "target_type": "Disease",
        "content_completeness": "full",
        "description": "食物→相关疾病",
    },
    "check_to_diseases": {
        "query_method": "get_diseases_by_check_node_id",
        "direction": "reverse",
        "source_type": "Check",
        "target_type": "Disease",
        "content_completeness": "full",
        "description": "检查→相关疾病",
    },
    "department_to_diseases": {
        "query_method": "get_diseases_by_department_node_id",
        "direction": "reverse",
        "source_type": "Department",
        "target_type": "Disease",
        "content_completeness": "full",
        "description": "科室→相关疾病",
    },
    "cure_to_diseases": {
        "query_method": "get_diseases_by_cure_node_id",
        "direction": "reverse",
        "source_type": "Cure",
        "target_type": "Disease",
        "content_completeness": "full",
        "description": "治疗方式→相关疾病",
    },
}

# ============================================================================
# DIMENSION_RECOMMENDATIONS - 每个维度的推荐路径和补充路径
# ============================================================================

DIMENSION_RECOMMENDATIONS: Dict[str, Dict] = {
    "disease_risk": {
        "primary_entity_types": ["Disease"],
        "recommended_paths": [
            "disease_attributes",
            "disease_to_complications",
            "disease_to_departments",
        ],
        "supplement_paths": [
            "symptom_to_diseases",
            "check_to_diseases",
        ],
    },
    "medication": {
        "primary_entity_types": ["Drug", "Disease"],
        "recommended_paths": [
            "disease_to_drugs_common",
            "disease_to_foods",
            "drug_to_diseases",
        ],
        "supplement_paths": [
            "disease_attributes",
        ],
    },
    "treatment": {
        "primary_entity_types": ["Disease"],
        "recommended_paths": [
            "disease_to_cures",
            "disease_attributes",
            "disease_to_complications",
        ],
        "supplement_paths": [
            "cure_to_diseases",
        ],
    },
    "dietary": {
        "primary_entity_types": ["Food", "Disease"],
        "recommended_paths": [
            "disease_to_foods",
            "disease_attributes",
            "food_to_diseases",
        ],
        "supplement_paths": [
            "disease_to_drugs_common",
        ],
    },
    "checkup": {
        "primary_entity_types": ["Disease", "Check"],
        "recommended_paths": [
            "disease_to_checks",
            "disease_attributes",
            "check_to_diseases",
        ],
        "supplement_paths": [
            "disease_to_symptoms",
        ],
    },
    "complication": {
        "primary_entity_types": ["Disease"],
        "recommended_paths": [
            "disease_to_complications",
            "disease_attributes",
            "disease_to_symptoms",
        ],
        "supplement_paths": [
            "symptom_to_diseases",
        ],
    },
    "prevention": {
        "primary_entity_types": ["Disease"],
        "recommended_paths": [
            "disease_attributes",
            "disease_to_complications",
            "disease_to_departments",
        ],
        "supplement_paths": [
            "check_to_diseases",
        ],
    },
    "susceptible": {
        "primary_entity_types": ["Disease", "Department"],
        "recommended_paths": [
            "disease_attributes",
            "disease_to_departments",
            "symptom_to_diseases",
        ],
        "supplement_paths": [
            "department_to_diseases",
        ],
    },
}


def get_path_info(path_name: str) -> Dict:
    """获取路径元信息，不存在返回空字典"""
    return PATH_REGISTRY.get(path_name, {})


def get_available_paths_summary() -> str:
    """生成Qwen3可读的路径摘要，用于PlanRetrieval prompt"""
    lines = []
    for name, info in PATH_REGISTRY.items():
        lines.append(
            f"- {name}: {info['description']} "
            f"(方向={info['direction']}, 源={info['source_type']}, 目标={info['target_type']})"
        )
    return "\n".join(lines)


def get_dimension_recommendations_summary() -> str:
    """生成Qwen3可读的维度推荐摘要，用于PlanRetrieval prompt"""
    lines = []
    for dim_name, rec in DIMENSION_RECOMMENDATIONS.items():
        rec_paths = ", ".join(rec["recommended_paths"])
        supp_paths = ", ".join(rec["supplement_paths"])
        entity_types = ", ".join(rec["primary_entity_types"])
        lines.append(
            f"- {dim_name}: 主要实体=[{entity_types}], "
            f"推荐路径=[{rec_paths}], 可补充路径=[{supp_paths}]"
        )
    return "\n".join(lines)


def validate_path_name(path_name: str) -> bool:
    """验证路径名是否在注册表中"""
    return path_name in PATH_REGISTRY


def get_recommended_paths_for_dimension(dimension: str) -> List[str]:
    """获取维度的推荐路径列表，用于降级兜底"""
    rec = DIMENSION_RECOMMENDATIONS.get(dimension, {})
    return rec.get("recommended_paths", [])


def get_supplement_paths_for_dimension(dimension: str) -> List[str]:
    """获取维度的补充路径列表"""
    rec = DIMENSION_RECOMMENDATIONS.get(dimension, {})
    return rec.get("supplement_paths", [])


def get_all_path_names() -> List[str]:
    """获取所有路径名列表"""
    return list(PATH_REGISTRY.keys())
