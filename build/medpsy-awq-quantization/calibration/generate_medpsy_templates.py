# -*- coding: utf-8 -*-
"""
MedPsy推理格式模板生成脚本

生成与MedPsy实际推理格式对齐的校准模板，覆盖5维度评估和6风险因子评估。
这些模板确保量化模型在MedPsy实际使用的prompt格式和输出JSON格式上保持精度。
"""

import json
import logging
import os
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "raw_benchmark")

MEDPSY_SYSTEM_PROMPT = "你是一位全科医生，擅长精炼评估。请在3秒内、不超过50字完成思考，然后直接输出JSON。"

HEALTH_DIMENSIONS = {
    "D1": {"name": "生理指标", "weight": 0.30, "sub_indicators": ["血压水平", "血糖水平", "血脂水平", "BMI指数", "心率"]},
    "D2": {"name": "用药情况", "weight": 0.20, "sub_indicators": ["用药依从性", "药物副作用", "药物相互作用"]},
    "D3": {"name": "治疗状况", "weight": 0.20, "sub_indicators": ["治疗效果", "康复进度", "并发症控制"]},
    "D4": {"name": "饮食状况", "weight": 0.15, "sub_indicators": ["饮食均衡性", "营养摄入", "饮食禁忌遵守"]},
    "D5": {"name": "检查情况", "weight": 0.15, "sub_indicators": ["检查完整性", "指标异常追踪", "复查及时性"]},
}

DISEASE_RISK_FACTORS = {
    "F1": {"name": "疾病严重程度", "weight": 0.25},
    "F2": {"name": "并发症风险", "weight": 0.20},
    "F3": {"name": "用药风险", "weight": 0.20},
    "F4": {"name": "生活习惯风险", "weight": 0.15},
    "F5": {"name": "复查监测风险", "weight": 0.10},
    "F6": {"name": "预防措施风险", "weight": 0.10},
}

USER_PROFILES = [
    {"age": 68, "gender": "男", "chronic_diseases": ["高血压", "糖尿病"], "medications": ["降压药", "二甲双胍"]},
    {"age": 72, "gender": "女", "chronic_diseases": ["冠心病", "高血脂"], "medications": ["他汀类", "阿司匹林"]},
    {"age": 65, "gender": "男", "chronic_diseases": ["慢性肾病"], "medications": ["ACEI", "利尿剂"]},
    {"age": 75, "gender": "女", "chronic_diseases": ["骨质疏松", "高血压"], "medications": ["钙片", "ARB类"]},
    {"age": 70, "gender": "男", "chronic_diseases": ["COPD", "高血压"], "medications": ["支气管扩张剂", "氨氯地平"]},
    {"age": 80, "gender": "女", "chronic_diseases": ["心房颤动", "糖尿病"], "medications": ["华法林", "胰岛素"]},
    {"age": 63, "gender": "男", "chronic_diseases": ["痛风", "高血脂"], "medications": ["别嘌醇", "非诺贝特"]},
    {"age": 77, "gender": "女", "chronic_diseases": ["类风湿", "骨质疏松"], "medications": ["甲氨蝶呤", "双膦酸盐"]},
]

ANOMALIES_LIST = [
    [{"item": "收缩压", "value": "165mmHg", "status": "偏高"}, {"item": "空腹血糖", "value": "8.2mmol/L", "status": "偏高"}],
    [{"item": "LDL-C", "value": "4.1mmol/L", "status": "偏高"}, {"item": "BMI", "value": "28.5", "status": "超重"}],
    [{"item": "血肌酐", "value": "180μmol/L", "status": "偏高"}, {"item": "eGFR", "value": "38", "status": "偏低"}],
    [{"item": "糖化血红蛋白", "value": "8.5%", "status": "偏高"}, {"item": "尿蛋白", "value": "++", "status": "阳性"}],
    [{"item": "骨密度T值", "value": "-3.0", "status": "偏低"}, {"item": "维生素D", "value": "15ng/mL", "status": "不足"}],
]

MEDICAL_ENTITIES = [
    {"Disease": ["高血压", "2型糖尿病"], "Drug": ["氨氯地平", "二甲双胍"], "Food": ["低盐饮食", "粗粮"]},
    {"Disease": ["冠心病"], "Drug": ["阿托伐他汀", "阿司匹林"], "Food": ["深海鱼", "坚果"]},
    {"Disease": ["慢性肾病3期"], "Drug": ["依那普利"], "Food": ["低蛋白饮食"]},
    {"Disease": ["骨质疏松"], "Drug": ["阿仑膦酸钠", "钙尔奇"], "Food": ["牛奶", "豆制品"]},
    {"Disease": ["COPD", "肺气肿"], "Drug": ["沙美特罗"], "Food": ["梨", "百合"]},
]

KNOWLEDGE_SAMPLES = [
    {"summary": "高血压患者应控制钠盐摄入，每日不超过5g", "refined_knowledge": ["高血压需长期服药控制", "血压目标<140/90mmHg"]},
    {"summary": "糖尿病患者需定期监测血糖和糖化血红蛋白", "refined_knowledge": ["空腹血糖目标4.4-7.0mmol/L"]},
    {"summary": "他汀类药物需监测肝功能和肌酸激酶", "refined_knowledge": ["LDL-C目标<2.6mmol/L"]},
    {"summary": "慢性肾病需限制蛋白质摄入并监测肾功能", "refined_knowledge": ["eGFR<60需积极干预"]},
    {"summary": "骨质疏松需补充钙和维生素D，预防跌倒", "refined_knowledge": ["T值<-2.5为骨质疏松"]},
]


def build_dimension_prompt(dim_id, dim_info, user_profile, anomalies, knowledge):
    sub_indicators = dim_info["sub_indicators"]
    format_instruction = (
        f"你是健康评估专家。评估维度: {dim_info['name']}(权重{dim_info['weight']})\n"
        f"子指标: {', '.join(sub_indicators)}\n\n"
        "严格按以下JSON格式输出，不要输出任何其他内容:\n"
        '{"dimension_score":0.72,"sub_indicator_scores":{"指标名":0.65},'
        '"dimension_reasoning":"总体评估"}\n\n'
        "对每个子指标评分(0-1)。\n\n以下是评估依据:\n"
    )
    user_profile_str = json.dumps(user_profile, ensure_ascii=False)
    anomalies_str = json.dumps(anomalies, ensure_ascii=False)
    data_section = f"用户={user_profile_str} 异常={anomalies_str} 风险={{}}\n"
    knowledge_str = json.dumps(knowledge, ensure_ascii=False)
    knowledge_section = f"知识={knowledge_str}"
    return format_instruction + data_section + knowledge_section


def build_risk_factor_prompt(factor_id, factor_info, user_profile, anomalies, medical_entities):
    format_instruction = (
        f"你是健康评估专家。评估风险因子: {factor_info['name']}(权重{factor_info['weight']})\n\n"
        "严格按以下JSON格式输出，不要输出任何其他内容:\n"
        '{"factor_score":45,"factor_reasoning":"评估理由","related_diseases":["疾病1"]}'
        "\n\n评估风险程度(0-100)并给简短理由(20字内)。\n\n以下是评估依据:\n"
    )
    user_profile_str = json.dumps(user_profile, ensure_ascii=False)
    anomalies_str = json.dumps(anomalies, ensure_ascii=False)
    entities_str = json.dumps(medical_entities, ensure_ascii=False)
    data_section = f"用户={user_profile_str} 异常={anomalies_str} 实体={entities_str}"
    return format_instruction + data_section


def generate_dimension_templates(count):
    templates = []
    for _ in range(count):
        dim_id = random.choice(list(HEALTH_DIMENSIONS.keys()))
        dim_info = HEALTH_DIMENSIONS[dim_id]
        user = random.choice(USER_PROFILES)
        anomalies = random.choice(ANOMALIES_LIST)
        knowledge = random.choice(KNOWLEDGE_SAMPLES)
        prompt = build_dimension_prompt(dim_id, dim_info, user, anomalies, knowledge)
        templates.append({
            "type": "medpsy_dimension",
            "dimension_id": dim_id,
            "dimension_name": dim_info["name"],
            "messages": [
                {"role": "system", "content": MEDPSY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "text": prompt,
        })
    return templates


def generate_risk_factor_templates(count):
    templates = []
    for _ in range(count):
        factor_id = random.choice(list(DISEASE_RISK_FACTORS.keys()))
        factor_info = DISEASE_RISK_FACTORS[factor_id]
        user = random.choice(USER_PROFILES)
        anomalies = random.choice(ANOMALIES_LIST)
        entities = random.choice(MEDICAL_ENTITIES)
        prompt = build_risk_factor_prompt(factor_id, factor_info, user, anomalies, entities)
        templates.append({
            "type": "medpsy_risk_factor",
            "factor_id": factor_id,
            "factor_name": factor_info["name"],
            "messages": [
                {"role": "system", "content": MEDPSY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "text": prompt,
        })
    return templates


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(42)

    dimension_templates = generate_dimension_templates(30)
    risk_factor_templates = generate_risk_factor_templates(20)
    all_templates = dimension_templates + risk_factor_templates

    with open(os.path.join(OUTPUT_DIR, "medpsy_templates.json"), "w", encoding="utf-8") as f:
        json.dump(all_templates, f, ensure_ascii=False, indent=2)

    logger.info(f"=== 生成统计 ===")
    logger.info(f"维度评估模板: {len(dimension_templates)}")
    logger.info(f"风险因子模板: {len(risk_factor_templates)}")
    logger.info(f"总计: {len(all_templates)}")


if __name__ == "__main__":
    main()
