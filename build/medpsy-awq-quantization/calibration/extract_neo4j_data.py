# -*- coding: utf-8 -*-
"""
Neo4j知识图谱数据提取脚本

从Neo4j图数据库提取医学实体和关系文本，用于AWQ校准数据构建。
按实体类型分层采样，跳过Producer实体。
"""

import json
import logging
import os
import random
from typing import Dict, List

from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Neo4j连接配置 — 从application.yaml读取
NEO4J_URI = "neo4j+s://627658bb.databases.neo4j.io"
NEO4J_USER = "627658bb"
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DATABASE = "627658bb"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "raw_neo4j")

# 采样策略配置
ENTITY_SAMPLING = {
    "Disease": {"rate": 1.0, "desc": "有长文本属性的全量纳入"},
    "Symptom": {"rate": 0.30, "desc": "30%采样"},
    "Drug": {"rate": 0.30, "desc": "30%采样"},
    "Food": {"rate": 0.20, "desc": "20%采样"},
    "Check": {"rate": 0.30, "desc": "30%采样"},
    "Cure": {"rate": 1.0, "desc": "全量"},
    "Department": {"rate": 1.0, "desc": "全量"},
}

# 关系采样分配
RELATION_SAMPLING = {
    "has_symptom": 200,
    "recommand_drug": 150,
    "need_check": 150,
    "recommand_eat": 100,
    "do_eat": 100,
    "no_eat": 100,
    "acompany_with": 80,
    "cure_way": 60,
    "belongs_to": 40,
    "common_drug": 40,
    "drugs_of": 20,
}

DISEASE_TEXT_PROPERTIES = ["desc", "cause", "prevent", "easy_get", "cure_lasttime", "cured_prob"]

PROP_CN = {
    "desc": "描述", "cause": "病因", "prevent": "预防",
    "easy_get": "易感人群", "cure_lasttime": "治疗周期", "cured_prob": "治愈概率",
}

TYPE_CN = {
    "Symptom": "症状", "Drug": "药品", "Food": "食物",
    "Check": "检查", "Cure": "治疗方式", "Department": "科室", "Disease": "疾病",
}

REL_CN = {
    "has_symptom": "有症状", "recommand_drug": "推荐用药", "need_check": "需要检查",
    "recommand_eat": "推荐食用", "do_eat": "宜吃", "no_eat": "忌吃",
    "acompany_with": "并发症", "cure_way": "治疗方式", "belongs_to": "所属科室",
    "common_drug": "常用药物", "drugs_of": "生产药品",
}


def get_driver():
    if not NEO4J_PASSWORD:
        raise ValueError("请设置环境变量 NEO4J_PASSWORD")
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def extract_entities(driver) -> Dict[str, List[dict]]:
    results = {}
    with driver.session(database=NEO4J_DATABASE) as session:
        for entity_type, config in ENTITY_SAMPLING.items():
            logger.info(f"提取 {entity_type} (采样率: {config['rate']})...")
            if entity_type == "Disease":
                query = """
                MATCH (n:Disease)
                WHERE n.desc IS NOT NULL AND n.desc <> ''
                RETURN n.name AS name, n.desc AS desc, n.cause AS cause,
                       n.prevent AS prevent, n.easy_get AS easy_get,
                       n.cure_lasttime AS cure_lasttime, n.cured_prob AS cured_prob
                """
            else:
                query = f"MATCH (n:{entity_type}) RETURN n.name AS name"
            records = session.run(query).data()
            if config["rate"] < 1.0:
                sample_size = max(1, int(len(records) * config["rate"]))
                records = random.sample(records, min(sample_size, len(records)))
            results[entity_type] = records
            logger.info(f"  {entity_type}: {len(records)} 条")
    return results


def extract_relations(driver) -> Dict[str, List[dict]]:
    results = {}
    with driver.session(database=NEO4J_DATABASE) as session:
        for rel_type, limit in RELATION_SAMPLING.items():
            logger.info(f"提取关系 {rel_type} (限制: {limit})...")
            query = f"""
            MATCH (s)-[r:{rel_type}]->(t)
            RETURN labels(s)[0] AS source_type, s.name AS source_name,
                   labels(t)[0] AS target_type, t.name AS target_name,
                   type(r) AS relation_type
            LIMIT {limit * 2}
            """
            records = session.run(query).data()
            if len(records) > limit:
                records = random.sample(records, limit)
            results[rel_type] = records
            logger.info(f"  {rel_type}: {len(records)} 条")
    return results


def entity_to_text(entity_type: str, record: dict) -> str:
    name = record.get("name", "")
    if entity_type == "Disease":
        parts = [f"疾病：{name}"]
        for prop in DISEASE_TEXT_PROPERTIES:
            value = record.get(prop)
            if value and isinstance(value, str) and value.strip():
                parts.append(f"{PROP_CN.get(prop, prop)}：{value.strip()}")
        return "，".join(parts)
    return f"{TYPE_CN.get(entity_type, entity_type)}：{name}"


def relation_to_text(record: dict) -> str:
    s_type = TYPE_CN.get(record.get("source_type", ""), record.get("source_type", ""))
    t_type = TYPE_CN.get(record.get("target_type", ""), record.get("target_type", ""))
    r_cn = REL_CN.get(record.get("relation_type", ""), record.get("relation_type", ""))
    return f"{s_type}：{record.get('source_name', '')}，{r_cn}，{t_type}：{record.get('target_name', '')}"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(42)

    driver = get_driver()
    try:
        entities = extract_entities(driver)
        entity_texts = []
        for entity_type, records in entities.items():
            for record in records:
                text = entity_to_text(entity_type, record)
                if text:
                    entity_texts.append({"type": "entity", "entity_type": entity_type, "text": text})

        relations = extract_relations(driver)
        relation_texts = []
        for rel_type, records in relations.items():
            for record in records:
                text = relation_to_text(record)
                if text:
                    relation_texts.append({"type": "relation", "relation_type": rel_type, "text": text})

        with open(os.path.join(OUTPUT_DIR, "entities.json"), "w", encoding="utf-8") as f:
            json.dump(entity_texts, f, ensure_ascii=False, indent=2)
        with open(os.path.join(OUTPUT_DIR, "relations.json"), "w", encoding="utf-8") as f:
            json.dump(relation_texts, f, ensure_ascii=False, indent=2)

        logger.info("=== 提取统计 ===")
        logger.info(f"实体文本总数: {len(entity_texts)}")
        for et, records in entities.items():
            logger.info(f"  {et}: {len(records)}")
        logger.info(f"关系文本总数: {len(relation_texts)}")
        for rt, records in relations.items():
            logger.info(f"  {rt}: {len(records)}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
