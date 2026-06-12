# -*- coding: utf-8 -*-
"""
校准数据集构建脚本

合并Neo4j实体/关系文本、CMB基准题目、MedPsy格式模板，
按比例分层采样生成最终AWQ校准数据集。

分配比例：
- Disease长文本属性: 40%
- 关系文本: 15%
- 其他实体: 10%
- CMB基准: 15%
- MedPsy格式模板: 20%
"""

import json
import logging
import os
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

NEO4J_ENTITIES_PATH = os.path.join(DATA_DIR, "raw_neo4j", "entities.json")
NEO4J_RELATIONS_PATH = os.path.join(DATA_DIR, "raw_neo4j", "relations.json")
CMB_DATA_PATH = os.path.join(DATA_DIR, "raw_benchmark", "cmb_data.json")
MEDPSY_TEMPLATES_PATH = os.path.join(DATA_DIR, "raw_benchmark", "medpsy_templates.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "calibration_dataset.json")

RATIOS = {
    "disease_text": 0.40,
    "relation_text": 0.15,
    "other_entity": 0.10,
    "cmb_benchmark": 0.15,
    "medpsy_template": 0.20,
}

MAX_CALIB_SAMPLES = 128
MAX_CALIB_SEQ_LEN = 512


def load_json(path: str) -> list:
    if not os.path.exists(path):
        logger.warning(f"文件不存在: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_entities(entities: list) -> tuple:
    disease_texts = []
    other_texts = []
    for item in entities:
        if item.get("entity_type") == "Disease":
            disease_texts.append(item)
        else:
            other_texts.append(item)
    return disease_texts, other_texts


def safe_sample(data: list, n: int) -> list:
    if len(data) <= n:
        return list(data)
    return random.sample(data, n)


def build_dataset(total_samples: int = None) -> list:
    entities = load_json(NEO4J_ENTITIES_PATH)
    relations = load_json(NEO4J_RELATIONS_PATH)
    cmb_data = load_json(CMB_DATA_PATH)
    medpsy_templates = load_json(MEDPSY_TEMPLATES_PATH)

    disease_texts, other_entities = classify_entities(entities)

    logger.info("=== 数据源统计 ===")
    logger.info(f"Disease长文本: {len(disease_texts)}")
    logger.info(f"其他实体: {len(other_entities)}")
    logger.info(f"关系文本: {len(relations)}")
    logger.info(f"CMB基准: {len(cmb_data)}")
    logger.info(f"MedPsy模板: {len(medpsy_templates)}")
    total_available = len(disease_texts) + len(other_entities) + len(relations) + len(cmb_data) + len(medpsy_templates)
    logger.info(f"可用总量: {total_available}")

    if total_available == 0:
        logger.error("无可用校准数据！请先运行数据提取脚本。")
        return []

    if total_samples is None:
        total_samples = total_available

    allocations = {}
    for category, ratio in RATIOS.items():
        allocations[category] = max(1, int(total_samples * ratio))

    logger.info("=== 分配方案 ===")
    for cat, count in allocations.items():
        logger.info(f"  {cat}: {count} (比例 {RATIOS[cat]*100:.0f}%)")

    sampled = []
    sampled.extend(safe_sample(disease_texts, allocations["disease_text"]))
    sampled.extend(safe_sample(relations, allocations["relation_text"]))
    sampled.extend(safe_sample(other_entities, allocations["other_entity"]))
    sampled.extend(safe_sample(cmb_data, allocations["cmb_benchmark"]))
    sampled.extend(safe_sample(medpsy_templates, allocations["medpsy_template"]))

    random.shuffle(sampled)

    calibration_texts = [item.get("text", "") for item in sampled if item.get("text")]

    stats = {
        "total_samples": len(calibration_texts),
        "allocations": allocations,
        "source_counts": {
            "disease_text": len(safe_sample(disease_texts, allocations["disease_text"])),
            "relation_text": len(safe_sample(relations, allocations["relation_text"])),
            "other_entity": len(safe_sample(other_entities, allocations["other_entity"])),
            "cmb_benchmark": len(safe_sample(cmb_data, allocations["cmb_benchmark"])),
            "medpsy_template": len(safe_sample(medpsy_templates, allocations["medpsy_template"])),
        },
        "ratios": RATIOS,
        "max_calib_samples": MAX_CALIB_SAMPLES,
        "max_calib_seq_len": MAX_CALIB_SEQ_LEN,
    }

    output = {"metadata": stats, "samples": sampled, "texts": calibration_texts}

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"=== 校准数据集构建完成 ===")
    logger.info(f"总样本数: {len(calibration_texts)}")
    logger.info(f"输出路径: {OUTPUT_PATH}")
    logger.info(f"AutoAWQ: max_calib_samples={MAX_CALIB_SAMPLES}, max_calib_seq_len={MAX_CALIB_SEQ_LEN}")

    return calibration_texts


if __name__ == "__main__":
    random.seed(42)
    build_dataset()
