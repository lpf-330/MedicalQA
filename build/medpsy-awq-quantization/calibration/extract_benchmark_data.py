# -*- coding: utf-8 -*-
"""
CMB专业医学基准数据提取脚本

从HuggingFace下载CMB(Chinese Medical Benchmark)数据集，
提取医学考试题目转换为文本格式，用于AWQ校准。
"""

import json
import logging
import os
import random

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "raw_benchmark")
TARGET_COUNT = 200


def extract_cmb_data() -> list:
    logger.info("加载CMB数据集...")
    configs = ["CMB-Exam", "CMB-Clin"]
    ds = None
    for cfg in configs:
        try:
            logger.info(f"尝试加载CMB config={cfg}...")
            ds = load_dataset("FreedomIntelligence/CMB", cfg, split="train")
            if ds and len(ds) > 0:
                break
        except Exception as e:
            logger.warning(f"CMB {cfg} train失败: {e}")
            try:
                ds = load_dataset("FreedomIntelligence/CMB", cfg, split="test")
                if ds and len(ds) > 0:
                    break
            except Exception as e2:
                logger.warning(f"CMB {cfg} test也失败: {e2}")
                ds = None
    if ds is None:
        logger.error("CMB数据集所有config均不可用")
        return []

    logger.info(f"CMB原始数据量: {len(ds)}")
    texts = []

    for item in ds:
        question = item.get("question", "")
        if not question:
            continue

        parts = [f"问题：{question}"]
        for opt_key in ["option_A", "option_B", "option_C", "option_D", "option_E"]:
            opt_val = item.get(opt_key)
            if opt_val:
                label = opt_key.replace("option_", "")
                parts.append(f"{label}. {opt_val}")

        answer = item.get("answer", "")
        if answer:
            parts.append(f"答案：{answer}")

        explanation = item.get("explanation", "")
        if explanation:
            parts.append(f"解析：{explanation}")

        text = "\n".join(parts)
        texts.append({
            "type": "cmb_question",
            "subject": item.get("subject", ""),
            "text": text,
        })

    logger.info(f"提取CMB题目: {len(texts)} 条")
    return texts


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(42)

    cmb_data = extract_cmb_data()

    if not cmb_data:
        logger.warning("CMB数据提取为空，将生成空文件，后续由MedPsy模板补充")

    if len(cmb_data) > TARGET_COUNT:
        cmb_data = random.sample(cmb_data, TARGET_COUNT)
        logger.info(f"采样到 {TARGET_COUNT} 条")

    with open(os.path.join(OUTPUT_DIR, "cmb_data.json"), "w", encoding="utf-8") as f:
        json.dump(cmb_data, f, ensure_ascii=False, indent=2)

    logger.info(f"=== CMB提取统计 ===")
    logger.info(f"总数: {len(cmb_data)}")


if __name__ == "__main__":
    main()
