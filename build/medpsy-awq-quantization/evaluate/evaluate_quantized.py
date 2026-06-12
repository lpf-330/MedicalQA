# -*- coding: utf-8 -*-
"""
量化模型评估脚本 — 通过SGLang HTTP接口调用量化模型采集输出
"""

import json
import logging
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(__file__))
from generate_baseline import (
    build_dimension_prompt, build_risk_factor_prompt,
    call_sglang, MEDPSY_SYSTEM_PROMPT, HEALTH_DIMENSIONS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
TEST_CASES_DIR = os.path.join(BASE_DIR, "test_cases")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SGLANG_URL = "http://localhost:30001/v1/chat/completions"
QUANTIZED_MODEL_NAME = "/home/project/MedicalQA/build/medpsy-awq-quantization/output/MedPsy-4B-AWQ"


def call_sglang_quantized(messages: list, max_tokens: int = 1536, temperature: float = 0.0) -> dict:
    payload = {
        "model": QUANTIZED_MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(SGLANG_URL, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    full_content = (reasoning + content).strip() if not content else content
    usage = data.get("usage", {})
    return {"content": full_content, "reasoning": reasoning, "usage": usage}


def evaluate_cases(cases: list, prompt_builder, result_type: str) -> list:
    results = []
    for case in cases:
        prompt = prompt_builder(case)
        messages = [
            {"role": "system", "content": MEDPSY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        case_id = case["id"]
        logger.info(f"评估 {case_id}...")
        try:
            start = time.time()
            result = call_sglang_quantized(messages)
            elapsed = time.time() - start
            entry = {
                "id": case_id, "type": result_type, "prompt": prompt,
                "response": result["content"], "elapsed_seconds": round(elapsed, 2),
                "usage": result["usage"], "error": None,
            }
            if result_type == "dimension":
                entry["dimension_id"] = case["dimension_id"]
                entry["dimension_name"] = case["dimension_name"]
            else:
                entry["factor_id"] = case["factor_id"]
                entry["factor_name"] = case["factor_name"]
            results.append(entry)
            logger.info(f"  完成: {elapsed:.2f}s, response_len={len(result['content'])}")
        except Exception as e:
            logger.error(f"  失败: {e}")
            entry = {
                "id": case_id, "type": result_type, "prompt": prompt,
                "response": None, "elapsed_seconds": None, "usage": None, "error": str(e),
            }
            results.append(entry)
        time.sleep(0.5)
    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(os.path.join(TEST_CASES_DIR, "dimension_eval_cases.json"), "r", encoding="utf-8") as f:
        dim_cases = json.load(f)
    with open(os.path.join(TEST_CASES_DIR, "risk_factor_cases.json"), "r", encoding="utf-8") as f:
        rf_cases = json.load(f)

    dim_results = evaluate_cases(dim_cases, build_dimension_prompt, "dimension")
    rf_results = evaluate_cases(rf_cases, build_risk_factor_prompt, "risk_factor")

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": "MedPsy-4B-AWQ (4-bit quantized)",
        "sglang_url": SGLANG_URL,
        "dimension_results": dim_results,
        "risk_factor_results": rf_results,
        "stats": {
            "dimension_total": len(dim_cases),
            "dimension_success": sum(1 for r in dim_results if r["error"] is None),
            "risk_factor_total": len(rf_cases),
            "risk_factor_success": sum(1 for r in rf_results if r["error"] is None),
        },
    }

    output_path = os.path.join(RESULTS_DIR, "quantized_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"量化模型评估完成: {output_path}")
    logger.info(f"维度: {output['stats']['dimension_success']}/{output['stats']['dimension_total']}")
    logger.info(f"风险因子: {output['stats']['risk_factor_success']}/{output['stats']['risk_factor_total']}")


if __name__ == "__main__":
    main()
