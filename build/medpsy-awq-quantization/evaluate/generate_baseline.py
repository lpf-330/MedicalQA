# -*- coding: utf-8 -*-
"""
基线采集脚本 — 通过SGLang HTTP接口调用原模型采集基线输出
"""

import json
import logging
import os
import time

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
TEST_CASES_DIR = os.path.join(BASE_DIR, "test_cases")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SGLANG_URL = "http://localhost:30001/v1/chat/completions"
MEDPSY_SYSTEM_PROMPT = "你是一位全科医生，擅长精炼评估。请在3秒内、不超过50字完成思考，然后直接输出JSON。"

HEALTH_DIMENSIONS = {
    "D1": {"name": "生理指标", "weight": 0.30},
    "D2": {"name": "用药情况", "weight": 0.20},
    "D3": {"name": "治疗状况", "weight": 0.20},
    "D4": {"name": "饮食状况", "weight": 0.15},
    "D5": {"name": "检查情况", "weight": 0.15},
}


def call_sglang(messages: list, max_tokens: int = 1536, temperature: float = 0.0) -> dict:
    payload = {
        "model": "/home/project/MedicalQA/base_models/MedPsy-4B",
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


def build_dimension_prompt(case: dict) -> str:
    dim_id = case["dimension_id"]
    dim_info = HEALTH_DIMENSIONS.get(dim_id, {"weight": 0.15})
    sub_indicators = case["sub_indicators"]
    format_instruction = (
        f"你是健康评估专家。评估维度: {case['dimension_name']}(权重{dim_info['weight']})\n"
        f"子指标: {', '.join(sub_indicators)}\n\n"
        "严格按以下JSON格式输出，不要输出任何其他内容:\n"
        '{"dimension_score":0.72,"sub_indicator_scores":{"指标名":0.65},'
        '"dimension_reasoning":"总体评估"}\n\n'
        "对每个子指标评分(0-1)。\n\n以下是评估依据:\n"
    )
    user_profile_str = json.dumps(case["user_profile"], ensure_ascii=False)
    anomalies_str = json.dumps(case["anomalies"], ensure_ascii=False)
    data_section = f"用户={user_profile_str} 异常={anomalies_str} 风险={{}}\n"
    knowledge_str = json.dumps(case.get("knowledge", {}), ensure_ascii=False)
    knowledge_section = f"知识={knowledge_str}"
    return format_instruction + data_section + knowledge_section


def build_risk_factor_prompt(case: dict) -> str:
    format_instruction = (
        f"你是健康评估专家。评估风险因子: {case['factor_name']}(权重{case['weight']})\n\n"
        "严格按以下JSON格式输出，不要输出任何其他内容:\n"
        '{"factor_score":45,"factor_reasoning":"评估理由","related_diseases":["疾病1"]}'
        "\n\n评估风险程度(0-100)并给简短理由(20字内)。\n\n以下是评估依据:\n"
    )
    user_profile_str = json.dumps(case["user_profile"], ensure_ascii=False)
    anomalies_str = json.dumps(case["anomalies"], ensure_ascii=False)
    entities_str = json.dumps(case.get("medical_entities", {}), ensure_ascii=False)
    data_section = f"用户={user_profile_str} 异常={anomalies_str} 实体={entities_str}"
    return format_instruction + data_section


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
            result = call_sglang(messages)
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

    logger.info(f"维度评估用例: {len(dim_cases)}, 风险因子用例: {len(rf_cases)}")

    dim_results = evaluate_cases(dim_cases, build_dimension_prompt, "dimension")
    rf_results = evaluate_cases(rf_cases, build_risk_factor_prompt, "risk_factor")

    baseline = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": "MedPsy-4B (float16 original)",
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

    output_path = os.path.join(RESULTS_DIR, "baseline_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)

    logger.info(f"基线采集完成: {output_path}")
    logger.info(f"维度: {baseline['stats']['dimension_success']}/{baseline['stats']['dimension_total']}")
    logger.info(f"风险因子: {baseline['stats']['risk_factor_success']}/{baseline['stats']['risk_factor_total']}")


if __name__ == "__main__":
    main()
