# -*- coding: utf-8 -*-
"""
对比分析脚本 — 对比原模型基线和量化模型输出
"""

import json
import logging
import math
import os
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def extract_json(text: str) -> dict | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def pearson_correlation(x: list, y: list) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def mean_absolute_error(x: list, y: list) -> float:
    if len(x) != len(y) or len(x) == 0:
        return float("inf")
    return sum(abs(xi - yi) for xi, yi in zip(x, y)) / len(x)


def compare_dimension_results(baseline: list, quantized: list) -> dict:
    baseline_by_id = {r["id"]: r for r in baseline if r.get("error") is None}
    quantized_by_id = {r["id"]: r for r in quantized if r.get("error") is None}
    common_ids = set(baseline_by_id.keys()) & set(quantized_by_id.keys())

    json_correct_b = 0
    json_correct_q = 0
    baseline_scores = []
    quantized_scores = []
    baseline_sub_cov = []
    quantized_sub_cov = []
    detail_results = []

    for cid in sorted(common_ids):
        b, q = baseline_by_id[cid], quantized_by_id[cid]
        b_json = extract_json(b.get("response", ""))
        q_json = extract_json(q.get("response", ""))
        b_valid, q_valid = b_json is not None, q_json is not None

        if b_valid: json_correct_b += 1
        if q_valid: json_correct_q += 1

        b_score = b_json.get("dimension_score") if b_valid else None
        q_score = q_json.get("dimension_score") if q_valid else None

        if b_score is not None and q_score is not None:
            try:
                baseline_scores.append(float(b_score))
                quantized_scores.append(float(q_score))
            except (ValueError, TypeError):
                pass

        b_sub = b_json.get("sub_indicator_scores", {}) if b_valid else {}
        q_sub = q_json.get("sub_indicator_scores", {}) if q_valid else {}
        if isinstance(b_sub, dict): baseline_sub_cov.append(len(b_sub))
        if isinstance(q_sub, dict): quantized_sub_cov.append(len(q_sub))

        detail_results.append({
            "id": cid, "baseline_valid_json": b_valid, "quantized_valid_json": q_valid,
            "baseline_score": b_score, "quantized_score": q_score,
            "baseline_sub_count": len(b_sub) if isinstance(b_sub, dict) else 0,
            "quantized_sub_count": len(q_sub) if isinstance(q_sub, dict) else 0,
        })

    total = len(common_ids) or 1
    return {
        "total_common_cases": len(common_ids),
        "json_correct_rate_baseline": json_correct_b / total,
        "json_correct_rate_quantized": json_correct_q / total,
        "json_correct_rate_diff": (json_correct_q - json_correct_b) / total,
        "pearson": pearson_correlation(baseline_scores, quantized_scores) if len(baseline_scores) >= 2 else None,
        "mae": mean_absolute_error(baseline_scores, quantized_scores) if baseline_scores else None,
        "avg_sub_coverage_baseline": sum(baseline_sub_cov) / len(baseline_sub_cov) if baseline_sub_cov else None,
        "avg_sub_coverage_quantized": sum(quantized_sub_cov) / len(quantized_sub_cov) if quantized_sub_cov else None,
        "score_pairs_count": len(baseline_scores),
        "details": detail_results,
    }


def compare_risk_factor_results(baseline: list, quantized: list) -> dict:
    baseline_by_id = {r["id"]: r for r in baseline if r.get("error") is None}
    quantized_by_id = {r["id"]: r for r in quantized if r.get("error") is None}
    common_ids = set(baseline_by_id.keys()) & set(quantized_by_id.keys())

    json_correct_b = 0
    json_correct_q = 0
    baseline_scores = []
    quantized_scores = []
    detail_results = []

    for cid in sorted(common_ids):
        b, q = baseline_by_id[cid], quantized_by_id[cid]
        b_json = extract_json(b.get("response", ""))
        q_json = extract_json(q.get("response", ""))
        b_valid, q_valid = b_json is not None, q_json is not None

        if b_valid: json_correct_b += 1
        if q_valid: json_correct_q += 1

        b_score = b_json.get("factor_score") if b_valid else None
        q_score = q_json.get("factor_score") if q_valid else None

        if b_score is not None and q_score is not None:
            try:
                baseline_scores.append(float(b_score))
                quantized_scores.append(float(q_score))
            except (ValueError, TypeError):
                pass

        detail_results.append({
            "id": cid, "baseline_valid_json": b_valid, "quantized_valid_json": q_valid,
            "baseline_score": b_score, "quantized_score": q_score,
        })

    total = len(common_ids) or 1
    return {
        "total_common_cases": len(common_ids),
        "json_correct_rate_baseline": json_correct_b / total,
        "json_correct_rate_quantized": json_correct_q / total,
        "json_correct_rate_diff": (json_correct_q - json_correct_b) / total,
        "pearson": pearson_correlation(baseline_scores, quantized_scores) if len(baseline_scores) >= 2 else None,
        "mae": mean_absolute_error(baseline_scores, quantized_scores) if baseline_scores else None,
        "score_pairs_count": len(baseline_scores),
        "details": detail_results,
    }


def compare_speed(baseline: list, quantized: list) -> dict:
    b_times = [r["elapsed_seconds"] for r in baseline if r.get("elapsed_seconds") is not None]
    q_times = [r["elapsed_seconds"] for r in quantized if r.get("elapsed_seconds") is not None]
    b_avg = sum(b_times) / len(b_times) if b_times else None
    q_avg = sum(q_times) / len(q_times) if q_times else None
    return {
        "baseline_avg_seconds": round(b_avg, 2) if b_avg else None,
        "quantized_avg_seconds": round(q_avg, 2) if q_avg else None,
        "speed_ratio": round(b_avg / q_avg, 2) if b_avg and q_avg and q_avg > 0 else None,
    }


def main():
    baseline_path = os.path.join(RESULTS_DIR, "baseline_results.json")
    quantized_path = os.path.join(RESULTS_DIR, "quantized_results.json")

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    with open(quantized_path, "r", encoding="utf-8") as f:
        quantized_data = json.load(f)

    dim_comp = compare_dimension_results(baseline_data["dimension_results"], quantized_data["dimension_results"])
    rf_comp = compare_risk_factor_results(baseline_data["risk_factor_results"], quantized_data["risk_factor_results"])

    all_b = baseline_data["dimension_results"] + baseline_data["risk_factor_results"]
    all_q = quantized_data["dimension_results"] + quantized_data["risk_factor_results"]
    speed = compare_speed(all_b, all_q)

    logger.info("=== 维度评估对比 ===")
    logger.info(f"JSON正确率: 原模型{dim_comp['json_correct_rate_baseline']:.2%} vs 量化{dim_comp['json_correct_rate_quantized']:.2%} (差异{dim_comp['json_correct_rate_diff']:+.2%})")
    if dim_comp["pearson"] is not None:
        logger.info(f"Pearson: {dim_comp['pearson']:.4f}, MAE: {dim_comp['mae']:.4f}")

    logger.info("=== 风险因子对比 ===")
    logger.info(f"JSON正确率: 原模型{rf_comp['json_correct_rate_baseline']:.2%} vs 量化{rf_comp['json_correct_rate_quantized']:.2%} (差异{rf_comp['json_correct_rate_diff']:+.2%})")
    if rf_comp["pearson"] is not None:
        logger.info(f"Pearson: {rf_comp['pearson']:.4f}, MAE: {rf_comp['mae']:.4f}")

    logger.info("=== 速度对比 ===")
    if speed["baseline_avg_seconds"]:
        logger.info(f"原模型: {speed['baseline_avg_seconds']}s, 量化: {speed['quantized_avg_seconds']}s, 比率: {speed['speed_ratio']}x")

    quality = "PASS"
    issues = []
    if dim_comp["pearson"] is not None and dim_comp["pearson"] < 0.80:
        quality = "FAIL"; issues.append(f"维度Pearson={dim_comp['pearson']:.4f}<0.80")
    if dim_comp["mae"] is not None and dim_comp["mae"] > 0.15:
        quality = "FAIL"; issues.append(f"维度MAE={dim_comp['mae']:.4f}>0.15")
    if rf_comp["pearson"] is not None and rf_comp["pearson"] < 0.80:
        quality = "FAIL"; issues.append(f"风险因子Pearson={rf_comp['pearson']:.4f}<0.80")
    if rf_comp["mae"] is not None and rf_comp["mae"] > 0.15:
        quality = "FAIL"; issues.append(f"风险因子MAE={rf_comp['mae']:.4f}>0.15")
    if dim_comp["json_correct_rate_diff"] < -0.05:
        quality = "FAIL"; issues.append(f"维度JSON正确率下降{dim_comp['json_correct_rate_diff']:.2%}")

    logger.info(f"=== 质量评估: {quality} {'(' + ', '.join(issues) + ')' if issues else ''} ===")

    comparison = {
        "dimension_comparison": dim_comp,
        "risk_factor_comparison": rf_comp,
        "speed_comparison": speed,
        "quality_assessment": {"result": quality, "issues": issues},
    }

    output_path = os.path.join(RESULTS_DIR, "comparison_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    logger.info(f"对比报告: {output_path}")


if __name__ == "__main__":
    main()
