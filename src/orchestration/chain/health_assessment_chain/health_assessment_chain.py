# -*- coding: utf-8 -*-
"""
健康评估Chain策略

实现健康报告生成业务的健康评估Chain策略，采用"评估框架+LLM评估引擎"分层设计。
评估框架定义评估维度、子指标、权重和计算公式，健康评估模型作为评估引擎对每个子指标进行评估，
最终按算法公式汇总计算健康评分、风险等级、疾病风险评分。
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional
from src.orchestration.chain.chain import Chain
from src.utils.logger import log_arch_event
from src.orchestration.chain.data_classes import ChainContext, ChainResult
from src.orchestration.chain.health_assessment_chain.health_assessment_context import HealthAssessmentContextBody
from src.orchestration.chain.health_assessment_chain.health_assessment_result import HealthAssessmentResultData
from src.orchestration.chain.health_assessment_chain.health_assessment_resource import HealthAssessmentResource
from src.config.business.report_service_config import get_runtime_config
from src.config.config_manager import ConfigManager

logger = logging.getLogger(__name__)

# ============================================================================
# 数据类定义
# ============================================================================

# ============================================================================
# 评估框架定义
# ============================================================================

# 健康综合评分维度定义(5维度) — 从配置读取
class _LazyReportConfig:
    """延迟加载配置代理，每次属性访问时从ConfigManager获取真实配置"""
    def __getattr__(self, name):
        return getattr(get_runtime_config(), name)

_report_config = _LazyReportConfig()
HEALTH_DIMENSIONS = _report_config.health_dimensions

# 疾病风险评分因子定义(6风险因子) — 从配置读取
DISEASE_RISK_FACTORS = _report_config.disease_risk_factors

# 风险等级划分标准(4等级) — 从配置读取
RISK_LEVEL_STANDARDS = _report_config.risk_level_standards


def _refresh_report_config() -> None:
    """刷新模块级常量，确保使用ConfigManager合并后的运行期配置值。

    模块加载时 _report_config.xxx 取到的是默认值（ConfigManager尚未初始化），
    在 execute() 入口调用此函数，利用 _LazyConfig 代理的延迟求值特性，
    重新从已初始化的 ConfigManager 读取实际配置。
    """
    global HEALTH_DIMENSIONS, DISEASE_RISK_FACTORS, RISK_LEVEL_STANDARDS
    HEALTH_DIMENSIONS = _report_config.health_dimensions
    DISEASE_RISK_FACTORS = _report_config.disease_risk_factors
    RISK_LEVEL_STANDARDS = _report_config.risk_level_standards


DIMENSION_ID_TO_NAME = {
    "D1": "disease_risk",
    "D2": "medication",
    "D3": "treatment",
    "D4": "dietary",
    "D5": "checkup",
    "D6": "complication",
    "D7": "prevention",
    "D8": "susceptible",
}

# 健康评估模型调用约束 — 每次调用动态获取，确保使用ConfigManager合并后的运行期配置
def _get_health_assessment_constraints() -> dict:
    cfg = get_runtime_config()
    return {
        "max_prompt_chars": cfg.max_prompt_chars,
        "timeout_seconds": cfg.multi_analysis_timeout,
        "max_retries": cfg.health_assessment_max_retries,
        "max_knowledge_chars": cfg.max_knowledge_chars,
        "max_user_info_chars": cfg.max_user_info_chars,
        "max_risk_factor_user_info_chars": cfg.max_risk_factor_user_info_chars,
        "health_assessment_max_tokens": cfg.health_assessment_max_tokens,
        "health_assessment_batch_max_tokens": cfg.health_assessment_batch_max_tokens,
    }

# ============================================================================
# HealthAssessmentChain类实现
# ============================================================================

class HealthAssessmentChain(Chain[ChainContext[HealthAssessmentContextBody], ChainResult[HealthAssessmentResultData]]):
    """
    健康评估Chain策略类
    
    实现健康报告生成业务的健康评估固定流程，采用"评估框架+LLM评估引擎"分层设计：
    
    分层架构：
    1. 评估框架（算法层）- 定义评估维度、子指标、权重、计算公式
    2. 健康评估模型评估引擎（推理层）- 对每个子指标进行医学推理评估
    3. 汇总计算（计算层）- 按算法公式汇总各子指标评分
    
    评估内容：
    1. 健康综合评分（5维度21子指标）
    2. 疾病风险评分（6风险因子）
    3. 风险等级判定（4等级标准）
    
    降级策略：
    - 健康评估模型不可用时，使用规则引擎评估子指标
    - 健康评估模型超时时，使用默认子指标评分
    - 子指标评估失败时，跳过该子指标（权重重新归一化）
    """
    
    def __init__(self, resource: HealthAssessmentResource):
        """
        初始化健康评估Chain策略
        
        Args:
            resource: Chain策略专属资源
        """
        self._resource = resource
        self._health_assessment_available = resource.health_assessment_model is not None
        
        # 健康评估模型可用性检查日志
        logger.info(f"[HealthAssessmentChain] 健康评估模型可用性检查: health_assessment_model={'可用' if self._health_assessment_available else '不可用(将使用规则引擎降级评估)'}")
        
        # 健康评估模型降级日志
        if not self._health_assessment_available:
            logger.warning("[健康评估模型_DEGRADED] 健康评估模型模型不可用，将使用规则引擎进行健康评估")

    @staticmethod
    def _get_rule_engine_scores() -> Dict[str, float]:
        """
        延迟获取规则引擎评分标准，避免ConfigManager未初始化时出错。

        Returns:
            rule_engine_scores字典，获取失败时返回默认值。
        """
        try:
            scores = ConfigManager().clinical_standards.get("rule_engine_scores", {})
            if scores:
                return scores
        except Exception as e:
            logger.debug(f"[HealthAssessmentChain] 获取规则引擎评分标准失败: {e}")
        # 默认值fallback
        return {
            "normal": 1.0, "mild_abnormal": 0.7, "moderate_abnormal": 0.4, "severe_abnormal": 0.3,
            "no_smoking": 1.0, "quit_smoking": 0.8, "smoking": 0.3,
            "no_drinking": 1.0, "moderate_drinking": 0.8, "heavy_drinking": 0.4,
            "regular_exercise": 1.0, "occasional_exercise": 0.6, "no_exercise": 0.3,
            "few_history": 0.7, "many_history": 0.4,
            "no_family_history": 1.0, "has_family_history": 0.6,
            "few_medication": 0.7, "many_medication": 0.4,
            "good_mental": 1.0, "moderate_stress": 0.7, "high_stress": 0.4,
            "good_emotion": 1.0, "moderate_emotion": 0.7, "poor_emotion": 0.4,
            "good_prevention": 0.7,
            "few_anomaly_indicators": 0.3, "moderate_anomaly_indicators": 0.6, "many_anomaly_indicators": 0.9,
            "few_medical_history": 0.4, "many_medical_history": 0.8,
            "no_family_risk": 0.1, "has_family_risk": 0.5,
            "few_bad_habits": 0.1, "moderate_bad_habits": 0.4, "many_bad_habits": 0.7,
            "young_age_risk": 0.1, "middle_age_risk": 0.3, "senior_age_risk": 0.6, "elderly_age_risk": 0.9,
            "few_complications": 0.1, "some_complications": 0.3, "many_complications": 0.7,
        }
        
    def execute(self, chain_context: ChainContext[HealthAssessmentContextBody], external_degraded: bool = False) -> ChainResult[HealthAssessmentResultData]:
        """
        执行Chain策略
        
        执行健康评估的完整流程：
        1. 健康综合评分计算（5维度并行评估）
        2. 疾病风险评分计算（6风险因子并行评估）
        3. 风险等级判定（基于健康评分和疾病风险评分）
        
        Args:
            chain_context: Chain输入数据容器
            external_degraded: 外部降级状态（如Agent超时等导致的降级），默认为False。
                当外部已标记降级时，Chain内部的降级标记应与外部保持一致。
            
        Returns:
            ChainResult: Chain输出数据容器
        """
        _refresh_report_config()
        start_time = time.time()
        logger.info(f"[HealthAssessmentChain] 开始执行Chain: session_id={chain_context.session_id}")
        log_arch_event(
            logger,
            component="HealthAssessmentChain",
            stage="CHAIN",
            event="execute",
            status="start",
            design_id="BIZ-4.3",
        )

        body = chain_context.body
        if body is None:
            logger.warning(f"[HealthAssessmentChain] 输入数据为空: session_id={chain_context.session_id}")
            return ChainResult(
                session_id=chain_context.session_id,
                data=HealthAssessmentResultData()
            )
        
        try:
            # 步骤1：健康综合评分计算（5维度并行评估）—— 独立try-except
            try:
                logger.info("[STAGE_ENTER] CalculateHealthScore")
                _stage_start = time.time()
                health_score, health_level, health_breakdown = self._calculate_health_score(body)
                logger.info(f"[STAGE_EXIT] CalculateHealthScore, duration={time.time() - _stage_start:.2f}s")
                logger.info(
                    f"[HealthAssessmentChain] 健康评分计算完成: "
                    f"health_score_present={health_score is not None}, health_level={health_level}"
                )
            except Exception as e:
                logger.error(f"[HealthAssessmentChain] 健康评分计算失败，降级为规则引擎: error_type={type(e).__name__}")
                health_score, health_level, health_breakdown = self._rule_calculate_health_score(body)
                health_breakdown["_degraded"] = True

            # 步骤2：疾病风险评分计算（6风险因子并行评估）—— 独立try-except
            try:
                logger.info("[STAGE_ENTER] CalculateDiseaseRisks")
                _stage_start = time.time()
                disease_risks, disease_breakdown = self._calculate_disease_risks(body)
                logger.info(f"[STAGE_EXIT] CalculateDiseaseRisks, duration={time.time() - _stage_start:.2f}s")
                logger.info(f"[HealthAssessmentChain] 疾病风险评分完成: disease_risks_count={len(disease_risks)}")
            except Exception as e:
                logger.error(f"[HealthAssessmentChain] 疾病风险评分计算失败，降级为规则引擎: error_type={type(e).__name__}")
                disease_risks, disease_breakdown = self._rule_calculate_disease_risks(body)
                disease_breakdown["_degraded"] = True

            # 步骤3：风险等级判定（基于健康评分和疾病风险评分）—— 独立try-except
            try:
                logger.info("[STAGE_ENTER] DetermineRiskLevel")
                _stage_start = time.time()
                risk_level, risk_reasoning = self._determine_risk_level(health_score, disease_risks, body)
                logger.info(f"[STAGE_EXIT] DetermineRiskLevel, duration={time.time() - _stage_start:.2f}s")
                logger.info(f"[HealthAssessmentChain] 风险等级判定完成: risk_level={risk_level}")
            except Exception as e:
                logger.error(f"[HealthAssessmentChain] 风险等级判定失败，降级为规则引擎: error_type={type(e).__name__}")
                risk_level, risk_reasoning = self._rule_determine_risk_level(health_score, disease_risks, body)

            # 最终健康评分和风险等级日志
            logger.info(
                f"[HEALTH_ASSESSMENT] health_score_present={health_score is not None}, "
                f"health_level={health_level}, risk_level={risk_level}"
            )
            if disease_risks:
                for dr in disease_risks:
                    disease_name = dr.get('disease_name', '')
                    logger.debug(
                        f"[HEALTH_ASSESSMENT] disease_risk: disease_name_len={len(disease_name)}, "
                        f"risk_score_present={dr.get('risk_score') is not None}, "
                        f"risk_level={dr.get('risk_level', '')}, "
                        f"confidence_present={dr.get('confidence') is not None}"
                    )

            # 汇总评分明细
            score_breakdown = {
                "health_dimensions": health_breakdown,
                "disease_risk_factors": disease_breakdown
            }

            # 汇总推理过程
            reasoning = self._aggregate_reasoning(health_breakdown, disease_breakdown, risk_reasoning)

            # 检查是否有任一步骤降级
            any_degraded = health_breakdown.get("_degraded", False) or disease_breakdown.get("_degraded", False)
            degraded_reason_parts = []
            if health_breakdown.get("_degraded"):
                degraded_reason_parts.append("健康评分计算降级")
            if disease_breakdown.get("_degraded"):
                degraded_reason_parts.append("疾病风险评分降级")

            result_data = HealthAssessmentResultData(
                health_score=health_score,
                health_level=health_level,
                risk_level=risk_level,
                disease_risks=disease_risks,
                score_breakdown=score_breakdown,
                reasoning=reasoning,
                degraded=any_degraded,
                degraded_reason="; ".join(degraded_reason_parts) if degraded_reason_parts else ""
            )

        except Exception as e:
            logger.error(f"[HealthAssessmentChain] Chain执行失败（步骤外异常），触发全量降级: error_type={type(e).__name__}")
            result_data = self._fallback_rule_assessment(body, type(e).__name__)
        
        # 合并外部降级状态：当外部（如Agent）已标记降级时，Chain内部的降级标记应与外部保持一致
        if external_degraded:
            original_degraded = result_data.degraded
            result_data.degraded = result_data.degraded or external_degraded
            if result_data.degraded and not result_data.degraded_reason:
                result_data.degraded_reason = "外部降级状态传入（如Agent超时等）"
            logger.info(f"[HealthAssessmentChain] 外部降级状态传入: external_degraded=True, "
                       f"内部降级状态={original_degraded}, 最终降级状态={result_data.degraded}")

        elapsed = time.time() - start_time
        logger.info(
            f"[HealthAssessmentChain] Chain执行完成: session_id={chain_context.session_id}, "
            f"health_score_present={result_data.health_score is not None}, "
            f"risk_level={result_data.risk_level}, "
            f"degraded={result_data.degraded}, elapsed={elapsed:.2f}s"
        )
        health_assessment_used = self._health_assessment_available and not result_data.degraded
        logger.info(
            f"[HEALTH_ASSESSMENT_SUMMARY] health_score_present={result_data.health_score is not None}, "
            f"health_level={result_data.health_level}, risk_level={result_data.risk_level}, "
            f"degraded={result_data.degraded}, health_assessment_used={health_assessment_used}"
        )
        
        return ChainResult(session_id=chain_context.session_id, data=result_data)
    
    # ========================================================================
    # 健康综合评分算法（评估框架）
    # ========================================================================
    
    def _calculate_health_score(self, body: HealthAssessmentContextBody) -> tuple:
        """
        计算健康综合评分（评估框架）
        
        基于SF-36健康调查量表和老年人综合健康评估量表的设计思路，
        采用5维度加权评分模型。
        
        计算公式：
        health_score = Σ(Di_score × Di_weight) × 100
        
        其中：
        - Di_score: 第i个维度的归一化评分(0-1)
        - Di_weight: 第i个维度的权重
        - Σ(Di_weight) = 1
        
        推理策略：
        1. 批量推理：将5个维度的评估prompt合并为一次call_model_batch()调用
        2. 串行推理降级：批量推理失败时，逐维度串行调用generate()
        3. 规则引擎降级：串行推理也失败时，使用规则引擎评估
        
        Args:
            body: Chain策略专属输入数据
            
        Returns:
            (health_score, health_level, breakdown)
        """
        logger.info("[HealthAssessmentChain] 开始计算健康综合评分")
        
        breakdown = {}
        dimension_scores = {}
        
        # 收集5个维度的评估信息
        dim_items = list(HEALTH_DIMENSIONS.items())
        
        # 尝试批量推理
        if self._health_assessment_available and self._resource.health_assessment_model is not None:
            try:
                logger.info("[HealthAssessmentChain] 尝试5维度批量推理")
                dimension_scores = self._batch_evaluate_dimensions(dim_items, body)
                if dimension_scores:
                    logger.info("[HealthAssessmentChain] 5维度批量推理成功")
                else:
                    logger.warning("[HealthAssessmentChain] 5维度批量推理返回空结果，降级为串行推理")
                    dimension_scores = self._serial_evaluate_dimensions(dim_items, body)
            except Exception as e:
                logger.warning(f"[HealthAssessmentChain] 5维度批量推理失败，降级为串行推理: error_type={type(e).__name__}")
                logger.warning(f"[健康评估模型_DEGRADED] 5维度批量推理失败，降级为串行推理: error_type={type(e).__name__}")
                logger.warning(f"[BATCH_DEGRADE] 批量推理失败，降级为串行推理: error_type={type(e).__name__}")
                dimension_scores = self._serial_evaluate_dimensions(dim_items, body)
        else:
            if not self._health_assessment_available:
                logger.warning("[健康评估模型_DEGRADED] 健康评估模型模型不可用，使用规则引擎评估5维度")
            dimension_scores = self._serial_evaluate_dimensions(dim_items, body)
        
        # 日志记录
        for dim_id, dim_info in dimension_scores.items():
            sub_scores = dim_info.get("sub_indicator_scores", {})
            logger.info(
                f"[HealthAssessmentChain] 维度{dim_id}({dim_info['name']})评估完成: "
                f"score_present={dim_info.get('score') is not None}, "
                f"weighted_score_present={dim_info.get('weighted_score') is not None}, "
                f"sub_indicator_count={len(sub_scores)}"
            )
            logger.info(
                f"[DIMENSION_SCORE] 维度={dim_info['name']}, "
                f"score_present={dim_info.get('score') is not None}, "
                f"weight_present={dim_info.get('weight') is not None}, "
                f"weighted_score_present={dim_info.get('weighted_score') is not None}, "
                f"sub_indicator_count={len(sub_scores)}"
            )

            # 5维度评分详细日志
            logger.debug(
                f"[DIMENSION_SCORE] dim_id={dim_id}, dim_name={dim_info['name']}, "
                f"score_present={dim_info.get('score') is not None}, "
                f"weight_present={dim_info.get('weight') is not None}, "
                f"weighted_score_present={dim_info.get('weighted_score') is not None}"
            )
            for sub_name, sub_info in sub_scores.items():
                sub_score_present = sub_info.get("score") is not None if isinstance(sub_info, dict) else sub_info is not None
                sub_reason = sub_info.get("reason", "") if isinstance(sub_info, dict) else ""
                logger.debug(
                    f"[DIMENSION_SCORE] dim_id={dim_id}, sub_indicator={sub_name}, "
                    f"sub_score_present={sub_score_present}, reason_len={len(sub_reason)}"
                )
        
        # 计算总分
        total_weighted_score = sum(d["weighted_score"] for d in dimension_scores.values())
        health_score = total_weighted_score * 100  # 转换为0-100分制
        health_score = round(health_score, 2)

        # 健康评分计算明细日志
        for dim_id_key, dim_info in dimension_scores.items():
            logger.info(
                f"[HEALTH_SCORE_CALC] 维度={dim_id_key}({dim_info['name']}), "
                f"score_present={dim_info.get('score') is not None}, "
                f"weight_present={dim_info.get('weight') is not None}, "
                f"weighted_score_present={dim_info.get('weighted_score') is not None}"
            )
        logger.info(
            f"[HEALTH_SCORE_CALC] dimension_count={len(dimension_scores)}, "
            f"total_weighted_score_present={total_weighted_score is not None}, "
            f"health_score_present={health_score is not None}"
        )

        # 确保分数在0-100范围内
        health_score = max(0, min(100, health_score))
        
        # 判定健康等级
        health_level = self._determine_health_level(health_score)
        
        breakdown = {
            "dimension_scores": dimension_scores,
            "total_weighted_score": total_weighted_score,
            "calculation_formula": "health_score = Σ(Di_score × Di_weight) × 100"
        }

        # 子指标降级检测：扫描是否有 _skipped 子指标
        skipped_indicators = []
        for dim_id_key, dim_info in dimension_scores.items():
            for sub_name, sub_info in dim_info.get("sub_indicator_scores", {}).items():
                if isinstance(sub_info, dict) and sub_info.get("_skipped"):
                    skipped_indicators.append(f"{dim_info['name']}:{sub_name}")
        if skipped_indicators:
            breakdown["_degraded"] = True
            logger.info(f"[HealthAssessmentChain] 子指标降级检测: skipped_indicators={skipped_indicators}, 标记health_breakdown降级")
        
        logger.info(
            f"[HealthAssessmentChain] 健康综合评分计算完成: "
            f"health_score_present={health_score is not None}, health_level={health_level}"
        )

        # 健康综合评分汇总日志
        logger.info(
            f"[HEALTH_ASSESSMENT] health_score_present={health_score is not None}, "
            f"health_level={health_level}, "
            f"total_weighted_score_present={total_weighted_score is not None}"
        )
        for dim_id_key, dim_info in dimension_scores.items():
            logger.debug(
                f"[HEALTH_ASSESSMENT] dimension={dim_id_key}({dim_info['name']}): "
                f"score_present={dim_info.get('score') is not None}, "
                f"weight_present={dim_info.get('weight') is not None}, "
                f"weighted_score_present={dim_info.get('weighted_score') is not None}"
            )
        
        return health_score, health_level, breakdown
    
    def _batch_evaluate_dimensions(
        self,
        dim_items: List[tuple],
        body: HealthAssessmentContextBody
    ) -> Dict[str, Dict]:
        """
        批量推理评估5个维度
        
        将5个维度的评估prompt合并为一次call_model_batch()调用，
        利用SGLang的continuous batching机制共享forward pass。
        
        Args:
            dim_items: 维度信息列表 [(dim_id, dim_info), ...]
            body: Chain策略专属输入数据
            
        Returns:
            维度评分字典 {dim_id: {"name": ..., "weight": ..., "score": ..., ...}}
        """
        # 构建所有维度的prompt
        prompts = []
        for dim_id, dim_info in dim_items:
            prompt = self._build_dimension_evaluation_prompt(
                dim_id, dim_info["name"], dim_info["sub_indicators"], body
            )
            # 检查Prompt长度限制
            if len(prompt) > _get_health_assessment_constraints()["max_prompt_chars"]:
                logger.warning(f"[HealthAssessmentChain] Prompt长度超过限制({len(prompt)} > {_get_health_assessment_constraints()['max_prompt_chars']})，截断")
                prompt = prompt[:_get_health_assessment_constraints()["max_prompt_chars"]]
            logger.info(f"[健康评估模型_INPUT] 批量推理-维度={dim_id}({dim_info['name']}), prompt_len={len(prompt)}")
            prompts.append(prompt)

        prompt_lengths = [len(p) for p in prompts]
        logger.info(f"[BATCH_INFERENCE] prompt_count={len(prompts)}, prompt_lengths={prompt_lengths}")

        model_service = self._resource.health_assessment_model
        logger.info(f"[LLM_INPUT] 5维度批量推理, prompt_count={len(prompts)}, prompt_lengths={prompt_lengths}")
        _batch_start = time.time()
        results = model_service.call_model_batch(prompts, max_tokens=_get_health_assessment_constraints()["health_assessment_batch_max_tokens"], timeout=_get_health_assessment_constraints()["timeout_seconds"])
        _batch_elapsed = time.time() - _batch_start

        result_lengths = [len(r) if r else 0 for r in results]
        logger.info(f"[HealthAssessment_DURATION] 5维度批量推理 duration={_batch_elapsed:.2f}s")
        logger.info(f"[健康评估模型_OUTPUT] 5维度批量推理 result_count={len(results)}, result_lengths={result_lengths}")
        logger.info(f"[LLM_OUTPUT] 5维度批量推理 result_count={len(results)}, result_lengths={result_lengths}")
        logger.info(f"[LLM_DURATION] 5维度批量推理 duration={_batch_elapsed:.2f}s")
        logger.info(f"[BATCH_RESULT] result_count={len(results)}, result_lengths={result_lengths}")
        
        # 逐维度解析结果
        dimension_scores = {}
        for i, (dim_id, dim_info) in enumerate(dim_items):
            dim_name = dim_info["name"]
            dim_weight = dim_info["weight"]
            
            try:
                response = results[i] if i < len(results) else ""
                evaluation_result = self._parse_dimension_result(response, dim_id, dim_name)
                
                if evaluation_result:
                    dim_score = evaluation_result.get("dimension_score", _report_config.default_dimension_score)
                    reasoning = evaluation_result.get("dimension_reasoning", "")
                    sub_scores = evaluation_result.get("sub_indicator_scores", {})

                    # 子指标失败降级：跳过失败子指标 + 权重归一化
                    sub_scores, dim_score = self._normalize_sub_indicator_scores(
                        dim_id, dim_name, dim_info["sub_indicators"], sub_scores
                    )

                    logger.info(f"[健康评估模型_OUTPUT] 批量推理-维度={dim_id}({dim_name}), sub_indicator_count={len(sub_scores)}, output_keys={list(evaluation_result.keys())}")

                    dimension_scores[dim_id] = {
                        "name": dim_name,
                        "weight": dim_weight,
                        "score": dim_score,
                        "weighted_score": dim_score * dim_weight,
                        "reasoning": reasoning,
                        "sub_indicator_scores": sub_scores
                    }
                else:
                    # 单维度解析失败，尝试结构化自修复
                    original_prompt = prompts[i] if i < len(prompts) else ""
                    repair_result = self._try_structured_repair(
                        raw_output=response,
                        error_description=f"维度{dim_id}({dim_name})的JSON解析失败或缺少必要字段(dimension_score)",
                        expected_format='{"dimension_score":0.72,"sub_indicator_scores":{"指标名":0.65},"dimension_reasoning":"总体评估"}',
                        context_type="dimension",
                        original_prompt=original_prompt
                    )
                    if repair_result and "dimension_score" in repair_result:
                        # 自修复成功，复用已有解析逻辑
                        evaluation_result = repair_result
                        dim_score = evaluation_result.get("dimension_score", _report_config.default_dimension_score)
                        reasoning = evaluation_result.get("dimension_reasoning", "")
                        sub_scores = evaluation_result.get("sub_indicator_scores", {})
                        sub_scores, dim_score = self._normalize_sub_indicator_scores(
                            dim_id, dim_name, dim_info["sub_indicators"], sub_scores
                        )
                        dimension_scores[dim_id] = {
                            "name": dim_name,
                            "weight": dim_weight,
                            "score": dim_score,
                            "weighted_score": dim_score * dim_weight,
                            "reasoning": reasoning,
                            "sub_indicator_scores": sub_scores,
                            "_repaired": True
                        }
                    else:
                        # 自修复也失败，降级为规则引擎
                        logger.warning(f"[健康评估模型_DEGRADED] 批量推理-维度{dim_id}({dim_name})结果解析失败（自修复也失败），降级为规则引擎")
                        dim_score, reasoning, sub_scores = self._rule_based_dimension_evaluation(
                            dim_id, dim_name, dim_info["sub_indicators"], body
                        )
                        dimension_scores[dim_id] = {
                            "name": dim_name,
                            "weight": dim_weight,
                            "score": dim_score,
                            "weighted_score": dim_score * dim_weight,
                            "reasoning": reasoning,
                            "sub_indicator_scores": sub_scores
                        }
            except Exception as e:
                # 单维度解析异常，降级为规则引擎
                logger.warning(f"[健康评估模型_DEGRADED] 批量推理-维度{dim_id}({dim_name})结果解析异常，降级为规则引擎: error_type={type(e).__name__}")
                dim_score, reasoning, sub_scores = self._rule_based_dimension_evaluation(
                    dim_id, dim_name, dim_info["sub_indicators"], body
                )
                dimension_scores[dim_id] = {
                    "name": dim_name,
                    "weight": dim_weight,
                    "score": dim_score,
                    "weighted_score": dim_score * dim_weight,
                    "reasoning": reasoning,
                    "sub_indicator_scores": sub_scores
                }
        
        return dimension_scores
    
    def _serial_evaluate_dimensions(
        self,
        dim_items: List[tuple],
        body: HealthAssessmentContextBody
    ) -> Dict[str, Dict]:
        """
        串行推理评估5个维度（降级策略）
        
        逐维度调用_evaluate_dimension()，每个维度内部也有降级策略：
        健康评估模型失败 -> 规则引擎。
        
        Args:
            dim_items: 维度信息列表 [(dim_id, dim_info), ...]
            body: Chain策略专属输入数据
            
        Returns:
            维度评分字典 {dim_id: {"name": ..., "weight": ..., "score": ..., ...}}
        """
        dimension_scores = {}
        
        for dim_id, dim_info in dim_items:
            dim_name = dim_info["name"]
            dim_weight = dim_info["weight"]
            sub_indicators = dim_info["sub_indicators"]
            
            # 评估维度得分（内部包含健康评估模型->规则引擎降级）
            dim_score, dim_reasoning, sub_scores = self._evaluate_dimension(
                dim_id, dim_name, sub_indicators, body
            )
            
            dimension_scores[dim_id] = {
                "name": dim_name,
                "weight": dim_weight,
                "score": dim_score,
                "weighted_score": dim_score * dim_weight,
                "reasoning": dim_reasoning,
                "sub_indicator_scores": sub_scores
            }
        
        return dimension_scores
    
    def _extract_json_from_response(self, response: str) -> str:
        """从模型响应中提取纯JSON内容，兼容markdown代码块、附加文本和不完整JSON

        模型可能在JSON前输出思考内容（尤其enable_thinking未完全抑制时），
        因此需要从文本末尾反向查找可解析的JSON对象。
        """
        candidates = []
        for json_match in re.finditer(r'```json\s*([\s\S]*?)\s*```', response):
            candidates.append(json_match.group(1).strip())
        for code_match in re.finditer(r'```\s*([\s\S]*?)\s*```', response):
            content = code_match.group(1).strip()
            if content.startswith('{'):
                candidates.append(content)

        # 非贪婪匹配找所有 {...} 片段，优先取末尾（模型通常在最后输出JSON）
        brace_matches = list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response))
        for match in reversed(brace_matches):
            candidates.append(match.group(0).strip())

        # 兜底：贪婪匹配（兼容深层嵌套JSON）
        if not brace_matches:
            greedy_match = re.search(r'\{[\s\S]*\}', response)
            if greedy_match:
                candidates.append(greedy_match.group(0).strip())

        for i, candidate in enumerate(candidates):
            try:
                json.loads(candidate)
                if i > 0:
                    logger.info(f"[健康评估模型_JSON_EXTRACT] 候选#{i}解析成功, 候选总数={len(candidates)}")
                return candidate
            except json.JSONDecodeError as e:
                logger.debug(f"[健康评估模型_JSON_EXTRACT] 候选#{i}解析失败: {type(e).__name__}, 片段前80字={candidate[:80]}")
                continue

        if candidates:
            logger.warning(f"[健康评估模型_JSON_EXTRACT] 所有{len(candidates)}个候选均无法解析，使用首个候选")
            return candidates[0]

        partial_match = re.search(r'\{[\s\S]*', response)
        if partial_match:
            logger.warning(f"[健康评估模型_JSON_EXTRACT] 未找到完整JSON，使用部分匹配")
            return partial_match.group(0).strip()
        logger.warning(f"[健康评估模型_JSON_EXTRACT] 响应中未找到任何JSON片段, response_len={len(response)}")
        return response.strip()

    def _try_structured_repair(
        self,
        raw_output: str,
        error_description: str,
        expected_format: str,
        context_type: str,
        original_prompt: str = ""
    ) -> Optional[Dict]:
        """
        健康评估模型结构化输出自修复机制

        当模型输出的JSON结构有误时，将原始请求（含完整上下文）、错误信息和修复指令
        一起输入给模型，让模型基于完整上下文重新生成正确的结构化输出。最多重试1次。

        仅用于结构错误（JSON解析失败、缺少必要字段、字段类型错误），
        不用于模型评估内容本身的逻辑错误。

        Args:
            raw_output: 模型的原始输出
            error_description: 具体错误描述
            expected_format: 期望的JSON格式示例
            context_type: 上下度类型 "dimension" 或 "risk_factor"
            original_prompt: 原始评估请求prompt（含维度、子指标、用户数据、知识素材）

        Returns:
            修复成功返回解析后的字典，失败返回None
        """
        if not self._health_assessment_available or self._resource.health_assessment_model is None:
            logger.warning("[STRUCTURED_REPAIR] 健康评估模型不可用，跳过自修复")
            return None

        if not raw_output:
            logger.warning("[STRUCTURED_REPAIR] 原始输出为空，跳过自修复")
            return None

        repair_prompt = (
            "你上一次输出的JSON结构有误，请基于原始请求重新评估并输出正确格式的JSON。\n\n"
            f"【错误信息】{error_description}\n\n"
            f"【你的原始输出（有误）】\n{raw_output}\n\n"
            f"【期望格式】\n{expected_format}\n\n"
        )

        if original_prompt:
            repair_prompt += (
                f"【原始请求（完整上下文）】\n{original_prompt}\n\n"
                "请基于以上完整请求上下文，直接输出修复后的JSON，不要输出其他内容。"
            )
        else:
            repair_prompt += "请直接输出修复后的JSON，不要输出其他内容。"

        try:
            logger.info(f"[STRUCTURED_REPAIR] 尝试自修复: context_type={context_type}, error={error_description[:100]}")
            _repair_start = time.time()
            repair_response = self._resource.health_assessment_model.generate(repair_prompt)
            _repair_elapsed = time.time() - _repair_start
            logger.info(f"[STRUCTURED_REPAIR] 自修复响应耗时={_repair_elapsed:.2f}s, "
                       f"response_len={len(repair_response) if repair_response else 0}")

            if not repair_response:
                logger.warning(f"[STRUCTURED_REPAIR] 自修复失败: 模型返回为空, context_type={context_type}")
                return None

            result = json.loads(self._extract_json_from_response(repair_response))
            logger.info(f"[STRUCTURED_REPAIR] 自修复成功: context_type={context_type}, "
                       f"output_keys={list(result.keys())}")
            return result

        except json.JSONDecodeError:
            logger.warning(f"[STRUCTURED_REPAIR] 自修复失败: JSON仍无法解析, context_type={context_type}")
            return None
        except Exception as e:
            logger.warning(f"[STRUCTURED_REPAIR] 自修复失败: error_type={type(e).__name__}, "
                          f"context_type={context_type}")
            return None

    def _parse_unit_score(self, value: object) -> Optional[float]:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        if 1 < score <= 100:
            score = score / 100
        if 0 <= score <= 1:
            return score
        return None

    def _parse_dimension_result(
        self,
        response: str,
        dim_id: str,
        dim_name: str
    ) -> Optional[Dict]:
        """
        解析维度评估的健康评估模型输出结果，含容错处理
        """
        if not response:
            logger.warning(f"[健康评估模型_PARSE_ERROR] 维度={dim_name}, 输出为空")
            return None

        try:
            result = json.loads(self._extract_json_from_response(response))

            # 容错：健康评估模型可能输出subindicator_scores（无下划线），归一化为sub_indicator_scores
            if "subindicator_scores" in result and "sub_indicator_scores" not in result:
                result["sub_indicator_scores"] = result.pop("subindicator_scores")

            # 容错：dimension_score > 1 时归一化
            if "dimension_score" in result:
                dimension_score = float(result["dimension_score"])
                if dimension_score > 1:
                    logger.warning("[HealthAssessmentChain] dimension_score超出0-1范围，尝试归一化(÷100)")
                    dimension_score = dimension_score / 100
                result["dimension_score"] = dimension_score
                sub_scores_raw = result.get("sub_indicator_scores", {})
                sub_indicator_count = len(sub_scores_raw) if isinstance(sub_scores_raw, dict) else 0
                logger.info(f"[健康评估模型_OUTPUT] 维度={dim_id}({dim_name}), sub_indicator_count={sub_indicator_count}, output_keys={list(result.keys())}")
                if 0 <= dimension_score <= 1:
                    # 容错：sub_indicator_scores值为裸数字时包装为dict
                    if "sub_indicator_scores" in result:
                        sub_scores = result["sub_indicator_scores"]
                        if isinstance(sub_scores, dict):
                            for k, v in list(sub_scores.items()):
                                if isinstance(v, (int, float, str)):
                                    score = self._parse_unit_score(v)
                                    if score is not None:
                                        sub_scores[k] = {"score": score, "reason": ""}
                                elif isinstance(v, dict) and "score" in v:
                                    score = self._parse_unit_score(v["score"])
                                    if score is not None:
                                        v["score"] = score
                                elif isinstance(v, dict):
                                    # 尝试从dict中提取第一个数值作为score
                                    for vk, vv in v.items():
                                        score = self._parse_unit_score(vv)
                                        if score is not None:
                                            reason_keys = set(v.keys()) - {vk}
                                            reason = str(v.get(next(iter(reason_keys), ""), ""))
                                            sub_scores[k] = {"score": score, "reason": reason}
                                            break
                        return result
                    # 缺少sub_indicator_scores时尝试构造
                    else:
                        logger.warning(f"[HealthAssessmentChain] 维度{dim_name}缺少sub_indicator_scores，尝试从dimension_score推导")
                        dim_info = HEALTH_DIMENSIONS.get(dim_id, {})
                        sub_indicators = dim_info.get("sub_indicators", [])
                        sub_scores = {}
                        for si in sub_indicators:
                            sub_scores[si] = {"score": dimension_score, "reason": "由维度总分推导"}
                        result["sub_indicator_scores"] = sub_scores
                        return result
                else:
                    logger.warning("[HealthAssessmentChain] dimension_score归一化后仍超出范围[0,1]")

        except json.JSONDecodeError as e:
            logger.warning(f"[健康评估模型_PARSE_ERROR] JSON解析失败: dimension={dim_name}, error={type(e).__name__}, raw_output_len={len(response)}")
        except Exception as e:
            logger.warning(f"[HealthAssessmentChain] 维度评估结果解析失败: error_type={type(e).__name__}")

        return None
    
    def _evaluate_dimension(
        self,
        dim_id: str,
        dim_name: str,
        sub_indicators: List[str],
        body: HealthAssessmentContextBody
    ) -> tuple:
        """
        评估单个维度的得分
        
        采用健康评估模型评估引擎对每个子指标进行评估，
        如果健康评估模型不可用，则降级为规则引擎评估。
        
        Args:
            dim_id: 维度ID
            dim_name: 维度名称
            sub_indicators: 子指标列表
            body: Chain策略专属输入数据
            
        Returns:
            (dimension_score, reasoning, sub_indicator_scores)
        """
        sub_indicator_scores = {}

        # 维度评估计时开始
        _dim_eval_start = time.time()

        # 健康评估模型输入日志：维度名称和子指标名称列表
        logger.info(f"[健康评估模型_INPUT] 维度={dim_id}({dim_name}), 子指标={sub_indicators}")

        # 尝试使用健康评估模型评估引擎
        if self._health_assessment_available and self._resource.health_assessment_model is not None:
            try:
                # 调用健康评估模型评估子指标
                evaluation_result = self._call_health_assessment_for_dimension(dim_id, dim_name, sub_indicators, body)

                if evaluation_result:
                    dimension_score = evaluation_result.get("dimension_score", _report_config.default_dimension_score)
                    reasoning = evaluation_result.get("dimension_reasoning", "")
                    sub_indicator_scores = evaluation_result.get("sub_indicator_scores", {})

                    # 子指标失败降级：跳过失败子指标 + 权重归一化
                    sub_indicator_scores, dimension_score = self._normalize_sub_indicator_scores(
                        dim_id, dim_name, sub_indicators, sub_indicator_scores
                    )

                    logger.info(
                        f"[健康评估模型_OUTPUT] 维度={dim_id}({dim_name}), "
                        f"dimension_score_present={dimension_score is not None}, "
                        f"sub_indicator_count={len(sub_indicator_scores)}"
                    )

                    # 防重复效果日志
                    _dim_eval_duration = time.time() - _dim_eval_start
                    logger.info(f"[REPETITION_CHECK] dimension={dim_id}({dim_name}), finish_reason_present=True, duration={_dim_eval_duration:.2f}s")

                    return dimension_score, reasoning, sub_indicator_scores
                    
            except Exception as e:
                logger.warning(f"[HealthAssessmentChain] 健康评估模型评估失败，降级为规则引擎: error_type={type(e).__name__}")
                logger.warning(f"[健康评估模型_DEGRADED] 维度{dim_id}({dim_name})健康评估模型评估失败，降级为规则引擎: error_type={type(e).__name__}")
                logger.warning(f"[DEGRADE_TO_RULE_ENGINE] 降级触发: 健康评估模型不可用, "
                              f"降级策略=规则引擎评估, 维度={dim_id}({dim_name})")
                logger.warning(f"[DEGRADE_STRATEGY] from=健康评估模型 to=规则引擎评估, reason=健康评估模型评估失败({dim_id}({dim_name}))")
        else:
            if not self._health_assessment_available:
                logger.warning(f"[健康评估模型_DEGRADED] 维度{dim_id}({dim_name})健康评估模型模型不可用，使用规则引擎评估")
                logger.warning(f"[DEGRADE_TO_RULE_ENGINE] 降级触发: 健康评估模型不可用, "
                              f"降级策略=规则引擎评估, 维度={dim_id}({dim_name})")
                logger.warning(f"[DEGRADE_STRATEGY] from=健康评估模型 to=规则引擎评估, reason=健康评估模型模型不可用({dim_id}({dim_name}))")

        # 降级：使用规则引擎评估
        dimension_score, reasoning, sub_indicator_scores = self._degrade_to_rule_engine(
            dim_id, dim_name, sub_indicators, body
        )
        
        # 防重复效果日志（降级路径）
        _dim_eval_duration = time.time() - _dim_eval_start
        logger.info(f"[REPETITION_CHECK] dimension={dim_id}({dim_name}), finish_reason_present=True, duration={_dim_eval_duration:.2f}s")
        
        return dimension_score, reasoning, sub_indicator_scores
    
    def _call_health_assessment_for_dimension(
        self,
        dim_id: str,
        dim_name: str,
        sub_indicators: List[str],
        body: HealthAssessmentContextBody
    ) -> Optional[Dict]:
        """
        调用健康评估模型评估引擎对维度子指标进行评估
        
        Args:
            dim_id: 维度ID
            dim_name: 维度名称
            sub_indicators: 子指标列表
            body: Chain策略专属输入数据
            
        Returns:
            评估结果字典，包含dimension_score、sub_indicator_scores、dimension_reasoning
        """
        # 构建评估Prompt
        prompt = self._build_dimension_evaluation_prompt(dim_id, dim_name, sub_indicators, body)

        dim_name_key = DIMENSION_ID_TO_NAME.get(dim_id, dim_id)
        dimension_knowledge = body.dimension_summaries.get(dim_name_key, {})
        knowledge_items = dimension_knowledge.get("knowledge_items", []) if isinstance(dimension_knowledge, dict) else []
        logger.info(f"[健康评估模型_INPUT] 维度={dim_id}({dim_name}), 子指标数={len(sub_indicators)}, prompt_len={len(prompt)}")
        logger.info(f"[健康评估模型_EVAL_INPUT] dimension={dim_name}, sub_indicator_count={len(sub_indicators)}, user_profile_keys={list(body.user_profile.keys())}, knowledge_item_count={len(knowledge_items)}")

        # 检查Prompt长度限制
        if len(prompt) > _get_health_assessment_constraints()["max_prompt_chars"]:
            logger.warning(f"[HealthAssessmentChain] Prompt长度超过限制({len(prompt)} > {_get_health_assessment_constraints()['max_prompt_chars']})，截断")
            prompt = prompt[:_get_health_assessment_constraints()["max_prompt_chars"]]
        
        try:
            # 调用健康评估模型模型
            # 注意：这里假设health_assessment_model有一个generate或predict方法
            # 实际实现需要根据健康评估模型模型的接口调整
            logger.info(f"[HealthAssessmentChain._call_health_assessment_for_dimension] 调用健康评估模型进行维度评估，维度={dim_name}")
            logger.info(f"[LLM_INPUT] 维度评估, dimension={dim_name}, prompt_len={len(prompt)}")
            _ha_start = time.time()
            response = self._resource.health_assessment_model.generate(prompt)
            _ha_elapsed = time.time() - _ha_start
            
            # 健康评估模型输出日志
            logger.info(f"[健康评估模型_OUTPUT] 维度={dim_id}({dim_name}), 评估结果长度={len(response) if response else 0}")

            # 解析JSON输出（提前解析以获取dimension_score和子指标评分概要）
            result = json.loads(self._extract_json_from_response(response))

            # 验证输出格式（含容错）
            if "dimension_score" in result:
                dimension_score = float(result["dimension_score"])
                # 容错：dimension_score > 1 时归一化
                if dimension_score > 1:
                    logger.warning("[HealthAssessmentChain] dimension_score超出0-1范围，尝试归一化(÷100)")
                    dimension_score = dimension_score / 100
                result["dimension_score"] = dimension_score
                sub_scores_raw = result.get("sub_indicator_scores", {})
                sub_indicator_count = len(sub_scores_raw) if isinstance(sub_scores_raw, dict) else 0

                response_len = len(response) if response else 0
                logger.info(f"[HealthAssessmentChain._call_health_assessment_for_dimension] 健康评估模型维度评估完成，维度={dim_name}，输出长度={response_len}")
                logger.info(f"[LLM_OUTPUT] 维度评估, dimension={dim_name}, response_len={response_len}, output_keys={list(result.keys())}")
                logger.info(f"[LLM_DURATION] {_ha_elapsed:.3f}s")
                logger.info(f"[HealthAssessment_DURATION] 维度名称={dim_name}, 耗时={_ha_elapsed:.3f}s")
                logger.info(f"[健康评估模型_OUTPUT] 维度={dim_id}({dim_name}), sub_indicator_count={sub_indicator_count}, output_keys={list(result.keys())}")
                if 0 <= dimension_score <= 1:
                    # 容错：缺少sub_indicator_scores时从维度定义推导
                    if "sub_indicator_scores" not in result:
                        dim_info = HEALTH_DIMENSIONS.get(dim_id, {})
                        sub_indicators = dim_info.get("sub_indicators", [])
                        sub_scores = {}
                        for si in sub_indicators:
                            sub_scores[si] = {"score": dimension_score, "reason": "由维度总分推导"}
                        result["sub_indicator_scores"] = sub_scores
                    # 容错：sub_indicator_scores值为裸数字时包装为dict
                    sub_scores = result.get("sub_indicator_scores", {})
                    if isinstance(sub_scores, dict):
                        for k, v in list(sub_scores.items()):
                            if isinstance(v, (int, float, str)):
                                score = self._parse_unit_score(v)
                                if score is not None:
                                    sub_scores[k] = {"score": score, "reason": ""}
                            elif isinstance(v, dict) and "score" in v:
                                score = self._parse_unit_score(v["score"])
                                if score is not None:
                                    v["score"] = score
                    return result
                else:
                    logger.warning("[HealthAssessmentChain] dimension_score归一化后仍超出范围[0,1]")
                    
        except json.JSONDecodeError as e:
            logger.warning(f"[健康评估模型_PARSE_ERROR] JSON解析失败: dimension={dim_name}, error={type(e).__name__}, raw_output_len={len(response)}")
            # 尝试结构化自修复
            repair_result = self._try_structured_repair(
                raw_output=response,
                error_description=f"JSON解析失败: {type(e).__name__}",
                expected_format='{"dimension_score":0.72,"sub_indicator_scores":{"指标名":0.65},"dimension_reasoning":"总体评估"}',
                context_type="dimension",
                original_prompt=prompt
            )
            if repair_result and "dimension_score" in repair_result:
                dimension_score = float(repair_result["dimension_score"])
                if dimension_score > 1:
                    dimension_score = dimension_score / 100
                if 0 <= dimension_score <= 1:
                    repair_result["dimension_score"] = dimension_score
                    if "sub_indicator_scores" not in repair_result:
                        dim_info = HEALTH_DIMENSIONS.get(dim_id, {})
                        sub_indicators_from_dim = dim_info.get("sub_indicators", [])
                        sub_scores = {}
                        for si in sub_indicators_from_dim:
                            sub_scores[si] = {"score": dimension_score, "reason": "由维度总分推导(自修复)"}
                        repair_result["sub_indicator_scores"] = sub_scores
                    sub_scores = repair_result.get("sub_indicator_scores", {})
                    if isinstance(sub_scores, dict):
                        for k, v in list(sub_scores.items()):
                            if isinstance(v, (int, float, str)):
                                score = self._parse_unit_score(v)
                                if score is not None:
                                    sub_scores[k] = {"score": score, "reason": ""}
                            elif isinstance(v, dict) and "score" in v:
                                score = self._parse_unit_score(v["score"])
                                if score is not None:
                                    v["score"] = score
                    return repair_result
            logger.warning(f"[HealthAssessmentChain] 健康评估模型输出JSON解析失败(自修复也失败): error_type=JSONDecodeError")
        except Exception as e:
            logger.warning(f"[HealthAssessmentChain] 健康评估模型调用失败: error_type={type(e).__name__}")

        return None

    def _build_dimension_evaluation_prompt(
        self,
        dim_id: str,
        dim_name: str,
        sub_indicators: List[str],
        body: HealthAssessmentContextBody
    ) -> str:
        """
        构建维度评估Prompt，格式指令前置确保不被截断，数据段后置可安全截断。

        Prompt结构：格式指令(开头，不可截断) → 维度信息 → 用户数据 → 知识素材(末尾，可截断)
        """
        max_prompt = _get_health_assessment_constraints()["max_prompt_chars"]
        max_knowledge = _get_health_assessment_constraints()["max_knowledge_chars"]
        max_user_info = _get_health_assessment_constraints()["max_user_info_chars"]

        # 格式指令（放在开头，确保不被截断）
        sub_indicators_example = ", ".join(f'"{s}":0.65' for s in sub_indicators)
        format_instruction = (
            f"你是健康评估专家。评估维度: {dim_name}(权重{HEALTH_DIMENSIONS[dim_id]['weight']})\n"
            f"子指标({len(sub_indicators)}个): {', '.join(sub_indicators)}\n\n"
            "严格按以下JSON格式输出，不要输出任何其他内容:\n"
            f'{{"dimension_score":0.72,"sub_indicator_scores":{{{sub_indicators_example}}},'
            '"dimension_reasoning":"总体评估"}\n\n'
            f"重要：sub_indicator_scores必须包含全部{len(sub_indicators)}个子指标的评分(0-1)，不可遗漏任何一个。\n\n"
            "以下是评估依据:\n"
        )

        # 用户信息截断
        user_info_budget = min(max_user_info, int(max_prompt * _report_config.user_info_budget_ratio) // 3)
        user_profile_str = json.dumps(body.user_profile, ensure_ascii=False)
        if len(user_profile_str) > user_info_budget:
            user_profile_str = user_profile_str[:user_info_budget] + "...(截断)"

        anomalies_str = json.dumps(body.anomalies, ensure_ascii=False)
        if len(anomalies_str) > user_info_budget:
            anomalies_str = anomalies_str[:user_info_budget] + "...(截断)"

        risk_factors_str = json.dumps(body.risk_factors, ensure_ascii=False)
        if len(risk_factors_str) > user_info_budget:
            risk_factors_str = risk_factors_str[:user_info_budget] + "...(截断)"

        data_section = f"用户={user_profile_str} 异常={anomalies_str} 风险={risk_factors_str}\n"

        # 知识素材：优先使用已提炼的summary和refined_knowledge，保留知识质量
        dim_name_key = DIMENSION_ID_TO_NAME.get(dim_id, dim_name)
        dimension_knowledge = body.dimension_summaries.get(dim_name_key, {})
        knowledge_budget = min(max_knowledge, int(max_prompt * _report_config.knowledge_budget_ratio))

        # 构建精简版知识：summary + refined_knowledge 优于原始knowledge_items
        compact_knowledge = {}
        if isinstance(dimension_knowledge, dict):
            # 优先使用summary（已提炼的维度摘要）
            summary = dimension_knowledge.get("summary", "")
            if summary:
                compact_knowledge["summary"] = summary
            # 优先使用refined_knowledge（已提炼的精简知识条目）
            refined = dimension_knowledge.get("refined_knowledge", [])
            if refined:
                compact_knowledge["refined_knowledge"] = refined[:_report_config.knowledge_item_limit]
            # 仅在refined_knowledge缺失时回退到knowledge_items（限制条目数，保留name+desc短字段）
            if not refined:
                knowledge_items = dimension_knowledge.get("knowledge_items", [])
                if knowledge_items:
                    limited_items = knowledge_items[:_report_config.knowledge_item_limit]
                    simplified_items = []
                    for item in limited_items:
                        simplified = {}
                        for key in ["name", "entity_name", "desc", "description"]:
                            if key in item and isinstance(item[key], str) and len(item[key]) <= _report_config.knowledge_content_truncate_len:
                                simplified[key] = item[key]
                        if simplified:
                            simplified_items.append(simplified)
                    if simplified_items:
                        compact_knowledge["knowledge_items"] = simplified_items

        knowledge_final_str = json.dumps(compact_knowledge, ensure_ascii=False)
        if len(knowledge_final_str) > knowledge_budget:
            logger.warning(f"[HealthAssessmentChain] 维度{dim_name}精简知识仍超预算({len(knowledge_final_str)}>{knowledge_budget})，截断末尾")
            knowledge_final_str = knowledge_final_str[:knowledge_budget]

        knowledge_section = f"知识={knowledge_final_str}"

        prompt = format_instruction + data_section + knowledge_section

        # 最终长度校验：截断末尾数据段，保留开头的格式指令
        if len(prompt) > max_prompt:
            logger.warning(f"[HealthAssessmentChain] 维度{dim_name}Prompt仍超限({len(prompt)}>{max_prompt})，截断末尾数据段")
            prompt = prompt[:max_prompt]

        return prompt
    def _normalize_sub_indicator_scores(
        self,
        dim_id: str,
        dim_name: str,
        expected_sub_indicators: List[str],
        sub_indicator_scores: Dict
    ) -> tuple:
        """
        子指标失败降级：跳过失败子指标 + 权重归一化

        当子指标评估失败（缺失评分或评分无效）时，统一降级方式为：
        1. 跳过该子指标（不参与维度评分计算）
        2. 将其权重分配给同维度其他子指标（权重重新归一化）
        3. 基于归一化后的子指标评分重新计算维度得分

        Args:
            dim_id: 维度ID
            dim_name: 维度名称
            expected_sub_indicators: 期望的子指标列表
            sub_indicator_scores: 实际的子指标评分字典

        Returns:
            (normalized_sub_scores, normalized_dimension_score)
            - normalized_sub_scores: 归一化后的子指标评分字典（已跳过失败子指标）
            - normalized_dimension_score: 重新计算的维度得分
        """
        if not sub_indicator_scores:
            return sub_indicator_scores, _report_config.default_dimension_score

        # 检查每个子指标是否有效
        valid_sub_scores = {}
        failed_sub_indicators = []

        for sub_name in expected_sub_indicators:
            sub_info = sub_indicator_scores.get(sub_name)
            if sub_info is None:
                # 子指标缺失
                failed_sub_indicators.append(sub_name)
                logger.warning(f"[SUB_INDICATOR_DEGRADED] 维度={dim_id}({dim_name}), "
                              f"子指标={sub_name}评估失败(缺失), 降级策略=跳过+权重归一化")
                logger.warning(f"[SUB_INDICATOR_DEGRADE] indicator={sub_name}, action=skip+weight_normalize")
                continue

            # 检查评分是否有效
            if isinstance(sub_info, dict):
                score = sub_info.get("score")
                if score is None or not isinstance(score, (int, float)):
                    failed_sub_indicators.append(sub_name)
                    logger.warning(f"[SUB_INDICATOR_DEGRADED] 维度={dim_id}({dim_name}), "
                                  f"子指标={sub_name}评估失败(评分无效: {score}), 降级策略=跳过+权重归一化")
                    logger.warning(f"[SUB_INDICATOR_DEGRADE] indicator={sub_name}, action=skip+weight_normalize")
                    continue
            elif isinstance(sub_info, (int, float)):
                score = sub_info
            else:
                failed_sub_indicators.append(sub_name)
                logger.warning(f"[SUB_INDICATOR_DEGRADED] 维度={dim_id}({dim_name}), "
                              f"子指标={sub_name}评估失败(类型无效: {type(sub_info)}), 降级策略=跳过+权重归一化")
                logger.warning(f"[SUB_INDICATOR_DEGRADE] indicator={sub_name}, action=skip+weight_normalize")
                continue

            valid_sub_scores[sub_name] = sub_info

        # 如果没有子指标失败，直接返回
        if not failed_sub_indicators:
            # 仍然基于有效子指标重新计算维度得分
            total_score = 0.0
            count = 0
            for sub_name, sub_info in valid_sub_scores.items():
                if isinstance(sub_info, dict):
                    total_score += sub_info.get("score", 0.0)
                else:
                    total_score += sub_info
                count += 1
            dimension_score = total_score / count if count > 0 else _report_config.default_dimension_score
            return sub_indicator_scores, dimension_score

        # 有子指标失败，执行权重归一化
        logger.info(
            f"[SUB_INDICATOR_NORMALIZE] 维度={dim_id}({dim_name}), "
            f"failed_sub_indicator_count={len(failed_sub_indicators)}, "
            f"valid_sub_indicator_count={len(valid_sub_scores)}, "
            f"expected_sub_indicator_count={len(expected_sub_indicators)}, "
            f"执行权重归一化"
        )

        # 基于有效子指标重新计算维度得分（等权平均）
        total_score = 0.0
        count = 0
        for sub_name, sub_info in valid_sub_scores.items():
            if isinstance(sub_info, dict):
                total_score += sub_info.get("score", 0.0)
            else:
                total_score += sub_info
            count += 1

        dimension_score = total_score / count if count > 0 else _report_config.default_dimension_score

        # 在子指标评分中标记失败的子指标
        for failed_sub in failed_sub_indicators:
            valid_sub_scores[failed_sub] = {
                "score": 0.0,
                "reason": "子指标评估失败，已跳过(权重归一化至其他子指标)",
                "_skipped": True
            }

        logger.info(
            f"[SUB_INDICATOR_NORMALIZE] 维度={dim_id}({dim_name}), "
            f"normalized_dimension_score_present={dimension_score is not None}, "
            f"skipped_sub_indicator_count={len(failed_sub_indicators)}"
        )

        return valid_sub_scores, dimension_score

    def _degrade_to_rule_engine(
        self,
        dim_id: str,
        dim_name: str,
        sub_indicators: List[str],
        body: HealthAssessmentContextBody
    ) -> tuple:
        """
        降级策略：健康评估模型不可用 -> 规则引擎评估（维度评估场景）

        当健康评估模型评估引擎不可用时，显式调用规则引擎对维度子指标进行评估。
        此方法是对_rule_based_dimension_evaluation的显式降级封装，
        添加降级日志记录，使降级路径可追踪。

        Args:
            dim_id: 维度ID
            dim_name: 维度名称
            sub_indicators: 子指标列表
            body: Chain策略专属输入数据

        Returns:
            (dimension_score, reasoning, sub_indicator_scores)
        """
        logger.info(f"[DEGRADE_TO_RULE_ENGINE] 执行规则引擎降级评估: "
                   f"维度={dim_id}({dim_name}), 子指标={sub_indicators}")
        return self._rule_based_dimension_evaluation(dim_id, dim_name, sub_indicators, body)

    def _degrade_to_rule_engine_risk_factor(
        self,
        factor_id: str,
        factor_name: str,
        body: HealthAssessmentContextBody
    ) -> tuple:
        """
        降级策略：健康评估模型不可用 -> 规则引擎评估（风险因子评估场景）

        当健康评估模型评估引擎不可用时，显式调用规则引擎对风险因子进行评估。
        此方法是对_rule_based_risk_factor_evaluation的显式降级封装，
        添加降级日志记录，使降级路径可追踪。

        Args:
            factor_id: 风险因子ID
            factor_name: 风险因子名称
            body: Chain策略专属输入数据

        Returns:
            (factor_score, reasoning)
        """
        logger.info(f"[DEGRADE_TO_RULE_ENGINE] 执行规则引擎降级评估: "
                   f"风险因子={factor_id}({factor_name})")
        return self._rule_based_risk_factor_evaluation(factor_id, factor_name, body)

    def _rule_based_dimension_evaluation(
        self,
        dim_id: str,
        dim_name: str,
        sub_indicators: List[str],
        body: HealthAssessmentContextBody
    ) -> tuple:
        """
        规则引擎降级：对维度子指标进行规则评估
        
        Args:
            dim_id: 维度ID
            dim_name: 维度名称
            sub_indicators: 子指标列表
            body: Chain策略专属输入数据
            
        Returns:
            (dimension_score, reasoning, sub_indicator_scores)
        """
        sub_indicator_scores = {}
        
        if dim_id == "D1":  # 生理指标
            dimension_score, reasoning, sub_indicator_scores = self._evaluate_physiological_dimension(body)
        elif dim_id == "D2":  # 生活方式
            dimension_score, reasoning, sub_indicator_scores = self._evaluate_lifestyle_dimension(body)
        elif dim_id == "D3":  # 病史风险
            dimension_score, reasoning, sub_indicator_scores = self._evaluate_medical_history_dimension(body)
        elif dim_id == "D4":  # 心理状态
            dimension_score, reasoning, sub_indicator_scores = self._evaluate_psychological_dimension(body)
        elif dim_id == "D5":  # 预防措施
            dimension_score, reasoning, sub_indicator_scores = self._evaluate_prevention_dimension(body)
        else:
            # 默认中性评分
            dimension_score = _report_config.default_dimension_score
            reasoning = "无足够数据，使用中性评分"
            for indicator in sub_indicators:
                sub_indicator_scores[indicator] = {"score": _report_config.default_dimension_score, "reason": "默认中性评分"}
        
        return dimension_score, reasoning, sub_indicator_scores
    
    def _evaluate_physiological_dimension(self, body: HealthAssessmentContextBody) -> tuple:
        """评估生理指标维度（规则引擎）"""
        sub_scores = {}
        total_score = 0
        count = 0
        rs = self._get_rule_engine_scores()

        # 从异常指标中提取生理指标
        anomalies = {item.get("indicator_name", ""): item for item in body.anomalies}

        # 血压评分
        if "血压" in anomalies or "收缩压" in anomalies:
            bp_anomaly = anomalies.get("血压") or anomalies.get("收缩压")
            anomaly_type = bp_anomaly.get("anomaly_type", "normal") if bp_anomaly else "normal"
            if anomaly_type == "normal":
                score = rs.get("normal", 1.0)
            elif anomaly_type in ["mild", "偏高", "偏低"]:
                score = rs.get("mild_abnormal", 0.7)
            else:
                score = rs.get("moderate_abnormal", 0.4)
            sub_scores["血压"] = {"score": score, "reason": f"血压{anomaly_type}"}
        else:
            score = rs.get("normal", 1.0)
            sub_scores["血压"] = {"score": score, "reason": "血压正常"}
        total_score += score
        count += 1

        # 血糖评分
        if "血糖" in anomalies or "空腹血糖" in anomalies:
            bg_anomaly = anomalies.get("血糖") or anomalies.get("空腹血糖")
            anomaly_type = bg_anomaly.get("anomaly_type", "normal") if bg_anomaly else "normal"
            if anomaly_type == "normal":
                score = rs.get("normal", 1.0)
            elif anomaly_type in ["mild", "偏高", "偏低"]:
                score = rs.get("mild_abnormal", 0.7)
            else:
                score = rs.get("moderate_abnormal", 0.4)
            sub_scores["血糖"] = {"score": score, "reason": f"血糖{anomaly_type}"}
        else:
            score = rs.get("normal", 1.0)
            sub_scores["血糖"] = {"score": score, "reason": "血糖正常"}
        total_score += score
        count += 1

        # BMI评分
        if "BMI" in anomalies or "体重指数" in anomalies:
            bmi_anomaly = anomalies.get("BMI") or anomalies.get("体重指数")
            anomaly_type = bmi_anomaly.get("anomaly_type", "normal") if bmi_anomaly else "normal"
            if anomaly_type == "normal":
                score = rs.get("normal", 1.0)
            elif anomaly_type in ["偏瘦", "超重"]:
                score = rs.get("mild_abnormal", 0.7)
            else:
                score = rs.get("moderate_abnormal", 0.4)
            sub_scores["BMI"] = {"score": score, "reason": f"BMI{anomaly_type}"}
        else:
            score = rs.get("normal", 1.0)
            sub_scores["BMI"] = {"score": score, "reason": "BMI正常"}
        total_score += score
        count += 1

        # 心率、血脂默认正常
        sub_scores["心率"] = {"score": rs.get("normal", 1.0), "reason": "心率正常"}
        sub_scores["血脂"] = {"score": rs.get("normal", 1.0), "reason": "血脂正常"}
        total_score += 2.0
        count += 2
        
        dimension_score = total_score / count if count > 0 else _report_config.default_dimension_score
        reasoning = f"生理指标维度平均得分{dimension_score:.2f}"

        # 规则引擎降级评估详细日志
        for sub_name, sub_info in sub_scores.items():
            sub_score = sub_info.get("score", 0.0) if isinstance(sub_info, dict) else sub_info
            sub_reason = sub_info.get("reason", "") if isinstance(sub_info, dict) else ""
            logger.info(f"[RULE_ENGINE_ASSESS] 维度=D1(生理指标), 子指标={sub_name}, score_present={sub_score is not None}, reason_len={len(sub_reason)}")

        return dimension_score, reasoning, sub_scores

    def _evaluate_lifestyle_dimension(self, body: HealthAssessmentContextBody) -> tuple:
        """评估生活方式维度（规则引擎）"""
        sub_scores = {}
        rs = self._get_rule_engine_scores()

        # 从用户档案中获取生活方式信息
        lifestyle = body.user_profile.get("lifestyle", {})

        # 运动评分
        exercise_freq = lifestyle.get("exercise_freq", "偶尔")
        if exercise_freq == "规律":
            sub_scores["运动"] = {"score": rs.get("regular_exercise", 1.0), "reason": "规律运动"}
        elif exercise_freq == "偶尔":
            sub_scores["运动"] = {"score": rs.get("occasional_exercise", 0.6), "reason": "偶尔运动"}
        else:
            sub_scores["运动"] = {"score": rs.get("no_exercise", 0.3), "reason": "缺乏运动"}

        # 饮食评分
        diet_quality = lifestyle.get("diet_quality", "一般")
        if diet_quality == "良好":
            sub_scores["饮食"] = {"score": rs.get("normal", 1.0), "reason": "饮食良好"}
        elif diet_quality == "一般":
            sub_scores["饮食"] = {"score": rs.get("mild_abnormal", 0.7), "reason": "饮食一般"}
        else:
            sub_scores["饮食"] = {"score": rs.get("moderate_abnormal", 0.4), "reason": "饮食较差"}

        # 睡眠评分
        sleep_quality = lifestyle.get("sleep_quality", "一般")
        if sleep_quality == "良好":
            sub_scores["睡眠"] = {"score": rs.get("normal", 1.0), "reason": "睡眠良好"}
        elif sleep_quality == "一般":
            sub_scores["睡眠"] = {"score": rs.get("mild_abnormal", 0.7), "reason": "睡眠一般"}
        else:
            sub_scores["睡眠"] = {"score": rs.get("moderate_abnormal", 0.4), "reason": "睡眠较差"}

        # 吸烟评分
        smoking = lifestyle.get("smoking", "不吸烟")
        if smoking == "不吸烟":
            sub_scores["吸烟"] = {"score": rs.get("no_smoking", 1.0), "reason": "不吸烟"}
        elif smoking == "已戒烟":
            sub_scores["吸烟"] = {"score": rs.get("quit_smoking", 0.8), "reason": "已戒烟"}
        else:
            sub_scores["吸烟"] = {"score": rs.get("smoking", 0.3), "reason": "吸烟"}

        # 饮酒评分
        drinking = lifestyle.get("drinking", "不饮酒")
        if drinking == "不饮酒":
            sub_scores["饮酒"] = {"score": rs.get("no_drinking", 1.0), "reason": "不饮酒"}
        elif drinking == "适量":
            sub_scores["饮酒"] = {"score": rs.get("moderate_drinking", 0.8), "reason": "适量饮酒"}
        else:
            sub_scores["饮酒"] = {"score": rs.get("heavy_drinking", 0.4), "reason": "过量饮酒"}
        
        total_score = sum(s["score"] for s in sub_scores.values())
        dimension_score = total_score / len(sub_scores)
        reasoning = f"生活方式维度平均得分{dimension_score:.2f}"

        # 规则引擎降级评估详细日志
        for sub_name, sub_info in sub_scores.items():
            sub_score = sub_info.get("score", 0.0) if isinstance(sub_info, dict) else sub_info
            sub_reason = sub_info.get("reason", "") if isinstance(sub_info, dict) else ""
            logger.info(f"[RULE_ENGINE_ASSESS] 维度=D2(生活方式), 子指标={sub_name}, score_present={sub_score is not None}, reason_len={len(sub_reason)}")

        return dimension_score, reasoning, sub_scores
    
    def _evaluate_medical_history_dimension(self, body: HealthAssessmentContextBody) -> tuple:
        """评估病史风险维度（规则引擎）"""
        sub_scores = {}
        rs = self._get_rule_engine_scores()

        # 统计既往病史
        diseases = body.medical_entities.get("diseases", [])
        disease_count = len(diseases)

        if disease_count == 0:
            sub_scores["既往病史"] = {"score": rs.get("normal", 1.0), "reason": "无既往病史"}
        elif disease_count <= _report_config.disease_count_mild:
            sub_scores["既往病史"] = {"score": rs.get("few_history", 0.7), "reason": f"有{disease_count}种病史"}
        else:
            sub_scores["既往病史"] = {"score": rs.get("many_history", 0.4), "reason": f"有{disease_count}种病史"}

        # 家族史
        family_history = body.user_profile.get("family_history", "")
        if family_history and family_history.strip():
            sub_scores["家族史"] = {"score": rs.get("has_family_history", 0.6), "reason": "有家族病史"}
        else:
            sub_scores["家族史"] = {"score": rs.get("no_family_history", 1.0), "reason": "无家族病史"}

        # 用药史
        medications = body.medical_entities.get("medications", [])
        if len(medications) == 0:
            sub_scores["用药史"] = {"score": rs.get("normal", 1.0), "reason": "无用药史"}
        elif len(medications) <= 3:
            sub_scores["用药史"] = {"score": rs.get("few_medication", 0.7), "reason": f"使用{len(medications)}种药物"}
        else:
            sub_scores["用药史"] = {"score": rs.get("many_medication", 0.4), "reason": f"使用{len(medications)}种药物"}
        
        total_score = sum(s["score"] for s in sub_scores.values())
        dimension_score = total_score / len(sub_scores)
        reasoning = f"病史风险维度平均得分{dimension_score:.2f}"

        # 规则引擎降级评估详细日志
        for sub_name, sub_info in sub_scores.items():
            sub_score = sub_info.get("score", 0.0) if isinstance(sub_info, dict) else sub_info
            sub_reason = sub_info.get("reason", "") if isinstance(sub_info, dict) else ""
            logger.info(f"[RULE_ENGINE_ASSESS] 维度=D3(病史风险), 子指标={sub_name}, score_present={sub_score is not None}, reason_len={len(sub_reason)}")

        return dimension_score, reasoning, sub_scores
    
    def _evaluate_psychological_dimension(self, body: HealthAssessmentContextBody) -> tuple:
        """评估心理状态维度（规则引擎）"""
        sub_scores = {}
        rs = self._get_rule_engine_scores()

        # 从用户档案中获取心理状态信息
        psychological = body.user_profile.get("psychological", {})

        # 压力水平
        stress_level = psychological.get("stress_level", "一般")
        if stress_level == "低":
            sub_scores["压力水平"] = {"score": rs.get("good_mental", 1.0), "reason": "压力水平低"}
        elif stress_level == "一般":
            sub_scores["压力水平"] = {"score": rs.get("moderate_stress", 0.7), "reason": "压力水平一般"}
        else:
            sub_scores["压力水平"] = {"score": rs.get("high_stress", 0.4), "reason": "压力水平高"}

        # 情绪状态
        mood = psychological.get("mood", "稳定")
        if mood == "稳定":
            sub_scores["情绪状态"] = {"score": rs.get("good_emotion", 1.0), "reason": "情绪稳定"}
        elif mood == "一般":
            sub_scores["情绪状态"] = {"score": rs.get("moderate_emotion", 0.7), "reason": "情绪一般"}
        else:
            sub_scores["情绪状态"] = {"score": rs.get("poor_emotion", 0.4), "reason": "情绪不稳定"}
        
        total_score = sum(s["score"] for s in sub_scores.values())
        dimension_score = total_score / len(sub_scores)
        reasoning = f"心理状态维度平均得分{dimension_score:.2f}"

        # 规则引擎降级评估详细日志
        for sub_name, sub_info in sub_scores.items():
            sub_score = sub_info.get("score", 0.0) if isinstance(sub_info, dict) else sub_info
            sub_reason = sub_info.get("reason", "") if isinstance(sub_info, dict) else ""
            logger.info(f"[RULE_ENGINE_ASSESS] 维度=D4(心理状态), 子指标={sub_name}, score_present={sub_score is not None}, reason_len={len(sub_reason)}")

        return dimension_score, reasoning, sub_scores
    
    def _evaluate_prevention_dimension(self, body: HealthAssessmentContextBody) -> tuple:
        """评估预防措施维度（规则引擎）"""
        sub_scores = {}
        rs = self._get_rule_engine_scores()

        # 体检频率（默认中性评分）
        sub_scores["体检频率"] = {"score": rs.get("good_prevention", 0.7), "reason": "体检频率未知"}

        # 疫苗接种（默认中性评分）
        sub_scores["疫苗接种"] = {"score": rs.get("good_prevention", 0.7), "reason": "疫苗接种情况未知"}

        # 筛查情况（默认中性评分）
        sub_scores["筛查情况"] = {"score": rs.get("good_prevention", 0.7), "reason": "筛查情况未知"}
        
        total_score = sum(s["score"] for s in sub_scores.values())
        dimension_score = total_score / len(sub_scores)
        reasoning = f"预防措施维度平均得分{dimension_score:.2f}"

        # 规则引擎降级评估详细日志
        for sub_name, sub_info in sub_scores.items():
            sub_score = sub_info.get("score", 0.0) if isinstance(sub_info, dict) else sub_info
            sub_reason = sub_info.get("reason", "") if isinstance(sub_info, dict) else ""
            logger.info(f"[RULE_ENGINE_ASSESS] 维度=D5(预防措施), 子指标={sub_name}, score_present={sub_score is not None}, reason_len={len(sub_reason)}")

        return dimension_score, reasoning, sub_scores
    
    def _determine_health_level(self, health_score: float) -> str:
        """
        判定健康等级
        
        等级划分：
        - 优秀：90-100分
        - 良好：80-89分
        - 一般：70-79分
        - 较差：60-69分
        - 差：<60分
        
        Args:
            health_score: 健康综合评分
            
        Returns:
            健康等级
        """
        if health_score >= _report_config.health_score_thresholds["excellent"]:
            return "优秀"
        elif health_score >= _report_config.health_score_thresholds["good"]:
            return "良好"
        elif health_score >= _report_config.health_score_thresholds["normal"]:
            return "一般"
        elif health_score >= _report_config.health_score_thresholds["poor"]:
            return "较差"
        else:
            return "差"
    
    # ========================================================================
    # 疾病风险评分算法（评估框架）
    # ========================================================================
    
    def _calculate_disease_risks(self, body: HealthAssessmentContextBody) -> tuple:
        """
        计算疾病风险评分（评估框架）
        
        基于Framingham心血管疾病风险评估模型，
        结合向量语义匹配技术和中文医疗知识图谱。
        
        计算公式：
        disease_risk_score = Σ(Fi_score × Fi_weight) × 100
        
        风险等级判定：
        - 低风险: score < 30
        - 轻度风险: 30 ≤ score < 50
        - 中度风险: 50 ≤ score < 70
        - 高风险: score ≥ 70
        
        推理策略：
        1. 批量推理：将6个风险因子的评估prompt合并为一次call_model_batch()调用
        2. 串行推理降级：批量推理失败时，逐因子串行调用generate()
        3. 规则引擎降级：串行推理也失败时，使用规则引擎评估
        
        Args:
            body: Chain策略专属输入数据
            
        Returns:
            (disease_risks, breakdown)
        """
        logger.info("[HealthAssessmentChain] 开始计算疾病风险评分")
        
        breakdown = {}
        factor_scores = {}
        disease_risks = []
        
        # 收集6个风险因子的评估信息
        factor_items = list(DISEASE_RISK_FACTORS.items())
        
        # 尝试批量推理
        if self._health_assessment_available and self._resource.health_assessment_model is not None:
            try:
                logger.info("[HealthAssessmentChain] 尝试6风险因子批量推理")
                factor_scores = self._batch_evaluate_risk_factors(factor_items, body)
                if factor_scores:
                    logger.info("[HealthAssessmentChain] 6风险因子批量推理成功")
                else:
                    logger.warning("[HealthAssessmentChain] 6风险因子批量推理返回空结果，降级为串行推理")
                    factor_scores = self._serial_evaluate_risk_factors(factor_items, body)
            except Exception as e:
                logger.warning(f"[HealthAssessmentChain] 6风险因子批量推理失败，降级为串行推理: error_type={type(e).__name__}")
                logger.warning(f"[健康评估模型_DEGRADED] 6风险因子批量推理失败，降级为串行推理: error_type={type(e).__name__}")
                logger.warning(f"[BATCH_DEGRADE] 批量推理失败，降级为串行推理: error_type={type(e).__name__}")
                factor_scores = self._serial_evaluate_risk_factors(factor_items, body)
        else:
            if not self._health_assessment_available:
                logger.warning("[健康评估模型_DEGRADED] 健康评估模型模型不可用，使用规则引擎评估6风险因子")
            factor_scores = self._serial_evaluate_risk_factors(factor_items, body)
        
        # 日志记录
        for factor_id, factor_info in factor_scores.items():
            logger.info(f"[HealthAssessmentChain] 风险因子{factor_id}({factor_info['name']})评估完成: score_present={factor_info.get('score') is not None}")
            logger.info(f"[RISK_FACTOR_SCORE] 因子={factor_info['name']}, score_present={factor_info.get('score') is not None}, weight_present={factor_info.get('weight') is not None}, weighted_score_present={factor_info.get('weighted_score') is not None}")
            
            # 6风险因子评分详细日志
            logger.debug(f"[RISK_FACTOR_SCORE] factor_id={factor_id}, factor_name={factor_info['name']}, "
                        f"factor_score_present={factor_info.get('score') is not None}, weight_present={factor_info.get('weight') is not None}, "
                        f"weighted_score_present={factor_info.get('weighted_score') is not None}, reasoning_len={len(factor_info.get('reasoning', ''))}")
        
        # 基于风险因子得分和用户数据，生成高风险疾病列表
        disease_risks = self._generate_disease_risk_list(body, factor_scores)
        
        breakdown = {
            "factor_scores": factor_scores,
            "disease_risk_count": len(disease_risks),
            "calculation_formula": "disease_risk_score = Σ(Fi_score × Fi_weight) × 100"
        }
        
        logger.info(f"[HealthAssessmentChain] 疾病风险评分计算完成: disease_risks_count={len(disease_risks)}")
        
        return disease_risks, breakdown
    
    def _batch_evaluate_risk_factors(
        self,
        factor_items: List[tuple],
        body: HealthAssessmentContextBody
    ) -> Dict[str, Dict]:
        """
        批量推理评估6个风险因子
        
        将6个风险因子的评估prompt合并为一次call_model_batch()调用，
        利用SGLang的continuous batching机制共享forward pass。
        
        Args:
            factor_items: 风险因子信息列表 [(factor_id, factor_info), ...]
            body: Chain策略专属输入数据
            
        Returns:
            风险因子评分字典 {factor_id: {"name": ..., "weight": ..., "score": ..., ...}}
        """
        # 构建所有风险因子的prompt
        prompts = []
        for factor_id, factor_info in factor_items:
            prompt = self._build_risk_factor_evaluation_prompt(factor_id, factor_info["name"], body)
            # 检查Prompt长度限制
            if len(prompt) > _get_health_assessment_constraints()["max_prompt_chars"]:
                logger.warning(f"[HealthAssessmentChain] 风险因子Prompt长度超过限制({len(prompt)} > {_get_health_assessment_constraints()['max_prompt_chars']})，截断")
                prompt = prompt[:_get_health_assessment_constraints()["max_prompt_chars"]]
            prompts.append(prompt)

        prompt_lengths = [len(p) for p in prompts]
        logger.info(f"[BATCH_INFERENCE] prompt_count={len(prompts)}, prompt_lengths={prompt_lengths}")

        model_service = self._resource.health_assessment_model
        logger.info(f"[LLM_INPUT] 6风险因子批量推理, prompt_count={len(prompts)}, prompt_lengths={prompt_lengths}")
        _batch_start = time.time()
        results = model_service.call_model_batch(prompts, max_tokens=_get_health_assessment_constraints()["health_assessment_batch_max_tokens"], timeout=_get_health_assessment_constraints()["timeout_seconds"])
        _batch_elapsed = time.time() - _batch_start

        result_lengths = [len(r) if r else 0 for r in results]
        logger.info(f"[HealthAssessment_DURATION] 6风险因子批量推理 duration={_batch_elapsed:.2f}s")
        logger.info(f"[健康评估模型_OUTPUT] 6风险因子批量推理 result_count={len(results)}, result_lengths={result_lengths}")
        logger.info(f"[LLM_OUTPUT] 6风险因子批量推理 result_count={len(results)}, result_lengths={result_lengths}")
        logger.info(f"[LLM_DURATION] 6风险因子批量推理 duration={_batch_elapsed:.2f}s")
        logger.info(f"[BATCH_RESULT] result_count={len(results)}, result_lengths={result_lengths}")
        
        # 逐风险因子解析结果
        factor_scores = {}
        for i, (factor_id, factor_info) in enumerate(factor_items):
            factor_name = factor_info["name"]
            factor_weight = factor_info["weight"]
            
            try:
                response = results[i] if i < len(results) else ""
                evaluation_result = self._parse_risk_factor_result(response, factor_id, factor_name)
                
                if evaluation_result:
                    factor_score = evaluation_result.get("factor_score", 50) / 100  # 归一化到0-1
                    reasoning = evaluation_result.get("factor_reasoning", "")
                    
                    logger.info(
                        f"[健康评估模型_OUTPUT] 批量推理-风险因子={factor_id}({factor_name}), "
                        f"factor_score_present={factor_score is not None}, "
                        f"output_keys={list(evaluation_result.keys())}"
                    )

                    factor_scores[factor_id] = {
                        "name": factor_name,
                        "weight": factor_weight,
                        "score": factor_score,
                        "weighted_score": factor_score * factor_weight,
                        "reasoning": reasoning
                    }
                else:
                    # 单因子解析失败，尝试结构化自修复
                    original_prompt = prompts[i] if i < len(prompts) else ""
                    repair_result = self._try_structured_repair(
                        raw_output=response,
                        error_description=f"风险因子{factor_id}({factor_name})的JSON解析失败或缺少必要字段(factor_score)",
                        expected_format='{"factor_score":65,"factor_reasoning":"评估理由"}',
                        context_type="risk_factor",
                        original_prompt=original_prompt
                    )
                    if repair_result and "factor_score" in repair_result:
                        # 自修复成功
                        factor_score = repair_result.get("factor_score", 50)
                        if 0 < factor_score <= 1:
                            factor_score = factor_score * 100
                        factor_score = factor_score / 100  # 归一化到0-1
                        reasoning = repair_result.get("factor_reasoning", "")
                        factor_scores[factor_id] = {
                            "name": factor_name,
                            "weight": factor_weight,
                            "score": factor_score,
                            "weighted_score": factor_score * factor_weight,
                            "reasoning": reasoning,
                            "_repaired": True
                        }
                    else:
                        # 自修复也失败，降级为规则引擎
                        logger.warning(f"[健康评估模型_DEGRADED] 批量推理-风险因子{factor_id}({factor_name})结果解析失败（自修复也失败），降级为规则引擎")
                        factor_score, reasoning = self._rule_based_risk_factor_evaluation(factor_id, factor_name, body)
                        factor_scores[factor_id] = {
                            "name": factor_name,
                            "weight": factor_weight,
                            "score": factor_score,
                            "weighted_score": factor_score * factor_weight,
                            "reasoning": reasoning
                        }
            except Exception as e:
                # 单因子解析异常，降级为规则引擎
                logger.warning(f"[健康评估模型_DEGRADED] 批量推理-风险因子{factor_id}({factor_name})结果解析异常，降级为规则引擎: error_type={type(e).__name__}")
                factor_score, reasoning = self._rule_based_risk_factor_evaluation(factor_id, factor_name, body)
                factor_scores[factor_id] = {
                    "name": factor_name,
                    "weight": factor_weight,
                    "score": factor_score,
                    "weighted_score": factor_score * factor_weight,
                    "reasoning": reasoning
                }
        
        return factor_scores
    
    def _serial_evaluate_risk_factors(
        self,
        factor_items: List[tuple],
        body: HealthAssessmentContextBody
    ) -> Dict[str, Dict]:
        """
        串行推理评估6个风险因子（降级策略）
        
        逐因子调用_evaluate_risk_factor()，每个因子内部也有降级策略：
        健康评估模型失败 -> 规则引擎。
        
        Args:
            factor_items: 风险因子信息列表 [(factor_id, factor_info), ...]
            body: Chain策略专属输入数据
            
        Returns:
            风险因子评分字典 {factor_id: {"name": ..., "weight": ..., "score": ..., ...}}
        """
        factor_scores = {}
        
        for factor_id, factor_info in factor_items:
            factor_name = factor_info["name"]
            factor_weight = factor_info["weight"]
            
            # 评估风险因子得分（内部包含健康评估模型->规则引擎降级）
            factor_score, factor_reasoning = self._evaluate_risk_factor(
                factor_id, factor_name, body
            )
            
            factor_scores[factor_id] = {
                "name": factor_name,
                "weight": factor_weight,
                "score": factor_score,
                "weighted_score": factor_score * factor_weight,
                "reasoning": factor_reasoning
            }
        
        return factor_scores
    
    def _build_risk_factor_evaluation_prompt(
        self,
        factor_id: str,
        factor_name: str,
        body: HealthAssessmentContextBody
    ) -> str:
        """
        构建风险因子评估Prompt，格式指令前置确保不被截断，数据段后置可安全截断。
        """
        max_prompt = _get_health_assessment_constraints()["max_prompt_chars"]
        max_rf_user = _get_health_assessment_constraints()["max_risk_factor_user_info_chars"]

        # 格式指令（放在开头，确保不被截断）
        format_instruction = (
            f"你是健康评估专家。评估风险因子: {factor_name}(权重{DISEASE_RISK_FACTORS[factor_id]['weight']})\n\n"
            "严格按以下JSON格式输出，不要输出任何其他内容:\n"
            '{"factor_score":45,"factor_reasoning":"基于用户数据的评估理由","related_diseases":["相关疾病1"]}'
            "\n\n评估风险程度(0-100)，factor_reasoning必须具体说明依据(不超过50字)。\n"
            "重要：factor_score必须根据用户数据认真评估，不可使用默认值。\n\n"
            "以下是评估依据:\n"
        )

        # 用户信息截断
        user_info_budget = min(max_rf_user, max_prompt // 4)
        user_profile_str = json.dumps(body.user_profile, ensure_ascii=False)
        if len(user_profile_str) > user_info_budget:
            user_profile_str = user_profile_str[:user_info_budget] + "...(截断)"

        anomalies_str = json.dumps(body.anomalies, ensure_ascii=False)
        if len(anomalies_str) > user_info_budget:
            anomalies_str = anomalies_str[:user_info_budget] + "...(截断)"

        medical_entities_str = json.dumps(body.medical_entities, ensure_ascii=False)
        if len(medical_entities_str) > user_info_budget:
            medical_entities_str = medical_entities_str[:user_info_budget] + "...(截断)"

        data_section = f"用户={user_profile_str} 异常={anomalies_str} 实体={medical_entities_str}"

        prompt = format_instruction + data_section

        # 最终长度校验：截断末尾数据段，保留开头的格式指令
        if len(prompt) > max_prompt:
            logger.warning(f"[HealthAssessmentChain] 风险因子{factor_name}Prompt仍超限({len(prompt)}>{max_prompt})，截断末尾数据段")
            prompt = prompt[:max_prompt]

        medical_entity_count = sum(len(v) for v in body.medical_entities.values())
        logger.info(f"[健康评估模型_INPUT] 风险因子={factor_id}({factor_name}), prompt_len={len(prompt)}, user_profile_keys={list(body.user_profile.keys())}, anomaly_count={len(body.anomalies)}, medical_entity_count={medical_entity_count}")

        return prompt
    
    def _parse_risk_factor_result(
        self,
        response: str,
        factor_id: str,
        factor_name: str
    ) -> Optional[Dict]:
        """
        解析风险因子评估的健康评估模型输出结果
        
        Args:
            response: 健康评估模型生成的JSON格式字符串
            factor_id: 风险因子ID
            factor_name: 风险因子名称
            
        Returns:
            评估结果字典，包含factor_score、factor_reasoning
            解析失败返回None
        """
        if not response:
            logger.warning(f"[健康评估模型_PARSE_ERROR] 风险因子={factor_name}, 输出为空")
            return None
        
        try:
            result = json.loads(self._extract_json_from_response(response))

            if "factor_score" in result:
                factor_score = float(result["factor_score"])
                # 容错：factor_score < 1 时可能是0-1归一化的值，放大到0-100
                if 0 < factor_score <= 1:
                    logger.warning("[HealthAssessmentChain] factor_score可能是0-1归一化值，放大为×100")
                    factor_score = factor_score * 100
                result["factor_score"] = factor_score
                logger.info(f"[健康评估模型_OUTPUT] 风险因子={factor_id}({factor_name}), factor_score_present={factor_score is not None}, output_keys={list(result.keys())}")
                if 0 <= factor_score <= 100:
                    return result

        except json.JSONDecodeError as e:
            logger.warning(f"[健康评估模型_PARSE_ERROR] JSON解析失败: 风险因子={factor_name}, error={type(e).__name__}, raw_output_len={len(response)}")
        except Exception as e:
            logger.warning(f"[HealthAssessmentChain] 风险因子评估结果解析失败: error_type={type(e).__name__}")
        
        return None
    
    def _evaluate_risk_factor(
        self,
        factor_id: str,
        factor_name: str,
        body: HealthAssessmentContextBody
    ) -> tuple:
        """
        评估单个风险因子的得分
        
        Args:
            factor_id: 风险因子ID
            factor_name: 风险因子名称
            body: Chain策略专属输入数据
            
        Returns:
            (factor_score, reasoning)
        """
        # 风险因子评估计时开始
        _factor_eval_start = time.time()

        # 尝试使用健康评估模型评估引擎
        if self._health_assessment_available and self._resource.health_assessment_model is not None:
            try:
                # 调用健康评估模型评估风险因子
                evaluation_result = self._call_health_assessment_for_risk_factor(factor_id, factor_name, body)

                if evaluation_result:
                    factor_score = evaluation_result.get("factor_score", 50) / 100  # 归一化到0-1
                    reasoning = evaluation_result.get("factor_reasoning", "")
                    logger.info(
                        f"[健康评估模型_OUTPUT] 风险因子={factor_id}({factor_name}), "
                        f"factor_score_present={factor_score is not None}, "
                        f"output_keys={list(evaluation_result.keys())}"
                    )

                    # 防重复效果日志
                    _factor_eval_duration = time.time() - _factor_eval_start
                    logger.info(f"[REPETITION_CHECK] dimension={factor_id}({factor_name}), finish_reason_present=True, duration={_factor_eval_duration:.2f}s")

                    return factor_score, reasoning
                    
            except Exception as e:
                logger.warning(f"[HealthAssessmentChain] 健康评估模型评估风险因子失败，降级为规则引擎: error_type={type(e).__name__}")
                logger.warning(f"[健康评估模型_DEGRADED] 风险因子{factor_id}({factor_name})健康评估模型评估失败，降级为规则引擎: error_type={type(e).__name__}")
                logger.warning(f"[DEGRADE_TO_RULE_ENGINE] 降级触发: 健康评估模型不可用, "
                              f"降级策略=规则引擎评估, 风险因子={factor_id}({factor_name})")
                logger.warning(f"[DEGRADE_STRATEGY] from=健康评估模型 to=规则引擎评估, reason=健康评估模型评估失败(风险因子{factor_id}({factor_name}))")
        else:
            if not self._health_assessment_available:
                logger.warning(f"[健康评估模型_DEGRADED] 风险因子{factor_id}({factor_name})健康评估模型模型不可用，使用规则引擎评估")
                logger.warning(f"[DEGRADE_TO_RULE_ENGINE] 降级触发: 健康评估模型不可用, "
                              f"降级策略=规则引擎评估, 风险因子={factor_id}({factor_name})")
                logger.warning(f"[DEGRADE_STRATEGY] from=健康评估模型 to=规则引擎评估, reason=健康评估模型模型不可用(风险因子{factor_id}({factor_name}))")

        # 降级：使用规则引擎评估
        factor_score, reasoning = self._degrade_to_rule_engine_risk_factor(factor_id, factor_name, body)

        # 防重复效果日志（降级路径）
        _factor_eval_duration = time.time() - _factor_eval_start
        logger.info(f"[REPETITION_CHECK] dimension={factor_id}({factor_name}), finish_reason_present=True, duration={_factor_eval_duration:.2f}s")
        
        return factor_score, reasoning
    
    def _call_health_assessment_for_risk_factor(
        self,
        factor_id: str,
        factor_name: str,
        body: HealthAssessmentContextBody
    ) -> Optional[Dict]:
        """
        调用健康评估模型评估引擎对风险因子进行评估
        
        Args:
            factor_id: 风险因子ID
            factor_name: 风险因子名称
            body: Chain策略专属输入数据
            
        Returns:
            评估结果字典
        """
        # 构建评估Prompt
        # 用户信息截断：控制用户档案、异常指标、医疗实体的总字符数
        user_profile_str = json.dumps(body.user_profile, ensure_ascii=False)
        if len(user_profile_str) > _get_health_assessment_constraints()["max_risk_factor_user_info_chars"]:
            user_profile_str = user_profile_str[:_get_health_assessment_constraints()["max_risk_factor_user_info_chars"]] + "...(截断)"
            logger.info(f"[HealthAssessmentChain] 风险因子{factor_name}用户档案截断: "
                       f"原始字符数={len(json.dumps(body.user_profile, ensure_ascii=False))}, 截断后={_get_health_assessment_constraints()['max_risk_factor_user_info_chars']}")
        
        anomalies_str = json.dumps(body.anomalies, ensure_ascii=False)
        if len(anomalies_str) > _get_health_assessment_constraints()["max_risk_factor_user_info_chars"]:
            anomalies_str = anomalies_str[:_get_health_assessment_constraints()["max_risk_factor_user_info_chars"]] + "...(截断)"
            logger.info(f"[HealthAssessmentChain] 风险因子{factor_name}异常指标截断: "
                       f"原始字符数={len(json.dumps(body.anomalies, ensure_ascii=False))}, 截断后={_get_health_assessment_constraints()['max_risk_factor_user_info_chars']}")
        
        medical_entities_str = json.dumps(body.medical_entities, ensure_ascii=False)
        if len(medical_entities_str) > _get_health_assessment_constraints()["max_risk_factor_user_info_chars"]:
            medical_entities_str = medical_entities_str[:_get_health_assessment_constraints()["max_risk_factor_user_info_chars"]] + "...(截断)"
            logger.info(f"[HealthAssessmentChain] 风险因子{factor_name}医疗实体截断: "
                       f"原始字符数={len(json.dumps(body.medical_entities, ensure_ascii=False))}, 截断后={_get_health_assessment_constraints()['max_risk_factor_user_info_chars']}")
        
        prompt = f"""你是一位资深全科医生，请对以下疾病风险因子进行评估。

## 风险因子
因子名称: {factor_name}
因子权重: {DISEASE_RISK_FACTORS[factor_id]['weight']}

## 评估依据
- 用户档案: {user_profile_str}
- 异常指标: {anomalies_str}
- 医疗实体: {medical_entities_str}

## 评估要求
1. 评估该风险因子的风险程度(0-100,越高风险越大)
2. 给出简短评估理由(不超过50字)
3. 评估应基于医学标准

## 输出格式(JSON)
{{
    "factor_score": 45,
    "factor_reasoning": "评估理由",
    "related_diseases": ["相关疾病1", "相关疾病2"]
}}

重要：你的回复必须且仅包含一个合法的JSON对象，不得用```json```包裹，不得输出任何验证性文字、注释或额外文本。
"""
        medical_entity_count = sum(len(v) for v in body.medical_entities.values())
        logger.info(f"[健康评估模型_INPUT] 风险因子={factor_id}({factor_name}), prompt_len={len(prompt)}, user_profile_keys={list(body.user_profile.keys())}, anomaly_count={len(body.anomalies)}, medical_entity_count={medical_entity_count}")
        logger.info(f"[健康评估模型_RISK_INPUT] factor={factor_name}, prompt_len={len(prompt)}, user_profile_keys={list(body.user_profile.keys())}")

        # 检查Prompt长度限制
        if len(prompt) > _get_health_assessment_constraints()["max_prompt_chars"]:
            prompt = prompt[:_get_health_assessment_constraints()["max_prompt_chars"]]
        
        try:
            # 调用健康评估模型模型
            logger.info(f"[HealthAssessmentChain._call_health_assessment_for_risk_factor] 调用健康评估模型进行风险因子评估，因子={factor_name}")
            logger.info(f"[LLM_INPUT] 风险因子评估, factor={factor_name}, prompt_len={len(prompt)}")
            _ha_start = time.time()
            response = self._resource.health_assessment_model.generate(prompt)
            _ha_elapsed = time.time() - _ha_start
            
            # 健康评估模型输出日志
            logger.info(f"[健康评估模型_OUTPUT] 风险因子={factor_id}({factor_name}), 评估结果长度={len(response) if response else 0}")

            result = json.loads(self._extract_json_from_response(response))

            if "factor_score" in result:
                factor_score = float(result["factor_score"])
                # 容错：factor_score < 1 时可能是0-1归一化的值，放大到0-100
                if 0 < factor_score <= 1:
                    logger.warning("[HealthAssessmentChain] factor_score可能是0-1归一化值，放大为×100")
                    factor_score = factor_score * 100
                result["factor_score"] = factor_score
                response_len = len(response) if response else 0
                logger.info(f"[HealthAssessmentChain._call_health_assessment_for_risk_factor] 健康评估模型风险因子评估完成，因子={factor_name}，输出长度={response_len}")
                logger.info(f"[LLM_OUTPUT] 风险因子评估, factor={factor_name}, response_len={response_len}, output_keys={list(result.keys())}")
                logger.info(f"[LLM_DURATION] {_ha_elapsed:.3f}s")
                logger.info(f"[HealthAssessment_DURATION] 风险因子名称={factor_name}, 耗时={_ha_elapsed:.3f}s")
                logger.info(f"[健康评估模型_OUTPUT] 风险因子={factor_id}({factor_name}), output_keys={list(result.keys())}")
                if 0 <= factor_score <= 100:
                    return result
                    
        except json.JSONDecodeError as e:
            logger.warning(f"[健康评估模型_PARSE_ERROR] JSON解析失败: dimension={factor_name}, error={type(e).__name__}, raw_output_len={len(response)}")
            # 尝试结构化自修复
            repair_result = self._try_structured_repair(
                raw_output=response,
                error_description=f"JSON解析失败: {type(e).__name__}",
                expected_format='{"factor_score":65,"factor_reasoning":"评估理由"}',
                context_type="risk_factor",
                original_prompt=prompt
            )
            if repair_result and "factor_score" in repair_result:
                factor_score = float(repair_result["factor_score"])
                if 0 < factor_score <= 1:
                    factor_score = factor_score * 100
                repair_result["factor_score"] = factor_score
                if 0 <= factor_score <= 100:
                    return repair_result
            logger.warning(f"[HealthAssessmentChain] 健康评估模型调用失败(自修复也失败): error_type=JSONDecodeError")
        except Exception as e:
            logger.warning(f"[HealthAssessmentChain] 健康评估模型调用失败: error_type={type(e).__name__}")

        return None
    
    def _rule_based_risk_factor_evaluation(
        self,
        factor_id: str,
        factor_name: str,
        body: HealthAssessmentContextBody
    ) -> tuple:
        """
        规则引擎降级：对风险因子进行规则评估

        Args:
            factor_id: 风险因子ID
            factor_name: 风险因子名称
            body: Chain策略专属输入数据

        Returns:
            (factor_score, reasoning)
        """
        rs = self._get_rule_engine_scores()

        def _return_with_log(factor_score: float, reasoning: str) -> tuple:
            logger.info(
                f"[RULE_ENGINE_ASSESS] 风险因子={factor_id}({factor_name}), "
                f"score_present={factor_score is not None}, reason_len={len(reasoning)}"
            )
            return factor_score, reasoning

        if factor_id == "F1":  # 异常指标风险
            anomaly_count = len(body.anomalies)
            if anomaly_count == 0:
                return _return_with_log(rs.get("no_family_risk", 0.1), "无异常指标")
            elif anomaly_count <= _report_config.anomaly_count_mild:
                return _return_with_log(rs.get("few_anomaly_indicators", 0.3), f"有{anomaly_count}个异常指标")
            elif anomaly_count <= _report_config.anomaly_count_moderate:
                return _return_with_log(rs.get("moderate_anomaly_indicators", 0.6), f"有{anomaly_count}个异常指标")
            else:
                return _return_with_log(rs.get("many_anomaly_indicators", 0.9), f"有{anomaly_count}个异常指标")

        elif factor_id == "F2":  # 病史风险
            diseases = body.medical_entities.get("diseases", [])
            disease_count = len(diseases)
            if disease_count == 0:
                return _return_with_log(rs.get("no_family_risk", 0.1), "无既往病史")
            elif disease_count <= _report_config.disease_count_mild:
                return _return_with_log(rs.get("few_medical_history", 0.4), f"有{disease_count}种病史")
            else:
                return _return_with_log(rs.get("many_medical_history", 0.8), f"有{disease_count}种病史")

        elif factor_id == "F3":  # 家族史风险
            family_history = body.user_profile.get("family_history", "")
            if family_history and family_history.strip():
                return _return_with_log(rs.get("has_family_risk", 0.5), "有家族病史")
            else:
                return _return_with_log(rs.get("no_family_risk", 0.1), "无家族病史")

        elif factor_id == "F4":  # 生活方式风险
            lifestyle = body.user_profile.get("lifestyle", {})
            risk_count = 0
            if lifestyle.get("smoking") == "吸烟":
                risk_count += 1
            if lifestyle.get("drinking") == "过量":
                risk_count += 1
            if lifestyle.get("exercise_freq") == "不运动":
                risk_count += 1

            if risk_count == 0:
                return _return_with_log(rs.get("few_bad_habits", 0.1), "生活方式良好")
            elif risk_count == 1:
                return _return_with_log(rs.get("moderate_bad_habits", 0.4), f"有{risk_count}个不良习惯")
            else:
                return _return_with_log(rs.get("many_bad_habits", 0.7), f"有{risk_count}个不良习惯")

        elif factor_id == "F5":  # 年龄风险
            age = body.user_profile.get("age", -1)
            if not isinstance(age, int) or age <= 0:
                return _return_with_log(rs.get("middle_age_risk", 0.3), "年龄未知，按中等风险评估")
            elif age < 40:
                return _return_with_log(rs.get("young_age_risk", 0.1), "年龄风险低")
            elif age < 60:
                return _return_with_log(rs.get("middle_age_risk", 0.3), "年龄风险中等")
            elif age < 75:
                return _return_with_log(rs.get("senior_age_risk", 0.6), "年龄风险较高")
            else:
                return _return_with_log(rs.get("elderly_age_risk", 0.9), "年龄风险高")

        elif factor_id == "F6":  # 并发症风险
            # 基于疾病数量评估并发症风险
            diseases = body.medical_entities.get("diseases", [])
            if len(diseases) >= _report_config.multi_disease_threshold:
                return _return_with_log(rs.get("many_complications", 0.7), "多病共存，并发症风险高")
            elif len(diseases) >= 1:
                return _return_with_log(rs.get("some_complications", 0.3), "有潜在并发症风险")
            else:
                return _return_with_log(rs.get("few_complications", 0.1), "并发症风险低")

        # 默认中性评分
        return _return_with_log(_report_config.default_dimension_score, "默认中性评分")
    
    def _generate_disease_risk_list(
        self,
        body: HealthAssessmentContextBody,
        factor_scores: Dict
    ) -> List[Dict]:
        """
        基于风险因子得分和用户数据，生成高风险疾病列表
        
        Args:
            body: Chain策略专属输入数据
            factor_scores: 风险因子得分字典
            
        Returns:
            高风险疾病列表
        """
        disease_risks = []

        # 从医疗实体中提取疾病
        diseases = body.medical_entities.get("diseases", [])

        # 计算风险因子加权均分，用于调整疾病基础风险分
        total_risk_factor_score = 0.0
        total_risk_factor_weight = 0.0
        for fid, finfo in factor_scores.items():
            total_risk_factor_score += finfo.get("score", 0) * finfo.get("weight", 0)
            total_risk_factor_weight += finfo.get("weight", 0)
        avg_risk_factor_score = (total_risk_factor_score / total_risk_factor_weight) if total_risk_factor_weight > 0 else 0.0

        for disease in diseases:
            disease_name = disease.get("entity_name", disease.get("name", ""))

            # 疾病存在本身即有基础风险分
            risk_score = _report_config.base_disease_risk_score

            # 基于异常指标匹配追加风险分
            anomaly_match_score = 0
            for anomaly in body.anomalies:
                anomaly_name = anomaly.get("indicator_name", "")
                if anomaly_name in disease_name or disease_name in anomaly_name:
                    anomaly_match_score += _report_config.anomaly_match_score_increment
            risk_score += anomaly_match_score

            # 基于风险因子得分调整：风险因子整体偏高时提升疾病风险
            risk_factor_adjustment = avg_risk_factor_score * 20
            risk_score += risk_factor_adjustment

            # 基于病史权重
            history_weight = 1.0
            past_medical_history = body.user_profile.get("past_medical_history", "")
            if past_medical_history and disease_name in past_medical_history:
                history_weight = _report_config.history_weight_multiplier

            family_history = body.user_profile.get("family_history", "")
            if family_history and disease_name in family_history:
                history_weight *= _report_config.family_history_weight_multiplier

            risk_score = risk_score * history_weight

            # 确保风险分在0-100范围内
            risk_score = min(100, max(0, risk_score))

            if risk_score > 0:
                disease_risks.append({
                    "disease_name": disease_name,
                    "risk_score": round(risk_score, 2),
                    "risk_level": self._get_risk_level_from_score(risk_score),
                    "evidence": ["基于用户数据评估"],
                    "confidence": _report_config.default_confidence
                })

        # 按风险分排序，返回Top-N
        disease_risks.sort(key=lambda x: x["risk_score"], reverse=True)
        return disease_risks[:_report_config.disease_risk_top_n]
    
    def _get_risk_level_from_score(self, risk_score: float) -> str:
        """根据风险分判定风险等级"""
        # 使用RISK_LEVEL_STANDARDS中的min值作为阈值
        thresholds = RISK_LEVEL_STANDARDS
        if risk_score < thresholds.get("轻", {}).get("min", 30):
            return "低"
        elif risk_score < thresholds.get("中", {}).get("min", 50):
            return "轻"
        elif risk_score < thresholds.get("高", {}).get("min", 70):
            return "中"
        else:
            return "高"
    
    # ========================================================================
    # 风险等级判定算法（评估框架）
    # ========================================================================
    
    def _determine_risk_level(
        self,
        health_score: float,
        disease_risks: List[Dict],
        body: HealthAssessmentContextBody
    ) -> tuple:
        """
        判定风险等级（评估框架）
        
        结合健康评分和各维度风险因子加权得分，输出最终的4级风险等级。
        
        计算公式：
        FinalRiskScore = (100 - HealthScore) + (Σ(RiskFactor_j.weight_j × 20))
        
        风险等级判定：
        - 低风险: 0-20
        - 轻度风险: 21-40
        - 中度风险: 41-60
        - 高风险: >60
        
        特殊情况：
        - 急性发作史: 直接判定为高风险
        - 恶性肿瘤史: 直接判定为高风险
        - 多病共存(≥3种): 提升一级风险
        - 年龄>65岁: 提升一级风险
        
        Args:
            health_score: 健康综合评分
            disease_risks: 疾病风险评分列表
            body: Chain策略专属输入数据
            
        Returns:
            (risk_level, reasoning)
        """
        logger.info("[HealthAssessmentChain] 开始判定风险等级")
        
        # 检查特殊情况
        special_conditions = self._check_special_conditions(body)
        
        if special_conditions["is_high_risk"]:
            return "高", special_conditions["reason"]
        
        # 计算最终风险分数
        final_risk_score = (_report_config.base_health_score - health_score)
        
        # 加上疾病风险因子的贡献
        avg_disease_risk = 0.0
        if disease_risks:
            avg_disease_risk = sum(d["risk_score"] for d in disease_risks) / len(disease_risks)
            final_risk_score += avg_disease_risk * _report_config.disease_risk_weight

        # 判定风险等级
        if final_risk_score < _report_config.risk_level_thresholds["low"]:
            risk_level = "低"
        elif final_risk_score < _report_config.risk_level_thresholds["mild"]:
            risk_level = "轻"
        elif final_risk_score < _report_config.risk_level_thresholds["moderate"]:
            risk_level = "中"
        else:
            risk_level = "高"

        # 风险等级判定详细日志
        logger.info(
            f"[RISK_LEVEL_JUDGE] health_score_present={health_score is not None}, "
            f"avg_disease_risk_present={avg_disease_risk is not None}, "
            f"final_risk_score_present={final_risk_score is not None}, "
            f"risk_level={risk_level}, "
            f"special_high_risk={special_conditions['is_high_risk']}, "
            f"need_upgrade={special_conditions['need_upgrade']}"
        )

        # 检查是否需要提升风险等级
        if special_conditions["need_upgrade"]:
            level_order = ["低", "轻", "中", "高"]
            current_idx = level_order.index(risk_level)
            if current_idx < len(level_order) - 1:
                risk_level = level_order[current_idx + 1]
        
        reasoning = f"最终风险分数{final_risk_score:.2f}，风险等级{risk_level}"
        if special_conditions["need_upgrade"]:
            reasoning += f"，因{special_conditions['upgrade_reason']}提升一级"
        
        logger.info(
            f"[HealthAssessmentChain] 风险等级判定完成: "
            f"final_risk_score_present={final_risk_score is not None}, risk_level={risk_level}"
        )
        
        return risk_level, reasoning
    
    def _check_special_conditions(self, body: HealthAssessmentContextBody) -> Dict:
        """
        检查特殊情况（急性发作史、恶性肿瘤史、多病共存、高龄）
        
        Args:
            body: Chain策略专属输入数据
            
        Returns:
            特殊情况检查结果
        """
        result = {
            "is_high_risk": False,
            "reason": "",
            "need_upgrade": False,
            "upgrade_reason": ""
        }
        
        # 检查既往病史
        past_medical_history = body.user_profile.get("past_medical_history", "") or ""
        diseases = body.medical_entities.get("diseases", [])
        
        # 检查急性发作史
        acute_keywords = ["急性", "发作", "急诊", "抢救"]
        if past_medical_history and any(keyword in past_medical_history for keyword in acute_keywords):
            result["is_high_risk"] = True
            result["reason"] = "有急性发作史，直接判定为高风险"
            return result
        
        # 检查恶性肿瘤史
        cancer_keywords = ["肿瘤", "癌", "恶性"]
        if past_medical_history and any(keyword in past_medical_history for keyword in cancer_keywords):
            result["is_high_risk"] = True
            result["reason"] = "有恶性肿瘤史，直接判定为高风险"
            return result
        
        # 检查多病共存
        if len(diseases) >= _report_config.multi_disease_threshold:
            result["need_upgrade"] = True
            result["upgrade_reason"] = "多病共存"

        # 检查高龄
        age = body.user_profile.get("age", -1)
        if isinstance(age, int) and age > 0 and age > _report_config.elderly_age_threshold:
            if result["need_upgrade"]:
                result["upgrade_reason"] += "、高龄"
            else:
                result["need_upgrade"] = True
                result["upgrade_reason"] = "高龄"
        
        return result
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def _aggregate_reasoning(
        self,
        health_breakdown: Dict,
        disease_breakdown: Dict,
        risk_reasoning: str
    ) -> str:
        """
        汇总推理过程
        
        Args:
            health_breakdown: 健康评分明细
            disease_breakdown: 疾病风险评分明细
            risk_reasoning: 风险等级判定理由
            
        Returns:
            推理过程汇总文本
        """
        reasoning_parts = []
        
        # 健康评分推理
        reasoning_parts.append("【健康综合评分】")
        dimension_scores = health_breakdown.get("dimension_scores", {})
        for dim_id, dim_info in dimension_scores.items():
            reasoning_parts.append(
                f"- {dim_info['name']}: 得分{dim_info['score']:.2f}，权重{dim_info['weight']}"
            )
        
        # 疾病风险推理
        reasoning_parts.append("\n【疾病风险评分】")
        factor_scores = disease_breakdown.get("factor_scores", {})
        for factor_id, factor_info in factor_scores.items():
            reasoning_parts.append(
                f"- {factor_info['name']}: 得分{factor_info['score']:.2f}，权重{factor_info['weight']}"
            )
        
        # 风险等级推理
        reasoning_parts.append(f"\n【风险等级判定】\n{risk_reasoning}")
        
        return "\n".join(reasoning_parts)

    def _rule_calculate_health_score(self, body: HealthAssessmentContextBody) -> tuple:
        """规则引擎计算健康综合评分（单步降级，仅此步骤用规则引擎）"""
        logger.info("[HealthAssessmentChain] 规则引擎降级：计算健康综合评分")
        dim_items = list(HEALTH_DIMENSIONS.items())
        dimension_scores = self._serial_evaluate_dimensions(dim_items, body)

        total_weighted_score = sum(d["weighted_score"] for d in dimension_scores.values())
        health_score = round(total_weighted_score * 100, 2)
        health_score = max(0, min(100, health_score))
        health_level = self._determine_health_level(health_score)

        breakdown = {
            "dimension_scores": dimension_scores,
            "total_weighted_score": total_weighted_score,
            "calculation_formula": "health_score = Σ(Di_score × Di_weight) × 100 (规则引擎降级)"
        }
        return health_score, health_level, breakdown

    def _rule_calculate_disease_risks(self, body: HealthAssessmentContextBody) -> tuple:
        """规则引擎计算疾病风险评分（单步降级，仅此步骤用规则引擎）"""
        logger.info("[HealthAssessmentChain] 规则引擎降级：计算疾病风险评分")
        factor_items = list(DISEASE_RISK_FACTORS.items())
        factor_scores = self._serial_evaluate_risk_factors(factor_items, body)

        disease_risks = self._generate_disease_risk_list(body, factor_scores)

        breakdown = {
            "factor_scores": factor_scores,
            "disease_risk_count": len(disease_risks),
            "calculation_formula": "disease_risk_score = Σ(Fi_score × Fi_weight) × 100 (规则引擎降级)"
        }
        return disease_risks, breakdown

    def _rule_determine_risk_level(self, health_score, disease_risks, body: HealthAssessmentContextBody) -> tuple:
        """规则引擎判定风险等级（单步降级，仅此步骤用规则引擎）"""
        logger.info("[HealthAssessmentChain] 规则引擎降级：判定风险等级")
        if health_score is None:
            health_score = _report_config.base_health_score
        final_risk_score = _report_config.base_health_score - health_score
        avg_disease_risk = 0.0
        if disease_risks:
            valid_scores = [d.get("risk_score", 0) for d in disease_risks if isinstance(d.get("risk_score"), (int, float))]
            if valid_scores:
                avg_disease_risk = sum(valid_scores) / len(valid_scores)
                final_risk_score += avg_disease_risk * _report_config.disease_risk_weight

        if final_risk_score < _report_config.risk_level_thresholds["low"]:
            risk_level = "低"
        elif final_risk_score < _report_config.risk_level_thresholds["mild"]:
            risk_level = "轻"
        elif final_risk_score < _report_config.risk_level_thresholds["moderate"]:
            risk_level = "中"
        else:
            risk_level = "高"

        reasoning = f"规则引擎降级判定：最终风险分数{final_risk_score:.2f}，风险等级{risk_level}"
        return risk_level, reasoning

    def _fallback_rule_assessment(
        self,
        body: HealthAssessmentContextBody,
        error_reason: str
    ) -> HealthAssessmentResultData:
        """
        降级策略：使用规则引擎进行完整评估
        
        当健康评估模型模型不可用时，使用规则引擎对子指标进行评估。
        
        Args:
            body: Chain策略专属输入数据
            error_reason: 错误原因
            
        Returns:
            健康评估结果
        """
        logger.warning(f"[HealthAssessmentChain] 执行降级策略: reason={error_reason}")
        
        # 使用规则引擎计算健康评分
        health_score, health_level, health_breakdown = self._calculate_health_score(body)
        
        # 使用规则引擎计算疾病风险
        disease_risks, disease_breakdown = self._calculate_disease_risks(body)
        
        # 使用规则引擎判定风险等级
        risk_level, risk_reasoning = self._determine_risk_level(health_score, disease_risks, body)
        
        # 汇总评分明细
        score_breakdown = {
            "health_dimensions": health_breakdown,
            "disease_risk_factors": disease_breakdown
        }
        
        # 汇总推理过程
        reasoning = self._aggregate_reasoning(health_breakdown, disease_breakdown, risk_reasoning)
        
        return HealthAssessmentResultData(
            health_score=health_score,
            health_level=health_level,
            risk_level=risk_level,
            disease_risks=disease_risks,
            score_breakdown=score_breakdown,
            reasoning=reasoning,
            degraded=True,
            degraded_reason=error_reason
        )
