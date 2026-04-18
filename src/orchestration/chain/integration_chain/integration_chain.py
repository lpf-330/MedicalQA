# -*- coding: utf-8 -*-
"""
整合计算Chain策略

实现健康报告生成业务的整合计算Chain策略，包含健康评分计算、风险等级判定、疾病风险评分、报告素材准备。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.orchestration.chain.chain import Chain
from src.orchestration.chain.data_classes import ChainContext, ChainResult

logger = logging.getLogger(__name__)


@dataclass
class IntegrationContextBody:
    """
    整合计算Chain策略专属输入数据类

    Attributes:
        dimension_results: 8个维度的评估结果
        knowledge_results: 知识检索结果
        anomalies: 异常指标
        risk_factors: 风险因子
    """
    dimension_results: Dict[str, Dict] = field(default_factory=dict)
    knowledge_results: List[Dict] = field(default_factory=list)
    anomalies: List[Dict] = field(default_factory=list)
    risk_factors: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "dimension_results": self.dimension_results,
            "knowledge_results": self.knowledge_results,
            "anomalies": self.anomalies,
            "risk_factors": self.risk_factors
        }


@dataclass
class IntegrationResultData:
    """
    整合计算Chain策略专属输出数据类

    Attributes:
        health_score: 健康综合评分，0-100
        health_level: 健康等级：优秀/良好/一般/较差/差
        risk_level: 风险等级：低/轻/中/高
        risk_diseases: 高风险疾病Top-5，包含disease_name、risk_score、confidence、evidence
        report_materials: 报告素材
    """
    health_score: int = 0
    health_level: str = "一般"
    risk_level: str = "低"
    risk_diseases: List[Dict] = field(default_factory=list)
    report_materials: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "health_score": self.health_score,
            "health_level": self.health_level,
            "risk_level": self.risk_level,
            "risk_diseases": self.risk_diseases,
            "report_materials": self.report_materials
        }


@dataclass
class IntegrationResource:
    """
    整合计算Chain策略专属资源类

    暂无外部资源依赖
    """
    pass


class IntegrationChain(Chain[ChainContext[IntegrationContextBody], ChainResult[IntegrationResultData]]):
    """
    整合计算Chain策略类

    实现健康报告生成业务的整合计算固定流程：
    1. 健康综合评分计算（算法2）
    2. 风险等级判定（算法3）
    3. 疾病风险评分（算法1）
    4. 报告素材准备
    """

    def __init__(self, resource: IntegrationResource):
        """
        初始化整合计算Chain策略

        Args:
            resource: Chain策略专属资源
        """
        self._resource = resource

    def execute(self, chain_context: ChainContext[IntegrationContextBody]) -> ChainResult[IntegrationResultData]:
        """
        执行Chain策略

        Args:
            chain_context: Chain输入数据容器

        Returns:
            ChainResult: Chain输出数据容器
        """
        start_time = time.time()
        logger.info(f"[IntegrationChain] 开始执行Chain: session_id={chain_context.session_id}")

        body = chain_context.body
        if body is None:
            logger.warning(f"[IntegrationChain] 输入数据为空: session_id={chain_context.session_id}")
            return ChainResult(
                session_id=chain_context.session_id,
                data=IntegrationResultData()
            )

        # 步骤1：健康综合评分计算（算法2）
        health_score = self._calculate_health_score(body)
        health_level = self._determine_health_level(health_score)
        logger.info(f"[IntegrationChain] 健康评分计算完成: health_score={health_score}, health_level={health_level}")

        # 步骤2：风险等级判定（算法3）
        risk_level = self._calculate_risk_level(body)
        logger.info(f"[IntegrationChain] 风险等级判定完成: risk_level={risk_level}")

        # 步骤3：疾病风险评分（算法1）
        risk_diseases = self._calculate_disease_risks(body)
        logger.info(f"[IntegrationChain] 疾病风险评分完成: risk_diseases_count={len(risk_diseases)}")

        # 步骤4：报告素材准备
        report_materials = self._prepare_report_materials(body, health_score, health_level, risk_level, risk_diseases)
        logger.info(f"[IntegrationChain] 报告素材准备完成")

        result_data = IntegrationResultData(
            health_score=health_score,
            health_level=health_level,
            risk_level=risk_level,
            risk_diseases=risk_diseases,
            report_materials=report_materials
        )

        elapsed = time.time() - start_time
        logger.info(f"[IntegrationChain] Chain执行完成: session_id={chain_context.session_id}, "
                   f"health_score={health_score}, risk_level={risk_level}, elapsed={elapsed:.2f}s")

        return ChainResult(session_id=chain_context.session_id, data=result_data)

    def _calculate_health_score(self, body: IntegrationContextBody) -> int:
        """
        计算健康综合评分（算法2）

        基于四个维度计算健康评分：
        - 维度A：基础生理指标（满分30分）
        - 维度B：生活方式（满分25分）
        - 维度C：病史情况（满分25分）
        - 维度D：其他综合（满分20分）

        Args:
            body: Chain策略专属输入数据

        Returns:
            健康综合评分（0-100）
        """
        logger.info("[IntegrationChain] 开始计算健康综合评分")

        # 维度A：基础生理指标（满分30分）
        score_a = self._calculate_dimension_a(body)
        logger.info(f"[IntegrationChain] 维度A（基础生理指标）得分: {score_a}/30")

        # 维度B：生活方式（满分25分）
        score_b = self._calculate_dimension_b(body)
        logger.info(f"[IntegrationChain] 维度B（生活方式）得分: {score_b}/25")

        # 维度C：病史情况（满分25分）
        score_c = self._calculate_dimension_c(body)
        logger.info(f"[IntegrationChain] 维度C（病史情况）得分: {score_c}/25")

        # 维度D：其他综合（满分20分）
        score_d = self._calculate_dimension_d(body)
        logger.info(f"[IntegrationChain] 维度D（其他综合）得分: {score_d}/20")

        # 总分计算
        total_score = score_a + score_b + score_c + score_d

        # 确保分数在0-100范围内
        total_score = max(0, min(100, total_score))

        logger.info(f"[IntegrationChain] 健康综合评分计算完成: total_score={total_score}")

        return total_score

    def _calculate_dimension_a(self, body: IntegrationContextBody) -> int:
        """
        计算维度A：基础生理指标（满分30分）

        评分规则：
        - 血压正常：+10分，血压偏高/偏低：+5分，血压严重异常：+0分
        - 血糖正常：+10分，血糖偏高/偏低：+5分，血糖严重异常：+0分
        - 心率正常：+5分，心率异常：+2分
        - 血氧正常：+5分，血氧偏低：+2分

        Args:
            body: Chain策略专属输入数据

        Returns:
            维度A得分（0-30分）
        """
        score = 0

        # 从异常指标中提取血压、血糖、心率、血氧的异常情况
        anomalies = body.anomalies
        anomaly_indicators = {item.get("indicator_name", ""): item for item in anomalies}

        # 血压评分（满分10分）
        if "血压" in anomaly_indicators or "收缩压" in anomaly_indicators or "舒张压" in anomaly_indicators:
            # 检查血压异常程度
            bp_anomaly = anomaly_indicators.get("血压") or anomaly_indicators.get("收缩压") or anomaly_indicators.get("舒张压")
            if bp_anomaly:
                severity = bp_anomaly.get("severity", "normal")
                if severity == "normal":
                    score += 10
                elif severity in ["mild", "偏高", "偏低"]:
                    score += 5
                else:
                    score += 0
            else:
                score += 10
        else:
            # 无血压异常记录，默认正常
            score += 10

        # 血糖评分（满分10分）
        if "血糖" in anomaly_indicators or "空腹血糖" in anomaly_indicators:
            bg_anomaly = anomaly_indicators.get("血糖") or anomaly_indicators.get("空腹血糖")
            if bg_anomaly:
                severity = bg_anomaly.get("severity", "normal")
                if severity == "normal":
                    score += 10
                elif severity in ["mild", "偏高", "偏低"]:
                    score += 5
                else:
                    score += 0
            else:
                score += 10
        else:
            score += 10

        # 心率评分（满分5分）
        if "心率" in anomaly_indicators:
            hr_anomaly = anomaly_indicators.get("心率")
            if hr_anomaly:
                severity = hr_anomaly.get("severity", "normal")
                if severity == "normal":
                    score += 5
                else:
                    score += 2
            else:
                score += 5
        else:
            score += 5

        # 血氧评分（满分5分）
        if "血氧" in anomaly_indicators or "血氧饱和度" in anomaly_indicators:
            spo2_anomaly = anomaly_indicators.get("血氧") or anomaly_indicators.get("血氧饱和度")
            if spo2_anomaly:
                severity = spo2_anomaly.get("severity", "normal")
                if severity == "normal":
                    score += 5
                else:
                    score += 2
            else:
                score += 5
        else:
            score += 5

        return score

    def _calculate_dimension_b(self, body: IntegrationContextBody) -> int:
        """
        计算维度B：生活方式（满分25分）

        评分规则：
        - 不吸烟：+10分，已戒烟：+7分，吸烟：+0分
        - 不饮酒：+10分，适量饮酒：+7分，过量饮酒：+0分
        - 规律运动：+5分，偶尔运动：+3分，不运动：+0分

        Args:
            body: Chain策略专属输入数据

        Returns:
            维度B得分（0-25分）
        """
        score = 0

        # 从维度结果中获取生活方式信息
        dimension_results = body.dimension_results

        # 吸烟情况评分（满分10分）
        lifestyle_data = dimension_results.get("dimension_7", {})  # 预防措施维度可能包含生活方式信息
        if lifestyle_data:
            smoking_status = lifestyle_data.get("smoking_status", "unknown")
            if smoking_status == "不吸烟":
                score += 10
            elif smoking_status == "已戒烟":
                score += 7
            else:
                score += 0
        else:
            # 默认不吸烟
            score += 10

        # 饮酒情况评分（满分10分）
        if lifestyle_data:
            drinking_status = lifestyle_data.get("drinking_status", "unknown")
            if drinking_status == "不饮酒":
                score += 10
            elif drinking_status == "适量饮酒":
                score += 7
            else:
                score += 0
        else:
            # 默认不饮酒
            score += 10

        # 运动情况评分（满分5分）
        if lifestyle_data:
            exercise_status = lifestyle_data.get("exercise_status", "unknown")
            if exercise_status == "规律运动":
                score += 5
            elif exercise_status == "偶尔运动":
                score += 3
            else:
                score += 0
        else:
            # 默认规律运动
            score += 5

        return score

    def _calculate_dimension_c(self, body: IntegrationContextBody) -> int:
        """
        计算维度C：病史情况（满分25分）

        评分规则：
        - 无既往病史：+25分
        - 1-2种病史：+15分
        - 3-5种病史：+8分
        - 6种以上病史：+0分

        Args:
            body: Chain策略专属输入数据

        Returns:
            维度C得分（0-25分）
        """
        # 从风险因子中统计病史数量
        risk_factors = body.risk_factors

        # 统计既往病史数量
        history_count = 0
        for factor in risk_factors:
            factor_type = factor.get("factor_type", "")
            if factor_type in ["既往病史", "past_medical_history"]:
                history_count += 1

        # 根据病史数量评分
        if history_count == 0:
            score = 25
        elif history_count <= 2:
            score = 15
        elif history_count <= 5:
            score = 8
        else:
            score = 0

        return score

    def _calculate_dimension_d(self, body: IntegrationContextBody) -> int:
        """
        计算维度D：其他综合（满分20分）

        评分规则：
        - BMI正常：+10分，BMI偏瘦/超重：+5分，BMI肥胖：+0分
        - 无家族病史：+10分，有家族病史：+5分

        Args:
            body: Chain策略专属输入数据

        Returns:
            维度D得分（0-20分）
        """
        score = 0

        # 从异常指标中获取BMI情况
        anomalies = body.anomalies
        anomaly_indicators = {item.get("indicator_name", ""): item for item in anomalies}

        # BMI评分（满分10分）
        if "BMI" in anomaly_indicators or "体重指数" in anomaly_indicators:
            bmi_anomaly = anomaly_indicators.get("BMI") or anomaly_indicators.get("体重指数")
            if bmi_anomaly:
                severity = bmi_anomaly.get("severity", "normal")
                if severity == "normal":
                    score += 10
                elif severity in ["偏瘦", "超重"]:
                    score += 5
                else:
                    score += 0
            else:
                score += 10
        else:
            score += 10

        # 家族病史评分（满分10分）
        risk_factors = body.risk_factors
        has_family_history = False
        for factor in risk_factors:
            factor_type = factor.get("factor_type", "")
            if factor_type in ["家族病史", "family_history"]:
                has_family_history = True
                break

        if has_family_history:
            score += 5
        else:
            score += 10

        return score

    def _determine_health_level(self, health_score: int) -> str:
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
        if health_score >= 90:
            return "优秀"
        elif health_score >= 80:
            return "良好"
        elif health_score >= 70:
            return "一般"
        elif health_score >= 60:
            return "较差"
        else:
            return "差"

    def _calculate_risk_level(self, body: IntegrationContextBody) -> str:
        """
        计算风险等级（算法3）

        计算公式：
        FinalRiskScore = (异常指标数 × 10 + 风险因子数 × 5 + 病史权重 × 3) / 10

        病史权重：
        - 无病史=0
        - 1-2种=1
        - 3-5种=2
        - 6种以上=3

        风险等级判定：
        - 低风险：FinalRiskScore < 20
        - 轻度风险：20 ≤ FinalRiskScore < 40
        - 中度风险：40 ≤ FinalRiskScore < 60
        - 高风险：FinalRiskScore ≥ 60

        Args:
            body: Chain策略专属输入数据

        Returns:
            风险等级
        """
        logger.info("[IntegrationChain] 开始计算风险等级")

        # 统计异常指标数
        anomaly_count = len(body.anomalies)

        # 统计风险因子数
        risk_factor_count = len(body.risk_factors)

        # 计算病史权重
        history_count = 0
        for factor in body.risk_factors:
            factor_type = factor.get("factor_type", "")
            if factor_type in ["既往病史", "past_medical_history"]:
                history_count += 1

        if history_count == 0:
            history_weight = 0
        elif history_count <= 2:
            history_weight = 1
        elif history_count <= 5:
            history_weight = 2
        else:
            history_weight = 3

        # 计算最终风险分数
        final_risk_score = (anomaly_count * 10 + risk_factor_count * 5 + history_weight * 3) / 10

        logger.info(f"[IntegrationChain] 风险等级计算: anomaly_count={anomaly_count}, "
                   f"risk_factor_count={risk_factor_count}, history_weight={history_weight}, "
                   f"final_risk_score={final_risk_score}")

        # 判定风险等级
        if final_risk_score < 20:
            return "低"
        elif final_risk_score < 40:
            return "轻"
        elif final_risk_score < 60:
            return "中"
        else:
            return "高"

    def _calculate_disease_risks(self, body: IntegrationContextBody) -> List[Dict]:
        """
        计算疾病风险评分（算法1）

        计算步骤：
        1. 基于异常指标和病史，从knowledge_results中提取相关疾病
        2. 计算疾病风险分：风险分 = (相关异常指标数 × 15 + 病史匹配度 × 20) × 置信度
        3. 病史权重叠加：既往病史×1.5，家族史×1.2
        4. 返回Top-5高风险疾病

        Args:
            body: Chain策略专属输入数据

        Returns:
            Top-5高风险疾病列表，每个疾病包含disease_name、risk_score、confidence、evidence
        """
        logger.info("[IntegrationChain] 开始计算疾病风险评分")

        # 从知识检索结果中提取疾病信息
        disease_candidates = {}

        # 步骤1：从knowledge_results中提取相关疾病
        for knowledge in body.knowledge_results:
            knowledge_type = knowledge.get("type", "")
            if knowledge_type == "disease":
                disease_name = knowledge.get("entity", "")
                if disease_name:
                    disease_candidates[disease_name] = {
                        "disease_name": disease_name,
                        "knowledge_data": knowledge.get("data", {}),
                        "score": knowledge.get("score", 0.0),
                        "related_anomalies": [],
                        "history_match": 0
                    }

        # 步骤2：匹配异常指标
        for anomaly in body.anomalies:
            anomaly_name = anomaly.get("indicator_name", "")
            anomaly_desc = anomaly.get("description", "")

            # 检查每个疾病是否与异常指标相关
            for disease_name, disease_info in disease_candidates.items():
                knowledge_data = disease_info["knowledge_data"]

                # 检查症状匹配
                symptoms = knowledge_data.get("symptoms", [])
                if isinstance(symptoms, list):
                    for symptom in symptoms:
                        if isinstance(symptom, dict):
                            symptom_name = symptom.get("name", "")
                        else:
                            symptom_name = str(symptom)

                        # 简单匹配：异常指标名称或描述包含症状名称
                        if anomaly_name in symptom_name or anomaly_desc in symptom_name or symptom_name in anomaly_name:
                            disease_info["related_anomalies"].append(anomaly_name)
                            break

        # 步骤3：匹配病史
        for factor in body.risk_factors:
            factor_type = factor.get("factor_type", "")
            factor_name = factor.get("factor_name", "")

            for disease_name, disease_info in disease_candidates.items():
                # 检查既往病史匹配
                if factor_type in ["既往病史", "past_medical_history"]:
                    if factor_name in disease_name or disease_name in factor_name:
                        disease_info["history_match"] += 1
                        disease_info["history_weight"] = 1.5

                # 检查家族病史匹配
                if factor_type in ["家族病史", "family_history"]:
                    if factor_name in disease_name or disease_name in factor_name:
                        disease_info["history_match"] += 1
                        disease_info["family_weight"] = 1.2

        # 步骤4：计算疾病风险分
        disease_risks = []
        for disease_name, disease_info in disease_candidates.items():
            # 计算风险分
            related_anomaly_count = len(set(disease_info["related_anomalies"]))
            history_match = disease_info["history_match"]

            # 基础风险分
            risk_score = (related_anomaly_count * 15 + history_match * 20)

            # 置信度（基于知识检索得分）
            confidence = min(1.0, disease_info["score"])

            # 应用置信度
            risk_score = risk_score * confidence

            # 病史权重叠加
            history_weight = disease_info.get("history_weight", 1.0)
            family_weight = disease_info.get("family_weight", 1.0)
            risk_score = risk_score * history_weight * family_weight

            # 确保风险分在0-100范围内
            risk_score = min(100, max(0, risk_score))

            # 构建证据列表
            evidence = []
            if disease_info["related_anomalies"]:
                evidence.append(f"相关异常指标: {', '.join(set(disease_info['related_anomalies']))}")
            if history_match > 0:
                evidence.append(f"病史匹配度: {history_match}")

            disease_risks.append({
                "disease_name": disease_name,
                "risk_score": round(risk_score, 2),
                "confidence": round(confidence, 2),
                "evidence": evidence
            })

        # 步骤5：排序并返回Top-5
        disease_risks.sort(key=lambda x: x["risk_score"], reverse=True)
        top_5_risks = disease_risks[:5]

        logger.info(f"[IntegrationChain] 疾病风险评分完成: top_5_risks_count={len(top_5_risks)}")

        return top_5_risks

    def _prepare_report_materials(
        self,
        body: IntegrationContextBody,
        health_score: int,
        health_level: str,
        risk_level: str,
        risk_diseases: List[Dict]
    ) -> Dict:
        """
        准备报告素材

        整合以下内容：
        1. 8个维度的评估结果
        2. 知识检索结果
        3. 异常指标和风险因子
        4. 生成报告摘要

        Args:
            body: Chain策略专属输入数据
            health_score: 健康综合评分
            health_level: 健康等级
            risk_level: 风险等级
            risk_diseases: 高风险疾病列表

        Returns:
            报告素材字典
        """
        logger.info("[IntegrationChain] 开始准备报告素材")

        # 整合8个维度的评估结果
        dimension_summaries = {}
        for dim_key, dim_value in body.dimension_results.items():
            if dim_value:
                dimension_summaries[dim_key] = {
                    "dimension_id": dim_key,
                    "summary": dim_value.get("summary", ""),
                    "key_findings": dim_value.get("key_findings", [])
                }

        # 整合知识检索结果
        knowledge_summary = []
        for knowledge in body.knowledge_results[:10]:  # 只取前10条
            knowledge_summary.append({
                "source": knowledge.get("source", ""),
                "type": knowledge.get("type", ""),
                "entity": knowledge.get("entity", ""),
                "relevance_score": knowledge.get("score", 0.0)
            })

        # 整合异常指标
        anomaly_summary = []
        for anomaly in body.anomalies:
            anomaly_summary.append({
                "indicator_name": anomaly.get("indicator_name", ""),
                "current_value": anomaly.get("current_value", ""),
                "normal_range": anomaly.get("normal_range", ""),
                "severity": anomaly.get("severity", "normal")
            })

        # 整合风险因子
        risk_factor_summary = []
        for factor in body.risk_factors:
            risk_factor_summary.append({
                "factor_type": factor.get("factor_type", ""),
                "factor_name": factor.get("factor_name", ""),
                "weight": factor.get("weight", 1.0)
            })

        # 生成报告摘要
        report_summary = self._generate_report_summary(
            health_score,
            health_level,
            risk_level,
            risk_diseases,
            anomaly_summary
        )

        report_materials = {
            "health_overview": {
                "health_score": health_score,
                "health_level": health_level,
                "risk_level": risk_level
            },
            "dimension_summaries": dimension_summaries,
            "knowledge_summary": knowledge_summary,
            "anomaly_summary": anomaly_summary,
            "risk_factor_summary": risk_factor_summary,
            "risk_diseases": risk_diseases,
            "report_summary": report_summary
        }

        logger.info("[IntegrationChain] 报告素材准备完成")

        return report_materials

    def _generate_report_summary(
        self,
        health_score: int,
        health_level: str,
        risk_level: str,
        risk_diseases: List[Dict],
        anomaly_summary: List[Dict]
    ) -> str:
        """
        生成报告摘要

        Args:
            health_score: 健康综合评分
            health_level: 健康等级
            risk_level: 风险等级
            risk_diseases: 高风险疾病列表
            anomaly_summary: 异常指标摘要

        Returns:
            报告摘要文本
        """
        summary_parts = []

        # 健康状况概述
        summary_parts.append(f"您的健康综合评分为{health_score}分，健康等级为{health_level}。")

        # 风险等级说明
        risk_level_desc = {
            "低": "整体风险较低",
            "轻": "存在轻度风险",
            "中": "存在中度风险，需要关注",
            "高": "存在高度风险，建议尽快就医"
        }
        summary_parts.append(f"风险等级为{risk_level}级，{risk_level_desc.get(risk_level, '')}。")

        # 异常指标说明
        if anomaly_summary:
            anomaly_count = len(anomaly_summary)
            summary_parts.append(f"共发现{anomaly_count}项异常指标。")

        # 高风险疾病说明
        if risk_diseases:
            high_risk_count = len([d for d in risk_diseases if d["risk_score"] >= 60])
            if high_risk_count > 0:
                summary_parts.append(f"需要重点关注{high_risk_count}种高风险疾病。")

        return " ".join(summary_parts)
