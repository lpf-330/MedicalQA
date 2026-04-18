# -*- coding: utf-8 -*-
"""
多维度分析Chain策略

实现健康报告生成业务的多维度分析Chain策略。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.orchestration.chain.chain import Chain
from src.orchestration.chain.data_classes import ChainContext, ChainResult
from src.orchestration.tool_call_handler.Impl.intent_classification_handler import IntentClassificationHandler

logger = logging.getLogger(__name__)


@dataclass
class MultiAnalysisContextBody:
    """
    多维度分析Chain策略专属输入数据类

    Attributes:
        validated_data: 校验后的数据
        degradation_level: 降级级别
    """
    validated_data: Dict = field(default_factory=dict)
    degradation_level: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "validated_data": self.validated_data,
            "degradation_level": self.degradation_level
        }


@dataclass
class MultiAnalysisResultData:
    """
    多维度分析Chain策略专属输出数据类

    Attributes:
        anomalies: 异常指标列表，包含指标名、异常类型、异常值、参考范围
        risk_factors: 风险因子列表，包含因子名、风险等级、依据
        medical_entities: 医疗实体列表
        analysis_summary: 分析摘要
    """
    anomalies: List[Dict] = field(default_factory=list)
    risk_factors: List[Dict] = field(default_factory=list)
    medical_entities: List[Dict] = field(default_factory=list)
    analysis_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "anomalies": self.anomalies,
            "risk_factors": self.risk_factors,
            "medical_entities": self.medical_entities,
            "analysis_summary": self.analysis_summary
        }


@dataclass
class MultiAnalysisResource:
    """
    多维度分析Chain策略专属资源类

    Attributes:
        intent_handler: 意图分类Handler（复用健康咨询的Handler）
        vector_encode_service: 向量编码服务（复用健康咨询的Service）
    """
    intent_handler: Optional[IntentClassificationHandler] = None
    vector_encode_service: Optional[Any] = None


class MultiAnalysisChain(Chain[ChainContext[MultiAnalysisContextBody], ChainResult[MultiAnalysisResultData]]):
    """
    多维度分析Chain策略类

    实现健康报告生成业务的多维度分析固定流程：
    1. 异常指标提取（监测数据异常值识别）
    2. 风险因子提取（基于病史和生活方式）
    3. 医疗实体提取（调用IntentClassificationHandler的extract_entities方法）
    4. 特殊规则应用
    5. 生成分析摘要
    """

    def __init__(self, resource: MultiAnalysisResource):
        """
        初始化多维度分析Chain策略

        Args:
            resource: Chain策略专属资源
        """
        self._resource = resource
        self._handler_degraded = False

    def execute(self, chain_context: ChainContext[MultiAnalysisContextBody]) -> ChainResult[MultiAnalysisResultData]:
        """
        执行Chain策略

        Args:
            chain_context: Chain输入数据容器

        Returns:
            ChainResult: Chain输出数据容器
        """
        start_time = time.time()
        logger.info(f"[MultiAnalysisChain] 开始执行Chain: session_id={chain_context.session_id}")

        body = chain_context.body
        if body is None or not body.validated_data:
            logger.warning(f"[MultiAnalysisChain] 输入数据为空: session_id={chain_context.session_id}")
            return ChainResult(
                session_id=chain_context.session_id,
                data=MultiAnalysisResultData()
            )

        self._handler_degraded = False

        # 步骤1：异常指标提取
        anomalies = self._extract_anomalies(body.validated_data)
        logger.info(f"[MultiAnalysisChain] 异常指标提取完成: anomaly_count={len(anomalies)}")

        # 步骤2：风险因子提取
        risk_factors = self._extract_risk_factors(body.validated_data)
        logger.info(f"[MultiAnalysisChain] 风险因子提取完成: risk_factor_count={len(risk_factors)}")

        # 步骤3：医疗实体提取
        medical_entities = self._extract_medical_entities(body.validated_data)
        logger.info(f"[MultiAnalysisChain] 医疗实体提取完成: entity_count={len(medical_entities)}")

        # 步骤4：特殊规则应用
        special_risks = self._apply_special_rules(anomalies, risk_factors, body.validated_data)
        risk_factors.extend(special_risks)
        logger.info(f"[MultiAnalysisChain] 特殊规则应用完成: special_risk_count={len(special_risks)}")

        # 步骤5：生成分析摘要
        analysis_summary = self._generate_analysis_summary(anomalies, risk_factors, medical_entities)
        logger.info(f"[MultiAnalysisChain] 分析摘要生成完成: summary_length={len(analysis_summary)}")

        result_data = MultiAnalysisResultData(
            anomalies=anomalies,
            risk_factors=risk_factors,
            medical_entities=medical_entities,
            analysis_summary=analysis_summary
        )

        elapsed = time.time() - start_time
        logger.info(f"[MultiAnalysisChain] Chain执行完成: session_id={chain_context.session_id}, "
                   f"anomalies={len(anomalies)}, risk_factors={len(risk_factors)}, "
                   f"medical_entities={len(medical_entities)}, elapsed={elapsed:.2f}s")

        return ChainResult(session_id=chain_context.session_id, data=result_data)

    def _extract_anomalies(self, validated_data: Dict) -> List[Dict]:
        """
        异常指标提取

        监测数据异常值识别（处理4个时间维度的数据）：
        - 血压异常：收缩压≥140或舒张压≥90为高血压，收缩压<90或舒张压<60为低血压
        - 血糖异常：空腹血糖≥7.0或餐后血糖≥11.1为高血糖，空腹血糖<3.9为低血糖
        - 心率异常：心率>100为心动过速，心率<60为心动过缓
        - 血氧异常：血氧<95%为低氧
        - 灌注指数异常：灌注指数<1.0为低灌注
        - 睡眠异常：睡眠时间<6小时为睡眠不足，>9小时为睡眠过多

        Args:
            validated_data: 校验后的数据

        Returns:
            异常指标列表
        """
        anomalies = []
        monitoring_data = validated_data.get("monitoring_data", {})

        # 血压异常检测（从latest维度获取最新数据）
        blood_pressure = monitoring_data.get("blood_pressure", {})
        if isinstance(blood_pressure, dict) and blood_pressure.get("latest"):
            latest_bp_list = blood_pressure["latest"]
            if isinstance(latest_bp_list, list) and latest_bp_list:
                # 获取最新一条血压数据
                latest_bp = latest_bp_list[-1] if isinstance(latest_bp_list[-1], dict) else {}
                systolic = latest_bp.get("systolic")
                diastolic = latest_bp.get("diastolic")

                if systolic is not None and diastolic is not None:
                    if systolic >= 140 or diastolic >= 90:
                        anomalies.append({
                            "indicator_name": "血压",
                            "anomaly_type": "高血压",
                            "anomaly_value": f"{systolic}/{diastolic} mmHg",
                            "reference_range": "收缩压<140 mmHg 且 舒张压<90 mmHg"
                        })
                    elif systolic < 90 or diastolic < 60:
                        anomalies.append({
                            "indicator_name": "血压",
                            "anomaly_type": "低血压",
                            "anomaly_value": f"{systolic}/{diastolic} mmHg",
                            "reference_range": "收缩压≥90 mmHg 且 舒张压≥60 mmHg"
                        })

        # 血糖异常检测（从latest维度获取最新数据）
        blood_glucose = monitoring_data.get("blood_glucose", {})
        if isinstance(blood_glucose, dict) and blood_glucose.get("latest"):
            latest_glucose_list = blood_glucose["latest"]
            if isinstance(latest_glucose_list, list) and latest_glucose_list:
                latest_glucose = latest_glucose_list[-1] if isinstance(latest_glucose_list[-1], dict) else {}
                glucose_value = latest_glucose.get("value")
                glucose_type = latest_glucose.get("type", "fasting")

                if glucose_value is not None:
                    if glucose_type == "fasting" or glucose_type == "空腹":
                        if glucose_value >= 7.0:
                            anomalies.append({
                                "indicator_name": "空腹血糖",
                                "anomaly_type": "高血糖",
                                "anomaly_value": f"{glucose_value} mmol/L",
                                "reference_range": "<7.0 mmol/L"
                            })
                        elif glucose_value < 3.9:
                            anomalies.append({
                                "indicator_name": "空腹血糖",
                                "anomaly_type": "低血糖",
                                "anomaly_value": f"{glucose_value} mmol/L",
                                "reference_range": "≥3.9 mmol/L"
                            })
                    else:
                        if glucose_value >= 11.1:
                            anomalies.append({
                                "indicator_name": "餐后血糖",
                                "anomaly_type": "高血糖",
                                "anomaly_value": f"{glucose_value} mmol/L",
                                "reference_range": "<11.1 mmol/L"
                            })

        # 心率异常检测（从latest维度获取最新数据）
        heart_rate = monitoring_data.get("heart_rate", {})
        if isinstance(heart_rate, dict) and heart_rate.get("latest"):
            latest_hr_list = heart_rate["latest"]
            if isinstance(latest_hr_list, list) and latest_hr_list:
                latest_hr = latest_hr_list[-1] if isinstance(latest_hr_list[-1], dict) else {}
                hr_value = latest_hr.get("value")

                if hr_value is not None:
                    if hr_value > 100:
                        anomalies.append({
                            "indicator_name": "心率",
                            "anomaly_type": "心动过速",
                            "anomaly_value": f"{hr_value} 次/分",
                            "reference_range": "60-100 次/分"
                        })
                    elif hr_value < 60:
                        anomalies.append({
                            "indicator_name": "心率",
                            "anomaly_type": "心动过缓",
                            "anomaly_value": f"{hr_value} 次/分",
                            "reference_range": "60-100 次/分"
                        })

        # 血氧异常检测（从latest维度获取最新数据）
        blood_oxygen = monitoring_data.get("blood_oxygen", {})
        if isinstance(blood_oxygen, dict) and blood_oxygen.get("latest"):
            latest_oxygen_list = blood_oxygen["latest"]
            if isinstance(latest_oxygen_list, list) and latest_oxygen_list:
                latest_oxygen = latest_oxygen_list[-1] if isinstance(latest_oxygen_list[-1], dict) else {}
                oxygen_value = latest_oxygen.get("value")

                if oxygen_value is not None and oxygen_value < 95:
                    anomalies.append({
                        "indicator_name": "血氧饱和度",
                        "anomaly_type": "低氧",
                        "anomaly_value": f"{oxygen_value}%",
                        "reference_range": "≥95%"
                    })

        # 灌注指数异常检测（从latest维度获取最新数据）
        perfusion_index = monitoring_data.get("perfusion_index", {})
        if isinstance(perfusion_index, dict) and perfusion_index.get("latest"):
            latest_pi_list = perfusion_index["latest"]
            if isinstance(latest_pi_list, list) and latest_pi_list:
                latest_pi = latest_pi_list[-1] if isinstance(latest_pi_list[-1], dict) else {}
                pi_value = latest_pi.get("value")

                if pi_value is not None and pi_value < 1.0:
                    anomalies.append({
                        "indicator_name": "灌注指数",
                        "anomaly_type": "低灌注",
                        "anomaly_value": f"{pi_value} PI",
                        "reference_range": "≥1.0 PI"
                    })

        # 睡眠异常检测（从latest维度获取最新数据）
        sleep = monitoring_data.get("sleep", {})
        if isinstance(sleep, dict) and sleep.get("latest"):
            latest_sleep_list = sleep["latest"]
            if isinstance(latest_sleep_list, list) and latest_sleep_list:
                latest_sleep = latest_sleep_list[-1] if isinstance(latest_sleep_list[-1], dict) else {}
                sleep_value = latest_sleep.get("value")

                if sleep_value is not None:
                    if sleep_value < 6:
                        anomalies.append({
                            "indicator_name": "睡眠",
                            "anomaly_type": "睡眠不足",
                            "anomaly_value": f"{sleep_value} 小时",
                            "reference_range": "6-9 小时"
                        })
                    elif sleep_value > 9:
                        anomalies.append({
                            "indicator_name": "睡眠",
                            "anomaly_type": "睡眠过多",
                            "anomaly_value": f"{sleep_value} 小时",
                            "reference_range": "6-9 小时"
                        })

        return anomalies

    def _extract_risk_factors(self, validated_data: Dict) -> List[Dict]:
        """
        风险因子提取

        基于病史和用户档案（处理字符串类型的病史字段）：
        - 高龄风险：年龄>=65岁
        - 多病共存：既往病史包含多种疾病
        - 家族遗传风险：有家族病史
        - 过敏史风险：有过敏史
        - 手术史风险：有手术史

        Args:
            validated_data: 校验后的数据

        Returns:
            风险因子列表
        """
        risk_factors = []
        user_profile = validated_data.get("user_profile", {})

        # 高龄风险（根据出生日期计算年龄）
        birth_date = user_profile.get("birth_date")
        if birth_date:
            try:
                from datetime import datetime
                birth = datetime.strptime(birth_date, "%Y-%m-%d")
                today = datetime.now()
                age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
                if age >= 65:
                    risk_factors.append({
                        "factor_name": "高龄",
                        "risk_level": "中",
                        "basis": f"年龄{age}岁，属于老年人群"
                    })
            except (ValueError, TypeError):
                pass

        # 既往病史风险（字符串类型）
        past_medical_history = user_profile.get("past_medical_history", "")
        if past_medical_history and past_medical_history.strip():
            # 统计既往病史中的疾病数量（简单统计逗号分隔）
            diseases = [d.strip() for d in past_medical_history.replace("、", ",").replace("，", ",").split(",") if d.strip()]
            if len(diseases) >= 3:
                risk_factors.append({
                    "factor_name": "多病共存",
                    "risk_level": "高",
                    "basis": f"既往病史包含多种疾病：{past_medical_history}"
                })
            elif len(diseases) > 0:
                risk_factors.append({
                    "factor_name": "既往病史",
                    "risk_level": "中",
                    "basis": f"既往病史：{past_medical_history}"
                })

        # 家族遗传风险（字符串类型）
        family_history = user_profile.get("family_history", "")
        if family_history and family_history.strip():
            risk_factors.append({
                "factor_name": "家族遗传",
                "risk_level": "中",
                "basis": f"有家族病史：{family_history}"
            })

        # 过敏史风险（字符串类型）
        allergy_history = user_profile.get("allergy_history", "")
        if allergy_history and allergy_history.strip():
            risk_factors.append({
                "factor_name": "过敏史",
                "risk_level": "低",
                "basis": f"有过敏史：{allergy_history}"
            })

        # 手术史风险（字符串类型）
        surgical_history = user_profile.get("surgical_history", "")
        if surgical_history and surgical_history.strip():
            risk_factors.append({
                "factor_name": "手术史",
                "risk_level": "低",
                "basis": f"有手术史：{surgical_history}"
            })

        return risk_factors

    def _extract_medical_entities(self, validated_data: Dict) -> List[Dict]:
        """
        医疗实体提取

        调用IntentClassificationHandler的extract_entities方法
        如果Handler不可用，使用规则引擎降级

        Args:
            validated_data: 校验后的数据

        Returns:
            医疗实体列表
        """
        medical_entities = []

        # 构建待提取的文本
        text_parts = []
        user_profile = validated_data.get("user_profile", {})

        # 添加既往病史（字符串类型）
        past_medical_history = user_profile.get("past_medical_history", "")
        if past_medical_history and past_medical_history.strip():
            text_parts.append(f"既往病史：{past_medical_history}")

        # 添加家族病史（字符串类型）
        family_history = user_profile.get("family_history", "")
        if family_history and family_history.strip():
            text_parts.append(f"家族病史：{family_history}")

        # 添加过敏史（字符串类型）
        allergy_history = user_profile.get("allergy_history", "")
        if allergy_history and allergy_history.strip():
            text_parts.append(f"过敏史：{allergy_history}")

        # 添加手术史（字符串类型）
        surgical_history = user_profile.get("surgical_history", "")
        if surgical_history and surgical_history.strip():
            text_parts.append(f"手术史：{surgical_history}")

        if not text_parts:
            logger.info("[MultiAnalysisChain] 无需提取医疗实体的文本内容")
            return medical_entities

        text = "。".join(text_parts)

        # 尝试使用Handler提取实体
        try:
            if self._resource.intent_handler is not None:
                logger.info(f"[MultiAnalysisChain] 使用Handler提取医疗实体: text_length={len(text)}")
                extract_result = self._resource.intent_handler.call_tool({
                    "method": "extract_entities",
                    "text": text
                })

                if isinstance(extract_result, list):
                    medical_entities = extract_result
                else:
                    medical_entities = extract_result.get("entities", [])

                logger.info(f"[MultiAnalysisChain] Handler提取医疗实体成功: entity_count={len(medical_entities)}")
            else:
                logger.warning("[MultiAnalysisChain] intent_handler未初始化，使用规则引擎降级")
                self._handler_degraded = True
                medical_entities = self._extract_entities_by_rules(text)
        except Exception as e:
            logger.error(f"[MultiAnalysisChain] Handler提取医疗实体失败，使用规则引擎降级: {str(e)}")
            self._handler_degraded = True
            medical_entities = self._extract_entities_by_rules(text)

        return medical_entities

    def _extract_entities_by_rules(self, text: str) -> List[Dict]:
        """
        使用规则引擎提取医疗实体（降级策略）

        Args:
            text: 待提取的文本

        Returns:
            医疗实体列表
        """
        entities = []

        # 常见疾病关键词
        disease_keywords = [
            "高血压", "糖尿病", "冠心病", "脑卒中", "慢性肾病",
            "肝炎", "胃炎", "肺炎", "哮喘", "关节炎",
            "肿瘤", "癌症", "心脏病", "中风"
        ]

        # 常见症状关键词
        symptom_keywords = [
            "头痛", "头晕", "恶心", "呕吐", "腹痛",
            "咳嗽", "发热", "乏力", "胸闷", "心悸",
            "失眠", "便秘", "腹泻", "水肿", "出血"
        ]

        # 提取疾病实体
        for keyword in disease_keywords:
            if keyword in text:
                entities.append({
                    "entity_name": keyword,
                    "entity_type": "Disease",
                    "confidence": 0.8
                })

        # 提取症状实体
        for keyword in symptom_keywords:
            if keyword in text:
                entities.append({
                    "entity_name": keyword,
                    "entity_type": "Symptom",
                    "confidence": 0.8
                })

        logger.info(f"[MultiAnalysisChain] 规则引擎提取医疗实体完成: entity_count={len(entities)}")
        return entities

    def _apply_special_rules(self, anomalies: List[Dict], risk_factors: List[Dict], validated_data: Dict) -> List[Dict]:
        """
        特殊规则应用

        - 高龄+高血压：心血管高风险
        - 糖尿病+肥胖：代谢综合征风险
        - 多病共存：综合评估风险

        Args:
            anomalies: 异常指标列表
            risk_factors: 风险因子列表
            validated_data: 校验后的数据

        Returns:
            特殊风险列表
        """
        special_risks = []

        # 检查是否存在高龄风险
        has_elderly_risk = any(rf["factor_name"] == "高龄" for rf in risk_factors)

        # 检查是否存在高血压异常
        has_hypertension = any(
            a["indicator_name"] == "血压" and a["anomaly_type"] == "高血压"
            for a in anomalies
        )

        # 高龄+高血压：心血管高风险
        if has_elderly_risk and has_hypertension:
            special_risks.append({
                "factor_name": "心血管高风险",
                "risk_level": "高",
                "basis": "高龄合并高血压，心血管事件风险显著增加"
            })

        # 检查是否存在糖尿病病史（字符串类型）
        user_profile = validated_data.get("user_profile", {})
        past_medical_history = user_profile.get("past_medical_history", "")
        has_diabetes = past_medical_history and "糖尿病" in past_medical_history

        # 检查是否存在肥胖（从血压数据推断或既往病史）
        has_obesity = past_medical_history and ("肥胖" in past_medical_history or "超重" in past_medical_history)

        # 糖尿病+肥胖：代谢综合征风险
        if has_diabetes and has_obesity:
            special_risks.append({
                "factor_name": "代谢综合征风险",
                "risk_level": "高",
                "basis": "糖尿病合并肥胖，代谢综合征风险显著增加"
            })

        # 检查是否存在多病共存
        has_multi_disease = any(
            rf["factor_name"] == "多病共存" for rf in risk_factors
        )

        # 多病共存：综合评估风险
        if has_multi_disease:
            special_risks.append({
                "factor_name": "综合评估风险",
                "risk_level": "高",
                "basis": "多病共存，需综合评估用药相互作用和疾病相互影响"
            })

        return special_risks

    def _generate_analysis_summary(self, anomalies: List[Dict], risk_factors: List[Dict], medical_entities: List[Dict]) -> str:
        """
        生成分析摘要

        Args:
            anomalies: 异常指标列表
            risk_factors: 风险因子列表
            medical_entities: 医疗实体列表

        Returns:
            分析摘要
        """
        summary_parts = []

        # 异常指标摘要
        if anomalies:
            anomaly_summary = "、".join([f"{a['indicator_name']}({a['anomaly_type']})" for a in anomalies])
            summary_parts.append(f"检测到{len(anomalies)}项异常指标：{anomaly_summary}")
        else:
            summary_parts.append("未检测到异常指标")

        # 风险因子摘要
        if risk_factors:
            high_risks = [rf for rf in risk_factors if rf["risk_level"] == "高"]
            medium_risks = [rf for rf in risk_factors if rf["risk_level"] == "中"]
            low_risks = [rf for rf in risk_factors if rf["risk_level"] == "低"]

            risk_summary_parts = []
            if high_risks:
                risk_summary_parts.append(f"高风险{len(high_risks)}项")
            if medium_risks:
                risk_summary_parts.append(f"中风险{len(medium_risks)}项")
            if low_risks:
                risk_summary_parts.append(f"低风险{len(low_risks)}项")

            summary_parts.append(f"识别到{len(risk_factors)}项风险因子（{'，'.join(risk_summary_parts)}）")
        else:
            summary_parts.append("未识别到风险因子")

        # 医疗实体摘要
        if medical_entities:
            disease_entities = [e for e in medical_entities if e.get("entity_type") == "Disease"]
            symptom_entities = [e for e in medical_entities if e.get("entity_type") == "Symptom"]

            entity_summary_parts = []
            if disease_entities:
                entity_summary_parts.append(f"{len(disease_entities)}种疾病")
            if symptom_entities:
                entity_summary_parts.append(f"{len(symptom_entities)}种症状")

            if entity_summary_parts:
                summary_parts.append(f"提取到{'、'.join(entity_summary_parts)}相关实体")

        # 降级提示
        if self._handler_degraded:
            summary_parts.append("（注：医疗实体提取使用规则引擎降级模式）")

        summary = "。".join(summary_parts) + "。"
        return summary
