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

        监测数据异常值识别：
        - 血压异常：收缩压≥140或舒张压≥90为高血压，收缩压<90或舒张压<60为低血压
        - 血糖异常：空腹血糖≥7.0或餐后血糖≥11.1为高血糖，空腹血糖<3.9为低血糖
        - 心率异常：心率>100为心动过速，心率<60为心动过缓
        - 血氧异常：血氧<95%为低氧
        - BMI异常：BMI<18.5为偏瘦，BMI≥24为超重，BMI≥28为肥胖

        Args:
            validated_data: 校验后的数据

        Returns:
            异常指标列表
        """
        anomalies = []

        # 血压异常检测
        blood_pressure = validated_data.get("blood_pressure", {})
        systolic = blood_pressure.get("systolic")  # 收缩压
        diastolic = blood_pressure.get("diastolic")  # 舒张压

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

        # 血糖异常检测
        blood_glucose = validated_data.get("blood_glucose", {})
        fasting_glucose = blood_glucose.get("fasting_glucose")  # 空腹血糖
        postprandial_glucose = blood_glucose.get("postprandial_glucose")  # 餐后血糖

        if fasting_glucose is not None:
            if fasting_glucose >= 7.0:
                anomalies.append({
                    "indicator_name": "空腹血糖",
                    "anomaly_type": "高血糖",
                    "anomaly_value": f"{fasting_glucose} mmol/L",
                    "reference_range": "<7.0 mmol/L"
                })
            elif fasting_glucose < 3.9:
                anomalies.append({
                    "indicator_name": "空腹血糖",
                    "anomaly_type": "低血糖",
                    "anomaly_value": f"{fasting_glucose} mmol/L",
                    "reference_range": "≥3.9 mmol/L"
                })

        if postprandial_glucose is not None:
            if postprandial_glucose >= 11.1:
                anomalies.append({
                    "indicator_name": "餐后血糖",
                    "anomaly_type": "高血糖",
                    "anomaly_value": f"{postprandial_glucose} mmol/L",
                    "reference_range": "<11.1 mmol/L"
                })

        # 心率异常检测
        heart_rate = validated_data.get("heart_rate")
        if heart_rate is not None:
            if heart_rate > 100:
                anomalies.append({
                    "indicator_name": "心率",
                    "anomaly_type": "心动过速",
                    "anomaly_value": f"{heart_rate} 次/分",
                    "reference_range": "60-100 次/分"
                })
            elif heart_rate < 60:
                anomalies.append({
                    "indicator_name": "心率",
                    "anomaly_type": "心动过缓",
                    "anomaly_value": f"{heart_rate} 次/分",
                    "reference_range": "60-100 次/分"
                })

        # 血氧异常检测
        blood_oxygen = validated_data.get("blood_oxygen")
        if blood_oxygen is not None:
            if blood_oxygen < 95:
                anomalies.append({
                    "indicator_name": "血氧饱和度",
                    "anomaly_type": "低氧",
                    "anomaly_value": f"{blood_oxygen}%",
                    "reference_range": "≥95%"
                })

        # BMI异常检测
        bmi = validated_data.get("bmi")
        if bmi is not None:
            if bmi >= 28:
                anomalies.append({
                    "indicator_name": "BMI",
                    "anomaly_type": "肥胖",
                    "anomaly_value": f"{bmi:.1f}",
                    "reference_range": "18.5-24"
                })
            elif bmi >= 24:
                anomalies.append({
                    "indicator_name": "BMI",
                    "anomaly_type": "超重",
                    "anomaly_value": f"{bmi:.1f}",
                    "reference_range": "18.5-24"
                })
            elif bmi < 18.5:
                anomalies.append({
                    "indicator_name": "BMI",
                    "anomaly_type": "偏瘦",
                    "anomaly_value": f"{bmi:.1f}",
                    "reference_range": "18.5-24"
                })

        return anomalies

    def _extract_risk_factors(self, validated_data: Dict) -> List[Dict]:
        """
        风险因子提取

        基于病史和生活方式：
        - 高龄风险：年龄≥65岁
        - 多病共存：既往病史≥3种
        - 家族遗传风险：有家族病史
        - 不良生活方式：吸烟、饮酒、缺乏运动

        Args:
            validated_data: 校验后的数据

        Returns:
            风险因子列表
        """
        risk_factors = []

        # 高龄风险
        age = validated_data.get("age")
        if age is not None and age >= 65:
            risk_factors.append({
                "factor_name": "高龄",
                "risk_level": "中",
                "basis": f"年龄{age}岁，属于老年人群"
            })

        # 多病共存风险
        medical_history = validated_data.get("medical_history", [])
        if isinstance(medical_history, list) and len(medical_history) >= 3:
            risk_factors.append({
                "factor_name": "多病共存",
                "risk_level": "高",
                "basis": f"既往病史{len(medical_history)}种：{', '.join(medical_history[:5])}"
            })

        # 家族遗传风险
        family_history = validated_data.get("family_history", [])
        if isinstance(family_history, list) and len(family_history) > 0:
            risk_factors.append({
                "factor_name": "家族遗传",
                "risk_level": "中",
                "basis": f"有家族病史：{', '.join(family_history[:5])}"
            })

        # 不良生活方式风险
        lifestyle = validated_data.get("lifestyle", {})

        # 吸烟
        smoking = lifestyle.get("smoking")
        if smoking:
            risk_factors.append({
                "factor_name": "吸烟",
                "risk_level": "高",
                "basis": "有吸烟习惯"
            })

        # 饮酒
        drinking = lifestyle.get("drinking")
        if drinking:
            risk_factors.append({
                "factor_name": "饮酒",
                "risk_level": "中",
                "basis": "有饮酒习惯"
            })

        # 缺乏运动
        exercise_frequency = lifestyle.get("exercise_frequency")
        if exercise_frequency is not None and exercise_frequency < 2:  # 每周运动少于2次
            risk_factors.append({
                "factor_name": "缺乏运动",
                "risk_level": "低",
                "basis": f"每周运动{exercise_frequency}次，运动不足"
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

        # 添加症状描述
        symptoms = validated_data.get("symptoms", [])
        if isinstance(symptoms, list) and symptoms:
            text_parts.append(f"症状：{', '.join(symptoms)}")

        # 添加既往病史
        medical_history = validated_data.get("medical_history", [])
        if isinstance(medical_history, list) and medical_history:
            text_parts.append(f"既往病史：{', '.join(medical_history)}")

        # 添加家族病史
        family_history = validated_data.get("family_history", [])
        if isinstance(family_history, list) and family_history:
            text_parts.append(f"家族病史：{', '.join(family_history)}")

        # 添加主诉
        chief_complaint = validated_data.get("chief_complaint")
        if chief_complaint:
            text_parts.append(f"主诉：{chief_complaint}")

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

        # 检查是否存在糖尿病病史
        medical_history = validated_data.get("medical_history", [])
        has_diabetes = isinstance(medical_history, list) and any(
            "糖尿病" in disease for disease in medical_history
        )

        # 检查是否存在肥胖
        has_obesity = any(
            a["indicator_name"] == "BMI" and a["anomaly_type"] == "肥胖"
            for a in anomalies
        )

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
