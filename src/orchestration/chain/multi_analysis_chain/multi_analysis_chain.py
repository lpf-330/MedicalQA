# -*- coding: utf-8 -*-
"""
多维度分析Chain策略

实现健康报告生成业务的多维度分析Chain策略。
"""

import logging
import time
from typing import Any, Dict, List, Optional
from src.config.business.report_service_config import get_runtime_config
from src.orchestration.chain.chain import Chain
from src.orchestration.chain.data_classes import ChainContext, ChainResult
from src.orchestration.chain.multi_analysis_chain.multi_analysis_context import MultiAnalysisContextBody
from src.orchestration.chain.multi_analysis_chain.multi_analysis_result import MultiAnalysisResultData
from src.orchestration.chain.multi_analysis_chain.multi_analysis_resource import MultiAnalysisResource
from src.orchestration.tool_call_handler.Impl.intent_classification_handler import IntentClassificationHandler

logger = logging.getLogger(__name__)

# 报告业务配置（dataclass，模块级创建安全）
class _LazyReportConfig:
    """延迟加载配置代理，每次属性访问时从ConfigManager获取真实配置"""
    def __getattr__(self, name):
        return getattr(get_runtime_config(), name)

_report_config = _LazyReportConfig()

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

        # 延迟获取临床标准值（依赖ConfigManager已加载）
        from src.config.config_manager import ConfigManager
        self._clinical = ConfigManager().clinical_standards

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
        entity_count = sum(len(v) for v in medical_entities.values()) if isinstance(medical_entities, dict) else 0
        logger.info(f"[MultiAnalysisChain] 医疗实体提取完成: entity_count={entity_count}")

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
        entity_count = sum(len(v) for v in medical_entities.values()) if isinstance(medical_entities, dict) else 0
        logger.info(f"[MultiAnalysisChain] Chain执行完成: session_id={chain_context.session_id}, "
                   f"anomalies={len(anomalies)}, risk_factors={len(risk_factors)}, "
                   f"medical_entities={entity_count}, elapsed={elapsed:.2f}s")

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
        bp_std = self._clinical.get("blood_pressure", {})
        bp_hyp_sys = bp_std.get("hypertension_systolic", 140)
        bp_hyp_dia = bp_std.get("hypertension_diastolic", 90)
        bp_hypo_sys = bp_std.get("hypotension_systolic", 90)
        bp_hypo_dia = bp_std.get("hypotension_diastolic", 60)

        blood_pressure = monitoring_data.get("blood_pressure", {})
        if isinstance(blood_pressure, dict) and blood_pressure.get("latest"):
            latest_bp_list = blood_pressure["latest"]
            if isinstance(latest_bp_list, list) and latest_bp_list:
                # 获取最新一条血压数据
                latest_bp = latest_bp_list[-1] if isinstance(latest_bp_list[-1], dict) else {}
                systolic = latest_bp.get("systolic")
                diastolic = latest_bp.get("diastolic")

                if systolic is not None and diastolic is not None:
                    if systolic >= bp_hyp_sys or diastolic >= bp_hyp_dia:
                        anomalies.append({
                            "indicator_name": "血压",
                            "anomaly_type": "高血压",
                            "anomaly_value": f"{systolic}/{diastolic} mmHg",
                            "reference_range": f"收缩压<{bp_hyp_sys} mmHg 且 舒张压<{bp_hyp_dia} mmHg"
                        })
                        logger.info(f"[ANOMALY_EXTRACT] indicator=血压, type=高血压, value={systolic}/{diastolic}, reference=收缩压<{bp_hyp_sys}且舒张压<{bp_hyp_dia}")
                    elif systolic < bp_hypo_sys or diastolic < bp_hypo_dia:
                        anomalies.append({
                            "indicator_name": "血压",
                            "anomaly_type": "低血压",
                            "anomaly_value": f"{systolic}/{diastolic} mmHg",
                            "reference_range": f"收缩压≥{bp_hypo_sys} mmHg 且 舒张压≥{bp_hypo_dia} mmHg"
                        })
                        logger.info(f"[ANOMALY_EXTRACT] indicator=血压, type=低血压, value={systolic}/{diastolic}, reference=收缩压≥{bp_hypo_sys}且舒张压≥{bp_hypo_dia}")

        # 血糖异常检测（从latest维度获取最新数据）
        bg_std = self._clinical.get("blood_glucose", {})
        bg_fasting_high = bg_std.get("fasting_high", 7.0)
        bg_fasting_low = bg_std.get("fasting_low", 3.9)
        bg_postprandial_high = bg_std.get("postprandial_high", 11.1)

        blood_glucose = monitoring_data.get("blood_glucose", {})
        if isinstance(blood_glucose, dict) and blood_glucose.get("latest"):
            latest_glucose_list = blood_glucose["latest"]
            if isinstance(latest_glucose_list, list) and latest_glucose_list:
                latest_glucose = latest_glucose_list[-1] if isinstance(latest_glucose_list[-1], dict) else {}
                glucose_value = latest_glucose.get("value")
                glucose_type = latest_glucose.get("type", "fasting")

                if glucose_value is not None:
                    if glucose_type == "fasting" or glucose_type == "空腹":
                        if glucose_value >= bg_fasting_high:
                            anomalies.append({
                                "indicator_name": "空腹血糖",
                                "anomaly_type": "高血糖",
                                "anomaly_value": f"{glucose_value} mmol/L",
                                "reference_range": f"<{bg_fasting_high} mmol/L"
                            })
                            logger.info(f"[ANOMALY_EXTRACT] indicator=空腹血糖, type=高血糖, value={glucose_value}, reference=<{bg_fasting_high}")
                        elif glucose_value < bg_fasting_low:
                            anomalies.append({
                                "indicator_name": "空腹血糖",
                                "anomaly_type": "低血糖",
                                "anomaly_value": f"{glucose_value} mmol/L",
                                "reference_range": f"≥{bg_fasting_low} mmol/L"
                            })
                            logger.info(f"[ANOMALY_EXTRACT] indicator=空腹血糖, type=低血糖, value={glucose_value}, reference=≥{bg_fasting_low}")
                    else:
                        if glucose_value >= bg_postprandial_high:
                            anomalies.append({
                                "indicator_name": "餐后血糖",
                                "anomaly_type": "高血糖",
                                "anomaly_value": f"{glucose_value} mmol/L",
                                "reference_range": f"<{bg_postprandial_high} mmol/L"
                            })
                            logger.info(f"[ANOMALY_EXTRACT] indicator=餐后血糖, type=高血糖, value={glucose_value}, reference=<{bg_postprandial_high}")

        # 心率异常检测（从latest维度获取最新数据）
        hr_std = self._clinical.get("heart_rate", {})
        hr_tachycardia = hr_std.get("tachycardia", 100)
        hr_bradycardia = hr_std.get("bradycardia", 60)

        heart_rate = monitoring_data.get("heart_rate", {})
        if isinstance(heart_rate, dict) and heart_rate.get("latest"):
            latest_hr_list = heart_rate["latest"]
            if isinstance(latest_hr_list, list) and latest_hr_list:
                latest_hr = latest_hr_list[-1] if isinstance(latest_hr_list[-1], dict) else {}
                hr_value = latest_hr.get("value")

                if hr_value is not None:
                    if hr_value > hr_tachycardia:
                        anomalies.append({
                            "indicator_name": "心率",
                            "anomaly_type": "心动过速",
                            "anomaly_value": f"{hr_value} 次/分",
                            "reference_range": f"{hr_bradycardia}-{hr_tachycardia} 次/分"
                        })
                        logger.info(f"[ANOMALY_EXTRACT] indicator=心率, type=心动过速, value={hr_value}, reference={hr_bradycardia}-{hr_tachycardia}")
                    elif hr_value < hr_bradycardia:
                        anomalies.append({
                            "indicator_name": "心率",
                            "anomaly_type": "心动过缓",
                            "anomaly_value": f"{hr_value} 次/分",
                            "reference_range": f"{hr_bradycardia}-{hr_tachycardia} 次/分"
                        })
                        logger.info(f"[ANOMALY_EXTRACT] indicator=心率, type=心动过缓, value={hr_value}, reference={hr_bradycardia}-{hr_tachycardia}")

        # 血氧异常检测（从latest维度获取最新数据）
        bo_std = self._clinical.get("blood_oxygen", {})
        bo_hypoxemia = bo_std.get("hypoxemia", 95)

        blood_oxygen = monitoring_data.get("blood_oxygen", {})
        if isinstance(blood_oxygen, dict) and blood_oxygen.get("latest"):
            latest_oxygen_list = blood_oxygen["latest"]
            if isinstance(latest_oxygen_list, list) and latest_oxygen_list:
                latest_oxygen = latest_oxygen_list[-1] if isinstance(latest_oxygen_list[-1], dict) else {}
                oxygen_value = latest_oxygen.get("value")

                if oxygen_value is not None and oxygen_value < bo_hypoxemia:
                    anomalies.append({
                        "indicator_name": "血氧饱和度",
                        "anomaly_type": "低氧",
                        "anomaly_value": f"{oxygen_value}%",
                        "reference_range": f"≥{bo_hypoxemia}%"
                    })
                    logger.info(f"[ANOMALY_EXTRACT] indicator=血氧饱和度, type=低氧, value={oxygen_value}%, reference=≥{bo_hypoxemia}%")

        # 灌注指数异常检测（从latest维度获取最新数据）
        pi_std = self._clinical.get("perfusion_index", {})
        pi_low = pi_std.get("low", 1.0)

        perfusion_index = monitoring_data.get("perfusion_index", {})
        if isinstance(perfusion_index, dict) and perfusion_index.get("latest"):
            latest_pi_list = perfusion_index["latest"]
            if isinstance(latest_pi_list, list) and latest_pi_list:
                latest_pi = latest_pi_list[-1] if isinstance(latest_pi_list[-1], dict) else {}
                pi_value = latest_pi.get("value")

                if pi_value is not None and pi_value < pi_low:
                    anomalies.append({
                        "indicator_name": "灌注指数",
                        "anomaly_type": "低灌注",
                        "anomaly_value": f"{pi_value} PI",
                        "reference_range": f"≥{pi_low} PI"
                    })
                    logger.info(f"[ANOMALY_EXTRACT] indicator=灌注指数, type=低灌注, value={pi_value}, reference=≥{pi_low}")

        # 睡眠异常检测（从latest维度获取最新数据）
        sleep_std = self._clinical.get("sleep", {})
        sleep_insufficient = sleep_std.get("insufficient", 6)
        sleep_excessive = sleep_std.get("excessive", 9)

        sleep = monitoring_data.get("sleep", {})
        if isinstance(sleep, dict) and sleep.get("latest"):
            latest_sleep_list = sleep["latest"]
            if isinstance(latest_sleep_list, list) and latest_sleep_list:
                latest_sleep = latest_sleep_list[-1] if isinstance(latest_sleep_list[-1], dict) else {}
                sleep_value = latest_sleep.get("value")

                if sleep_value is not None:
                    if sleep_value < sleep_insufficient:
                        anomalies.append({
                            "indicator_name": "睡眠",
                            "anomaly_type": "睡眠不足",
                            "anomaly_value": f"{sleep_value} 小时",
                            "reference_range": f"{sleep_insufficient}-{sleep_excessive} 小时"
                        })
                        logger.info(f"[ANOMALY_EXTRACT] indicator=睡眠, type=睡眠不足, value={sleep_value}, reference={sleep_insufficient}-{sleep_excessive}")
                    elif sleep_value > sleep_excessive:
                        anomalies.append({
                            "indicator_name": "睡眠",
                            "anomaly_type": "睡眠过多",
                            "anomaly_value": f"{sleep_value} 小时",
                            "reference_range": f"{sleep_insufficient}-{sleep_excessive} 小时"
                        })
                        logger.info(f"[ANOMALY_EXTRACT] indicator=睡眠, type=睡眠过多, value={sleep_value}, reference={sleep_insufficient}-{sleep_excessive}")

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

        # 高龄风险阈值
        ra_std = self._clinical.get("risk_assessment", {})
        elderly_age = ra_std.get("elderly_age", 65)
        multi_disease_count = ra_std.get("multi_disease_count", 3)

        # 高龄风险（根据出生日期计算年龄，优先birth_date，回退到age字段）
        birth_date = user_profile.get("birth_date")
        age = None
        if birth_date:
            try:
                from datetime import datetime
                birth = datetime.strptime(birth_date, "%Y-%m-%d")
                today = datetime.now()
                age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
            except (ValueError, TypeError):
                pass
        if age is None:
            profile_age = user_profile.get("age")
            if isinstance(profile_age, (int, float)) and profile_age > 0:
                age = int(profile_age)
        if age is not None and age >= elderly_age:
            risk_factors.append({
                "factor_name": "高龄",
                "risk_level": "中",
                "basis": f"年龄{age}岁，属于老年人群"
            })
            logger.info(f"[RISK_FACTOR_EXTRACT] factor=高龄, level=中, basis=年龄{age}岁，属于老年人群")

        # 既往病史风险（字符串类型）
        past_medical_history = user_profile.get("past_medical_history", "")
        if past_medical_history and past_medical_history.strip():
            # 统计既往病史中的疾病数量（简单统计逗号分隔）
            diseases = [d.strip() for d in past_medical_history.replace("、", ",").replace("，", ",").split(",") if d.strip()]
            if len(diseases) >= multi_disease_count:
                risk_factors.append({
                    "factor_name": "多病共存",
                    "risk_level": "高",
                    "basis": f"既往病史包含多种疾病：{past_medical_history}"
                })
                logger.info(f"[RISK_FACTOR_EXTRACT] factor=多病共存, level=高, basis=既往病史包含多种疾病：{past_medical_history}")
            elif len(diseases) > 0:
                risk_factors.append({
                    "factor_name": "既往病史",
                    "risk_level": "中",
                    "basis": f"既往病史：{past_medical_history}"
                })
                logger.info(f"[RISK_FACTOR_EXTRACT] factor=既往病史, level=中, basis=既往病史：{past_medical_history}")

        # 家族遗传风险（字符串类型）
        family_history = user_profile.get("family_history", "")
        if family_history and family_history.strip():
            risk_factors.append({
                "factor_name": "家族遗传",
                "risk_level": "中",
                "basis": f"有家族病史：{family_history}"
            })
            logger.info(f"[RISK_FACTOR_EXTRACT] factor=家族遗传, level=中, basis=有家族病史：{family_history}")

        # 过敏史风险（字符串类型）
        allergy_history = user_profile.get("allergy_history", "")
        if allergy_history and allergy_history.strip():
            risk_factors.append({
                "factor_name": "过敏史",
                "risk_level": "低",
                "basis": f"有过敏史：{allergy_history}"
            })
            logger.info(f"[RISK_FACTOR_EXTRACT] factor=过敏史, level=低, basis=有过敏史：{allergy_history}")

        # 手术史风险（字符串类型）
        surgical_history = user_profile.get("surgical_history", "")
        if surgical_history and surgical_history.strip():
            risk_factors.append({
                "factor_name": "手术史",
                "risk_level": "低",
                "basis": f"有手术史：{surgical_history}"
            })
            logger.info(f"[RISK_FACTOR_EXTRACT] factor=手术史, level=低, basis=有手术史：{surgical_history}")

        return risk_factors

    def _extract_medical_entities(self, validated_data: Dict) -> Dict[str, List]:
        """
        医疗实体提取

        通过intent_handler（NER模型）进行实体提取，NER不可用时回退到规则引擎降级

        Args:
            validated_data: 校验后的数据

        Returns:
            医疗实体字典，按类型分类存储
        """
        entities_list = []

        text_parts = []
        user_profile = validated_data.get("user_profile", {})

        past_medical_history = user_profile.get("past_medical_history", "")
        if past_medical_history and past_medical_history.strip():
            text_parts.append(f"既往病史：{past_medical_history}")

        family_history = user_profile.get("family_history", "")
        if family_history and family_history.strip():
            text_parts.append(f"家族病史：{family_history}")

        allergy_history = user_profile.get("allergy_history", "")
        if allergy_history and allergy_history.strip():
            text_parts.append(f"过敏史：{allergy_history}")

        surgical_history = user_profile.get("surgical_history", "")
        if surgical_history and surgical_history.strip():
            text_parts.append(f"手术史：{surgical_history}")

        if not text_parts:
            logger.info("[MultiAnalysisChain] 无需提取医疗实体的文本内容")
            return {"diseases": [], "symptoms": [], "medications": [], "examinations": [], "other": []}

        text = "。".join(text_parts)

        ner_extracted = False
        ner_handler = self._resource.ner_handler if self._resource else None
        if ner_handler is not None:
            try:
                handler_result = ner_handler.call_tool({"method": "extract_entities", "text": text})
                if isinstance(handler_result, list):
                    entities_list = handler_result
                elif isinstance(handler_result, dict):
                    entities_list = handler_result.get("entities", [])
                else:
                    entities_list = []
                critical_types = {"disease", "dis", "symptom", "sym", "medication", "dru", "drug"}
                has_critical_entity = any(
                    e.get("entity_type", "").lower() in critical_types
                    for e in entities_list
                )
                if len(entities_list) >= 2 and has_critical_entity:
                    ner_extracted = True
                    logger.info(f"[MultiAnalysisChain] NER模型提取医疗实体成功: entity_count={len(entities_list)}")
                else:
                    reason = "实体过少" if len(entities_list) < 2 else "缺少关键医疗实体(disease/symptom/medication)"
                    logger.warning(f"[MultiAnalysisChain] NER模型输出质量不足(entity_count={len(entities_list)}, has_critical={has_critical_entity}), 原因={reason}, 回退到规则引擎")
            except Exception as e:
                logger.warning(f"[MultiAnalysisChain] NER模型提取实体失败，回退到规则引擎: {str(e)}")

        if not ner_extracted:
            logger.info(f"[MultiAnalysisChain] 使用规则引擎提取医疗实体: text_length={len(text)}")
            entities_list = self._extract_entities_by_rules(text)

        # NER模型返回的BIO标签映射（如B-DIS、I-DIS等）+ 常见别名
        disease_types = ["disease", "疾病", "disease_entity", "疾病实体", "dis", "b-dis", "i-dis", "body_part"]
        symptom_types = ["symptom", "症状", "symptom_entity", "症状实体", "sym", "b-sym", "i-sym"]
        medication_types = ["medication", "药物", "drug", "medication_entity", "药物实体", "med", "b-med", "i-med", "b-dru", "i-dru"]
        procedure_types = ["procedure", "检查", "examination", "检查项目", "procedure_entity", "检查实体", "pro", "b-pro", "i-pro", "medical_item", "surgery"]

        # 所有已知类型的小写集合，用于筛选"other"类别
        all_known_types = set(disease_types + symptom_types + medication_types + procedure_types)

        medical_entities = {
            "diseases": [e for e in entities_list if e.get("entity_type", "").lower() in disease_types],
            "symptoms": [e for e in entities_list if e.get("entity_type", "").lower() in symptom_types],
            "medications": [e for e in entities_list if e.get("entity_type", "").lower() in medication_types],
            "examinations": [e for e in entities_list if e.get("entity_type", "").lower() in procedure_types],
            "other": [e for e in entities_list if e.get("entity_type", "").lower() not in all_known_types]
        }

        logger.info(f"[MultiAnalysisChain] 实体分类结果: diseases={len(medical_entities['diseases'])}, "
                   f"symptoms={len(medical_entities['symptoms'])}, "
                   f"medications={len(medical_entities['medications'])}, "
                   f"examinations={len(medical_entities['examinations'])}, "
                   f"other={len(medical_entities.get('other', []))}")
        for e in entities_list:
            entity_name = e.get('name', e.get('entity_name', '未知'))
            entity_type = e.get('entity_type', '未知')
            # 确定分类归属
            if entity_type.lower() in disease_types:
                category = "diseases"
            elif entity_type.lower() in symptom_types:
                category = "symptoms"
            elif entity_type.lower() in medication_types:
                category = "medications"
            elif entity_type.lower() in procedure_types:
                category = "examinations"
            else:
                category = "other"
            logger.info(f"[ENTITY_CLASSIFY] name={entity_name}, entity_type={entity_type}, category={category}")

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
            "高血压", "2型糖尿病", "糖尿病", "1型糖尿病", "冠心病", "脑卒中", "慢性肾病",
            "肝炎", "胃炎", "肺炎", "哮喘", "关节炎", "高血脂", "高血糖",
            "肿瘤", "癌症", "心脏病", "中风", "心绞痛", "心肌梗死", "心衰",
            "肾功能不全", "肝硬化", "胃溃疡", "肠炎", "甲亢", "甲减",
            "骨质疏松", "痛风", "贫血", "慢性支气管炎", "肺气肿",
        ]

        # 常见症状关键词
        symptom_keywords = [
            "头痛", "头晕", "恶心", "呕吐", "腹痛",
            "咳嗽", "发热", "乏力", "胸闷", "心悸",
            "失眠", "便秘", "腹泻", "水肿", "出血",
            "口渴", "多尿", "多饮", "气短", "胸痛",
            "心慌", "耳鸣", "视力模糊", "肢体麻木",
        ]

        # 常见药物关键词
        medication_keywords = [
            "硝苯地平", "二甲双胍", "阿司匹林", "阿托伐他汀",
            "氨氯地平", "缬沙坦", "氯沙坦", "美托洛尔",
            "格列美脲", "胰岛素", "阿卡波糖", "瑞格列奈",
        ]

        # 提取疾病实体（长关键词优先匹配）
        rule_confidence = _report_config.rule_engine_confidence
        for keyword in sorted(disease_keywords, key=len, reverse=True):
            if keyword in text:
                entities.append({
                    "entity_name": keyword,
                    "entity_type": "Disease",
                    "confidence": rule_confidence
                })

        # 提取症状实体
        for keyword in sorted(symptom_keywords, key=len, reverse=True):
            if keyword in text:
                entities.append({
                    "entity_name": keyword,
                    "entity_type": "Symptom",
                    "confidence": rule_confidence
                })

        # 提取药物实体
        for keyword in medication_keywords:
            if keyword in text:
                entities.append({
                    "entity_name": keyword,
                    "entity_type": "Medication",
                    "confidence": rule_confidence
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
            logger.info("[SPECIAL_RULE] rule=高龄+高血压, triggered=True, result=心血管高风险")
        else:
            logger.info(f"[SPECIAL_RULE] rule=高龄+高血压, triggered=False, has_elderly_risk={has_elderly_risk}, has_hypertension={has_hypertension}")

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
            logger.info("[SPECIAL_RULE] rule=糖尿病+肥胖, triggered=True, result=代谢综合征风险")
        else:
            logger.info(f"[SPECIAL_RULE] rule=糖尿病+肥胖, triggered=False, has_diabetes={has_diabetes}, has_obesity={has_obesity}")

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
            logger.info("[SPECIAL_RULE] rule=多病共存, triggered=True, result=综合评估风险")
        else:
            logger.info("[SPECIAL_RULE] rule=多病共存, triggered=False")

        return special_risks

    def _generate_analysis_summary(self, anomalies: List[Dict], risk_factors: List[Dict], medical_entities: Dict[str, List]) -> str:
        """
        生成分析摘要

        Args:
            anomalies: 异常指标列表
            risk_factors: 风险因子列表
            medical_entities: 医疗实体字典

        Returns:
            分析摘要
        """
        summary_parts = []

        if anomalies:
            anomaly_summary = "、".join([f"{a['indicator_name']}({a['anomaly_type']})" for a in anomalies])
            summary_parts.append(f"检测到{len(anomalies)}项异常指标：{anomaly_summary}")
        else:
            summary_parts.append("未检测到异常指标")

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

        if medical_entities and isinstance(medical_entities, dict):
            disease_entities = medical_entities.get("diseases", [])
            symptom_entities = medical_entities.get("symptoms", [])
            medication_entities = medical_entities.get("medications", [])
            procedure_entities = medical_entities.get("examinations", [])
            other_entities = medical_entities.get("other", [])

            entity_summary_parts = []
            if disease_entities:
                entity_summary_parts.append(f"{len(disease_entities)}种疾病")
            if symptom_entities:
                entity_summary_parts.append(f"{len(symptom_entities)}种症状")
            if medication_entities:
                entity_summary_parts.append(f"{len(medication_entities)}种药物")
            if procedure_entities:
                entity_summary_parts.append(f"{len(procedure_entities)}种检查项目")
            if other_entities:
                entity_summary_parts.append(f"{len(other_entities)}种其他实体")

            if entity_summary_parts:
                summary_parts.append(f"提取到{'、'.join(entity_summary_parts)}相关实体")

        if self._handler_degraded:
            summary_parts.append("（注：医疗实体提取使用规则引擎降级模式）")

        summary = "。".join(summary_parts) + "。"
        return summary
