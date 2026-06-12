# -*- coding: utf-8 -*-
"""
数据准备Chain策略

实现健康报告生成业务的数据准备Chain策略。
"""

import logging
import time
from typing import Any, Dict, List
from src.config.business.report_service_config import get_runtime_config
from src.orchestration.chain.chain import Chain
from src.orchestration.chain.data_classes import ChainContext, ChainResult
from src.orchestration.chain.data_prepare_chain.data_prepare_context import DataPrepareContextBody
from src.orchestration.chain.data_prepare_chain.data_prepare_result import DataPrepareResultData
from src.orchestration.chain.data_prepare_chain.data_prepare_resource import DataPrepareResource

logger = logging.getLogger(__name__)

class _LazyReportConfig:
    """延迟加载配置代理，每次属性访问时从ConfigManager获取真实配置"""
    def __getattr__(self, name):
        return getattr(get_runtime_config(), name)

_report_config = _LazyReportConfig()

class DataPrepareChain(Chain[ChainContext[DataPrepareContextBody], ChainResult[DataPrepareResultData]]):
    """
    数据准备Chain策略类

    实现数据准备的固定流程：
    1. 参数校验（必填字段检查）
    2. 数据标准化（单位统一、格式转换）
    3. 空值处理（缺失字段标记）
    4. 完整性判断（计算降级级别）
    """

    # 核心字段定义 - 6项监测指标
    CORE_MONITORING_FIELDS = [
        "heart_rate",       # 心率
        "blood_glucose",    # 血糖
        "perfusion_index",  # 灌注指数
        "blood_oxygen",     # 血氧
        "sleep",            # 睡眠
        "blood_pressure"    # 血压
    ]

    # 核心用户档案字段
    CORE_PROFILE_FIELDS = [
        "gender",           # 性别
        "birth_date",       # 出生日期
        "height",           # 身高
        "weight"            # 体重
    ]

    def __init__(self, resource: DataPrepareResource):
        """
        初始化数据准备Chain策略

        Args:
            resource: Chain策略专属资源
        """
        self._resource = resource

    def execute(self, chain_context: ChainContext[DataPrepareContextBody]) -> ChainResult[DataPrepareResultData]:
        """
        执行Chain策略

        Args:
            chain_context: Chain输入数据容器

        Returns:
            ChainResult: Chain输出数据容器
        """
        start_time = time.time()
        logger.info(f"[DataPrepareChain] 开始执行Chain: session_id={chain_context.session_id}")

        body = chain_context.body
        if body is None:
            logger.warning(f"[DataPrepareChain] 输入数据为空: session_id={chain_context.session_id}")
            return ChainResult(
                session_id=chain_context.session_id,
                data=DataPrepareResultData(
                    validated_data={},
                    degradation_level=3,
                    missing_fields=["task_id", "monitoring_data", "user_profile"],
                    data_completeness=0.0
                )
            )

        try:
            # 1. 参数校验
            missing_fields = self._validate_required_fields(body)
            logger.info(f"[DataPrepareChain] 参数校验完成: missing_fields={missing_fields}")

            # 2. 数据标准化
            validated_data = self._standardize_data(body)
            logger.info("[DataPrepareChain] 数据标准化完成")

            # 3. 空值处理和缺失字段标记
            validated_data, missing_fields = self._handle_missing_fields(validated_data, missing_fields)
            logger.info(f"[DataPrepareChain] 空值处理完成: total_missing_fields={len(missing_fields)}")

            # 4. 计算数据完整度
            data_completeness = self._calculate_completeness(validated_data, missing_fields)
            logger.info(f"[DataPrepareChain] 数据完整度计算完成: completeness={data_completeness:.2%}")

            # 5. 计算降级级别
            degradation_level = self._calculate_degradation_level(data_completeness, missing_fields)
            logger.info(f"[DataPrepareChain] 降级级别计算完成: degradation_level={degradation_level}")

            result_data = DataPrepareResultData(
                validated_data=validated_data,
                degradation_level=degradation_level,
                missing_fields=missing_fields,
                data_completeness=data_completeness
            )

            elapsed = time.time() - start_time
            logger.info(f"[DataPrepareChain] Chain执行完成: session_id={chain_context.session_id}, "
                       f"degradation_level={degradation_level}, "
                       f"data_completeness={data_completeness:.2%}, "
                       f"elapsed={elapsed:.2f}s")

            return ChainResult(session_id=chain_context.session_id, data=result_data)

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[DataPrepareChain] Chain执行异常: session_id={chain_context.session_id}, "
                        f"error={str(e)}, elapsed={elapsed:.2f}s")
            return ChainResult(
                session_id=chain_context.session_id,
                data=DataPrepareResultData(
                    validated_data={},
                    degradation_level=3,
                    missing_fields=[],
                    data_completeness=0.0
                )
            )

    def _validate_required_fields(self, body: DataPrepareContextBody) -> List[str]:
        """
        参数校验：检查必填字段

        Args:
            body: Chain策略专属输入数据

        Returns:
            缺失字段列表
        """
        missing_fields = []

        # 检查task_id
        if not body.task_id:
            missing_fields.append("task_id")

        # 检查monitoring_data
        if not body.monitoring_data:
            missing_fields.append("monitoring_data")

        # 检查user_profile
        if not body.user_profile:
            missing_fields.append("user_profile")

        return missing_fields

    def _standardize_data(self, body: DataPrepareContextBody) -> Dict[str, Any]:
        """
        数据标准化：单位统一、格式转换

        Args:
            body: Chain策略专属输入数据

        Returns:
            标准化后的数据
        """
        validated_data = {
            "task_id": body.task_id,
            "monitoring_data": {},
            "user_profile": {}
        }

        # 标准化监测数据
        if body.monitoring_data:
            validated_data["monitoring_data"] = self._standardize_monitoring_data(body.monitoring_data)

        # 标准化用户档案
        if body.user_profile:
            validated_data["user_profile"] = self._standardize_user_profile(body.user_profile)

        return validated_data

    def _standardize_monitoring_data(self, monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准化监测数据

        处理6项监测指标，每项指标包含4个时间维度：
        - latest: 当日最新3-5次数据
        - daily_stats: 最近30天日统计
        - weekly_stats: 最近12周周统计
        - monthly_stats: 最近6个月月统计

        Args:
            monitoring_data: 原始监测数据

        Returns:
            标准化后的监测数据
        """
        standardized = {}

        # 心率标准化（统一为bpm）
        if "heart_rate" in monitoring_data:
            hr_data = monitoring_data["heart_rate"]
            if isinstance(hr_data, dict):
                standardized["heart_rate"] = self._standardize_time_dimension_data(hr_data, "heart_rate")

        # 血糖标准化（统一为mmol/L）
        if "blood_glucose" in monitoring_data:
            glucose_data = monitoring_data["blood_glucose"]
            if isinstance(glucose_data, dict):
                standardized["blood_glucose"] = self._standardize_time_dimension_data(glucose_data, "blood_glucose")

        # 灌注指数标准化
        if "perfusion_index" in monitoring_data:
            pi_data = monitoring_data["perfusion_index"]
            if isinstance(pi_data, dict):
                standardized["perfusion_index"] = self._standardize_time_dimension_data(pi_data, "perfusion_index")

        # 血氧饱和度标准化（统一为%）
        if "blood_oxygen" in monitoring_data:
            oxygen_data = monitoring_data["blood_oxygen"]
            if isinstance(oxygen_data, dict):
                standardized["blood_oxygen"] = self._standardize_time_dimension_data(oxygen_data, "blood_oxygen")

        # 睡眠数据标准化
        if "sleep" in monitoring_data:
            sleep_data = monitoring_data["sleep"]
            if isinstance(sleep_data, dict):
                standardized["sleep"] = self._standardize_time_dimension_data(sleep_data, "sleep")

        # 血压标准化（统一为mmHg）
        if "blood_pressure" in monitoring_data:
            bp_data = monitoring_data["blood_pressure"]
            if isinstance(bp_data, dict):
                standardized["blood_pressure"] = self._standardize_time_dimension_data(bp_data, "blood_pressure")

        # 保留其他原始数据
        for key, value in monitoring_data.items():
            if key not in ["heart_rate", "blood_glucose", "perfusion_index", "blood_oxygen", "sleep", "blood_pressure"]:
                standardized[key] = value

        return standardized

    def _standardize_time_dimension_data(self, data: Dict[str, Any], indicator_type: str) -> Dict[str, Any]:
        """
        标准化单个指标的时间维度数据

        Args:
            data: 原始时间维度数据
            indicator_type: 指标类型

        Returns:
            标准化后的时间维度数据
        """
        standardized = {}
        time_dimensions = ["latest", "daily_stats", "weekly_stats", "monthly_stats"]

        for dimension in time_dimensions:
            current_data = data.get("current")
            if dimension in data:
                dimension_data = data[dimension]
                if dimension == "latest" and not dimension_data and current_data is not None:
                    standardized[dimension] = current_data if isinstance(current_data, list) else [current_data]
                elif dimension_data is not None:
                    standardized[dimension] = dimension_data
                else:
                    standardized[dimension] = []
            elif dimension == "latest" and current_data is not None:
                standardized[dimension] = current_data if isinstance(current_data, list) else [current_data]
            else:
                standardized[dimension] = []

        return standardized

    def _standardize_user_profile(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准化用户档案

        处理与SpringBoot后端users表对齐的字段：
        - user_id: 用户ID
        - gender: 性别
        - birth_date: 出生日期
        - height: 身高
        - weight: 体重
        - past_medical_history: 既往病史（字符串类型）
        - family_history: 家族病史（字符串类型）
        - allergy_history: 过敏史（字符串类型）
        - surgical_history: 手术史（字符串类型）
        - medical_compliance: 用药医嘱（字符串类型）

        Args:
            user_profile: 原始用户档案

        Returns:
            标准化后的用户档案
        """
        standardized = {}

        # 用户ID
        if "user_id" in user_profile:
            standardized["user_id"] = int(user_profile["user_id"]) if user_profile["user_id"] is not None else None

        # 性别
        if "gender" in user_profile:
            gender = user_profile["gender"]
            # 统一性别格式
            if isinstance(gender, str):
                if gender.lower() in ["男", "male", "m"]:
                    standardized["gender"] = "male"
                elif gender.lower() in ["女", "female", "f"]:
                    standardized["gender"] = "female"
                else:
                    standardized["gender"] = gender.lower()

        # 出生日期
        if "birth_date" in user_profile:
            standardized["birth_date"] = str(user_profile["birth_date"]) if user_profile["birth_date"] else None

        # 从birth_date计算age；若外部已提供age，则保留该值作为兜底
        birth_date = standardized.get("birth_date", "")
        if birth_date:
            try:
                from datetime import datetime
                birth = datetime.strptime(birth_date, "%Y-%m-%d")
                today = datetime.now()
                age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
                standardized["age"] = age
            except (ValueError, TypeError):
                standardized["age"] = user_profile.get("age", -1)
        else:
            standardized["age"] = user_profile.get("age", -1)

        # 身高
        if "height" in user_profile:
            try:
                standardized["height"] = float(user_profile["height"])
            except (ValueError, TypeError):
                standardized["height"] = None

        # 体重
        if "weight" in user_profile:
            try:
                standardized["weight"] = float(user_profile["weight"])
            except (ValueError, TypeError):
                standardized["weight"] = None

        # 既往病史（字符串类型，按配置截断）
        if "past_medical_history" in user_profile or "medical_history" in user_profile:
            medical_history = user_profile.get("past_medical_history")
            if medical_history is None:
                medical_history = user_profile.get("medical_history")
            if medical_history is not None:
                history_str = str(medical_history)
                limit = _report_config.past_medical_history_limit
                if len(history_str) > limit:
                    history_str = history_str[:limit] + "..."
                standardized["past_medical_history"] = history_str
            else:
                standardized["past_medical_history"] = ""

        # 家族病史（字符串类型，按配置截断）
        if "family_history" in user_profile:
            family_history = user_profile["family_history"]
            if family_history is not None:
                history_str = str(family_history)
                limit = _report_config.family_history_limit
                if len(history_str) > limit:
                    history_str = history_str[:limit] + "..."
                standardized["family_history"] = history_str
            else:
                standardized["family_history"] = ""

        # 过敏史（字符串类型）
        if "allergy_history" in user_profile:
            allergy_history = user_profile["allergy_history"]
            if allergy_history is not None:
                standardized["allergy_history"] = str(allergy_history)
            else:
                standardized["allergy_history"] = ""

        # 手术史（字符串类型）
        if "surgical_history" in user_profile:
            surgical_history = user_profile["surgical_history"]
            if surgical_history is not None:
                standardized["surgical_history"] = str(surgical_history)
            else:
                standardized["surgical_history"] = ""

        # 用药医嘱（字符串类型）
        if "medical_compliance" in user_profile or "medication_history" in user_profile:
            medical_compliance = user_profile.get("medical_compliance")
            if medical_compliance is None:
                medical_compliance = user_profile.get("medication_history")
            if medical_compliance is not None:
                standardized["medical_compliance"] = str(medical_compliance)
            else:
                standardized["medical_compliance"] = ""

        # 保留其他原始数据
        for key, value in user_profile.items():
            if key not in ["user_id", "gender", "birth_date", "age", "height", "weight",
                          "past_medical_history", "medical_history", "family_history", "allergy_history",
                          "surgical_history", "medical_compliance", "medication_history"]:
                standardized[key] = value

        return standardized

    def _handle_missing_fields(self, validated_data: Dict[str, Any], missing_fields: List[str]) -> tuple:
        """
        空值处理：标记缺失字段

        Args:
            validated_data: 已标准化的数据
            missing_fields: 已识别的缺失字段列表

        Returns:
            处理后的数据和更新后的缺失字段列表
        """
        # 检查监测数据中的核心字段
        monitoring_data = validated_data.get("monitoring_data", {})

        # 检查6项监测指标
        for monitoring_field in self.CORE_MONITORING_FIELDS:
            if monitoring_field not in monitoring_data or not monitoring_data[monitoring_field]:
                missing_fields.append(f"monitoring_data.{monitoring_field}")
            elif isinstance(monitoring_data[monitoring_field], dict):
                # 检查是否有有效的时间维度数据
                has_valid_data = False
                for dimension in ["latest", "daily_stats", "weekly_stats", "monthly_stats"]:
                    if dimension in monitoring_data[monitoring_field] and monitoring_data[monitoring_field][dimension]:
                        has_valid_data = True
                        break
                if not has_valid_data:
                    missing_fields.append(f"monitoring_data.{monitoring_field}")

        # 检查用户档案中的核心字段
        user_profile = validated_data.get("user_profile", {})
        for profile_field in self.CORE_PROFILE_FIELDS:
            if profile_field not in user_profile or not user_profile[profile_field]:
                missing_fields.append(f"user_profile.{profile_field}")

        # 为缺失的核心字段设置默认值
        for monitoring_field in self.CORE_MONITORING_FIELDS:
            if monitoring_field not in monitoring_data:
                validated_data["monitoring_data"][monitoring_field] = {
                    "latest": [],
                    "daily_stats": [],
                    "weekly_stats": [],
                    "monthly_stats": []
                }

        for profile_field in self.CORE_PROFILE_FIELDS:
            if profile_field not in user_profile:
                if profile_field in ["height", "weight"]:
                    validated_data["user_profile"][profile_field] = None
                else:
                    validated_data["user_profile"][profile_field] = ""

        # 为病史字段设置默认值
        history_fields = ["past_medical_history", "family_history", "allergy_history", "surgical_history", "medical_compliance"]
        for history_field in history_fields:
            if history_field not in user_profile:
                validated_data["user_profile"][history_field] = ""

        # 记录缺失字段详情日志
        if missing_fields:
            core_missing = [f for f in missing_fields if any(core_field in f for core_field in self.CORE_MONITORING_FIELDS + self.CORE_PROFILE_FIELDS)]
            other_missing = [f for f in missing_fields if f not in core_missing]
            logger.info(f"[DataPrepareChain] [MISSING_FIELDS] total={len(missing_fields)}, core_missing={core_missing}, other_missing={other_missing}")

        return validated_data, missing_fields

    def _calculate_completeness(self, validated_data: Dict[str, Any], missing_fields: List[str]) -> float:
        """
        计算数据完整度

        Args:
            validated_data: 已标准化的数据
            missing_fields: 缺失字段列表

        Returns:
            数据完整度（0.0-1.0）
        """
        # 计算总字段数
        total_fields = 0
        filled_fields = 0

        # 统计监测数据字段
        monitoring_data = validated_data.get("monitoring_data", {})
        for key, value in monitoring_data.items():
            total_fields += 1
            if value is not None and value != "" and value != [] and value != {} and value != "未知" and value != -1:
                filled_fields += 1

        # 统计用户档案字段
        user_profile = validated_data.get("user_profile", {})
        for key, value in user_profile.items():
            total_fields += 1
            if value is not None and value != "" and value != [] and value != {} and value != "未知" and value != -1:
                filled_fields += 1

        # 计算完整度
        if total_fields == 0:
            return 0.0

        completeness = filled_fields / total_fields
        return round(completeness, 2)

    def _calculate_degradation_level(self, data_completeness: float, missing_fields: List[str]) -> int:
        """
        计算降级级别

        降级级别规则：
        - 0级：数据完整度≥高阈值，所有核心字段都有数据
        - 1级：数据完整度≥高阈值但有核心字段缺失，或完整度中阈值-高阈值且核心字段缺失<=轻度
        - 2级：数据完整度低阈值-中阈值，核心字段缺失<=中度
        - 3级：数据完整度<低阈值，或核心字段缺失>中度

        Args:
            data_completeness: 数据完整度
            missing_fields: 缺失字段列表

        Returns:
            降级级别（0-3）
        """
        # 统计缺失的核心字段数量
        core_missing_count = 0
        for missing_field in missing_fields:
            if any(core_field in missing_field for core_field in self.CORE_MONITORING_FIELDS + self.CORE_PROFILE_FIELDS):
                core_missing_count += 1

        # 从配置读取降级阈值
        completeness_high = _report_config.degradation_completeness_high
        completeness_medium = _report_config.degradation_completeness_medium
        completeness_low = _report_config.degradation_completeness_low
        core_missing_mild = _report_config.degradation_core_missing_mild
        core_missing_moderate = _report_config.degradation_core_missing_moderate

        # 根据完整度和核心字段缺失情况判断降级级别
        if core_missing_count > core_missing_moderate:
            rule_desc = f"completeness={data_completeness:.2f}, core_missing_count={core_missing_count}, rule=核心字段缺失>{core_missing_moderate}, level=3"
            logger.info(f"[DataPrepareChain] [DEGRADATION_CALC] {rule_desc}")
            return 3
        if data_completeness >= completeness_high and core_missing_count == 0:
            rule_desc = f"completeness={data_completeness:.2f}, core_missing_count={core_missing_count}, rule=≥{completeness_high:.0%}且核心字段无缺失, level=0"
            logger.info(f"[DataPrepareChain] [DEGRADATION_CALC] {rule_desc}")
            return 0
        elif data_completeness >= completeness_high and core_missing_count > 0:
            rule_desc = f"completeness={data_completeness:.2f}, core_missing_count={core_missing_count}, rule=≥{completeness_high:.0%}但核心字段有缺失, level=1"
            logger.info(f"[DataPrepareChain] [DEGRADATION_CALC] {rule_desc}")
            return 1
        elif data_completeness >= completeness_medium and core_missing_count <= core_missing_mild:
            rule_desc = f"completeness={data_completeness:.2f}, core_missing_count={core_missing_count}, rule={completeness_medium:.0%}-{completeness_high:.0%}且核心字段缺失<={core_missing_mild}, level=1"
            logger.info(f"[DataPrepareChain] [DEGRADATION_CALC] {rule_desc}")
            return 1
        elif data_completeness >= completeness_low and core_missing_count <= core_missing_moderate:
            rule_desc = f"completeness={data_completeness:.2f}, core_missing_count={core_missing_count}, rule={completeness_low:.0%}-{completeness_medium:.0%}且核心字段缺失<={core_missing_moderate}, level=2"
            logger.info(f"[DataPrepareChain] [DEGRADATION_CALC] {rule_desc}")
            return 2
        else:
            rule_desc = f"completeness={data_completeness:.2f}, core_missing_count={core_missing_count}, rule=<{completeness_low:.0%}或核心字段缺失>{core_missing_moderate}, level=3"
            logger.info(f"[DataPrepareChain] [DEGRADATION_CALC] {rule_desc}")
            return 3
