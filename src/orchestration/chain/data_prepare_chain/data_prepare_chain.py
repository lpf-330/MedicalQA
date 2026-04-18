# -*- coding: utf-8 -*-
"""
数据准备Chain策略

实现健康报告生成业务的数据准备Chain策略。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.orchestration.chain.chain import Chain
from src.orchestration.chain.data_classes import ChainContext, ChainResult

logger = logging.getLogger(__name__)


@dataclass
class DataPrepareContextBody:
    """
    数据准备Chain策略专属输入数据类

    Attributes:
        monitoring_data: 监测数据（心率、血糖、灌注指数、血氧、睡眠、血压），每项包含4个时间维度
        user_profile: 用户档案（user_id, gender, birth_date, height, weight, past_medical_history, family_history, allergy_history, surgical_history, medical_compliance）
        task_id: 任务ID
    """
    monitoring_data: Dict[str, Any] = field(default_factory=dict)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    task_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "monitoring_data": self.monitoring_data,
            "user_profile": self.user_profile,
            "task_id": self.task_id
        }


@dataclass
class DataPrepareResultData:
    """
    数据准备Chain策略专属输出数据类

    Attributes:
        validated_data: 校验后的数据
        degradation_level: 降级级别（0-3）
        missing_fields: 缺失字段列表
        data_completeness: 数据完整度（0.0-1.0）
    """
    validated_data: Dict[str, Any] = field(default_factory=dict)
    degradation_level: int = 0
    missing_fields: List[str] = field(default_factory=list)
    data_completeness: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "validated_data": self.validated_data,
            "degradation_level": self.degradation_level,
            "missing_fields": self.missing_fields,
            "data_completeness": self.data_completeness
        }


@dataclass
class DataPrepareResource:
    """
    数据准备Chain策略专属资源类

    暂无外部资源依赖
    """
    pass


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
            logger.info(f"[DataPrepareChain] 数据标准化完成")

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

        # 时间维度列表
        time_dimensions = ["latest", "daily_stats", "weekly_stats", "monthly_stats"]

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
            if dimension in data:
                dimension_data = data[dimension]
                if dimension_data is not None:
                    standardized[dimension] = dimension_data
                else:
                    standardized[dimension] = []
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

        # 既往病史（字符串类型）
        if "past_medical_history" in user_profile:
            medical_history = user_profile["past_medical_history"]
            if medical_history is not None:
                standardized["past_medical_history"] = str(medical_history)
            else:
                standardized["past_medical_history"] = ""

        # 家族病史（字符串类型）
        if "family_history" in user_profile:
            family_history = user_profile["family_history"]
            if family_history is not None:
                standardized["family_history"] = str(family_history)
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
        if "medical_compliance" in user_profile:
            medical_compliance = user_profile["medical_compliance"]
            if medical_compliance is not None:
                standardized["medical_compliance"] = str(medical_compliance)
            else:
                standardized["medical_compliance"] = ""

        # 保留其他原始数据
        for key, value in user_profile.items():
            if key not in ["user_id", "gender", "birth_date", "height", "weight",
                          "past_medical_history", "family_history", "allergy_history",
                          "surgical_history", "medical_compliance"]:
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
        for field in self.CORE_MONITORING_FIELDS:
            if field not in monitoring_data or not monitoring_data[field]:
                missing_fields.append(f"monitoring_data.{field}")
            elif isinstance(monitoring_data[field], dict):
                # 检查是否有有效的时间维度数据
                has_valid_data = False
                for dimension in ["latest", "daily_stats", "weekly_stats", "monthly_stats"]:
                    if dimension in monitoring_data[field] and monitoring_data[field][dimension]:
                        has_valid_data = True
                        break
                if not has_valid_data:
                    missing_fields.append(f"monitoring_data.{field}")

        # 检查用户档案中的核心字段
        user_profile = validated_data.get("user_profile", {})
        for field in self.CORE_PROFILE_FIELDS:
            if field not in user_profile or not user_profile[field]:
                missing_fields.append(f"user_profile.{field}")

        # 为缺失的核心字段设置默认值
        for field in self.CORE_MONITORING_FIELDS:
            if field not in monitoring_data:
                validated_data["monitoring_data"][field] = {
                    "latest": [],
                    "daily_stats": [],
                    "weekly_stats": [],
                    "monthly_stats": []
                }

        for field in self.CORE_PROFILE_FIELDS:
            if field not in user_profile:
                if field in ["height", "weight"]:
                    validated_data["user_profile"][field] = None
                else:
                    validated_data["user_profile"][field] = ""

        # 为病史字段设置默认值
        history_fields = ["past_medical_history", "family_history", "allergy_history", "surgical_history", "medical_compliance"]
        for field in history_fields:
            if field not in user_profile:
                validated_data["user_profile"][field] = ""

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
            if value is not None and value != [] and value != {}:
                filled_fields += 1

        # 统计用户档案字段
        user_profile = validated_data.get("user_profile", {})
        for key, value in user_profile.items():
            total_fields += 1
            if value is not None and value != [] and value != {}:
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
        - 0级：数据完整度≥90%，所有核心字段都有数据
        - 1级：数据完整度70%-89%，部分核心字段缺失
        - 2级：数据完整度50%-69%，多个核心字段缺失
        - 3级：数据完整度<50%，大量核心字段缺失

        Args:
            data_completeness: 数据完整度
            missing_fields: 缺失字段列表

        Returns:
            降级级别（0-3）
        """
        # 统计缺失的核心字段数量
        core_missing_count = 0
        for field in missing_fields:
            if any(core_field in field for core_field in self.CORE_MONITORING_FIELDS + self.CORE_PROFILE_FIELDS):
                core_missing_count += 1

        # 根据完整度和核心字段缺失情况判断降级级别
        if data_completeness >= 0.9 and core_missing_count == 0:
            return 0
        elif data_completeness >= 0.7 and core_missing_count <= 2:
            return 1
        elif data_completeness >= 0.5 and core_missing_count <= 4:
            return 2
        else:
            return 3
