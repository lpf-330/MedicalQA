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
        monitoring_data: 监测数据（血压、血糖、心率、血氧、BMI、睡眠等）
        user_profile: 用户档案（基本信息、既往病史、家族病史、生活方式）
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

    # 核心字段定义
    CORE_MONITORING_FIELDS = [
        "blood_pressure",  # 血压
        "blood_glucose",   # 血糖
        "heart_rate"       # 心率
    ]

    CORE_PROFILE_FIELDS = [
        "age",    # 年龄
        "gender", # 性别
        "medical_history"  # 既往病史
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

        Args:
            monitoring_data: 原始监测数据

        Returns:
            标准化后的监测数据
        """
        standardized = {}

        # 血压标准化（统一为mmHg）
        if "blood_pressure" in monitoring_data:
            bp_data = monitoring_data["blood_pressure"]
            if isinstance(bp_data, dict):
                # 收缩压
                if "systolic" in bp_data:
                    standardized["systolic_pressure"] = self._standardize_pressure(bp_data["systolic"])
                # 舒张压
                if "diastolic" in bp_data:
                    standardized["diastolic_pressure"] = self._standardize_pressure(bp_data["diastolic"])
            elif isinstance(bp_data, (int, float)):
                # 如果是单个数值，视为收缩压
                standardized["systolic_pressure"] = float(bp_data)

        # 血糖标准化（统一为mmol/L）
        if "blood_glucose" in monitoring_data:
            glucose_data = monitoring_data["blood_glucose"]
            if isinstance(glucose_data, dict):
                # 空腹血糖
                if "fasting" in glucose_data:
                    standardized["fasting_glucose"] = self._standardize_glucose(glucose_data["fasting"])
                # 餐后血糖
                if "postprandial" in glucose_data:
                    standardized["postprandial_glucose"] = self._standardize_glucose(glucose_data["postprandial"])
            elif isinstance(glucose_data, (int, float)):
                standardized["fasting_glucose"] = float(glucose_data)

        # 心率标准化（统一为bpm）
        if "heart_rate" in monitoring_data:
            hr_data = monitoring_data["heart_rate"]
            if isinstance(hr_data, dict):
                if "value" in hr_data:
                    standardized["heart_rate"] = float(hr_data["value"])
            elif isinstance(hr_data, (int, float)):
                standardized["heart_rate"] = float(hr_data)

        # 血氧饱和度标准化（统一为%）
        if "blood_oxygen" in monitoring_data:
            oxygen_data = monitoring_data["blood_oxygen"]
            if isinstance(oxygen_data, dict):
                if "value" in oxygen_data:
                    standardized["blood_oxygen"] = float(oxygen_data["value"])
            elif isinstance(oxygen_data, (int, float)):
                standardized["blood_oxygen"] = float(oxygen_data)

        # BMI标准化
        if "bmi" in monitoring_data:
            bmi_data = monitoring_data["bmi"]
            if isinstance(bmi_data, dict):
                if "value" in bmi_data:
                    standardized["bmi"] = float(bmi_data["value"])
            elif isinstance(bmi_data, (int, float)):
                standardized["bmi"] = float(bmi_data)

        # 睡眠数据标准化
        if "sleep" in monitoring_data:
            sleep_data = monitoring_data["sleep"]
            if isinstance(sleep_data, dict):
                if "duration" in sleep_data:
                    standardized["sleep_duration"] = float(sleep_data["duration"])
                if "quality" in sleep_data:
                    standardized["sleep_quality"] = str(sleep_data["quality"])

        # 保留其他原始数据
        for key, value in monitoring_data.items():
            if key not in ["blood_pressure", "blood_glucose", "heart_rate", "blood_oxygen", "bmi", "sleep"]:
                standardized[key] = value

        return standardized

    def _standardize_user_profile(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准化用户档案

        Args:
            user_profile: 原始用户档案

        Returns:
            标准化后的用户档案
        """
        standardized = {}

        # 基本信息
        if "age" in user_profile:
            standardized["age"] = int(user_profile["age"])

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

        # 既往病史
        if "medical_history" in user_profile:
            medical_history = user_profile["medical_history"]
            if isinstance(medical_history, list):
                standardized["medical_history"] = medical_history
            elif isinstance(medical_history, str):
                standardized["medical_history"] = [medical_history]
            else:
                standardized["medical_history"] = []

        # 家族病史
        if "family_history" in user_profile:
            family_history = user_profile["family_history"]
            if isinstance(family_history, list):
                standardized["family_history"] = family_history
            elif isinstance(family_history, str):
                standardized["family_history"] = [family_history]
            else:
                standardized["family_history"] = []

        # 生活方式
        if "lifestyle" in user_profile:
            lifestyle = user_profile["lifestyle"]
            if isinstance(lifestyle, dict):
                standardized["lifestyle"] = lifestyle

        # 保留其他原始数据
        for key, value in user_profile.items():
            if key not in ["age", "gender", "medical_history", "family_history", "lifestyle"]:
                standardized[key] = value

        return standardized

    def _standardize_pressure(self, value: Any) -> float:
        """
        标准化血压值（统一为mmHg）

        Args:
            value: 原始血压值

        Returns:
            标准化后的血压值（mmHg）
        """
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, dict):
            if "value" in value:
                return float(value["value"])
            elif "mmHg" in value:
                return float(value["mmHg"])
        return 0.0

    def _standardize_glucose(self, value: Any) -> float:
        """
        标准化血糖值（统一为mmol/L）

        Args:
            value: 原始血糖值

        Returns:
            标准化后的血糖值（mmol/L）
        """
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, dict):
            if "value" in value:
                return float(value["value"])
            elif "mmol_L" in value:
                return float(value["mmol_L"])
            elif "mg_dL" in value:
                # mg/dL转mmol/L：除以18
                return float(value["mg_dL"]) / 18.0
        return 0.0

    def _handle_missing_fields(self, validated_data: Dict[str, Any], missing_fields: List[str]) -> tuple:
        """
        空值处理：标记缺失字段

        Args:
            validated_data: 已标准化的数据
            missing_fields: 已识别的缺失字段列表

        Returns:
            处理后的数据和更新后的缺失字段列表
        """
        # 检查监测数据中的核心字段（使用标准化后的字段名）
        monitoring_data = validated_data.get("monitoring_data", {})
        
        # 血压：检查收缩压和舒张压
        if "systolic_pressure" not in monitoring_data or monitoring_data["systolic_pressure"] is None:
            missing_fields.append("monitoring_data.blood_pressure")
        if "diastolic_pressure" not in monitoring_data or monitoring_data["diastolic_pressure"] is None:
            # 舒张压缺失不单独标记，因为血压已经标记
            pass
        
        # 血糖：检查空腹血糖
        if "fasting_glucose" not in monitoring_data or monitoring_data["fasting_glucose"] is None:
            missing_fields.append("monitoring_data.blood_glucose")
        
        # 心率
        if "heart_rate" not in monitoring_data or monitoring_data["heart_rate"] is None:
            missing_fields.append("monitoring_data.heart_rate")

        # 检查用户档案中的核心字段
        user_profile = validated_data.get("user_profile", {})
        for field in self.CORE_PROFILE_FIELDS:
            if field not in user_profile or not user_profile[field]:
                missing_fields.append(f"user_profile.{field}")

        # 为缺失的核心字段设置默认值
        if "systolic_pressure" not in monitoring_data:
            validated_data["monitoring_data"]["systolic_pressure"] = None
        if "diastolic_pressure" not in monitoring_data:
            validated_data["monitoring_data"]["diastolic_pressure"] = None
        if "fasting_glucose" not in monitoring_data:
            validated_data["monitoring_data"]["fasting_glucose"] = None
        if "heart_rate" not in monitoring_data:
            validated_data["monitoring_data"]["heart_rate"] = None

        if "age" not in user_profile:
            validated_data["user_profile"]["age"] = None
        if "gender" not in user_profile:
            validated_data["user_profile"]["gender"] = None
        if "medical_history" not in user_profile:
            validated_data["user_profile"]["medical_history"] = []

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
