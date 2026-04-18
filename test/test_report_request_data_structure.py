"""
健康报告生成业务接口数据结构测试模块

该模块测试健康报告生成业务接口的数据结构，确保数据结构与SpringBoot后端数据库表结构对齐。
测试覆盖以下场景：
1. 完整的监测数据（包含4个时间维度）
2. 部分时间维度的监测数据
3. 字符串类型的病史字段
4. 空值处理逻辑
5. 端到端的报告生成流程
"""

import pytest
from typing import Dict, Any, Optional
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from src.schemas.report_request import (
    MonitoringData,
    UserProfile,
    ReportRequestBody,
    ReportRequest
)
from src.schemas.base_request import BaseRequest


# ==============================================================================
# 测试数据工厂类
# ==============================================================================

class TestDataFactory:
    """
    测试数据工厂类

    提供创建各种测试数据的静态方法，确保测试数据符合SpringBoot后端数据库结构。
    """

    @staticmethod
    def create_complete_heart_rate_data() -> Dict[str, Any]:
        """
        创建完整的心率数据（包含4个时间维度）

        Returns:
            Dict[str, Any]: 完整的心率数据
        """
        return {
            "latest": [
                {"value": 72, "unit": "bpm", "time": "2024-01-15 08:00:00"},
                {"value": 75, "unit": "bpm", "time": "2024-01-15 12:00:00"},
                {"value": 70, "unit": "bpm", "time": "2024-01-15 18:00:00"}
            ],
            "daily_stats": [
                {"date": "2024-01-15", "avg": 72, "max": 85, "min": 62},
                {"date": "2024-01-14", "avg": 71, "max": 83, "min": 60},
                {"date": "2024-01-13", "avg": 73, "max": 88, "min": 65}
            ],
            "weekly_stats": [
                {"week": "2024-W2", "avg": 72, "trend": "stable"},
                {"week": "2024-W1", "avg": 71, "trend": "rising"}
            ],
            "monthly_stats": [
                {"month": "2024-01", "avg": 72, "trend": "stable"},
                {"month": "2023-12", "avg": 70, "trend": "stable"}
            ]
        }

    @staticmethod
    def create_complete_blood_glucose_data() -> Dict[str, Any]:
        """
        创建完整的血糖数据（包含4个时间维度）

        Returns:
            Dict[str, Any]: 完整的血糖数据
        """
        return {
            "latest": [
                {"value": 5.5, "unit": "mmol/L", "type": "fasting", "time": "2024-01-15 07:00:00"},
                {"value": 7.2, "unit": "mmol/L", "type": "postprandial", "time": "2024-01-15 09:30:00"}
            ],
            "daily_stats": [
                {"date": "2024-01-15", "avg": 6.0, "max": 7.5, "min": 5.2},
                {"date": "2024-01-14", "avg": 5.8, "max": 7.2, "min": 5.0}
            ],
            "weekly_stats": [
                {"week": "2024-W2", "avg": 5.9, "trend": "stable"},
                {"week": "2024-W1", "avg": 5.7, "trend": "rising"}
            ],
            "monthly_stats": [
                {"month": "2024-01", "avg": 5.8, "trend": "stable"},
                {"month": "2023-12", "avg": 5.6, "trend": "stable"}
            ]
        }

    @staticmethod
    def create_complete_perfusion_index_data() -> Dict[str, Any]:
        """
        创建完整的灌注指数数据（包含4个时间维度）

        Returns:
            Dict[str, Any]: 完整的灌注指数数据
        """
        return {
            "latest": [
                {"value": 3.5, "unit": "PI", "time": "2024-01-15 08:00:00"},
                {"value": 3.8, "unit": "PI", "time": "2024-01-15 14:00:00"}
            ],
            "daily_stats": [
                {"date": "2024-01-15", "avg": 3.6, "max": 4.0, "min": 3.2},
                {"date": "2024-01-14", "avg": 3.4, "max": 3.9, "min": 3.0}
            ],
            "weekly_stats": [
                {"week": "2024-W2", "avg": 3.5, "trend": "stable"},
                {"week": "2024-W1", "avg": 3.3, "trend": "stable"}
            ],
            "monthly_stats": [
                {"month": "2024-01", "avg": 3.4, "trend": "stable"},
                {"month": "2023-12", "avg": 3.2, "trend": "stable"}
            ]
        }

    @staticmethod
    def create_complete_blood_oxygen_data() -> Dict[str, Any]:
        """
        创建完整的血氧数据（包含4个时间维度）

        Returns:
            Dict[str, Any]: 完整的血氧数据
        """
        return {
            "latest": [
                {"value": 98.5, "unit": "%", "time": "2024-01-15 08:00:00"},
                {"value": 97.8, "unit": "%", "time": "2024-01-15 12:00:00"}
            ],
            "daily_stats": [
                {"date": "2024-01-15", "avg": 98.0, "max": 99.0, "min": 97.0},
                {"date": "2024-01-14", "avg": 97.8, "max": 98.5, "min": 96.5}
            ],
            "weekly_stats": [
                {"week": "2024-W2", "avg": 98.1, "trend": "stable"},
                {"week": "2024-W1", "avg": 97.9, "trend": "stable"}
            ],
            "monthly_stats": [
                {"month": "2024-01", "avg": 98.0, "trend": "stable"},
                {"month": "2023-12", "avg": 97.8, "trend": "stable"}
            ]
        }

    @staticmethod
    def create_complete_sleep_data() -> Dict[str, Any]:
        """
        创建完整的睡眠数据（包含4个时间维度）

        Returns:
            Dict[str, Any]: 完整的睡眠数据
        """
        return {
            "latest": [
                {"value": 7.5, "unit": "hours", "time": "2024-01-15 06:00:00"},
                {"value": 6.8, "unit": "hours", "time": "2024-01-14 06:30:00"}
            ],
            "daily_stats": [
                {"date": "2024-01-15", "avg": 7.2, "max": 8.0, "min": 6.5},
                {"date": "2024-01-14", "avg": 7.0, "max": 7.5, "min": 6.0}
            ],
            "weekly_stats": [
                {"week": "2024-W2", "avg": 7.1, "trend": "stable"},
                {"week": "2024-W1", "avg": 6.8, "trend": "rising"}
            ],
            "monthly_stats": [
                {"month": "2024-01", "avg": 7.0, "trend": "stable"},
                {"month": "2023-12", "avg": 6.5, "trend": "stable"}
            ]
        }

    @staticmethod
    def create_complete_blood_pressure_data() -> Dict[str, Any]:
        """
        创建完整的血压数据（包含4个时间维度）

        Returns:
            Dict[str, Any]: 完整的血压数据
        """
        return {
            "latest": [
                {"systolic": 120, "diastolic": 80, "unit": "mmHg", "time": "2024-01-15 08:00:00"},
                {"systolic": 118, "diastolic": 78, "unit": "mmHg", "time": "2024-01-15 12:00:00"}
            ],
            "daily_stats": [
                {"date": "2024-01-15", "avg_systolic": 118, "avg_diastolic": 79},
                {"date": "2024-01-14", "avg_systolic": 120, "avg_diastolic": 80}
            ],
            "weekly_stats": [
                {"week": "2024-W2", "avg_systolic": 119, "avg_diastolic": 78, "trend": "stable"},
                {"week": "2024-W1", "avg_systolic": 121, "avg_diastolic": 80, "trend": "rising"}
            ],
            "monthly_stats": [
                {"month": "2024-01", "avg_systolic": 120, "avg_diastolic": 79, "trend": "stable"},
                {"month": "2023-12", "avg_systolic": 118, "avg_diastolic": 78, "trend": "stable"}
            ]
        }

    @staticmethod
    def create_complete_monitoring_data() -> MonitoringData:
        """
        创建完整的监测数据（包含6项指标，每项包含4个时间维度）

        Returns:
            MonitoringData: 完整的监测数据对象
        """
        return MonitoringData(
            heart_rate=TestDataFactory.create_complete_heart_rate_data(),
            blood_glucose=TestDataFactory.create_complete_blood_glucose_data(),
            perfusion_index=TestDataFactory.create_complete_perfusion_index_data(),
            blood_oxygen=TestDataFactory.create_complete_blood_oxygen_data(),
            sleep=TestDataFactory.create_complete_sleep_data(),
            blood_pressure=TestDataFactory.create_complete_blood_pressure_data()
        )

    @staticmethod
    def create_complete_user_profile() -> UserProfile:
        """
        创建完整的用户档案

        Returns:
            UserProfile: 完整的用户档案对象
        """
        return UserProfile(
            user_id=1,
            gender="male",
            birth_date="1955-03-15",
            height=170.0,
            weight=75.0,
            past_medical_history="冠心病史5年、高血脂3年、2020年脑梗死",
            family_history="父亲有高血压、母亲有糖尿病",
            allergy_history="青霉素过敏、海鲜过敏",
            surgical_history="2020年胆囊切除术、2018年髋关节置换术",
            medical_compliance="好"
        )

    @staticmethod
    def create_complete_report_request() -> ReportRequest:
        """
        创建完整的健康报告生成请求

        Returns:
            ReportRequest: 完整的请求对象
        """
        return ReportRequest(
            request_id="req-test-001",
            timestamp="2024-01-15T10:00:00",
            user_id="user-001",
            client_info={"client_type": "web", "version": "1.0.0"},
            body=ReportRequestBody(
                task_id="task-001",
                monitoring_data=TestDataFactory.create_complete_monitoring_data(),
                user_profile=TestDataFactory.create_complete_user_profile(),
                session_id="session-001"
            )
        )


# ==============================================================================
# 测试类1: 测试完整的监测数据（包含4个时间维度）
# ==============================================================================

class TestCompleteMonitoringData:
    """
    测试完整的监测数据（包含4个时间维度）

    测试包含6项监测指标（heart_rate, blood_glucose, perfusion_index, blood_oxygen, sleep, blood_pressure）
    测试每项监测指标包含4个时间维度（latest, daily_stats, weekly_stats, monthly_stats）
    测试数据格式符合SpringBoot后端数据库结构
    """

    def test_complete_heart_rate_data_structure(self):
        """测试心率数据结构包含完整的4个时间维度"""
        heart_rate_data = TestDataFactory.create_complete_heart_rate_data()
        monitoring_data = MonitoringData(heart_rate=heart_rate_data)

        assert monitoring_data.heart_rate is not None
        assert "latest" in monitoring_data.heart_rate
        assert "daily_stats" in monitoring_data.heart_rate
        assert "weekly_stats" in monitoring_data.heart_rate
        assert "monthly_stats" in monitoring_data.heart_rate

        # 验证latest数据格式
        latest = monitoring_data.heart_rate["latest"]
        assert isinstance(latest, list)
        assert len(latest) >= 1
        assert "value" in latest[0]
        assert "unit" in latest[0]
        assert "time" in latest[0]

        # 验证daily_stats数据格式
        daily_stats = monitoring_data.heart_rate["daily_stats"]
        assert isinstance(daily_stats, list)
        assert "date" in daily_stats[0]
        assert "avg" in daily_stats[0]

        # 验证weekly_stats数据格式
        weekly_stats = monitoring_data.heart_rate["weekly_stats"]
        assert isinstance(weekly_stats, list)
        assert "week" in weekly_stats[0]
        assert "avg" in weekly_stats[0]

        # 验证monthly_stats数据格式
        monthly_stats = monitoring_data.heart_rate["monthly_stats"]
        assert isinstance(monthly_stats, list)
        assert "month" in monthly_stats[0]
        assert "avg" in monthly_stats[0]

    def test_complete_blood_glucose_data_structure(self):
        """测试血糖数据结构包含完整的4个时间维度"""
        blood_glucose_data = TestDataFactory.create_complete_blood_glucose_data()
        monitoring_data = MonitoringData(blood_glucose=blood_glucose_data)

        assert monitoring_data.blood_glucose is not None
        assert "latest" in monitoring_data.blood_glucose
        assert "daily_stats" in monitoring_data.blood_glucose
        assert "weekly_stats" in monitoring_data.blood_glucose
        assert "monthly_stats" in monitoring_data.blood_glucose

        # 验证latest数据格式（血糖包含type字段）
        latest = monitoring_data.blood_glucose["latest"]
        assert isinstance(latest, list)
        assert "value" in latest[0]
        assert "unit" in latest[0]
        assert "type" in latest[0]  # fasting或postprandial

    def test_complete_blood_pressure_data_structure(self):
        """测试血压数据结构包含完整的4个时间维度"""
        blood_pressure_data = TestDataFactory.create_complete_blood_pressure_data()
        monitoring_data = MonitoringData(blood_pressure=blood_pressure_data)

        assert monitoring_data.blood_pressure is not None
        assert "latest" in monitoring_data.blood_pressure
        assert "daily_stats" in monitoring_data.blood_pressure
        assert "weekly_stats" in monitoring_data.blood_pressure
        assert "monthly_stats" in monitoring_data.blood_pressure

        # 验证latest数据格式（血压包含systolic和diastolic）
        latest = monitoring_data.blood_pressure["latest"]
        assert isinstance(latest, list)
        assert "systolic" in latest[0]
        assert "diastolic" in latest[0]
        assert "unit" in latest[0]

        # 验证daily_stats数据格式
        daily_stats = monitoring_data.blood_pressure["daily_stats"]
        assert "avg_systolic" in daily_stats[0]
        assert "avg_diastolic" in daily_stats[0]

    def test_all_six_monitoring_indicators(self):
        """测试包含全部6项监测指标"""
        monitoring_data = TestDataFactory.create_complete_monitoring_data()

        # 验证6项指标都存在
        assert monitoring_data.heart_rate is not None
        assert monitoring_data.blood_glucose is not None
        assert monitoring_data.perfusion_index is not None
        assert monitoring_data.blood_oxygen is not None
        assert monitoring_data.sleep is not None
        assert monitoring_data.blood_pressure is not None

    def test_monitoring_data_model_dump(self):
        """测试监测数据序列化为字典"""
        monitoring_data = TestDataFactory.create_complete_monitoring_data()
        data_dict = monitoring_data.model_dump()

        assert isinstance(data_dict, dict)
        assert "heart_rate" in data_dict
        assert "blood_glucose" in data_dict
        assert "perfusion_index" in data_dict
        assert "blood_oxygen" in data_dict
        assert "sleep" in data_dict
        assert "blood_pressure" in data_dict

        # 验证序列化后的数据结构完整
        assert "latest" in data_dict["heart_rate"]
        assert "daily_stats" in data_dict["heart_rate"]


# ==============================================================================
# 测试类2: 测试部分时间维度的监测数据
# ==============================================================================

class TestPartialTimeDimensionData:
    """
    测试部分时间维度的监测数据

    测试仅包含latest时间维度的数据
    测试仅包含daily_stats和weekly_stats时间维度的数据
    测试缺失部分时间维度时的处理
    """

    def test_only_latest_time_dimension(self):
        """测试仅包含latest时间维度的数据"""
        heart_rate_data = {
            "latest": [
                {"value": 72, "unit": "bpm", "time": "2024-01-15 08:00:00"}
            ]
        }

        monitoring_data = MonitoringData(heart_rate=heart_rate_data)

        assert monitoring_data.heart_rate is not None
        assert "latest" in monitoring_data.heart_rate
        # 其他时间维度不存在
        assert "daily_stats" not in monitoring_data.heart_rate
        assert "weekly_stats" not in monitoring_data.heart_rate
        assert "monthly_stats" not in monitoring_data.heart_rate

    def test_daily_and_weekly_stats_only(self):
        """测试仅包含daily_stats和weekly_stats时间维度的数据"""
        blood_pressure_data = {
            "daily_stats": [
                {"date": "2024-01-15", "avg_systolic": 118, "avg_diastolic": 79}
            ],
            "weekly_stats": [
                {"week": "2024-W2", "avg_systolic": 119, "avg_diastolic": 78, "trend": "stable"}
            ]
        }

        monitoring_data = MonitoringData(blood_pressure=blood_pressure_data)

        assert monitoring_data.blood_pressure is not None
        assert "daily_stats" in monitoring_data.blood_pressure
        assert "weekly_stats" in monitoring_data.blood_pressure
        # latest和monthly_stats不存在
        assert "latest" not in monitoring_data.blood_pressure
        assert "monthly_stats" not in monitoring_data.blood_pressure

    def test_missing_time_dimensions_handling(self):
        """测试缺失部分时间维度时的处理"""
        # 创建只有部分时间维度的数据
        blood_glucose_data = {
            "latest": [
                {"value": 5.5, "unit": "mmol/L", "type": "fasting", "time": "2024-01-15 07:00:00"}
            ],
            "monthly_stats": [
                {"month": "2024-01", "avg": 5.8, "trend": "stable"}
            ]
        }

        monitoring_data = MonitoringData(blood_glucose=blood_glucose_data)

        assert monitoring_data.blood_glucose is not None
        assert "latest" in monitoring_data.blood_glucose
        assert "monthly_stats" in monitoring_data.blood_glucose
        # daily_stats和weekly_stats缺失
        assert "daily_stats" not in monitoring_data.blood_glucose
        assert "weekly_stats" not in monitoring_data.blood_glucose

    def test_empty_time_dimension_list(self):
        """测试时间维度为空列表的情况"""
        heart_rate_data = {
            "latest": [],
            "daily_stats": [],
            "weekly_stats": [],
            "monthly_stats": []
        }

        monitoring_data = MonitoringData(heart_rate=heart_rate_data)

        assert monitoring_data.heart_rate is not None
        assert monitoring_data.heart_rate["latest"] == []
        assert monitoring_data.heart_rate["daily_stats"] == []

    def test_single_indicator_with_partial_dimensions(self):
        """测试单个指标包含部分时间维度"""
        monitoring_data = MonitoringData(
            heart_rate={
                "latest": [{"value": 72, "unit": "bpm", "time": "2024-01-15 08:00:00"}]
            },
            blood_pressure={
                "daily_stats": [{"date": "2024-01-15", "avg_systolic": 118, "avg_diastolic": 79}]
            }
        )

        # 心率只有latest
        assert monitoring_data.heart_rate is not None
        assert "latest" in monitoring_data.heart_rate

        # 血压只有daily_stats
        assert monitoring_data.blood_pressure is not None
        assert "daily_stats" in monitoring_data.blood_pressure


# ==============================================================================
# 测试类3: 测试字符串类型的病史字段
# ==============================================================================

class TestStringTypeHistoryFields:
    """
    测试字符串类型的病史字段

    测试past_medical_history, family_history, allergy_history, surgical_history字段为字符串类型
    测试空字符串和None值的处理
    测试包含多个病史信息的字符串
    """

    def test_all_history_fields_are_string_type(self):
        """测试所有病史字段都是字符串类型"""
        user_profile = TestDataFactory.create_complete_user_profile()

        assert isinstance(user_profile.past_medical_history, str)
        assert isinstance(user_profile.family_history, str)
        assert isinstance(user_profile.allergy_history, str)
        assert isinstance(user_profile.surgical_history, str)

    def test_empty_string_history_fields(self):
        """测试空字符串的病史字段"""
        user_profile = UserProfile(
            user_id=1,
            gender="male",
            birth_date="1955-03-15",
            past_medical_history="",
            family_history="",
            allergy_history="",
            surgical_history=""
        )

        assert user_profile.past_medical_history == ""
        assert user_profile.family_history == ""
        assert user_profile.allergy_history == ""
        assert user_profile.surgical_history == ""

    def test_none_history_fields(self):
        """测试None值的病史字段"""
        user_profile = UserProfile(
            user_id=1,
            gender="male",
            birth_date="1955-03-15",
            past_medical_history=None,
            family_history=None,
            allergy_history=None,
            surgical_history=None
        )

        assert user_profile.past_medical_history is None
        assert user_profile.family_history is None
        assert user_profile.allergy_history is None
        assert user_profile.surgical_history is None

    def test_multiple_history_items_in_string(self):
        """测试包含多个病史信息的字符串"""
        user_profile = UserProfile(
            user_id=1,
            past_medical_history="冠心病史5年、高血脂3年、2020年脑梗死、糖尿病前期",
            family_history="父亲有高血压、母亲有糖尿病、哥哥有冠心病",
            allergy_history="青霉素过敏、海鲜过敏、花粉过敏",
            surgical_history="2020年胆囊切除术、2018年髋关节置换术、2015年阑尾切除术"
        )

        # 验证字符串包含多个病史项
        assert "冠心病史" in user_profile.past_medical_history
        assert "高血脂" in user_profile.past_medical_history
        assert "脑梗死" in user_profile.past_medical_history

        assert "父亲" in user_profile.family_history
        assert "母亲" in user_profile.family_history

    def test_medical_compliance_string_field(self):
        """测试用药医嘱字符串字段"""
        user_profile = UserProfile(
            medical_compliance="好"
        )

        assert user_profile.medical_compliance == "好"

        # 测试不同的用药医嘱值
        user_profile_2 = UserProfile(medical_compliance="一般")
        assert user_profile_2.medical_compliance == "一般"

        user_profile_3 = UserProfile(medical_compliance="差")
        assert user_profile_3.medical_compliance == "差"

    def test_history_fields_model_dump(self):
        """测试病史字段序列化"""
        user_profile = TestDataFactory.create_complete_user_profile()
        data_dict = user_profile.model_dump()

        assert isinstance(data_dict["past_medical_history"], str)
        assert isinstance(data_dict["family_history"], str)
        assert isinstance(data_dict["allergy_history"], str)
        assert isinstance(data_dict["surgical_history"], str)


# ==============================================================================
# 测试类4: 测试空值处理逻辑
# ==============================================================================

class TestNullValueHandling:
    """
    测试空值处理逻辑

    测试监测数据为空的情况
    测试用户档案部分字段为空的情况
    测试所有字段都为空的情况
    """

    def test_empty_monitoring_data(self):
        """测试监测数据为空的情况"""
        monitoring_data = MonitoringData()

        assert monitoring_data.heart_rate is None
        assert monitoring_data.blood_glucose is None
        assert monitoring_data.perfusion_index is None
        assert monitoring_data.blood_oxygen is None
        assert monitoring_data.sleep is None
        assert monitoring_data.blood_pressure is None

    def test_partial_empty_user_profile(self):
        """测试用户档案部分字段为空的情况"""
        user_profile = UserProfile(
            user_id=1,
            gender="male",
            birth_date="1955-03-15",
            height=170.0,
            weight=75.0
            # 病史字段未设置，应为None
        )

        assert user_profile.user_id == 1
        assert user_profile.gender == "male"
        assert user_profile.birth_date == "1955-03-15"
        assert user_profile.past_medical_history is None
        assert user_profile.family_history is None
        assert user_profile.allergy_history is None
        assert user_profile.surgical_history is None
        assert user_profile.medical_compliance is None

    def test_all_fields_empty(self):
        """测试所有字段都为空的情况"""
        user_profile = UserProfile()
        monitoring_data = MonitoringData()

        # 用户档案所有字段为空
        assert user_profile.user_id is None
        assert user_profile.gender is None
        assert user_profile.birth_date is None
        assert user_profile.height is None
        assert user_profile.weight is None

        # 监测数据所有字段为空
        assert monitoring_data.heart_rate is None
        assert monitoring_data.blood_glucose is None
        assert monitoring_data.blood_pressure is None

    def test_empty_dict_vs_none(self):
        """测试空字典与None的区别"""
        # 空字典
        monitoring_data_1 = MonitoringData(heart_rate={})
        assert monitoring_data_1.heart_rate is not None
        assert monitoring_data_1.heart_rate == {}

        # None
        monitoring_data_2 = MonitoringData(heart_rate=None)
        assert monitoring_data_2.heart_rate is None

    def test_whitespace_only_history_fields(self):
        """测试仅包含空白字符的病史字段"""
        user_profile = UserProfile(
            past_medical_history="   ",
            family_history="\t\n",
            allergy_history=""
        )

        # 字段值应为空白字符串，不是None
        assert user_profile.past_medical_history == "   "
        assert user_profile.family_history == "\t\n"
        assert user_profile.allergy_history == ""


# ==============================================================================
# 测试类5: 测试端到端的报告生成流程
# ==============================================================================

class TestEndToEndReportGeneration:
    """
    测试端到端的报告生成流程

    测试从ReportRequest到ReportController的完整流程
    测试数据验证逻辑
    测试数据标准化逻辑
    """

    def test_complete_report_request_creation(self):
        """测试完整的报告请求创建"""
        request = TestDataFactory.create_complete_report_request()

        assert request.request_id == "req-test-001"
        assert request.user_id == "user-001"
        assert request.body.task_id == "task-001"
        assert request.body.session_id == "session-001"

        # 验证监测数据
        assert request.body.monitoring_data.heart_rate is not None
        assert request.body.monitoring_data.blood_glucose is not None

        # 验证用户档案
        assert request.body.user_profile.user_id == 1
        assert request.body.user_profile.gender == "male"

    def test_report_request_getter_methods(self):
        """测试ReportRequest的getter方法"""
        request = TestDataFactory.create_complete_report_request()

        assert request.get_task_id() == "task-001"
        assert request.get_session_id() == "session-001"
        assert request.get_monitoring_data() is not None
        assert request.get_user_profile() is not None

    def test_report_request_has_indicator_methods(self):
        """测试ReportRequest的指标判断方法"""
        request = TestDataFactory.create_complete_report_request()

        assert request.has_heart_rate() is True
        assert request.has_blood_glucose() is True
        assert request.has_perfusion_index() is True
        assert request.has_blood_oxygen() is True
        assert request.has_sleep() is True
        assert request.has_blood_pressure() is True

    def test_report_request_has_history_methods(self):
        """测试ReportRequest的病史判断方法"""
        request = TestDataFactory.create_complete_report_request()

        assert request.has_past_medical_history() is True
        assert request.has_family_history() is True
        assert request.has_allergy_history() is True
        assert request.has_surgical_history() is True
        assert request.has_medical_compliance() is True

    def test_report_request_missing_indicators(self):
        """测试缺少部分指标的请求"""
        monitoring_data = MonitoringData(
            heart_rate={"latest": [{"value": 72, "unit": "bpm", "time": "2024-01-15 08:00:00"}]}
        )
        user_profile = UserProfile(user_id=1, gender="male")

        request = ReportRequest(
            request_id="req-test-002",
            body=ReportRequestBody(
                task_id="task-002",
                monitoring_data=monitoring_data,
                user_profile=user_profile
            )
        )

        assert request.has_heart_rate() is True
        assert request.has_blood_glucose() is False
        assert request.has_blood_pressure() is False

    def test_report_request_missing_history(self):
        """测试缺少病史信息的请求"""
        user_profile = UserProfile(
            user_id=1,
            gender="male",
            birth_date="1955-03-15"
            # 没有病史字段
        )

        request = ReportRequest(
            request_id="req-test-003",
            body=ReportRequestBody(
                task_id="task-003",
                monitoring_data=MonitoringData(heart_rate={"latest": [{"value": 72}]}),
                user_profile=user_profile
            )
        )

        assert request.has_past_medical_history() is False
        assert request.has_family_history() is False
        assert request.has_allergy_history() is False

    def test_report_request_validation_success(self):
        """测试请求验证成功的情况"""
        request = TestDataFactory.create_complete_report_request()

        # 验证请求基本有效性
        assert request.validate_request() is True

    def test_report_request_model_dump(self):
        """测试请求序列化为字典"""
        request = TestDataFactory.create_complete_report_request()
        data_dict = request.model_dump()

        assert "request_id" in data_dict
        assert "body" in data_dict
        assert "monitoring_data" in data_dict["body"]
        assert "user_profile" in data_dict["body"]

    def test_report_request_json_serialization(self):
        """测试请求JSON序列化"""
        request = TestDataFactory.create_complete_report_request()
        json_str = request.model_dump_json()

        assert isinstance(json_str, str)
        assert "task-001" in json_str
        assert "heart_rate" in json_str

    def test_report_controller_validation_with_valid_request(self):
        """测试ReportController验证有效请求"""
        from src.controller.report_controller import ReportController
        from fastapi import HTTPException

        # 创建Mock服务
        mock_service = Mock()
        controller = ReportController(mock_service)

        request = TestDataFactory.create_complete_report_request()

        # 验证不应抛出异常
        try:
            controller._validate_request(request)
            validation_passed = True
        except HTTPException:
            validation_passed = False

        assert validation_passed is True

    def test_report_controller_validation_with_missing_body(self):
        """测试ReportController验证缺少body的请求（Pydantic验证）"""
        from src.controller.report_controller import ReportController
        from pydantic import ValidationError

        mock_service = Mock()
        controller = ReportController(mock_service)

        # Pydantic不允许body为None，会抛出ValidationError
        with pytest.raises(ValidationError) as exc_info:
            ReportRequest(
                request_id="req-test",
                body=None  # type: ignore
            )

        # 验证错误信息包含body字段
        assert "body" in str(exc_info.value)

    def test_report_controller_validation_with_missing_task_id(self):
        """测试ReportController验证缺少task_id的请求"""
        from src.controller.report_controller import ReportController
        from fastapi import HTTPException

        mock_service = Mock()
        controller = ReportController(mock_service)

        request = ReportRequest(
            request_id="req-test",
            body=ReportRequestBody(
                task_id="",  # 空task_id
                monitoring_data=MonitoringData(heart_rate={"latest": [{"value": 72}]}),
                user_profile=UserProfile(user_id=1)
            )
        )

        with pytest.raises(HTTPException) as exc_info:
            controller._validate_request(request)

        assert exc_info.value.status_code == 400

    def test_report_controller_validation_with_missing_monitoring_data(self):
        """测试ReportController验证缺少监测数据的请求"""
        from src.controller.report_controller import ReportController
        from fastapi import HTTPException

        mock_service = Mock()
        controller = ReportController(mock_service)

        request = ReportRequest(
            request_id="req-test",
            body=ReportRequestBody(
                task_id="task-001",
                monitoring_data=MonitoringData(),  # 所有指标都为None
                user_profile=UserProfile(user_id=1)
            )
        )

        with pytest.raises(HTTPException) as exc_info:
            controller._validate_request(request)

        assert exc_info.value.status_code == 400
        assert "至少需要包含一项监测指标" in str(exc_info.value.detail)

    def test_report_service_build_agent_context(self):
        """测试ReportService构建AgentContext"""
        from src.service.report_service import ReportService

        # 创建Mock Agent
        mock_agent = Mock()
        mock_agent.resources = None

        service = ReportService(mock_agent)
        request = TestDataFactory.create_complete_report_request()

        context = service._build_agent_context(request)

        assert context.session_id == "session-001"
        assert context.body is not None
        assert context.body.task_id == "task-001"

    def test_report_service_build_agent_context_without_session_id(self):
        """测试ReportService在没有session_id时使用task_id"""
        from src.service.report_service import ReportService

        mock_agent = Mock()
        mock_agent.resources = None

        service = ReportService(mock_agent)

        # 创建没有session_id的请求
        request = ReportRequest(
            request_id="req-test",
            body=ReportRequestBody(
                task_id="task-no-session",
                monitoring_data=TestDataFactory.create_complete_monitoring_data(),
                user_profile=TestDataFactory.create_complete_user_profile()
                # 没有session_id
            )
        )

        context = service._build_agent_context(request)

        # session_id应该使用task_id
        assert context.session_id == "task-no-session"


# ==============================================================================
# 测试类6: 测试数据结构边界情况
# ==============================================================================

class TestDataStructureEdgeCases:
    """
    测试数据结构边界情况

    测试极端值、特殊字符、边界条件等
    """

    def test_extreme_values_for_vital_signs(self):
        """测试生命体征的极端值"""
        # 极高心率
        monitoring_data = MonitoringData(
            heart_rate={
                "latest": [{"value": 200, "unit": "bpm", "time": "2024-01-15 08:00:00"}]
            }
        )
        assert monitoring_data.heart_rate["latest"][0]["value"] == 200

        # 极低血氧
        monitoring_data = MonitoringData(
            blood_oxygen={
                "latest": [{"value": 70.0, "unit": "%", "time": "2024-01-15 08:00:00"}]
            }
        )
        assert monitoring_data.blood_oxygen["latest"][0]["value"] == 70.0

    def test_special_characters_in_history_fields(self):
        """测试病史字段中的特殊字符"""
        user_profile = UserProfile(
            past_medical_history="糖尿病（2型）、高血压[III期]、冠心病{不稳定型}",
            family_history="父亲：高血压、母亲：糖尿病",
            allergy_history="青霉素(过敏)、磺胺类[禁用]"
        )

        assert "（2型）" in user_profile.past_medical_history
        assert "：" in user_profile.family_history

    def test_very_long_history_string(self):
        """测试非常长的病史字符串"""
        long_history = "、".join([f"疾病{i}" for i in range(100)])

        user_profile = UserProfile(
            past_medical_history=long_history
        )

        # 100个疾病用顿号连接，每个疾病约5字符，总长度约489字符
        assert len(user_profile.past_medical_history) > 400
        assert "疾病0" in user_profile.past_medical_history
        assert "疾病99" in user_profile.past_medical_history

    def test_unicode_characters_in_data(self):
        """测试数据中的Unicode字符"""
        monitoring_data = MonitoringData(
            heart_rate={
                "latest": [{"value": 72, "unit": "次/分", "time": "2024-01-15 08:00:00"}]
            }
        )

        assert "次/分" in str(monitoring_data.heart_rate)

    def test_negative_values_handling(self):
        """测试负值处理"""
        # 理论上生命体征不应为负值，但测试数据结构是否能接受
        monitoring_data = MonitoringData(
            perfusion_index={
                "latest": [{"value": -1.0, "unit": "PI", "time": "2024-01-15 08:00:00"}]
            }
        )

        # Pydantic应该接受任何值（除非有验证器）
        assert monitoring_data.perfusion_index["latest"][0]["value"] == -1.0

    def test_float_precision_in_values(self):
        """测试浮点数精度"""
        monitoring_data = MonitoringData(
            blood_glucose={
                "latest": [{"value": 5.555555555, "unit": "mmol/L", "time": "2024-01-15 08:00:00"}]
            }
        )

        # 验证浮点数精度保持
        assert monitoring_data.blood_glucose["latest"][0]["value"] == 5.555555555


# ==============================================================================
# 测试类7: 测试数据结构向后兼容性
# ==============================================================================

class TestDataBackwardCompatibility:
    """
    测试数据结构向后兼容性

    确保新数据结构能够处理旧格式的数据
    """

    def test_minimal_valid_request(self):
        """测试最小有效请求"""
        request = ReportRequest(
            body=ReportRequestBody(
                task_id="minimal-task",
                monitoring_data=MonitoringData(heart_rate={"latest": [{"value": 72}]}),
                user_profile=UserProfile(user_id=1)
            )
        )

        assert request.body.task_id == "minimal-task"
        assert request.body.monitoring_data.heart_rate is not None

    def test_request_without_optional_fields(self):
        """测试没有可选字段的请求"""
        request = ReportRequest(
            body=ReportRequestBody(
                task_id="task-no-optional",
                monitoring_data=MonitoringData(heart_rate={"latest": [{"value": 72}]}),
                user_profile=UserProfile()
            )
        )

        # 可选字段应为None
        assert request.body.user_profile.gender is None
        assert request.body.user_profile.birth_date is None
        assert request.body.session_id is None

    def test_user_profile_with_partial_fields(self):
        """测试只有部分字段的用户档案"""
        user_profile = UserProfile(
            user_id=1,
            gender="male"
            # 其他字段未设置
        )

        assert user_profile.user_id == 1
        assert user_profile.gender == "male"
        assert user_profile.birth_date is None
        assert user_profile.height is None


# ==============================================================================
# 运行测试
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
