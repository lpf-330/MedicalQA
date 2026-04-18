"""
健康报告生成请求数据类模块

该模块定义了健康报告生成API请求的数据结构。
数据结构与SpringBoot后端数据库表结构对齐，符合《项目需求设计v1.1》要求。
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from .base_request import BaseRequest


class MonitoringData(BaseModel):
    """
    监测数据类

    包含用户的各项健康监测指标数据，支持6项监测指标，每项指标包含4个时间维度的数据。

    Attributes:
        heart_rate (Optional[Dict[str, Any]]): 心率数据，包含latest、daily_stats、weekly_stats、monthly_stats四个时间维度
        blood_glucose (Optional[Dict[str, Any]]): 血糖数据，包含latest、daily_stats、weekly_stats、monthly_stats四个时间维度
        perfusion_index (Optional[Dict[str, Any]]): 灌注指数数据，包含latest、daily_stats、weekly_stats、monthly_stats四个时间维度
        blood_oxygen (Optional[Dict[str, Any]]): 血氧数据，包含latest、daily_stats、weekly_stats、monthly_stats四个时间维度
        sleep (Optional[Dict[str, Any]]): 睡眠数据，包含latest、daily_stats、weekly_stats、monthly_stats四个时间维度
        blood_pressure (Optional[Dict[str, Any]]): 血压数据，包含latest、daily_stats、weekly_stats、monthly_stats四个时间维度

    时间维度说明:
        - latest: 当日最新3-5次数据（List[Dict]），用于识别当前异常状态、判断即时风险
        - daily_stats: 最近30天日统计数据（List[Dict]），用于分析日内变异、发现周期性异常
        - weekly_stats: 最近12周周统计数据（List[Dict]），用于判断改善/恶化方向、评估干预效果
        - monthly_stats: 最近6个月月统计数据（List[Dict]），用于评估慢性病风险、计算长期平均值

    Example:
        >>> monitoring_data = MonitoringData(
        ...     heart_rate={
        ...         "latest": [{"value": 72, "unit": "bpm", "time": "2024-01-01 08:00:00"}],
        ...         "daily_stats": [{"date": "2024-01-01", "avg": 70, "max": 85, "min": 62}],
        ...         "weekly_stats": [{"week": "2024-W1", "avg": 71, "trend": "stable"}],
        ...         "monthly_stats": [{"month": "2024-01", "avg": 72, "trend": "stable"}]
        ...     },
        ...     blood_pressure={
        ...         "latest": [{"systolic": 120, "diastolic": 80, "unit": "mmHg", "time": "2024-01-01 08:00:00"}],
        ...         "daily_stats": [{"date": "2024-01-01", "avg_systolic": 118, "avg_diastolic": 79}],
        ...         "weekly_stats": [{"week": "2024-W1", "avg_systolic": 119, "avg_diastolic": 78, "trend": "stable"}],
        ...         "monthly_stats": [{"month": "2024-01", "avg_systolic": 120, "avg_diastolic": 80, "trend": "rising"}]
        ...     }
        ... )
    """

    heart_rate: Optional[Dict[str, Any]] = Field(
        default=None,
        description="心率数据，包含latest（当日最新3-5次）、daily_stats（最近30天日统计）、weekly_stats（最近12周周统计）、monthly_stats（最近6个月月统计）四个时间维度",
        examples=[{
            "latest": [{"value": 72, "unit": "bpm", "time": "2024-01-01 08:00:00"}],
            "daily_stats": [{"date": "2024-01-01", "avg": 70, "max": 85, "min": 62}],
            "weekly_stats": [{"week": "2024-W1", "avg": 71, "trend": "stable"}],
            "monthly_stats": [{"month": "2024-01", "avg": 72, "trend": "stable"}]
        }]
    )

    blood_glucose: Optional[Dict[str, Any]] = Field(
        default=None,
        description="血糖数据，包含latest（当日最新3-5次）、daily_stats（最近30天日统计）、weekly_stats（最近12周周统计）、monthly_stats（最近6个月月统计）四个时间维度",
        examples=[{
            "latest": [{"value": 5.5, "unit": "mmol/L", "type": "fasting", "time": "2024-01-01 08:00:00"}],
            "daily_stats": [{"date": "2024-01-01", "avg": 5.8, "max": 7.2, "min": 5.0}],
            "weekly_stats": [{"week": "2024-W1", "avg": 5.6, "trend": "stable"}],
            "monthly_stats": [{"month": "2024-01", "avg": 5.7, "trend": "rising"}]
        }]
    )

    perfusion_index: Optional[Dict[str, Any]] = Field(
        default=None,
        description="灌注指数数据，包含latest（当日最新3-5次）、daily_stats（最近30天日统计）、weekly_stats（最近12周周统计）、monthly_stats（最近6个月月统计）四个时间维度",
        examples=[{
            "latest": [{"value": 3.5, "unit": "PI", "time": "2024-01-01 08:00:00"}],
            "daily_stats": [{"date": "2024-01-01", "avg": 3.2, "max": 4.0, "min": 2.8}],
            "weekly_stats": [{"week": "2024-W1", "avg": 3.3, "trend": "stable"}],
            "monthly_stats": [{"month": "2024-01", "avg": 3.4, "trend": "stable"}]
        }]
    )

    blood_oxygen: Optional[Dict[str, Any]] = Field(
        default=None,
        description="血氧数据，包含latest（当日最新3-5次）、daily_stats（最近30天日统计）、weekly_stats（最近12周周统计）、monthly_stats（最近6个月月统计）四个时间维度",
        examples=[{
            "latest": [{"value": 98.5, "unit": "%", "time": "2024-01-01 08:00:00"}],
            "daily_stats": [{"date": "2024-01-01", "avg": 98.0, "max": 99.0, "min": 97.0}],
            "weekly_stats": [{"week": "2024-W1", "avg": 98.2, "trend": "stable"}],
            "monthly_stats": [{"month": "2024-01", "avg": 98.1, "trend": "stable"}]
        }]
    )

    sleep: Optional[Dict[str, Any]] = Field(
        default=None,
        description="睡眠数据，包含latest（当日最新3-5次）、daily_stats（最近30天日统计）、weekly_stats（最近12周周统计）、monthly_stats（最近6个月月统计）四个时间维度",
        examples=[{
            "latest": [{"value": 7.5, "unit": "hours", "time": "2024-01-01 08:00:00"}],
            "daily_stats": [{"date": "2024-01-01", "avg": 7.2, "max": 8.0, "min": 6.5}],
            "weekly_stats": [{"week": "2024-W1", "avg": 7.3, "trend": "stable"}],
            "monthly_stats": [{"month": "2024-01", "avg": 7.4, "trend": "stable"}]
        }]
    )

    blood_pressure: Optional[Dict[str, Any]] = Field(
        default=None,
        description="血压数据，包含latest（当日最新3-5次）、daily_stats（最近30天日统计）、weekly_stats（最近12周周统计）、monthly_stats（最近6个月月统计）四个时间维度",
        examples=[{
            "latest": [{"systolic": 120, "diastolic": 80, "unit": "mmHg", "time": "2024-01-01 08:00:00"}],
            "daily_stats": [{"date": "2024-01-01", "avg_systolic": 118, "avg_diastolic": 79}],
            "weekly_stats": [{"week": "2024-W1", "avg_systolic": 119, "avg_diastolic": 78, "trend": "stable"}],
            "monthly_stats": [{"month": "2024-01", "avg_systolic": 120, "avg_diastolic": 80, "trend": "rising"}]
        }]
    )


class UserProfile(BaseModel):
    """
    用户档案数据类

    包含用户的基本信息和健康档案，字段与SpringBoot后端users表对齐。
    所有病史字段均为字符串文本类型（str），便于自然语言处理。

    Attributes:
        user_id (Optional[int]): 用户ID，对应SpringBoot后端users表的id字段
        gender (Optional[str]): 性别，对应SpringBoot后端users表的gender字段，值为"male"、"female"或"other"
        birth_date (Optional[str]): 出生日期，对应SpringBoot后端users表的birth_date字段，格式为YYYY-MM-DD
        height (Optional[float]): 身高(cm)，对应SpringBoot后端users表的height字段
        weight (Optional[float]): 体重(kg)，对应SpringBoot后端users表的weight字段
        past_medical_history (Optional[str]): 既往病史，对应SpringBoot后端users表的past_medical_history字段，字符串文本类型
        family_history (Optional[str]): 家族遗传病史，对应SpringBoot后端users表的family_history字段，字符串文本类型
        allergy_history (Optional[str]): 过敏史，对应SpringBoot后端users表的allergy_history字段，字符串文本类型
        surgical_history (Optional[str]): 手术史，对应SpringBoot后端users表的surgical_history字段，字符串文本类型
        medical_compliance (Optional[str]): 用药医嘱，对应SpringBoot后端users表的medical_compliance字段，字符串文本类型

    Example:
        >>> user_profile = UserProfile(
        ...     user_id=1,
        ...     gender="male",
        ...     birth_date="1955-03-15",
        ...     height=170.0,
        ...     weight=75.0,
        ...     past_medical_history="冠心病史5年、高血脂3年、2020年脑梗死",
        ...     family_history="父亲有高血压、母亲有糖尿病",
        ...     allergy_history="青霉素过敏、海鲜过敏",
        ...     surgical_history="2020年胆囊切除术、2018年髋关节置换术",
        ...     medical_compliance="好"
        ... )
    """

    user_id: Optional[int] = Field(
        default=None,
        description="用户ID，对应SpringBoot后端users表的id字段",
        examples=[1, 2, 3]
    )

    gender: Optional[str] = Field(
        default=None,
        description="性别，对应SpringBoot后端users表的gender字段，值为male、female或other",
        examples=["male", "female", "other"]
    )

    birth_date: Optional[str] = Field(
        default=None,
        description="出生日期，对应SpringBoot后端users表的birth_date字段，格式为YYYY-MM-DD",
        examples=["1955-03-15", "1960-08-20"]
    )

    height: Optional[float] = Field(
        default=None,
        description="身高(cm)，对应SpringBoot后端users表的height字段",
        examples=[170.0, 175.0, 160.0]
    )

    weight: Optional[float] = Field(
        default=None,
        description="体重(kg)，对应SpringBoot后端users表的weight字段",
        examples=[75.0, 65.0, 80.0]
    )

    past_medical_history: Optional[str] = Field(
        default=None,
        description="既往病史，对应SpringBoot后端users表的past_medical_history字段，字符串文本类型",
        examples=["冠心病史5年、高血脂3年、2020年脑梗死"]
    )

    family_history: Optional[str] = Field(
        default=None,
        description="家族遗传病史，对应SpringBoot后端users表的family_history字段，字符串文本类型",
        examples=["父亲有高血压、母亲有糖尿病"]
    )

    allergy_history: Optional[str] = Field(
        default=None,
        description="过敏史，对应SpringBoot后端users表的allergy_history字段，字符串文本类型",
        examples=["青霉素过敏、海鲜过敏"]
    )

    surgical_history: Optional[str] = Field(
        default=None,
        description="手术史，对应SpringBoot后端users表的surgical_history字段，字符串文本类型",
        examples=["2020年胆囊切除术、2018年髋关节置换术"]
    )

    medical_compliance: Optional[str] = Field(
        default=None,
        description="用药医嘱，对应SpringBoot后端users表的medical_compliance字段，字符串文本类型",
        examples=["好", "一般", "差"]
    )


class ReportRequestBody(BaseModel):
    """
    健康报告生成请求体数据类

    包含健康报告生成特有的请求数据结构。

    Attributes:
        task_id (str): 任务标识符
        monitoring_data (MonitoringData): 监测数据，包含各项健康指标
        user_profile (UserProfile): 用户档案，包含基本信息、病史等
        session_id (Optional[str]): 会话ID，用于多轮对话的会话标识

    Example:
        >>> body = ReportRequestBody(
        ...     task_id="task-001",
        ...     monitoring_data=MonitoringData(heart_rate={"latest": [{"value": 72, "unit": "bpm"}]}),
        ...     user_profile=UserProfile(user_id=1, gender="male", birth_date="1955-03-15"),
        ...     session_id="session-001"
        ... )
    """

    task_id: str = Field(
        ...,
        description="任务标识符",
        examples=["task-001", "task-abc123"]
    )

    monitoring_data: MonitoringData = Field(
        ...,
        description="监测数据，包含各项健康指标"
    )

    user_profile: UserProfile = Field(
        ...,
        description="用户档案，包含基本信息、病史等"
    )

    session_id: Optional[str] = Field(
        default=None,
        description="会话ID，用于多轮对话的会话标识"
    )


class ReportRequest(BaseRequest[ReportRequestBody]):
    """
    健康报告生成请求数据类

    继承BaseRequest，包含健康报告生成特有的属性。
    用于接收和处理健康报告生成相关的API请求。

    Attributes:
        继承自BaseRequest:
            - request_id (str): 请求唯一标识符
            - timestamp (str): 请求时间戳
            - body (ReportRequestBody): 健康报告生成请求体
            - user_id (Optional[str]): 用户ID
            - client_info (Optional[dict]): 客户端信息

    Example:
        >>> request = ReportRequest(
        ...     request_id="req-123456",
        ...     user_id="user-001",
        ...     body=ReportRequestBody(
        ...         task_id="task-001",
        ...         monitoring_data=MonitoringData(heart_rate={"latest": [{"value": 72}]}),
        ...         user_profile=UserProfile(user_id=1, gender="male")
        ...     )
        ... )
        >>> request.body.task_id
        'task-001'
    """

    body: ReportRequestBody = Field(
        ...,
        description="健康报告生成请求体"
    )

    class Config:
        """Pydantic配置类"""
        json_schema_extra = {
            "example": {
                "request_id": "req-123456789abc",
                "timestamp": "2024-01-01T12:00:00",
                "user_id": "user-001",
                "client_info": {
                    "client_type": "web",
                    "version": "1.0.0"
                },
                "body": {
                    "task_id": "task-001",
                    "monitoring_data": {
                        "heart_rate": {
                            "latest": [{"value": 72, "unit": "bpm", "time": "2024-01-01 08:00:00"}],
                            "daily_stats": [{"date": "2024-01-01", "avg": 70, "max": 85, "min": 62}],
                            "weekly_stats": [{"week": "2024-W1", "avg": 71, "trend": "stable"}],
                            "monthly_stats": [{"month": "2024-01", "avg": 72, "trend": "stable"}]
                        },
                        "blood_glucose": {
                            "latest": [{"value": 5.5, "unit": "mmol/L", "type": "fasting", "time": "2024-01-01 08:00:00"}],
                            "daily_stats": [{"date": "2024-01-01", "avg": 5.8, "max": 7.2, "min": 5.0}],
                            "weekly_stats": [{"week": "2024-W1", "avg": 5.6, "trend": "stable"}],
                            "monthly_stats": [{"month": "2024-01", "avg": 5.7, "trend": "rising"}]
                        },
                        "blood_oxygen": {
                            "latest": [{"value": 98.5, "unit": "%", "time": "2024-01-01 08:00:00"}],
                            "daily_stats": [{"date": "2024-01-01", "avg": 98.0, "max": 99.0, "min": 97.0}],
                            "weekly_stats": [{"week": "2024-W1", "avg": 98.2, "trend": "stable"}],
                            "monthly_stats": [{"month": "2024-01", "avg": 98.1, "trend": "stable"}]
                        },
                        "sleep": {
                            "latest": [{"value": 7.5, "unit": "hours", "time": "2024-01-01 08:00:00"}],
                            "daily_stats": [{"date": "2024-01-01", "avg": 7.2, "max": 8.0, "min": 6.5}],
                            "weekly_stats": [{"week": "2024-W1", "avg": 7.3, "trend": "stable"}],
                            "monthly_stats": [{"month": "2024-01", "avg": 7.4, "trend": "stable"}]
                        },
                        "blood_pressure": {
                            "latest": [{"systolic": 120, "diastolic": 80, "unit": "mmHg", "time": "2024-01-01 08:00:00"}],
                            "daily_stats": [{"date": "2024-01-01", "avg_systolic": 118, "avg_diastolic": 79}],
                            "weekly_stats": [{"week": "2024-W1", "avg_systolic": 119, "avg_diastolic": 78, "trend": "stable"}],
                            "monthly_stats": [{"month": "2024-01", "avg_systolic": 120, "avg_diastolic": 80, "trend": "rising"}]
                        },
                        "perfusion_index": {
                            "latest": [{"value": 3.5, "unit": "PI", "time": "2024-01-01 08:00:00"}],
                            "daily_stats": [{"date": "2024-01-01", "avg": 3.2, "max": 4.0, "min": 2.8}],
                            "weekly_stats": [{"week": "2024-W1", "avg": 3.3, "trend": "stable"}],
                            "monthly_stats": [{"month": "2024-01", "avg": 3.4, "trend": "stable"}]
                        }
                    },
                    "user_profile": {
                        "user_id": 1,
                        "gender": "male",
                        "birth_date": "1955-03-15",
                        "height": 170.0,
                        "weight": 75.0,
                        "past_medical_history": "冠心病史5年、高血脂3年、2020年脑梗死",
                        "family_history": "父亲有高血压、母亲有糖尿病",
                        "allergy_history": "青霉素过敏、海鲜过敏",
                        "surgical_history": "2020年胆囊切除术",
                        "medical_compliance": "好"
                    },
                    "session_id": "session-001"
                }
            }
        }

    def get_task_id(self) -> str:
        """
        获取任务ID

        Returns:
            str: 任务标识符

        Example:
            >>> request = ReportRequest(body=ReportRequestBody(task_id="task-001"))
            >>> request.get_task_id()
            'task-001'
        """
        return self.body.task_id

    def get_monitoring_data(self) -> MonitoringData:
        """
        获取监测数据

        Returns:
            MonitoringData: 监测数据对象

        Example:
            >>> monitoring_data = MonitoringData(heart_rate={"latest": [{"value": 72}]})
            >>> request = ReportRequest(body=ReportRequestBody(task_id="task-001", monitoring_data=monitoring_data))
            >>> request.get_monitoring_data()
            MonitoringData(heart_rate={'latest': [{'value': 72}]}, blood_glucose=None, ...)
        """
        return self.body.monitoring_data

    def get_user_profile(self) -> UserProfile:
        """
        获取用户档案

        Returns:
            UserProfile: 用户档案对象

        Example:
            >>> user_profile = UserProfile(user_id=1, gender="male")
            >>> request = ReportRequest(body=ReportRequestBody(task_id="task-001", user_profile=user_profile))
            >>> request.get_user_profile()
            UserProfile(user_id=1, gender='male', birth_date=None, ...)
        """
        return self.body.user_profile

    def get_session_id(self) -> Optional[str]:
        """
        获取会话ID

        Returns:
            Optional[str]: 会话ID，如果未设置则返回None

        Example:
            >>> request = ReportRequest(body=ReportRequestBody(task_id="task-001", session_id="session-001"))
            >>> request.get_session_id()
            'session-001'
        """
        return self.body.session_id

    def has_blood_pressure(self) -> bool:
        """
        判断是否有血压数据

        Returns:
            bool: 如果有血压数据返回True，否则返回False

        Example:
            >>> monitoring_data = MonitoringData(blood_pressure={"latest": [{"systolic": 120, "diastolic": 80}]})
            >>> request = ReportRequest(body=ReportRequestBody(task_id="task-001", monitoring_data=monitoring_data))
            >>> request.has_blood_pressure()
            True
        """
        return self.body.monitoring_data.blood_pressure is not None

    def has_blood_glucose(self) -> bool:
        """
        判断是否有血糖数据

        Returns:
            bool: 如果有血糖数据返回True，否则返回False

        Example:
            >>> monitoring_data = MonitoringData(blood_glucose={"latest": [{"value": 5.5}]})
            >>> request = ReportRequest(body=ReportRequestBody(task_id="task-001", monitoring_data=monitoring_data))
            >>> request.has_blood_glucose()
            True
        """
        return self.body.monitoring_data.blood_glucose is not None

    def has_heart_rate(self) -> bool:
        """
        判断是否有心率数据

        Returns:
            bool: 如果有心率数据返回True，否则返回False

        Example:
            >>> monitoring_data = MonitoringData(heart_rate={"latest": [{"value": 72}]})
            >>> request = ReportRequest(body=ReportRequestBody(task_id="task-001", monitoring_data=monitoring_data))
            >>> request.has_heart_rate()
            True
        """
        return self.body.monitoring_data.heart_rate is not None

    def has_blood_oxygen(self) -> bool:
        """
        判断是否有血氧数据

        Returns:
            bool: 如果有血氧数据返回True，否则返回False

        Example:
            >>> monitoring_data = MonitoringData(blood_oxygen={"latest": [{"value": 98.5}]})
            >>> request = ReportRequest(body=ReportRequestBody(task_id="task-001", monitoring_data=monitoring_data))
            >>> request.has_blood_oxygen()
            True
        """
        return self.body.monitoring_data.blood_oxygen is not None

    def has_sleep(self) -> bool:
        """
        判断是否有睡眠数据

        Returns:
            bool: 如果有睡眠数据返回True，否则返回False

        Example:
            >>> monitoring_data = MonitoringData(sleep={"latest": [{"value": 7.5}]})
            >>> request = ReportRequest(body=ReportRequestBody(task_id="task-001", monitoring_data=monitoring_data))
            >>> request.has_sleep()
            True
        """
        return self.body.monitoring_data.sleep is not None

    def has_perfusion_index(self) -> bool:
        """
        判断是否有灌注指数数据

        Returns:
            bool: 如果有灌注指数数据返回True，否则返回False

        Example:
            >>> monitoring_data = MonitoringData(perfusion_index={"latest": [{"value": 3.5}]})
            >>> request = ReportRequest(body=ReportRequestBody(task_id="task-001", monitoring_data=monitoring_data))
            >>> request.has_perfusion_index()
            True
        """
        return self.body.monitoring_data.perfusion_index is not None

    def has_past_medical_history(self) -> bool:
        """
        判断是否有既往病史

        Returns:
            bool: 如果有既往病史返回True，否则返回False

        Example:
            >>> user_profile = UserProfile(past_medical_history="冠心病史5年、高血脂3年")
            >>> request = ReportRequest(body=ReportRequestBody(task_id="task-001", user_profile=user_profile))
            >>> request.has_past_medical_history()
            True
        """
        return self.body.user_profile.past_medical_history is not None and len(self.body.user_profile.past_medical_history.strip()) > 0

    def has_family_history(self) -> bool:
        """
        判断是否有家族病史

        Returns:
            bool: 如果有家族病史返回True，否则返回False

        Example:
            >>> user_profile = UserProfile(family_history="父亲有高血压、母亲有糖尿病")
            >>> request = ReportRequest(body=ReportRequestBody(task_id="task-001", user_profile=user_profile))
            >>> request.has_family_history()
            True
        """
        return self.body.user_profile.family_history is not None and len(self.body.user_profile.family_history.strip()) > 0

    def has_allergy_history(self) -> bool:
        """
        判断是否有过敏史

        Returns:
            bool: 如果有过敏史返回True，否则返回False

        Example:
            >>> user_profile = UserProfile(allergy_history="青霉素过敏、海鲜过敏")
            >>> request = ReportRequest(body=ReportRequestBody(task_id="task-001", user_profile=user_profile))
            >>> request.has_allergy_history()
            True
        """
        return self.body.user_profile.allergy_history is not None and len(self.body.user_profile.allergy_history.strip()) > 0

    def has_surgical_history(self) -> bool:
        """
        判断是否有手术史

        Returns:
            bool: 如果有手术史返回True，否则返回False

        Example:
            >>> user_profile = UserProfile(surgical_history="2020年胆囊切除术")
            >>> request = ReportRequest(body=ReportRequestBody(task_id="task-001", user_profile=user_profile))
            >>> request.has_surgical_history()
            True
        """
        return self.body.user_profile.surgical_history is not None and len(self.body.user_profile.surgical_history.strip()) > 0

    def has_medical_compliance(self) -> bool:
        """
        判断是否有用药医嘱

        Returns:
            bool: 如果有用药医嘱返回True，否则返回False

        Example:
            >>> user_profile = UserProfile(medical_compliance="好")
            >>> request = ReportRequest(body=ReportRequestBody(task_id="task-001", user_profile=user_profile))
            >>> request.has_medical_compliance()
            True
        """
        return self.body.user_profile.medical_compliance is not None and len(self.body.user_profile.medical_compliance.strip()) > 0
