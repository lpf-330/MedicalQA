"""
健康报告生成请求数据类模块

该模块定义了健康报告生成API请求的数据结构。
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from .base_request import BaseRequest


class MonitoringData(BaseModel):
    """
    监测数据类

    包含用户的各项健康监测指标数据。

    Attributes:
        blood_pressure (Optional[Dict[str, float]]): 血压数据，包含systolic（收缩压）和diastolic（舒张压）
        blood_sugar (Optional[Dict[str, float]]): 血糖数据，包含fasting（空腹血糖）和postprandial（餐后血糖）
        heart_rate (Optional[int]): 心率，单位：次/分钟
        blood_oxygen (Optional[float]): 血氧饱和度，百分比
        bmi (Optional[float]): 体重指数
        sleep_data (Optional[Dict[str, Any]]): 睡眠数据，包含duration（时长）、quality（质量）等
        temperature (Optional[float]): 体温，单位：摄氏度
        weight (Optional[float]): 体重，单位：千克
        height (Optional[float]): 身高，单位：厘米
        steps (Optional[int]): 步数

    Example:
        >>> monitoring_data = MonitoringData(
        ...     blood_pressure={"systolic": 120.0, "diastolic": 80.0},
        ...     heart_rate=75,
        ...     blood_oxygen=98.5,
        ...     bmi=23.5
        ... )
    """

    blood_pressure: Optional[Dict[str, float]] = Field(
        default=None,
        description="血压数据，包含systolic（收缩压）和diastolic（舒张压）",
        examples=[{"systolic": 120.0, "diastolic": 80.0}]
    )

    blood_sugar: Optional[Dict[str, float]] = Field(
        default=None,
        description="血糖数据，包含fasting（空腹血糖）和postprandial（餐后血糖）",
        examples=[{"fasting": 5.5, "postprandial": 7.2}]
    )

    heart_rate: Optional[int] = Field(
        default=None,
        description="心率，单位：次/分钟",
        examples=[75, 80]
    )

    blood_oxygen: Optional[float] = Field(
        default=None,
        description="血氧饱和度，百分比",
        examples=[98.5, 97.0]
    )

    bmi: Optional[float] = Field(
        default=None,
        description="体重指数",
        examples=[23.5, 25.0]
    )

    sleep_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="睡眠数据，包含duration（时长）、quality（质量）等",
        examples=[{"duration": 7.5, "quality": "good"}]
    )

    temperature: Optional[float] = Field(
        default=None,
        description="体温，单位：摄氏度",
        examples=[36.5, 37.2]
    )

    weight: Optional[float] = Field(
        default=None,
        description="体重，单位：千克",
        examples=[65.0, 70.0]
    )

    height: Optional[float] = Field(
        default=None,
        description="身高，单位：厘米",
        examples=[175.0, 160.0]
    )

    steps: Optional[int] = Field(
        default=None,
        description="步数",
        examples=[8000, 10000]
    )


class UserProfile(BaseModel):
    """
    用户档案数据类

    包含用户的基本信息、病史、生活方式等档案数据。

    Attributes:
        basic_info (Optional[Dict[str, Any]]): 基本信息，包含age（年龄）、gender（性别）、name（姓名）等
        past_medical_history (Optional[List[str]]): 既往病史列表
        family_history (Optional[List[str]]): 家族病史列表
        lifestyle (Optional[Dict[str, Any]]): 生活方式，包含smoking（吸烟）、drinking（饮酒）、exercise（运动）等
        allergies (Optional[List[str]]): 过敏史列表
        medications (Optional[List[str]]): 当前用药列表

    Example:
        >>> user_profile = UserProfile(
        ...     basic_info={"age": 45, "gender": "male", "name": "张三"},
        ...     past_medical_history=["高血压", "糖尿病"],
        ...     lifestyle={"smoking": False, "drinking": "偶尔", "exercise": "每周3次"}
        ... )
    """

    basic_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="基本信息，包含age（年龄）、gender（性别）、name（姓名）等",
        examples=[{"age": 45, "gender": "male", "name": "张三"}]
    )

    past_medical_history: Optional[List[str]] = Field(
        default=None,
        description="既往病史列表",
        examples=[["高血压", "糖尿病", "冠心病"]]
    )

    family_history: Optional[List[str]] = Field(
        default=None,
        description="家族病史列表",
        examples=[["高血压", "糖尿病"]]
    )

    lifestyle: Optional[Dict[str, Any]] = Field(
        default=None,
        description="生活方式，包含smoking（吸烟）、drinking（饮酒）、exercise（运动）等",
        examples=[{"smoking": False, "drinking": "偶尔", "exercise": "每周3次"}]
    )

    allergies: Optional[List[str]] = Field(
        default=None,
        description="过敏史列表",
        examples=[["青霉素", "海鲜"]]
    )

    medications: Optional[List[str]] = Field(
        default=None,
        description="当前用药列表",
        examples=[["阿司匹林", "降压药"]]
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
        ...     monitoring_data=MonitoringData(heart_rate=75),
        ...     user_profile=UserProfile(basic_info={"age": 45, "gender": "male"}),
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
        ...         monitoring_data=MonitoringData(heart_rate=75),
        ...         user_profile=UserProfile(basic_info={"age": 45})
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
                        "blood_pressure": {"systolic": 120.0, "diastolic": 80.0},
                        "blood_sugar": {"fasting": 5.5, "postprandial": 7.2},
                        "heart_rate": 75,
                        "blood_oxygen": 98.5,
                        "bmi": 23.5,
                        "sleep_data": {"duration": 7.5, "quality": "good"},
                        "temperature": 36.5,
                        "weight": 65.0,
                        "height": 175.0,
                        "steps": 8000
                    },
                    "user_profile": {
                        "basic_info": {"age": 45, "gender": "male", "name": "张三"},
                        "past_medical_history": ["高血压", "糖尿病"],
                        "family_history": ["高血压"],
                        "lifestyle": {"smoking": False, "drinking": "偶尔", "exercise": "每周3次"},
                        "allergies": ["青霉素"],
                        "medications": ["阿司匹林"]
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
            >>> monitoring_data = MonitoringData(heart_rate=75)
            >>> request = ReportRequest(body=ReportRequestBody(task_id="task-001", monitoring_data=monitoring_data))
            >>> request.get_monitoring_data()
            MonitoringData(blood_pressure=None, blood_sugar=None, heart_rate=75, ...)
        """
        return self.body.monitoring_data

    def get_user_profile(self) -> UserProfile:
        """
        获取用户档案

        Returns:
            UserProfile: 用户档案对象

        Example:
            >>> user_profile = UserProfile(basic_info={"age": 45})
            >>> request = ReportRequest(body=ReportRequestBody(task_id="task-001", user_profile=user_profile))
            >>> request.get_user_profile()
            UserProfile(basic_info={'age': 45}, past_medical_history=None, ...)
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
            >>> monitoring_data = MonitoringData(blood_pressure={"systolic": 120.0, "diastolic": 80.0})
            >>> request = ReportRequest(body=ReportRequestBody(task_id="task-001", monitoring_data=monitoring_data))
            >>> request.has_blood_pressure()
            True
        """
        return self.body.monitoring_data.blood_pressure is not None

    def has_blood_sugar(self) -> bool:
        """
        判断是否有血糖数据

        Returns:
            bool: 如果有血糖数据返回True，否则返回False

        Example:
            >>> monitoring_data = MonitoringData(blood_sugar={"fasting": 5.5, "postprandial": 7.2})
            >>> request = ReportRequest(body=ReportRequestBody(task_id="task-001", monitoring_data=monitoring_data))
            >>> request.has_blood_sugar()
            True
        """
        return self.body.monitoring_data.blood_sugar is not None

    def has_heart_rate(self) -> bool:
        """
        判断是否有心率数据

        Returns:
            bool: 如果有心率数据返回True，否则返回False

        Example:
            >>> monitoring_data = MonitoringData(heart_rate=75)
            >>> request = ReportRequest(body=ReportRequestBody(task_id="task-001", monitoring_data=monitoring_data))
            >>> request.has_heart_rate()
            True
        """
        return self.body.monitoring_data.heart_rate is not None

    def has_sleep_data(self) -> bool:
        """
        判断是否有睡眠数据

        Returns:
            bool: 如果有睡眠数据返回True，否则返回False

        Example:
            >>> monitoring_data = MonitoringData(sleep_data={"duration": 7.5, "quality": "good"})
            >>> request = ReportRequest(body=ReportRequestBody(task_id="task-001", monitoring_data=monitoring_data))
            >>> request.has_sleep_data()
            True
        """
        return self.body.monitoring_data.sleep_data is not None

    def has_past_medical_history(self) -> bool:
        """
        判断是否有既往病史

        Returns:
            bool: 如果有既往病史返回True，否则返回False

        Example:
            >>> user_profile = UserProfile(past_medical_history=["高血压", "糖尿病"])
            >>> request = ReportRequest(body=ReportRequestBody(task_id="task-001", user_profile=user_profile))
            >>> request.has_past_medical_history()
            True
        """
        return self.body.user_profile.past_medical_history is not None and len(self.body.user_profile.past_medical_history) > 0

    def has_family_history(self) -> bool:
        """
        判断是否有家族病史

        Returns:
            bool: 如果有家族病史返回True，否则返回False

        Example:
            >>> user_profile = UserProfile(family_history=["高血压"])
            >>> request = ReportRequest(body=ReportRequestBody(task_id="task-001", user_profile=user_profile))
            >>> request.has_family_history()
            True
        """
        return self.body.user_profile.family_history is not None and len(self.body.user_profile.family_history) > 0

    def has_allergies(self) -> bool:
        """
        判断是否有过敏史

        Returns:
            bool: 如果有过敏史返回True，否则返回False

        Example:
            >>> user_profile = UserProfile(allergies=["青霉素"])
            >>> request = ReportRequest(body=ReportRequestBody(task_id="task-001", user_profile=user_profile))
            >>> request.has_allergies()
            True
        """
        return self.body.user_profile.allergies is not None and len(self.body.user_profile.allergies) > 0
