"""
健康报告生成响应数据类模块

该模块定义了健康报告生成API响应的数据结构。
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from .base_response import BaseResponse


class ReportResponseData(BaseModel):
    """
    健康报告生成响应数据类

    包含健康报告生成特有的响应数据结构。

    Attributes:
        result (str): 报告内容，Markdown格式
        event_type (str): SSE事件类型，默认"message"
        health_score (int): 健康综合评分，0-100
        health_level (str): 健康等级：优秀/良好/一般/较差/差
        risk_level (str): 风险等级：低/轻/中/高
        risk_diseases (List[Dict[str, Any]]): 高风险疾病列表，包含disease_name、risk_score、confidence、evidence
        sources (List[str]): 知识来源列表
        word_count (int): 报告字数
        error_code (int): 错误码，0表示无错误
        error_message (str): 错误消息
        session_id (Optional[str]): 会话ID
        dimension_results (Optional[Dict[str, Any]]): 各维度评估结果
        recommendations (Optional[List[str]]): 健康建议列表
        follow_up_date (Optional[str]): 建议复查日期

    Example:
        >>> data = ReportResponseData(
        ...     result="# 健康报告\\n## 综合评估\\n您的健康状况良好...",
        ...     health_score=85,
        ...     health_level="良好",
        ...     risk_level="低",
        ...     risk_diseases=[{"disease_name": "高血压", "risk_score": 30, "confidence": 0.8}],
        ...     sources=["医学知识库", "临床指南"],
        ...     word_count=1500,
        ...     recommendations=["保持规律作息", "适量运动", "定期体检"]
        ... )
    """

    result: str = Field(
        ...,
        description="报告内容，Markdown格式",
        examples=["# 健康报告\\n## 综合评估\\n您的健康状况良好..."]
    )

    event_type: str = Field(
        default="message",
        description="SSE事件类型",
        examples=["message", "end", "error"]
    )

    health_score: int = Field(
        ...,
        description="健康综合评分，0-100",
        ge=0,
        le=100,
        examples=[85, 72, 90]
    )

    health_level: str = Field(
        ...,
        description="健康等级：优秀/良好/一般/较差/差",
        examples=["优秀", "良好", "一般", "较差", "差"]
    )

    risk_level: str = Field(
        ...,
        description="风险等级：低/轻/中/高",
        examples=["低", "轻", "中", "高"]
    )

    risk_diseases: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="高风险疾病列表，包含disease_name、risk_score、confidence、evidence",
        examples=[
            [
                {
                    "disease_name": "高血压",
                    "risk_score": 30,
                    "confidence": 0.8,
                    "evidence": "血压偏高，有家族病史"
                },
                {
                    "disease_name": "糖尿病",
                    "risk_score": 25,
                    "confidence": 0.7,
                    "evidence": "空腹血糖偏高"
                }
            ]
        ]
    )

    sources: List[str] = Field(
        default_factory=list,
        description="知识来源列表",
        examples=[
            ["医学知识库-高血压章节", "临床指南-健康评估"]
        ]
    )

    word_count: int = Field(
        default=0,
        description="报告字数",
        examples=[1500, 2000]
    )

    error_code: int = Field(
        default=0,
        description="错误码，0表示无错误",
        examples=[0, 1001, 2001]
    )

    error_message: str = Field(
        default="",
        description="错误消息",
        examples=["", "模型服务不可用", "请求参数错误"]
    )

    session_id: Optional[str] = Field(
        default=None,
        description="会话ID"
    )

    dimension_results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="各维度评估结果",
        examples=[
            {
                "心血管": {"score": 85, "level": "良好", "details": "心血管功能正常"},
                "代谢": {"score": 78, "level": "一般", "details": "血糖略高"},
                "生活方式": {"score": 90, "level": "优秀", "details": "生活方式健康"}
            }
        ]
    )

    recommendations: Optional[List[str]] = Field(
        default=None,
        description="健康建议列表",
        examples=[
            ["保持规律作息", "适量运动", "定期体检", "控制饮食"]
        ]
    )

    follow_up_date: Optional[str] = Field(
        default=None,
        description="建议复查日期",
        examples=["2024-04-01", "2024-06-15"]
    )


class ReportResponse(BaseResponse[ReportResponseData]):
    """
    健康报告生成响应数据类

    继承BaseResponse，包含健康报告生成特有的属性。
    用于返回健康报告生成相关的API响应。

    Attributes:
        继承自BaseResponse:
            - status_code (int): 响应状态码
            - message (str): 响应消息
            - data (ReportResponseData): 健康报告生成响应数据
            - timestamp (str): 响应时间戳
            - request_id (Optional[str]): 请求ID

    Example:
        >>> response = ReportResponse(
        ...     status_code=200,
        ...     message="报告生成成功",
        ...     data=ReportResponseData(
        ...         result="# 健康报告\\n您的健康状况良好",
        ...         health_score=85,
        ...         health_level="良好",
        ...         risk_level="低",
        ...         recommendations=["保持规律作息", "适量运动"]
        ...     ),
        ...     request_id="req-123456"
        ... )
        >>> response.data.health_score
        85
    """

    data: Optional[ReportResponseData] = Field(
        default=None,
        description="健康报告生成响应数据"
    )

    class Config:
        """Pydantic配置类"""
        json_schema_extra = {
            "example": {
                "status_code": 200,
                "message": "报告生成成功",
                "data": {
                    "result": "# 健康报告\\n## 综合评估\\n您的健康状况良好，各项指标基本正常。\\n\\n## 详细分析\\n...",
                    "event_type": "message",
                    "health_score": 85,
                    "health_level": "良好",
                    "risk_level": "低",
                    "risk_diseases": [
                        {
                            "disease_name": "高血压",
                            "risk_score": 30,
                            "confidence": 0.8,
                            "evidence": "血压偏高，有家族病史"
                        }
                    ],
                    "sources": [
                        "医学知识库-高血压章节",
                        "临床指南-健康评估"
                    ],
                    "word_count": 1500,
                    "error_code": 0,
                    "error_message": "",
                    "session_id": "session-001",
                    "dimension_results": {
                        "心血管": {"score": 85, "level": "良好", "details": "心血管功能正常"},
                        "代谢": {"score": 78, "level": "一般", "details": "血糖略高"},
                        "生活方式": {"score": 90, "level": "优秀", "details": "生活方式健康"}
                    },
                    "recommendations": [
                        "保持规律作息",
                        "适量运动",
                        "定期体检",
                        "控制饮食"
                    ],
                    "follow_up_date": "2024-04-01"
                },
                "timestamp": "2024-01-01T12:00:00",
                "request_id": "req-123456789abc"
            }
        }

    def get_result(self) -> Optional[str]:
        """
        获取报告内容

        Returns:
            Optional[str]: 报告内容（Markdown格式），如果未设置则返回None

        Example:
            >>> response = ReportResponse(data=ReportResponseData(result="# 健康报告", health_score=85, health_level="良好", risk_level="低"))
            >>> response.get_result()
            '# 健康报告'
        """
        if self.data:
            return self.data.result
        return None

    def get_health_score(self) -> Optional[int]:
        """
        获取健康综合评分

        Returns:
            Optional[int]: 健康综合评分（0-100），如果未设置则返回None

        Example:
            >>> response = ReportResponse(data=ReportResponseData(result="test", health_score=85, health_level="良好", risk_level="低"))
            >>> response.get_health_score()
            85
        """
        if self.data:
            return self.data.health_score
        return None

    def get_health_level(self) -> Optional[str]:
        """
        获取健康等级

        Returns:
            Optional[str]: 健康等级（优秀/良好/一般/较差/差），如果未设置则返回None

        Example:
            >>> response = ReportResponse(data=ReportResponseData(result="test", health_score=85, health_level="良好", risk_level="低"))
            >>> response.get_health_level()
            '良好'
        """
        if self.data:
            return self.data.health_level
        return None

    def get_risk_level(self) -> Optional[str]:
        """
        获取风险等级

        Returns:
            Optional[str]: 风险等级（低/轻/中/高），如果未设置则返回None

        Example:
            >>> response = ReportResponse(data=ReportResponseData(result="test", health_score=85, health_level="良好", risk_level="低"))
            >>> response.get_risk_level()
            '低'
        """
        if self.data:
            return self.data.risk_level
        return None

    def get_risk_diseases(self) -> Optional[List[Dict[str, Any]]]:
        """
        获取高风险疾病列表

        Returns:
            Optional[List[Dict[str, Any]]]: 高风险疾病列表，如果未设置则返回None

        Example:
            >>> diseases = [{"disease_name": "高血压", "risk_score": 30}]
            >>> response = ReportResponse(data=ReportResponseData(result="test", health_score=85, health_level="良好", risk_level="低", risk_diseases=diseases))
            >>> response.get_risk_diseases()
            [{'disease_name': '高血压', 'risk_score': 30}]
        """
        if self.data:
            return self.data.risk_diseases
        return None

    def get_sources(self) -> Optional[List[str]]:
        """
        获取知识来源列表

        Returns:
            Optional[List[str]]: 知识来源列表，如果未设置则返回None

        Example:
            >>> sources = ["医学知识库", "临床指南"]
            >>> response = ReportResponse(data=ReportResponseData(result="test", health_score=85, health_level="良好", risk_level="低", sources=sources))
            >>> response.get_sources()
            ['医学知识库', '临床指南']
        """
        if self.data:
            return self.data.sources
        return None

    def get_session_id(self) -> Optional[str]:
        """
        获取会话ID

        Returns:
            Optional[str]: 会话ID，如果未设置则返回None

        Example:
            >>> response = ReportResponse(data=ReportResponseData(result="test", health_score=85, health_level="良好", risk_level="低", session_id="session-001"))
            >>> response.get_session_id()
            'session-001'
        """
        if self.data:
            return self.data.session_id
        return None

    def get_dimension_results(self) -> Optional[Dict[str, Any]]:
        """
        获取各维度评估结果

        Returns:
            Optional[Dict[str, Any]]: 各维度评估结果，如果未设置则返回None

        Example:
            >>> dimensions = {"心血管": {"score": 85, "level": "良好"}}
            >>> response = ReportResponse(data=ReportResponseData(result="test", health_score=85, health_level="良好", risk_level="低", dimension_results=dimensions))
            >>> response.get_dimension_results()
            {'心血管': {'score': 85, 'level': '良好'}}
        """
        if self.data:
            return self.data.dimension_results
        return None

    def get_recommendations(self) -> Optional[List[str]]:
        """
        获取健康建议列表

        Returns:
            Optional[List[str]]: 健康建议列表，如果未设置则返回None

        Example:
            >>> recommendations = ["保持规律作息", "适量运动"]
            >>> response = ReportResponse(data=ReportResponseData(result="test", health_score=85, health_level="良好", risk_level="低", recommendations=recommendations))
            >>> response.get_recommendations()
            ['保持规律作息', '适量运动']
        """
        if self.data:
            return self.data.recommendations
        return None

    def get_follow_up_date(self) -> Optional[str]:
        """
        获取建议复查日期

        Returns:
            Optional[str]: 建议复查日期，如果未设置则返回None

        Example:
            >>> response = ReportResponse(data=ReportResponseData(result="test", health_score=85, health_level="良好", risk_level="低", follow_up_date="2024-04-01"))
            >>> response.get_follow_up_date()
            '2024-04-01'
        """
        if self.data:
            return self.data.follow_up_date
        return None

    def has_risk_diseases(self) -> bool:
        """
        判断是否有高风险疾病

        Returns:
            bool: 如果有高风险疾病返回True，否则返回False

        Example:
            >>> diseases = [{"disease_name": "高血压", "risk_score": 30}]
            >>> response = ReportResponse(data=ReportResponseData(result="test", health_score=85, health_level="良好", risk_level="低", risk_diseases=diseases))
            >>> response.has_risk_diseases()
            True
        """
        if self.data and self.data.risk_diseases:
            return len(self.data.risk_diseases) > 0
        return False

    def has_recommendations(self) -> bool:
        """
        判断是否有健康建议

        Returns:
            bool: 如果有健康建议返回True，否则返回False

        Example:
            >>> recommendations = ["保持规律作息"]
            >>> response = ReportResponse(data=ReportResponseData(result="test", health_score=85, health_level="良好", risk_level="低", recommendations=recommendations))
            >>> response.has_recommendations()
            True
        """
        if self.data and self.data.recommendations:
            return len(self.data.recommendations) > 0
        return False

    def has_dimension_results(self) -> bool:
        """
        判断是否有各维度评估结果

        Returns:
            bool: 如果有各维度评估结果返回True，否则返回False

        Example:
            >>> dimensions = {"心血管": {"score": 85}}
            >>> response = ReportResponse(data=ReportResponseData(result="test", health_score=85, health_level="良好", risk_level="低", dimension_results=dimensions))
            >>> response.has_dimension_results()
            True
        """
        if self.data and self.data.dimension_results:
            return len(self.data.dimension_results) > 0
        return False

    def is_high_risk(self) -> bool:
        """
        判断是否为高风险

        Returns:
            bool: 如果风险等级为"高"返回True，否则返回False

        Example:
            >>> response = ReportResponse(data=ReportResponseData(result="test", health_score=85, health_level="良好", risk_level="高"))
            >>> response.is_high_risk()
            True
        """
        if self.data:
            return self.data.risk_level == "高"
        return False

    def get_word_count(self) -> Optional[int]:
        """
        获取报告字数

        Returns:
            Optional[int]: 报告字数，如果未设置则返回None

        Example:
            >>> response = ReportResponse(data=ReportResponseData(result="test", health_score=85, health_level="良好", risk_level="低", word_count=1500))
            >>> response.get_word_count()
            1500
        """
        if self.data:
            return self.data.word_count
        return None
