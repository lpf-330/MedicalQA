"""
健康咨询响应数据类模块

该模块定义了健康咨询API响应的数据结构。
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from .base_response import BaseResponse


class ConsultResponseData(BaseModel):
    """
    健康咨询响应数据类
    
    包含健康咨询特有的响应数据结构。
    
    Attributes:
        result (str): 咨询的主要结果内容
        suggestions (Optional[List[str]]): 健康建议列表
        related_knowledge (Optional[List[Dict[str, str]]]): 相关的健康知识
        session_id (Optional[str]): 会话ID，用于多轮对话的会话标识
        follow_up_questions (Optional[List[str]]): 建议用户进一步咨询的问题
        confidence (Optional[float]): 咨询结果的置信度，范围0-1
        sources (Optional[List[Dict[str, str]]]): 咨询结果的知识来源
    
    Example:
        >>> data = ConsultResponseData(
        ...     result="根据您的描述，头痛可能由多种原因引起...",
        ...     suggestions=["保持充足睡眠", "适当运动", "定期检查血压"],
        ...     session_id="session-001"
        ... )
    """
    
    result: str = Field(
        ...,
        description="咨询的主要结果内容",
        examples=["根据您的描述，头痛可能由多种原因引起，包括压力、睡眠不足、高血压等。"]
    )
    
    suggestions: Optional[List[str]] = Field(
        default=None,
        description="健康建议列表",
        examples=[
            ["保持充足睡眠", "适当运动", "定期检查血压", "减少咖啡因摄入"]
        ]
    )
    
    related_knowledge: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="相关的健康知识",
        examples=[
            [
                {
                    "title": "高血压的症状",
                    "content": "高血压常见症状包括头痛、头晕、心悸等...",
                    "source": "医学知识库"
                }
            ]
        ]
    )
    
    session_id: Optional[str] = Field(
        default=None,
        description="会话ID，用于多轮对话的会话标识"
    )
    
    follow_up_questions: Optional[List[str]] = Field(
        default=None,
        description="建议用户进一步咨询的问题",
        examples=[
            ["您的头痛通常在什么时间发生？", "您是否有高血压家族史？"]
        ]
    )
    
    confidence: Optional[float] = Field(
        default=None,
        description="咨询结果的置信度，范围0-1",
        ge=0.0,
        le=1.0,
        examples=[0.85]
    )
    
    sources: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="咨询结果的知识来源",
        examples=[
            [
                {
                    "type": "medical_database",
                    "name": "医学知识库",
                    "url": "https://example.com/knowledge/123"
                }
            ]
        ]
    )


class ConsultResponse(BaseResponse[ConsultResponseData]):
    """
    健康咨询响应数据类
    
    继承BaseResponse，包含健康咨询特有的属性。
    用于返回健康咨询相关的API响应。
    
    Attributes:
        继承自BaseResponse:
            - status_code (int): 响应状态码
            - message (str): 响应消息
            - data (ConsultResponseData): 健康咨询响应数据
            - timestamp (str): 响应时间戳
            - request_id (Optional[str]): 请求ID
    
    Example:
        >>> response = ConsultResponse(
        ...     status_code=200,
        ...     message="咨询成功",
        ...     data=ConsultResponseData(
        ...         result="根据您的描述，头痛可能由多种原因引起...",
        ...         suggestions=["保持充足睡眠", "适当运动"],
        ...         session_id="session-001"
        ...     ),
        ...     request_id="req-123456"
        ... )
        >>> response.data.result
        '根据您的描述，头痛可能由多种原因引起...'
    """
    
    data: Optional[ConsultResponseData] = Field(
        default=None,
        description="健康咨询响应数据"
    )
    
    class Config:
        """Pydantic配置类"""
        json_schema_extra = {
            "example": {
                "status_code": 200,
                "message": "咨询成功",
                "data": {
                    "result": "根据您的描述，头痛可能由多种原因引起，包括压力、睡眠不足、高血压等。",
                    "suggestions": [
                        "保持充足睡眠",
                        "适当运动",
                        "定期检查血压",
                        "减少咖啡因摄入"
                    ],
                    "related_knowledge": [
                        {
                            "title": "高血压的症状",
                            "content": "高血压常见症状包括头痛、头晕、心悸等...",
                            "source": "医学知识库"
                        }
                    ],
                    "session_id": "session-001",
                    "follow_up_questions": [
                        "您的头痛通常在什么时间发生？",
                        "您是否有高血压家族史？"
                    ],
                    "confidence": 0.85,
                    "sources": [
                        {
                            "type": "medical_database",
                            "name": "医学知识库",
                            "url": "https://example.com/knowledge/123"
                        }
                    ]
                },
                "timestamp": "2024-01-01T12:00:00",
                "request_id": "req-123456789abc"
            }
        }
    
    def get_result(self) -> Optional[str]:
        """
        获取咨询结果
        
        Returns:
            Optional[str]: 咨询的主要结果内容，如果未设置则返回None
        
        Example:
            >>> response = ConsultResponse(data=ConsultResponseData(result="头痛可能由多种原因引起"))
            >>> response.get_result()
            '头痛可能由多种原因引起'
        """
        if self.data:
            return self.data.result
        return None
    
    def get_suggestions(self) -> Optional[List[str]]:
        """
        获取健康建议列表
        
        Returns:
            Optional[List[str]]: 健康建议列表，如果未设置则返回None
        
        Example:
            >>> response = ConsultResponse(data=ConsultResponseData(result="test", suggestions=["多休息", "多喝水"]))
            >>> response.get_suggestions()
            ['多休息', '多喝水']
        """
        if self.data:
            return self.data.suggestions
        return None
    
    def get_session_id(self) -> Optional[str]:
        """
        获取会话ID
        
        Returns:
            Optional[str]: 会话ID，如果未设置则返回None
        
        Example:
            >>> response = ConsultResponse(data=ConsultResponseData(result="test", session_id="session-001"))
            >>> response.get_session_id()
            'session-001'
        """
        if self.data:
            return self.data.session_id
        return None
    
    def get_follow_up_questions(self) -> Optional[List[str]]:
        """
        获取后续问题列表
        
        Returns:
            Optional[List[str]]: 建议用户进一步咨询的问题列表，如果未设置则返回None
        
        Example:
            >>> questions = ["您头痛多久了？", "有家族病史吗？"]
            >>> response = ConsultResponse(data=ConsultResponseData(result="test", follow_up_questions=questions))
            >>> response.get_follow_up_questions()
            ['您头痛多久了？', '有家族病史吗？']
        """
        if self.data:
            return self.data.follow_up_questions
        return None
    
    def get_confidence(self) -> Optional[float]:
        """
        获取咨询结果的置信度
        
        Returns:
            Optional[float]: 置信度，范围0-1，如果未设置则返回None
        
        Example:
            >>> response = ConsultResponse(data=ConsultResponseData(result="test", confidence=0.85))
            >>> response.get_confidence()
            0.85
        """
        if self.data:
            return self.data.confidence
        return None
    
    def has_suggestions(self) -> bool:
        """
        判断是否有健康建议
        
        Returns:
            bool: 如果有健康建议返回True，否则返回False
        
        Example:
            >>> response = ConsultResponse(data=ConsultResponseData(result="test", suggestions=["多休息"]))
            >>> response.has_suggestions()
            True
        """
        if self.data and self.data.suggestions:
            return len(self.data.suggestions) > 0
        return False
    
    def has_follow_up_questions(self) -> bool:
        """
        判断是否有后续问题
        
        Returns:
            bool: 如果有后续问题返回True，否则返回False
        
        Example:
            >>> response = ConsultResponse(data=ConsultResponseData(result="test", follow_up_questions=["问题1"]))
            >>> response.has_follow_up_questions()
            True
        """
        if self.data and self.data.follow_up_questions:
            return len(self.data.follow_up_questions) > 0
        return False
