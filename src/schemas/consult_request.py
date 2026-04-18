"""
健康咨询请求数据类模块

该模块定义了健康咨询API请求的数据结构。
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from .base_request import BaseRequest


class ChatMessage(BaseModel):

    role: str = Field(
        ...,
        description="消息角色",
        examples=["user", "assistant"]
    )

    content: str = Field(
        ...,
        description="消息内容",
        examples=["我最近总是头痛", "请问您的头痛持续多长时间了？"]
    )


class ConsultRequestBody(BaseModel):
    """
    健康咨询请求体数据类
    
    包含健康咨询特有的请求数据结构。
    
    Attributes:
        task_id (str): 任务标识符
        chat_history (List[ChatMessage]): 对话历史列表
        question (str): 用户提出的健康咨询问题
        session_id (Optional[str]): 会话ID，用于多轮对话的会话标识
        conversation_history (Optional[List[Dict[str, str]]]): 对话历史，包含之前的对话记录
        user_profile (Optional[Dict[str, Any]]): 用户健康档案信息，如年龄、性别、病史等
        context (Optional[Dict[str, Any]]): 额外的上下文信息
    
    Example:
        >>> body = ConsultRequestBody(
        ...     task_id="task-001",
        ...     chat_history=[ChatMessage(role="user", content="我最近总是头痛")],
        ...     question="我最近总是头痛，应该怎么办？",
        ...     session_id="session-001",
        ...     user_profile={"age": 45, "gender": "male"}
        ... )
    """

    task_id: str = Field(
        ...,
        description="任务标识符",
        examples=["task-001", "task-abc123"]
    )

    chat_history: List[ChatMessage] = Field(
        ...,
        description="对话历史列表",
        examples=[
            [
                {"role": "user", "content": "我最近总是头痛"},
                {"role": "assistant", "content": "请问您的头痛持续多长时间了？"}
            ]
        ]
    )
    
    question: str = Field(
        ...,
        description="用户提出的健康咨询问题",
        examples=["我最近总是头痛，应该怎么办？", "高血压患者应该注意什么？"]
    )
    
    session_id: Optional[str] = Field(
        default=None,
        description="会话ID，用于多轮对话的会话标识"
    )
    
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="对话历史，包含之前的对话记录",
        examples=[
            [
                {"role": "user", "content": "我最近总是头痛"},
                {"role": "assistant", "content": "请问您的头痛持续多长时间了？"}
            ]
        ]
    )
    
    user_profile: Optional[Dict[str, Any]] = Field(
        default=None,
        description="用户健康档案信息，如年龄、性别、病史等",
        examples=[
            {
                "age": 45,
                "gender": "male",
                "medical_history": ["高血压", "糖尿病"],
                "allergies": ["青霉素"]
            }
        ]
    )
    
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="额外的上下文信息"
    )


class ConsultRequest(BaseRequest[ConsultRequestBody]):
    """
    健康咨询请求数据类
    
    继承BaseRequest，包含健康咨询特有的属性。
    用于接收和处理健康咨询相关的API请求。
    
    Attributes:
        继承自BaseRequest:
            - request_id (str): 请求唯一标识符
            - timestamp (str): 请求时间戳
            - body (ConsultRequestBody): 健康咨询请求体
            - user_id (Optional[str]): 用户ID
            - client_info (Optional[dict]): 客户端信息
    
    Example:
        >>> request = ConsultRequest(
        ...     request_id="req-123456",
        ...     user_id="user-001",
        ...     body=ConsultRequestBody(
        ...         question="我最近总是头痛，应该怎么办？",
        ...         session_id="session-001",
        ...         user_profile={"age": 45, "gender": "male"}
        ...     )
        ... )
        >>> request.body.question
        '我最近总是头痛，应该怎么办？'
    """
    
    body: ConsultRequestBody = Field(
        ...,
        description="健康咨询请求体"
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
                    "chat_history": [
                        {"role": "user", "content": "我最近总是头痛"},
                        {"role": "assistant", "content": "请问您的头痛持续多长时间了？"}
                    ],
                    "question": "我最近总是头痛，应该怎么办？",
                    "session_id": "session-001",
                    "conversation_history": [
                        {"role": "user", "content": "我最近总是头痛"},
                        {"role": "assistant", "content": "请问您的头痛持续多长时间了？"}
                    ],
                    "user_profile": {
                        "age": 45,
                        "gender": "male",
                        "medical_history": ["高血压"],
                        "allergies": ["青霉素"]
                    },
                    "context": {}
                }
            }
        }
    
    def get_question(self) -> str:
        """
        获取用户提出的问题
        
        Returns:
            str: 用户提出的健康咨询问题
        
        Example:
            >>> request = ConsultRequest(body=ConsultRequestBody(question="头痛怎么办？"))
            >>> request.get_question()
            '头痛怎么办？'
        """
        return self.body.question
    
    def get_session_id(self) -> Optional[str]:
        """
        获取会话ID
        
        Returns:
            Optional[str]: 会话ID，如果未设置则返回None
        
        Example:
            >>> request = ConsultRequest(body=ConsultRequestBody(question="test", session_id="session-001"))
            >>> request.get_session_id()
            'session-001'
        """
        return self.body.session_id
    
    def get_conversation_history(self) -> Optional[List[Dict[str, str]]]:
        """
        获取对话历史
        
        Returns:
            Optional[List[Dict[str, str]]]: 对话历史列表，如果未设置则返回None
        
        Example:
            >>> history = [{"role": "user", "content": "头痛"}]
            >>> request = ConsultRequest(body=ConsultRequestBody(question="test", conversation_history=history))
            >>> request.get_conversation_history()
            [{'role': 'user', 'content': '头痛'}]
        """
        return self.body.conversation_history
    
    def get_user_profile(self) -> Optional[Dict[str, Any]]:
        """
        获取用户健康档案信息
        
        Returns:
            Optional[Dict[str, Any]]: 用户健康档案信息，如果未设置则返回None
        
        Example:
            >>> profile = {"age": 45, "gender": "male"}
            >>> request = ConsultRequest(body=ConsultRequestBody(question="test", user_profile=profile))
            >>> request.get_user_profile()
            {'age': 45, 'gender': 'male'}
        """
        return self.body.user_profile
    
    def has_conversation_history(self) -> bool:
        """
        判断是否有对话历史
        
        Returns:
            bool: 如果有对话历史返回True，否则返回False
        
        Example:
            >>> request = ConsultRequest(body=ConsultRequestBody(question="test"))
            >>> request.has_conversation_history()
            False
        """
        return self.body.conversation_history is not None and len(self.body.conversation_history) > 0
    
    def has_user_profile(self) -> bool:
        """
        判断是否有用户健康档案
        
        Returns:
            bool: 如果有用户健康档案返回True，否则返回False
        
        Example:
            >>> request = ConsultRequest(body=ConsultRequestBody(question="test", user_profile={"age": 45}))
            >>> request.has_user_profile()
            True
        """
        return self.body.user_profile is not None and len(self.body.user_profile) > 0
