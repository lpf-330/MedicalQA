"""
基础请求数据类模块

该模块定义了所有API请求的基础数据结构，提供统一的请求格式。
"""

from typing import Any, Generic, TypeVar, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import uuid

# 定义泛型类型变量，用于请求体的类型
T = TypeVar('T')


class BaseRequest(BaseModel, Generic[T]):
    """
    基础请求数据类
    
    所有API请求的基础类，提供统一的请求格式，包含请求ID、时间戳等通用属性。
    支持泛型，可以适配不同类型的请求体数据。
    
    Attributes:
        request_id (str): 请求唯一标识符，用于请求追踪和日志关联
        timestamp (str): 请求时间戳，ISO格式的时间字符串
        body (Optional[T]): 请求体数据，泛型类型，根据具体业务包含不同的数据结构
        user_id (Optional[str]): 用户ID，标识请求发起者
        client_info (Optional[dict]): 客户端信息，包含客户端类型、版本等
    
    Example:
        >>> request = BaseRequest[dict](
        ...     request_id="req-123456",
        ...     body={"question": "什么是高血压？"},
        ...     user_id="user-001"
        ... )
        >>> request.model_dump()
        {
            'request_id': 'req-123456',
            'timestamp': '2024-01-01T12:00:00',
            'body': {'question': '什么是高血压？'},
            'user_id': 'user-001',
            'client_info': None
        }
    """
    
    request_id: str = Field(
        default_factory=lambda: f"req-{uuid.uuid4().hex[:12]}",
        description="请求唯一标识符，用于请求追踪和日志关联",
        examples=["req-123456789abc", "req-def456789ghi"]
    )
    
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="请求时间戳，ISO格式的时间字符串"
    )
    
    body: Optional[T] = Field(
        default=None,
        description="请求体数据，泛型类型，根据具体业务包含不同的数据结构"
    )
    
    user_id: Optional[str] = Field(
        default=None,
        description="用户ID，标识请求发起者"
    )
    
    client_info: Optional[dict] = Field(
        default=None,
        description="客户端信息，包含客户端类型、版本等"
    )
    
    class Config:
        """Pydantic配置类"""
        json_schema_extra = {
            "example": {
                "request_id": "req-123456789abc",
                "timestamp": "2024-01-01T12:00:00",
                "body": {"key": "value"},
                "user_id": "user-001",
                "client_info": {
                    "client_type": "web",
                    "version": "1.0.0"
                }
            }
        }
    
    def to_dict(self) -> dict:
        """
        将请求对象转换为字典格式
        
        Returns:
            dict: 包含所有属性的字典
        
        Example:
            >>> request = BaseRequest(request_id="req-123", body={"question": "test"})
            >>> request.to_dict()
            {'request_id': 'req-123', 'timestamp': '...', 'body': {'question': 'test'}, 'user_id': None, 'client_info': None}
        """
        return self.model_dump()
    
    def get_request_id(self) -> str:
        """
        获取请求ID
        
        Returns:
            str: 请求唯一标识符
        
        Example:
            >>> request = BaseRequest(request_id="req-123")
            >>> request.get_request_id()
            'req-123'
        """
        return self.request_id
    
    def get_user_id(self) -> Optional[str]:
        """
        获取用户ID
        
        Returns:
            Optional[str]: 用户ID，如果未设置则返回None
        
        Example:
            >>> request = BaseRequest(user_id="user-001")
            >>> request.get_user_id()
            'user-001'
        """
        return self.user_id
    
    def get_client_type(self) -> Optional[str]:
        """
        获取客户端类型
        
        Returns:
            Optional[str]: 客户端类型，如果未设置则返回None
        
        Example:
            >>> request = BaseRequest(client_info={"client_type": "web"})
            >>> request.get_client_type()
            'web'
        """
        if self.client_info and "client_type" in self.client_info:
            return self.client_info["client_type"]
        return None
    
    def validate_request(self) -> bool:
        """
        验证请求的基本有效性
        
        Returns:
            bool: 如果请求基本有效返回True，否则返回False
        
        Example:
            >>> request = BaseRequest(request_id="req-123")
            >>> request.validate_request()
            True
        """
        # 检查request_id是否为空
        if not self.request_id or not self.request_id.strip():
            return False
        
        # 检查timestamp是否为空
        if not self.timestamp or not self.timestamp.strip():
            return False
        
        return True
