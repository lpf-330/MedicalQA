"""
基础响应数据类模块

该模块定义了所有API响应的基础数据结构，提供统一的响应格式。
"""

from typing import Any, Generic, TypeVar, Optional
from datetime import datetime
from pydantic import BaseModel, Field

# 定义泛型类型变量，用于响应数据的类型
T = TypeVar('T')


class BaseResponse(BaseModel, Generic[T]):
    """
    基础响应数据类
    
    所有API响应的基础类，提供统一的响应格式，包含状态码、消息、数据等通用属性。
    支持泛型，可以适配不同类型的响应数据。
    
    Attributes:
        status_code (int): 响应状态码，200表示成功，其他值表示不同的错误类型
        message (str): 响应消息，描述响应的简要信息
        data (Optional[T]): 响应数据，泛型类型，根据具体业务返回不同的数据结构
        timestamp (str): 响应时间戳，ISO格式的时间字符串
        request_id (Optional[str]): 请求ID，用于请求追踪和日志关联
    
    Example:
        >>> response = BaseResponse[dict](
        ...     status_code=200,
        ...     message="操作成功",
        ...     data={"user_id": "123", "name": "张三"}
        ... )
        >>> response.model_dump()
        {
            'status_code': 200,
            'message': '操作成功',
            'data': {'user_id': '123', 'name': '张三'},
            'timestamp': '2024-01-01T12:00:00',
            'request_id': None
        }
    """
    
    status_code: int = Field(
        default=200,
        description="响应状态码，200表示成功，其他值表示不同的错误类型",
        examples=[200, 400, 500]
    )
    
    message: str = Field(
        default="操作成功",
        description="响应消息，描述响应的简要信息",
        examples=["操作成功", "参数错误", "服务器内部错误"]
    )
    
    data: Optional[T] = Field(
        default=None,
        description="响应数据，泛型类型，根据具体业务返回不同的数据结构"
    )
    
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="响应时间戳，ISO格式的时间字符串"
    )
    
    request_id: Optional[str] = Field(
        default=None,
        description="请求ID，用于请求追踪和日志关联"
    )
    
    class Config:
        """Pydantic配置类"""
        json_schema_extra = {
            "example": {
                "status_code": 200,
                "message": "操作成功",
                "data": {"key": "value"},
                "timestamp": "2024-01-01T12:00:00",
                "request_id": "req-123456"
            }
        }
    
    def is_success(self) -> bool:
        """
        判断响应是否成功
        
        Returns:
            bool: 如果状态码为200返回True，否则返回False
        
        Example:
            >>> response = BaseResponse(status_code=200, message="成功")
            >>> response.is_success()
            True
        """
        return self.status_code == 200
    
    def to_dict(self) -> dict:
        """
        将响应对象转换为字典格式
        
        Returns:
            dict: 包含所有属性的字典
        
        Example:
            >>> response = BaseResponse(status_code=200, message="成功")
            >>> response.to_dict()
            {'status_code': 200, 'message': '成功', 'data': None, 'timestamp': '...', 'request_id': None}
        """
        return self.model_dump()
    
    @classmethod
    def success(cls, data: Optional[T] = None, message: str = "操作成功", request_id: Optional[str] = None) -> "BaseResponse[T]":
        """
        创建成功的响应对象
        
        Args:
            data: 响应数据
            message: 响应消息，默认为"操作成功"
            request_id: 请求ID
        
        Returns:
            BaseResponse[T]: 成功的响应对象
        
        Example:
            >>> response = BaseResponse.success(data={"user_id": "123"}, message="查询成功")
            >>> response.status_code
            200
        """
        return cls(
            status_code=200,
            message=message,
            data=data,
            request_id=request_id
        )
    
    @classmethod
    def error(cls, status_code: int = 500, message: str = "操作失败", request_id: Optional[str] = None) -> "BaseResponse[T]":
        """
        创建错误的响应对象
        
        Args:
            status_code: 错误状态码，默认为500
            message: 错误消息，默认为"操作失败"
            request_id: 请求ID
        
        Returns:
            BaseResponse[T]: 错误的响应对象
        
        Example:
            >>> response = BaseResponse.error(status_code=400, message="参数错误")
            >>> response.status_code
            400
        """
        return cls(
            status_code=status_code,
            message=message,
            data=None,
            request_id=request_id
        )
