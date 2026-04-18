"""
schemas包

该包负责项目数据类管理，定义所有API请求和响应的数据结构。
"""

from .base_request import BaseRequest
from .base_response import BaseResponse
from .consult_request import ConsultRequest, ConsultRequestBody, ChatMessage
from .consult_response import ConsultResponse, ConsultResponseData
from .report_request import ReportRequest, ReportRequestBody, MonitoringData, UserProfile
from .report_response import ReportResponse, ReportResponseData

__all__ = [
    "BaseRequest",
    "BaseResponse",
    "ChatMessage",
    "ConsultRequest",
    "ConsultRequestBody",
    "ConsultResponse",
    "ConsultResponseData",
    "ReportRequest",
    "ReportRequestBody",
    "MonitoringData",
    "UserProfile",
    "ReportResponse",
    "ReportResponseData",
]
