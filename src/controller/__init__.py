"""
接入层包

该包包含接入层的核心类，负责HTTP协议处理、请求参数校验、协议转换。
"""

from src.controller.consult_controller import ConsultController
from src.controller.report_controller import ReportController

__all__ = ['ConsultController', 'ReportController']
