"""
服务层包

该包包含服务层的核心类，负责业务逻辑的初步封装。
"""

from src.service.consult_service import ConsultService
from src.service.report_service import ReportService

__all__ = ['ConsultService', 'ReportService']
