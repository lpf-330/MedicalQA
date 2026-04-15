"""
编排层模型业务服务包

该包提供模型业务服务的设计规范，包括：
- ModelBusinessService: 模型业务服务接口

重要说明：
    模型业务服务不是模型服务，它是模型服务根据不同业务场景所定制的服务。
    例如：
    - ConsultModelService: 健康咨询业务场景下的模型服务
    - ReportModelService: 健康报告业务场景下的模型服务
"""

from src.orchestration.model_business_service.model_business_service import ModelBusinessService

__all__ = ['ModelBusinessService']
