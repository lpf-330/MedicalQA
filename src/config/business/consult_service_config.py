# -*- coding: utf-8 -*-
"""
健康咨询业务配置文件

定义健康咨询业务的配置参数和所需的资源配置引用。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.config.base_config import BusinessConfig


@dataclass
class ConsultServiceConfig(BusinessConfig):
    """
    健康咨询业务配置类
    
    属性：
        business_id: 业务ID（文件名作为唯一标识）
        resource_configs: 所需的资源配置文件名列表
        max_retries: 最大重试次数
        timeout: 超时时间（秒）
        enable_knowledge_retrieval: 是否启用知识检索
        enable_model_consultation: 是否启用模型咨询
    """
    
    business_id: str = "consult_service"
    resource_configs: List[str] = field(default_factory=lambda: ["neo4j_config", "vllm_config"])
    
    max_retries: int = 3
    timeout: int = 60
    enable_knowledge_retrieval: bool = True
    enable_model_consultation: bool = True
    
    def validate(self) -> bool:
        """
        验证配置有效性
        
        Returns:
            bool: 配置是否有效
        """
        if not super().validate():
            return False
        
        if not self.resource_configs:
            print("警告: resource_configs 不能为空")
            return False
        if self.max_retries < 0:
            print("警告: max_retries 不能为负数")
            return False
        if self.timeout < 0:
            print("警告: timeout 不能为负数")
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        导出配置为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        base_dict = super().to_dict()
        base_dict.update({
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "enable_knowledge_retrieval": self.enable_knowledge_retrieval,
            "enable_model_consultation": self.enable_model_consultation,
        })
        return base_dict


business_config = ConsultServiceConfig()
