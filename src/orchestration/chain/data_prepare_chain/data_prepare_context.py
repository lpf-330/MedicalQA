# -*- coding: utf-8 -*-
"""
数据准备Chain策略专属输入数据类
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class DataPrepareContextBody:
    """
    数据准备Chain策略专属输入数据类

    Attributes:
        monitoring_data: 监测数据（心率、血糖、灌注指数、血氧、睡眠、血压），每项包含4个时间维度
        user_profile: 用户档案（user_id, gender, birth_date, height, weight, past_medical_history, family_history, allergy_history, surgical_history, medical_compliance）
        task_id: 任务ID
    """
    monitoring_data: Dict[str, Any] = field(default_factory=dict)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    task_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "monitoring_data": self.monitoring_data,
            "user_profile": self.user_profile,
            "task_id": self.task_id
        }
