# -*- coding: utf-8 -*-
"""
编排层Chain模式结果数据类

该模块定义了ChainResult类，用于chain策略的输出数据封装。
"""

from dataclasses import dataclass
from typing import Generic, TypeVar, Optional

T = TypeVar('T')


@dataclass
class ChainResult(Generic[T]):
    """
    ChainResult类 - Chain输出数据容器类

    chain输出数据容器类，通过泛型组合chain业务策略专属result。
    用于封装chain策略执行后的输出数据。

    Attributes:
        session_id: 任务id
        data: chain策略的专属输出数据，泛型T由具体业务策略定义
    """

    session_id: str
    data: Optional[T] = None

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id不能为空")

    def to_dict(self) -> dict:
        data_dict = None
        if self.data is not None:
            if hasattr(self.data, 'to_dict'):
                data_dict = self.data.to_dict()
            elif hasattr(self.data, '__dict__'):
                data_dict = self.data.__dict__
            else:
                data_dict = self.data

        return {
            "session_id": self.session_id,
            "data": data_dict
        }

    def is_success(self) -> bool:
        return self.data is not None

    def __repr__(self) -> str:
        data_type = type(self.data).__name__ if self.data is not None else "None"
        return f"ChainResult(session_id='{self.session_id}', data_type={data_type})"
