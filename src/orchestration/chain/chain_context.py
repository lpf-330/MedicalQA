# -*- coding: utf-8 -*-
"""
编排层Chain模式上下文数据类

该模块定义了ChainContext类，用于chain策略的输入数据封装。
"""

from dataclasses import dataclass
from typing import Generic, TypeVar, Optional

T = TypeVar('T')


@dataclass
class ChainContext(Generic[T]):
    """
    ChainContext类 - Chain输入数据容器类

    chain输入数据容器类，通过泛型组合chain业务策略专属context。
    用于封装chain策略执行所需的输入数据。

    Attributes:
        session_id: 任务id
        body: chain策略的专属输入数据，泛型T由具体业务策略定义
    """

    session_id: str
    body: Optional[T] = None

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id不能为空")

    def to_dict(self) -> dict:
        body_dict = None
        if self.body is not None:
            if hasattr(self.body, 'to_dict'):
                body_dict = self.body.to_dict()
            elif hasattr(self.body, '__dict__'):
                body_dict = self.body.__dict__
            else:
                body_dict = self.body

        return {
            "session_id": self.session_id,
            "body": body_dict
        }

    def __repr__(self) -> str:
        body_type = type(self.body).__name__ if self.body is not None else "None"
        return f"ChainContext(session_id='{self.session_id}', body_type={body_type})"
