# -*- coding: utf-8 -*-
"""
编排层Agent模式上下文数据类

该模块定义了AgentContext类，用于agent策略的输入数据封装。
"""

from dataclasses import dataclass
from typing import Generic, TypeVar, Optional

T = TypeVar('T')


@dataclass
class AgentContext(Generic[T]):
    """
    AgentContext类 - Agent输入数据容器类

    agent输入数据容器类，通过泛型组合agent业务策略专属context。
    用于封装agent策略执行所需的输入数据。

    Attributes:
        session_id: 任务id
        current_state: 记录当前Agent实例的执行状态
        body: agent策略的专属输入数据，泛型T由具体业务策略定义
    """

    session_id: str
    current_state: str = "INIT"
    body: Optional[T] = None

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id不能为空")
        if not self.current_state:
            raise ValueError("current_state不能为空")

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
            "current_state": self.current_state,
            "body": body_dict
        }

    def update_state(self, new_state: str) -> None:
        if not new_state:
            raise ValueError("new_state不能为空")
        self.current_state = new_state

    def __repr__(self) -> str:
        body_type = type(self.body).__name__ if self.body is not None else "None"
        return (
            f"AgentContext("
            f"session_id='{self.session_id}', "
            f"current_state='{self.current_state}', "
            f"body_type={body_type})"
        )
