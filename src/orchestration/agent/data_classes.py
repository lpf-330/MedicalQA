"""
编排层Agent模式数据类

该模块定义了AgentContext和AgentResult类，用于agent策略的输入输出数据封装。
"""

from dataclasses import dataclass, field
from typing import Generic, TypeVar, Optional

# 定义泛型类型变量
T = TypeVar('T')


@dataclass
class AgentContext(Generic[T]):
    """
    AgentContext类 - Agent输入数据容器类

    agent输入数据容器类，通过泛型组合agent业务策略专属context。
    用于封装agent策略执行所需的输入数据。

    使用示例：
        >>> # 定义agent策略专属context数据类
        >>> @dataclass
        ... class MyAgentContextBody:
        ...     query: str
        ...     user_profile: dict
        ...     conversation_history: list
        ...
        >>> # 创建AgentContext实例
        >>> body = MyAgentContextBody(query="咨询问题", user_profile={}, conversation_history=[])
        >>> agent_context = AgentContext(
        ...     session_id="session_001",
        ...     current_state="INIT",
        ...     body=body
        ... )

    Attributes:
        session_id: 任务id
        current_state: 记录当前Agent实例的执行状态
        body: agent策略的专属输入数据，泛型T由具体业务策略定义
    """

    session_id: str
    current_state: str = "INIT"
    body: Optional[T] = None

    def __post_init__(self) -> None:
        """
        初始化后验证参数

        Raises:
            ValueError: session_id或current_state为空时抛出
        """
        if not self.session_id:
            raise ValueError("session_id不能为空")
        if not self.current_state:
            raise ValueError("current_state不能为空")

    def to_dict(self) -> dict:
        """
        将AgentContext转换为字典格式

        Returns:
            dict: 包含session_id、current_state和body的字典

        Example:
            >>> agent_context.to_dict()
            {"session_id": "session_001", "current_state": "INIT", "body": {...}}
        """
        # 如果body有to_dict方法，则调用它
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
        """
        更新当前状态

        Args:
            new_state: 新的状态值

        Raises:
            ValueError: new_state为空时抛出
        """
        if not new_state:
            raise ValueError("new_state不能为空")
        self.current_state = new_state

    def __repr__(self) -> str:
        """返回AgentContext的字符串表示"""
        body_type = type(self.body).__name__ if self.body is not None else "None"
        return (
            f"AgentContext("
            f"session_id='{self.session_id}', "
            f"current_state='{self.current_state}', "
            f"body_type={body_type})"
        )


@dataclass
class AgentResult(Generic[T]):
    """
    AgentResult类 - Agent输出数据容器类

    agent输出数据容器类，通过泛型组合agent业务策略专属result。
    用于封装agent策略执行后的输出数据。

    使用示例：
        >>> # 定义agent策略专属result数据类
        >>> @dataclass
        ... class MyAgentResultData:
        ...     answer: str
        ...     confidence: float
        ...     sources: list
        ...
        >>> # 创建AgentResult实例
        >>> data = MyAgentResultData(answer="回答内容", confidence=0.95, sources=[])
        >>> agent_result = AgentResult(session_id="session_001", data=data)

    Attributes:
        session_id: 任务id
        data: agent策略的专属输出数据，泛型T由具体业务策略定义
    """

    session_id: str
    data: Optional[T] = None

    def __post_init__(self) -> None:
        """
        初始化后验证参数

        Raises:
            ValueError: session_id为空时抛出
        """
        if not self.session_id:
            raise ValueError("session_id不能为空")

    def to_dict(self) -> dict:
        """
        将AgentResult转换为字典格式

        Returns:
            dict: 包含session_id和data的字典

        Example:
            >>> agent_result.to_dict()
            {"session_id": "session_001", "data": {...}}
        """
        # 如果data有to_dict方法，则调用它
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
        """
        判断agent执行是否成功

        Returns:
            bool: 如果data不为None，返回True；否则返回False
        """
        return self.data is not None

    def __repr__(self) -> str:
        """返回AgentResult的字符串表示"""
        data_type = type(self.data).__name__ if self.data is not None else "None"
        return f"AgentResult(session_id='{self.session_id}', data_type={data_type})"
