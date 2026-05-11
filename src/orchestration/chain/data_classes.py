# AI辅助生成：GLM-5，2026-04-15
"""
编排层Chain模式数据类

该模块定义了ChainContext和ChainResult类，用于chain策略的输入输出数据封装。
"""

from dataclasses import dataclass
from typing import Generic, TypeVar, Optional

# 定义泛型类型变量
T = TypeVar('T')


@dataclass
class ChainContext(Generic[T]):
    """
    ChainContext类 - Chain输入数据容器类

    chain输入数据容器类，通过泛型组合chain业务策略专属context。
    用于封装chain策略执行所需的输入数据。

    使用示例：
        >>> # 定义chain策略专属context数据类
        >>> @dataclass
        ... class MyChainContextBody:
        ...     query: str
        ...     context: dict
        ...
        >>> # 创建ChainContext实例
        >>> body = MyChainContextBody(query="查询内容", context={})
        >>> chain_context = ChainContext(session_id="session_001", body=body)

    Attributes:
        session_id: 任务id
        body: chain策略的专属输入数据，泛型T由具体业务策略定义
    """

    session_id: str
    body: Optional[T] = None

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
        将ChainContext转换为字典格式

        Returns:
            dict: 包含session_id和body的字典

        Example:
            >>> chain_context.to_dict()
            {"session_id": "session_001", "body": {...}}
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
            "body": body_dict
        }

    def __repr__(self) -> str:
        """返回ChainContext的字符串表示"""
        body_type = type(self.body).__name__ if self.body is not None else "None"
        return f"ChainContext(session_id='{self.session_id}', body_type={body_type})"


@dataclass
class ChainResult(Generic[T]):
    """
    ChainResult类 - Chain输出数据容器类

    chain输出数据容器类，通过泛型组合chain业务策略专属result。
    用于封装chain策略执行后的输出数据。

    使用示例：
        >>> # 定义chain策略专属result数据类
        >>> @dataclass
        ... class MyChainResultData:
        ...     answer: str
        ...     confidence: float
        ...
        >>> # 创建ChainResult实例
        >>> data = MyChainResultData(answer="回答内容", confidence=0.95)
        >>> chain_result = ChainResult(session_id="session_001", data=data)

    Attributes:
        session_id: 任务id
        data: chain策略的专属输出数据，泛型T由具体业务策略定义
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
        将ChainResult转换为字典格式

        Returns:
            dict: 包含session_id和data的字典

        Example:
            >>> chain_result.to_dict()
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
        判断chain执行是否成功

        Returns:
            bool: 如果data不为None，返回True；否则返回False
        """
        return self.data is not None

    def __repr__(self) -> str:
        """返回ChainResult的字符串表示"""
        data_type = type(self.data).__name__ if self.data is not None else "None"
        return f"ChainResult(session_id='{self.session_id}', data_type={data_type})"
