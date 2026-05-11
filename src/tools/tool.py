# AI辅助生成：GLM-5，2026-04-15
"""
Tool工具层核心接口

该模块定义了Tool接口，作为所有tool必须实现的统一接口。
Tool接口定义了工具的基本行为，包括初始化资源和释放资源。
"""

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """
    Tool接口 - 工具的基础接口

    作为所有tool必须实现的统一接口，定义了工具的基本行为。
    每个Tool封装一个独立的原子能力，如知识检索、内容校验等。

    重要说明：
        Tool工具层不包含模型调用相关的Tool。
        模型调用由编排层的ConsultModelService和ReportModelService负责。
        Tool层只保留真正的"工具"——即对外部系统的操作能力封装。

    生命周期：
        1. 创建Tool实例
        2. 调用_init_resource()初始化资源
        3. 使用Tool提供的功能
        4. 调用release_source()释放资源
    """

    @abstractmethod
    def _init_resource(self) -> None:
        """
        初始化所用资源

        在Tool实例创建后调用，用于初始化工具所需的各种资源。
        例如：建立数据库连接、加载模型、初始化客户端等。

        该方法为私有方法，由Tool内部或MCP代理层调用。

        Raises:
            ResourceException: 资源初始化失败时抛出
            BusinessException: 业务逻辑错误时抛出
        """
        pass

    @abstractmethod
    def release_source(self) -> None:
        """
        释放所用资源

        在Tool使用完毕后调用，用于释放工具占用的各种资源。
        例如：关闭数据库连接、释放模型内存、关闭客户端等。

        该方法为公共方法，由外部调用者（如MCP代理层）调用。

        注意：
            - 该方法应该是幂等的，即多次调用不会产生副作用
            - 即使资源未初始化，调用该方法也不应抛出异常

        Raises:
            ResourceException: 资源释放失败时抛出
        """
        pass

    def __enter__(self) -> 'Tool':
        """
        上下文管理器入口方法

        适配with语法，进入上下文时初始化资源。

        Returns:
            Tool: 当前Tool实例
        """
        self._init_resource()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        上下文管理器退出方法

        适配with语法，在上下文执行结束时自动释放资源。

        Args:
            exc_type: 异常类型
            exc_val: 异常值
            exc_tb: 异常堆栈信息
        """
        self.release_source()
