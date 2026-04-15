"""
编排层模型业务服务接口

该模块定义了ModelBusinessService接口，是模型业务服务的核心抽象。
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Any, Dict, List

# 定义泛型类型变量
I = TypeVar('I')  # 输入数据类型
O = TypeVar('O')  # 输出数据类型


class ModelBusinessService(ABC, Generic[I, O]):
    """
    ModelBusinessService接口 - 模型业务服务接口

    每一个在不同业务下定义的模型服务调用类必须实现的接口。
    其实现类为agent策略和chain策略提供模型服务。

    重要说明：
        模型业务服务不是模型服务，它是模型服务根据不同业务场景所定制的服务。
        例如：
        - ConsultModelService: 健康咨询业务场景下的模型服务
        - ReportModelService: 健康报告业务场景下的模型服务

    职责：
        - 初始化模型
        - 调用模型服务
        - 释放模型资源

    使用示例：
        >>> class ConsultModelService(ModelBusinessService[List[Dict], str]):
        ...     def __init__(self, model_config: Dict[str, Any]):
        ...         self._model_config = model_config
        ...         self._model = None
        ...
        ...     def _init_model(self) -> None:
        ...         # 初始化模型实例
        ...         self._model = load_model(self._model_config)
        ...
        ...     def call_model(self, messages: List[Dict]) -> str:
        ...         if self._model is None:
        ...             raise ValueError("model未初始化")
        ...         # 调用模型生成回答
        ...         response = self._model.generate(messages)
        ...         return response
        ...
        ...     def release(self) -> None:
        ...         if self._model is not None:
        ...             self._model.close()
        ...             self._model = None

    生命周期：
        1. 创建ModelBusinessService实例
        2. 调用_init_model初始化模型
        3. 调用call_model使用模型服务
        4. 调用release释放模型资源

    泛型参数：
        I: 模型调用的输入数据类型（通常是消息列表）
        O: 模型调用的输出数据类型（通常是生成的文本）
    """

    @abstractmethod
    def _init_model(self) -> None:
        """
        初始化模型

        在ModelBusinessService实例创建后调用，用于初始化模型实例。
        该方法为私有方法，由agent策略或chain策略内部调用。

        Raises:
            ResourceException: 资源初始化失败时抛出
            ConfigException: 配置错误时抛出

        Example:
            >>> model_service._init_model()
        """
        pass

    @abstractmethod
    def call_model(self, messages: I) -> O:
        """
        使用模型服务

        通过模型实例调用模型服务，生成输出。
        输入类型和输出类型由实现该接口的类型的泛型决定。

        Args:
            messages: 模型调用的输入数据，通常是消息列表

        Returns:
            O: 模型调用的输出数据，通常是生成的文本

        Raises:
            ParamException: 参数错误时抛出
            BusinessException: 业务逻辑错误时抛出
            ResourceException: 资源访问错误时抛出

        Example:
            >>> messages = [
            ...     {"role": "system", "content": "你是一个健康咨询助手"},
            ...     {"role": "user", "content": "我最近感觉头痛，应该怎么办？"}
            ... ]
            >>> response = model_service.call_model(messages)
            >>> print(response)
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """
        释放模型资源

        在ModelBusinessService使用完毕后调用，用于释放模型资源。
        该方法应该是幂等的，即多次调用不会产生副作用。

        Example:
            >>> model_service.release()
        """
        pass

    def __repr__(self) -> str:
        """返回ModelBusinessService的字符串表示"""
        return f"{self.__class__.__name__}()"
