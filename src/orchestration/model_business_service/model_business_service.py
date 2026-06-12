# AI辅助生成：GLM-5，2026-04-15
"""
编排层模型业务服务接口

该模块定义了ModelBusinessService接口，是模型业务服务的核心抽象。

资源获取时机说明：
- 系统启动时：Pool根据min_idle配置预创建资源实例（处于空闲状态）
- 处理请求时：调用acquire()获取资源，处理完成后立即释放
- 禁止在初始化时获取资源并长期持有
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Optional

I = TypeVar('I')
O = TypeVar('O')


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

    资源获取时机：
        - 系统启动时：Pool根据min_idle配置预创建资源实例（处于空闲状态）
        - 处理请求时：调用acquire()获取资源，处理完成后立即释放
        - 禁止在初始化时获取资源并长期持有

    职责：
        - 调用模型服务
        - 管理资源生命周期（获取和释放）

    使用示例：
        >>> class ConsultModelService(ModelBusinessService[List[Dict], str]):
        ...     def call_model(self, messages: List[Dict]) -> str:
        ...         # 在处理请求时获取资源，处理完成后自动释放
        ...         with GlobalResourceManager.acquire(ResourceType.REASONING_MODEL, ConfigId.REASONING_CONFIG) as handle:
        ...             adapter = handle.get_client()
        ...             return adapter.generate(messages=messages)

    泛型参数：
        I: 模型调用的输入数据类型（通常是消息列表）
        O: 模型调用的输出数据类型（通常是生成的文本）
    """

    @abstractmethod
    def _init_model(self) -> None:
        """
        初始化模型

        在ModelBusinessService实例创建后调用，用于初始化模型相关配置。
        该方法为私有方法，由子类实现具体的初始化逻辑。

        重要说明：
            根据资源获取时机规范，不应在此方法中获取资源并长期持有。
            资源应在call_model方法中临时获取，使用后立即释放。

        子类实现示例：
            >>> def _init_model(self) -> None:
            ...     # 初始化模型配置
            ...     self._validate_config()
            ...     # 不在此获取资源，资源在call_model中临时获取

        Raises:
            ParamException: 配置参数错误时抛出
            ResourceException: 资源初始化失败时抛出
        """
        pass

    @abstractmethod
    def call_model(self, messages: I) -> O:
        """
        使用模型服务

        通过模型实例调用模型服务，生成输出。
        在此方法内部获取资源，处理完成后释放资源。

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
    def call_model_batch(self, prompts: List[str], max_tokens: Optional[int] = None, timeout: Optional[int] = None) -> List[str]:
        """
        批量调用模型服务

        将多个prompt一次性提交给模型引擎进行批量推理，
        利用推理引擎的continuous batching机制共享forward pass，减少引擎运行次数。

        Args:
            prompts: 输入提示列表，每个元素是一个独立的评估prompt
            max_tokens: 最大生成token数，默认从配置类读取
            timeout: 单个prompt超时时间（秒），默认从配置类读取

        Returns:
            List[str]: 每个prompt对应的生成结果列表

        Raises:
            ParamException: 参数错误时抛出
            BusinessException: 业务逻辑错误时抛出
            ResourceException: 资源访问错误时抛出

        Example:
            >>> prompts = ["评估维度1...", "评估维度2...", "评估维度3..."]
            >>> results = model_service.call_model_batch(prompts, max_tokens=256)
            >>> print(len(results))  # 3
        """
        pass

    def release_model(self) -> None:
        """
        释放模型资源

        由于资源在每次调用后已释放，此方法保持兼容性但默认无需操作。
        子类可以覆盖此方法以实现特定的清理逻辑。

        Example:
            >>> model_service.release_model()
        """
        pass

    def __repr__(self) -> str:
        """返回ModelBusinessService的字符串表示"""
        return f"{self.__class__.__name__}()"
