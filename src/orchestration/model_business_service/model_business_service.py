"""
编排层模型业务服务接口

该模块定义了ModelBusinessService接口，是模型业务服务的核心抽象。

资源获取时机说明：
- 系统启动时：Pool根据min_idle配置预创建资源实例（处于空闲状态）
- 处理请求时：调用acquire()获取资源，处理完成后立即释放
- 禁止在初始化时获取资源并长期持有
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Any, Dict, List, Iterator

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
        ...         with GlobalResourceManager.acquire("vllm_model", "vllm_config") as handle:
        ...             client = handle.get_client()
        ...             return client.generate(messages)

    泛型参数：
        I: 模型调用的输入数据类型（通常是消息列表）
        O: 模型调用的输出数据类型（通常是生成的文本）
    """

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

    def release(self) -> None:
        """
        释放模型资源

        由于资源在每次调用后已释放，此方法保持兼容性但默认无需操作。
        子类可以覆盖此方法以实现特定的清理逻辑。

        Example:
            >>> model_service.release()
        """
        pass

    def __repr__(self) -> str:
        """返回ModelBusinessService的字符串表示"""
        return f"{self.__class__.__name__}()"
