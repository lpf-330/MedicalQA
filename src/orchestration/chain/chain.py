# AI辅助生成：GLM-5，2026-04-15
"""
编排层Chain模式接口

该模块定义了Chain接口，是chain策略的核心抽象。
"""

import logging
from abc import ABC, abstractmethod
from typing import Generator, Generic, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from src.orchestration.chain.data_classes import ChainContext, ChainResult

I = TypeVar('I')
O = TypeVar('O')

logger = logging.getLogger(__name__)


class Chain(ABC, Generic[I, O]):
    """
    Chain接口 - Chain策略接口

    chain策略接口，每一个chain业务策略必须实现的接口。
    chain策略是固定流程的业务策略，常作为agent策略中的一部分。

    与Agent策略的区别：
        - Agent策略：基于状态机的业务策略，支持状态转换和动态流程
        - Chain策略：固定流程的业务策略，执行固定的处理步骤

    使用示例：
        >>> class MyChain(Chain[MyContextBody, MyResultData]):
        ...     def execute(self, chain_context: ChainContext[MyContextBody]) -> ChainResult[MyResultData]:
        ...         # 实现具体的chain逻辑
        ...         result_data = MyResultData(...)
        ...         return ChainResult(session_id=chain_context.session_id, data=result_data)

    生命周期：
        1. 创建Chain实例
        2. 准备ChainContext输入数据
        3. 调用execute方法执行chain逻辑
        4. 获取ChainResult输出数据

    泛型参数：
        I: chain策略的专属输入数据类型（通过ChainContext包装）
        O: chain策略的专属输出数据类型（通过ChainResult包装）
    """

    @abstractmethod
    def execute(self, chain_context: 'ChainContext[I]') -> 'ChainResult[O]':
        """
        执行chain策略

        每个chain策略必须实现的方法，是chain策略的执行方法。
        输入数据为chain策略设置经ChainContext容器包装后的、chain策略专属context数据类。
        输出数据为经ChainResult容器包装后的、chain策略专属result数据类。

        Args:
            chain_context: chain策略的输入数据容器，包含session_id和body

        Returns:
            ChainResult[O]: chain策略的输出数据容器，包含session_id和data

        Raises:
            ParamException: 参数错误时抛出
            BusinessException: 业务逻辑错误时抛出
            ResourceException: 资源访问错误时抛出

        Example:
            >>> chain = MyChain()
            >>> context = ChainContext(session_id="session_001", body=my_context_body)
            >>> result = chain.execute(context)
            >>> print(result.data)
        """
        logger.info(f"[Chain.execute] Chain开始执行: chain_class={self.__class__.__name__}, session_id={getattr(chain_context, 'session_id', 'N/A')}")
        pass

    def execute_stream(self, chain_context: 'ChainContext[I]') -> Generator[str, None, 'ChainResult[O]']:
        """
        流式执行chain策略

        可选实现，支持流式输出中间结果。默认实现回退到execute()。
        子类如需流式支持应重写此方法。

        Args:
            chain_context: chain策略的输入数据容器

        Yields:
            str: 流式输出的中间文本片段

        Returns:
            ChainResult[O]: chain策略的最终输出数据容器
        """
        logger.info(f"[Chain.execute_stream] Chain回退到同步执行: chain_class={self.__class__.__name__}")
        result = self.execute(chain_context)
        yield ""
        return result

    def __repr__(self) -> str:
        """返回Chain的字符串表示"""
        return f"{self.__class__.__name__}()"
