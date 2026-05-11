# AI辅助生成：GLM-5，2026-04-15
"""
编排层Agent模式策略接口

该模块定义了AgentStrategy接口，是agent策略的核心抽象。
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from src.orchestration.agent.data_classes import AgentContext, AgentResult
    from src.orchestration.agent.agent_resource import AgentResource

# 定义泛型类型变量
I = TypeVar('I')  # 输入数据类型
O = TypeVar('O')  # 输出数据类型


class AgentStrategy(ABC, Generic[I, O]):
    """
    AgentStrategy接口 - Agent策略接口

    agent策略接口，每一个agent业务策略必须实现的接口。
    agent策略是基于状态机的业务策略，是编排的核心内容。

    与Chain策略的区别：
        - Agent策略：基于状态机的业务策略，支持状态转换和动态流程
        - Chain策略：固定流程的业务策略，执行固定的处理步骤

    使用示例：
        >>> class ConsultStrategy(AgentStrategy[ConsultContextBody, ConsultResultData]):
        ...     def execute(
        ...         self,
        ...         context: AgentContext[ConsultContextBody],
        ...         resource: AgentResource
        ...     ) -> AgentResult[ConsultResultData]:
        ...         # 实现具体的agent策略逻辑
        ...         # 使用状态机管理状态转换
        ...         # 使用模型服务调用AI模型
        ...         # 使用chain执行固定流程
        ...         # 使用tool handlers调用工具
        ...         result_data = ConsultResultData(...)
        ...         return AgentResult(session_id=context.session_id, data=result_data)

    生命周期：
        1. 创建AgentStrategy实例
        2. 准备AgentContext输入数据和AgentResource资源
        3. 调用execute方法执行agent策略
        4. 获取AgentResult输出数据

    泛型参数：
        I: agent策略的专属输入数据类型（通过AgentContext包装）
        O: agent策略的专属输出数据类型（通过AgentResult包装）
    """

    @abstractmethod
    def execute(
        self,
        context: 'AgentContext[I]',
        resource: 'AgentResource'
    ) -> 'AgentResult[O]':
        """
        执行agent策略

        每个agent策略必须实现的执行方法。
        输入数据为agent策略设置经AgentContext容器包装后的、agent策略专属context数据类和agent资源类。
        输出数据为经AgentResult容器包装后的、agent策略专属result数据类。

        Args:
            context: agent策略的输入数据容器，包含session_id、current_state和body
            resource: agent策略的资源类，包含状态机、模型服务、chain实例、tool调用实例等

        Returns:
            AgentResult[O]: agent策略的输出数据容器，包含session_id和data

        Raises:
            ParamException: 参数错误时抛出
            BusinessException: 业务逻辑错误时抛出
            ResourceException: 资源访问错误时抛出

        Example:
            >>> strategy = ConsultStrategy()
            >>> context = AgentContext(session_id="session_001", current_state="INIT", body=my_context_body)
            >>> resource = AgentResource(state_machine=sm, model_service=ms, ...)
            >>> result = strategy.execute(context, resource)
            >>> print(result.data)
        """
        pass

    def __repr__(self) -> str:
        """返回AgentStrategy的字符串表示"""
        return f"{self.__class__.__name__}()"
