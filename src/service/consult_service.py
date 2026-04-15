"""
服务层健康咨询服务模块

该模块定义了ConsultService类，是健康咨询业务的服务类。
"""

import logging
import time
from typing import TYPE_CHECKING, TypeVar, TypeAlias, Any

if TYPE_CHECKING:
    from src.orchestration.agent.agent import Agent
    from src.orchestration.agent.data_classes import AgentContext, AgentResult

logger = logging.getLogger(__name__)

T = TypeVar('T')

ConsultContext: TypeAlias = 'AgentContext[Any]'
ConsultResult: TypeAlias = 'AgentResult[Any]'


class ConsultService:
    """
    ConsultService类 - 健康咨询服务类

    健康咨询服务，组合agent策略和资源，调用agent入口方法。
    被ConsultController依赖使用。

    职责：
        - 接收Controller传递的请求上下文
        - 组合编排层的策略与资源
        - 驱动完整的健康咨询业务流程
        - 管理请求生命周期内的状态和资源

    使用示例：
        >>> # 创建agent实例（通常在应用启动时创建）
        >>> agent = Agent(strategy=consult_strategy, resources=agent_resource)
        >>> # 创建ConsultService实例
        >>> service = ConsultService(agent=agent)
        >>> # 处理健康咨询请求
        >>> context = AgentContext(session_id="session_001", current_state="INIT", body=consult_context_body)
        >>> result = service.process_consult(context)
        >>> print(result.data)

    Attributes:
        _agent: Agent实例，用于执行健康咨询编排逻辑
    """

    def __init__(self, agent: 'Agent[Any, Any]') -> None:
        """
        初始化ConsultService实例

        Args:
            agent: Agent实例，必须已配置好健康咨询策略和资源

        Raises:
            ValueError: agent为None时抛出
        """
        if agent is None:
            raise ValueError("agent不能为None")

        self._agent: 'Agent[Any, Any]' = agent
        logger.info("[ConsultService] 服务初始化完成")

    @property
    def agent(self) -> 'Agent[Any, Any]':
        """
        获取Agent实例（只读属性）

        Returns:
            Agent[Any, Any]: Agent实例
        """
        return self._agent

    def process_consult(self, context: ConsultContext) -> ConsultResult:
        """
        健康咨询服务方法

        组合agent策略和资源，调用agent入口方法。
        该方法接收健康咨询上下文，通过Agent执行编排逻辑，返回健康咨询结果。

        Args:
            context: 健康咨询上下文，包含session_id、current_state和body
                     body应包含用户问题、对话历史、用户档案等信息

        Returns:
            ConsultResult: 健康咨询结果，包含session_id和data
                          data应包含咨询结果、建议、置信度等信息

        Raises:
            ValueError: context为None或无效时抛出
            ParamException: 参数错误时抛出
            BusinessException: 业务逻辑错误时抛出
            ResourceException: 资源访问错误时抛出

        Example:
            >>> # 创建上下文
            >>> context = AgentContext(
            ...     session_id="session_001",
            ...     current_state="INIT",
            ...     body=consult_context_body
            ... )
            >>> # 处理咨询
            >>> result = service.process_consult(context)
            >>> print(result.is_success())
            True
        """
        start_time = time.time()
        
        if context is None:
            logger.error("[ConsultService] context为None")
            raise ValueError("context不能为None")

        if not hasattr(context, 'session_id') or not context.session_id:
            logger.error("[ConsultService] session_id为空")
            raise ValueError("context.session_id不能为空")

        logger.info(f"[ConsultService] 开始处理咨询: session_id={context.session_id}")

        result = self._agent.run(context)

        elapsed = time.time() - start_time
        logger.info(f"[ConsultService] 咨询处理完成: session_id={context.session_id}, elapsed={elapsed:.2f}s")

        return result

    def __repr__(self) -> str:
        """返回ConsultService的字符串表示"""
        return f"ConsultService(agent={self._agent})"
