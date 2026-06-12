# AI辅助生成：GLM-5，2026-04-15
"""
编排层Agent模式容器类

该模块定义了Agent类，是agent的组合容器类。
"""

import logging
from typing import Generic, TypeVar, TYPE_CHECKING

from src.utils.logger import log_arch_event

if TYPE_CHECKING:
    from src.orchestration.agent.data_classes import AgentContext, AgentResult
    from src.orchestration.agent.agent_resource import AgentResource
    from src.orchestration.agent.agent_strategy import AgentStrategy

I = TypeVar('I')
O = TypeVar('O')

logger = logging.getLogger(__name__)


class Agent(Generic[I, O]):
    """
    Agent类 - Agent组合容器类

    agent的组合容器类，它本身不实现编排逻辑，它负责组合agent策略、资源。
    被服务层各业务的服务类依赖使用。

    职责：
        - 组合agent策略和资源
        - 提供统一的执行入口
        - 管理agent的执行流程

    使用示例：
        >>> # 创建agent策略
        >>> strategy = ConsultStrategy()
        >>> # 创建agent资源
        >>> resource = AgentResource(
        ...     state_machine=state_machine,
        ...     model_service=model_service,
        ...     chain_registry={"knowledge_chain": knowledge_chain},
        ...     tool_handlers={"neo4j_tool": neo4j_handler}
        ... )
        >>> # 创建Agent实例
        >>> agent = Agent(strategy=strategy, resources=resource)
        >>> # 执行agent
        >>> context = AgentContext(session_id="session_001", current_state="INIT", body=my_context_body)
        >>> result = agent.run(context)
        >>> print(result.data)

    Attributes:
        _strategy: agent策略
        _resources: agent策略专属资源类
    """

    def __init__(
        self,
        strategy: 'AgentStrategy[I, O]',
        resources: 'AgentResource'
    ) -> None:
        """
        初始化Agent实例

        Args:
            strategy: agent策略，必须实现AgentStrategy接口
            resources: agent策略专属资源类

        Raises:
            ValueError: strategy或resources为None时抛出
        """
        if strategy is None:
            raise ValueError("strategy不能为None")
        if resources is None:
            raise ValueError("resources不能为None")

        self._strategy: 'AgentStrategy[I, O]' = strategy
        self._resources: 'AgentResource' = resources
        logger.info(f"[Agent.__init__] Agent实例创建: strategy={type(strategy).__name__}, resources={resources}")

    @property
    def strategy(self) -> 'AgentStrategy[I, O]':
        """
        获取agent策略（只读属性）

        Returns:
            AgentStrategy[I, O]: agent策略
        """
        return self._strategy

    @property
    def resources(self) -> 'AgentResource':
        """
        获取agent资源（只读属性）

        Returns:
            AgentResource: agent策略专属资源类
        """
        return self._resources

    def run(self, context: 'AgentContext[I]') -> 'AgentResult[O]':
        """
        编排方法的执行入口

        内部实际上是调用agent策略的执行方法。
        输入数据为agent策略设置经AgentContext容器包装后的、agent策略专属context数据类。
        输出数据为经AgentResult容器包装后的、agent策略专属result数据类。

        Args:
            context: agent策略的输入数据容器，包含session_id、current_state和body

        Returns:
            AgentResult[O]: agent策略的输出数据容器，包含session_id和data

        Raises:
            ParamException: 参数错误时抛出
            BusinessException: 业务逻辑错误时抛出
            ResourceException: 资源访问错误时抛出

        Example:
            >>> context = AgentContext(session_id="session_001", current_state="INIT", body=my_context_body)
            >>> result = agent.run(context)
            >>> print(result.data)
        """
        if context is None:
            raise ValueError("context不能为None")

        logger.info(f"[Agent.run] Agent开始执行: strategy={type(self._strategy).__name__}, session_id={getattr(context, 'session_id', 'N/A')}, current_state={getattr(context, 'current_state', 'N/A')}")
        log_arch_event(
            logger,
            component="Agent",
            stage="ORCHESTRATION",
            event="agent_run_start",
            status="start",
            design_id="ARCH-3.1",
        )

        result = self._strategy.execute(context, self._resources)

        logger.info(f"[Agent.run] Agent执行完成: strategy={type(self._strategy).__name__}, session_id={getattr(result, 'session_id', 'N/A')}")
        logger.info(f"[AGENT_STRATEGY_EXEC] event=end, strategy_type={type(self._strategy).__name__}, session_id={getattr(result, 'session_id', 'N/A')}")
        return result

    def update_resources(self, resources: 'AgentResource') -> None:
        """
        更新agent资源

        Args:
            resources: 新的agent资源

        Raises:
            ValueError: resources为None时抛出
        """
        if resources is None:
            raise ValueError("resources不能为None")
        self._resources = resources

    def __repr__(self) -> str:
        """返回Agent的字符串表示"""
        strategy_name = type(self._strategy).__name__
        return (
            f"Agent("
            f"strategy={strategy_name}, "
            f"resources={self._resources})"
        )
