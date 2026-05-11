# AI辅助生成：GLM-5，2026-04-15
"""
编排层Agent模式资源类

该模块定义了AgentResource类，为agent策略提供资源支持。
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.orchestration.state_machine.state_machine import StateMachine
    from src.orchestration.chain.chain import Chain
    from src.orchestration.tool_call_handler.tool_call_handler import ToolCallHandler
    from src.orchestration.model_business_service.model_business_service import ModelBusinessService


@dataclass
class AgentResource:
    """
    AgentResource类 - Agent资源类

    agent资源，为agent策略提供资源支持。
    包含agent策略执行所需的各种资源，如状态机、模型服务、chain实例、tool调用实例等。

    使用示例：
        >>> resource = AgentResource(
        ...     state_machine=state_machine,
        ...     model_service=model_service,
        ...     chain_registry={"knowledge_chain": knowledge_chain},
        ...     tool_handlers={"neo4j_tool": neo4j_handler}
        ... )

    Attributes:
        state_machine: 状态机，为agent策略提供状态管理支持
        model_service: 模型业务服务。注：模型业务服务不是模型服务，它是模型服务根据不同业务场景所定制的服务
        chain_registry: 注册所使用的Chain实例资源，key为chain名称，value为Chain实例
        tool_handlers: 注册所使用的Tool调用实例资源，key为tool名称，value为ToolCallHandler实例
    """

    state_machine: Optional['StateMachine'] = None
    model_service: Optional['ModelBusinessService'] = None
    chain_registry: Dict[str, 'Chain'] = field(default_factory=dict)
    tool_handlers: Dict[str, 'ToolCallHandler'] = field(default_factory=dict)

    def get_chain(self, chain_name: str) -> Optional['Chain']:
        """
        获取指定名称的Chain实例

        Args:
            chain_name: Chain实例的名称

        Returns:
            Optional[Chain]: Chain实例，如果不存在则返回None

        Example:
            >>> chain = resource.get_chain("knowledge_chain")
        """
        return self.chain_registry.get(chain_name)

    def has_chain(self, chain_name: str) -> bool:
        """
        检查是否存在指定名称的Chain实例

        Args:
            chain_name: Chain实例的名称

        Returns:
            bool: 是否存在该Chain实例
        """
        return chain_name in self.chain_registry

    def register_chain(self, chain_name: str, chain: 'Chain') -> None:
        """
        注册Chain实例

        Args:
            chain_name: Chain实例的名称
            chain: Chain实例

        Raises:
            ValueError: chain_name为空或chain为None时抛出

        Example:
            >>> resource.register_chain("knowledge_chain", knowledge_chain)
        """
        if not chain_name:
            raise ValueError("chain_name不能为空")
        if chain is None:
            raise ValueError("chain不能为None")

        self.chain_registry[chain_name] = chain

    def unregister_chain(self, chain_name: str) -> Optional['Chain']:
        """
        注销Chain实例

        Args:
            chain_name: Chain实例的名称

        Returns:
            Optional[Chain]: 被注销的Chain实例，如果不存在则返回None

        Example:
            >>> chain = resource.unregister_chain("knowledge_chain")
        """
        return self.chain_registry.pop(chain_name, None)

    def get_tool_handler(self, tool_name: str) -> Optional['ToolCallHandler']:
        """
        获取指定名称的ToolCallHandler实例

        Args:
            tool_name: ToolCallHandler实例的名称

        Returns:
            Optional[ToolCallHandler]: ToolCallHandler实例，如果不存在则返回None

        Example:
            >>> handler = resource.get_tool_handler("neo4j_tool")
        """
        return self.tool_handlers.get(tool_name)

    def has_tool_handler(self, tool_name: str) -> bool:
        """
        检查是否存在指定名称的ToolCallHandler实例

        Args:
            tool_name: ToolCallHandler实例的名称

        Returns:
            bool: 是否存在该ToolCallHandler实例
        """
        return tool_name in self.tool_handlers

    def register_tool_handler(self, tool_name: str, handler: 'ToolCallHandler') -> None:
        """
        注册ToolCallHandler实例

        Args:
            tool_name: ToolCallHandler实例的名称
            handler: ToolCallHandler实例

        Raises:
            ValueError: tool_name为空或handler为None时抛出

        Example:
            >>> resource.register_tool_handler("neo4j_tool", neo4j_handler)
        """
        if not tool_name:
            raise ValueError("tool_name不能为空")
        if handler is None:
            raise ValueError("handler不能为None")

        self.tool_handlers[tool_name] = handler

    def unregister_tool_handler(self, tool_name: str) -> Optional['ToolCallHandler']:
        """
        注销ToolCallHandler实例

        Args:
            tool_name: ToolCallHandler实例的名称

        Returns:
            Optional[ToolCallHandler]: 被注销的ToolCallHandler实例，如果不存在则返回None

        Example:
            >>> handler = resource.unregister_tool_handler("neo4j_tool")
        """
        return self.tool_handlers.pop(tool_name, None)

    def get_all_chain_names(self) -> list:
        """
        获取所有已注册的Chain实例名称列表

        Returns:
            list: Chain实例名称列表
        """
        return list(self.chain_registry.keys())

    def get_all_tool_names(self) -> list:
        """
        获取所有已注册的ToolCallHandler实例名称列表

        Returns:
            list: ToolCallHandler实例名称列表
        """
        return list(self.tool_handlers.keys())

    def clear_chains(self) -> None:
        """
        清空所有Chain实例
        """
        self.chain_registry.clear()

    def clear_tool_handlers(self) -> None:
        """
        清空所有ToolCallHandler实例
        """
        self.tool_handlers.clear()

    def clear_all(self) -> None:
        """
        清空所有资源
        """
        self.state_machine = None
        self.model_service = None
        self.chain_registry.clear()
        self.tool_handlers.clear()

    def to_dict(self) -> Dict[str, Any]:
        """
        将AgentResource转换为字典格式

        Returns:
            Dict[str, Any]: 包含资源信息的字典
        """
        return {
            "has_state_machine": self.state_machine is not None,
            "has_model_service": self.model_service is not None,
            "chain_count": len(self.chain_registry),
            "chain_names": list(self.chain_registry.keys()),
            "tool_count": len(self.tool_handlers),
            "tool_names": list(self.tool_handlers.keys())
        }

    def __repr__(self) -> str:
        """返回AgentResource的字符串表示"""
        return (
            f"AgentResource("
            f"has_state_machine={self.state_machine is not None}, "
            f"has_model_service={self.model_service is not None}, "
            f"chain_count={len(self.chain_registry)}, "
            f"tool_count={len(self.tool_handlers)})"
        )
