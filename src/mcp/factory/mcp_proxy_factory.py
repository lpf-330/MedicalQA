# AI辅助生成：GLM-5，2026-04-15
"""
MCP代理层工厂类

该模块定义了MCPProxyFactory类，负责管理、缓存所有MCP代理tool。
"""

import logging
from typing import Dict, Optional, TYPE_CHECKING

from src.mcp.factory.tool_proxy_config import ProxyType, ToolProxyConfig
from src.utils.logger import log_arch_event

if TYPE_CHECKING:
    from src.mcp.proxy.interfaces import MCPTool

logger = logging.getLogger(__name__)


class MCPProxyFactory:
    """
    MCPProxyFactory类 - MCP代理工厂

    所有MCP代理tool的工厂，直接管理、缓存所有MCP代理tool。
    使用单例模式，确保全局唯一的工厂实例。

    职责：
        - 管理MCP代理tool的实例缓存
        - 管理MCP代理tool的配置缓存
        - 创建、获取、删除MCP代理tool实例

    使用示例：
        >>> factory = MCPProxyFactory.get_instance()
        >>> factory.init_factory(configs)
        >>> tool_proxy = factory.get_tool_proxy_instance("neo4j_tool")
        >>> result = tool_proxy.call("query", {"cypher": "..."})

    Attributes:
        _instance: 单例实例（类属性）
        _initialized: 初始化标志（类属性）
        _proxy_cache: MCP代理tool的实例缓存
        _configs: MCP代理tool的配置缓存
    """

    _instance: Optional['MCPProxyFactory'] = None
    _initialized: bool = False

    def __new__(cls) -> 'MCPProxyFactory':
        """
        创建单例实例

        Returns:
            MCPProxyFactory: 工厂实例
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """
        初始化MCP代理工厂

        使用_initialized标志避免重复初始化。
        """
        if MCPProxyFactory._initialized:
            return

        self._proxy_cache: Dict[str, 'MCPTool'] = {}
        self._configs: Dict[str, 'ToolProxyConfig'] = {}
        MCPProxyFactory._initialized = True
        logger.info("[MCPProxyFactory.__init__] MCP代理工厂初始化完成")

    @classmethod
    def get_instance(cls) -> 'MCPProxyFactory':
        """
        获取MCP代理工厂的单例实例

        Returns:
            MCPProxyFactory: 工厂实例
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        重置单例实例

        用于测试或需要重新初始化工厂的场景。
        """
        if cls._instance is not None:
            # 清理所有缓存的代理实例
            for proxy in cls._instance._proxy_cache.values():
                try:
                    proxy.release_tool(None)
                except Exception as e:
                    logger.debug(f"[MCPProxyFactory] 释放代理实例资源失败: {e}")

            cls._instance._proxy_cache.clear()
            cls._instance._configs.clear()

        cls._instance = None
        cls._initialized = False

    @property
    def proxy_cache(self) -> Dict[str, 'MCPTool']:
        """
        获取MCP代理tool的实例缓存（只读属性）

        Returns:
            Dict[str, MCPTool]: 代理实例缓存字典
        """
        return self._proxy_cache.copy()

    @property
    def configs(self) -> Dict[str, 'ToolProxyConfig']:
        """
        获取MCP代理tool的配置缓存（只读属性）

        Returns:
            Dict[str, ToolProxyConfig]: 配置缓存字典
        """
        return self._configs.copy()

    def initialize(self, configs: Dict[str, 'ToolProxyConfig']) -> None:
        """初始化MCP代理工厂 — 公共接口，替代_init_factory私有方法"""
        self._init_factory(configs)

    def _init_factory(self, configs: Dict[str, 'ToolProxyConfig']) -> None:
        """
        初始化MCP代理工厂

        通过传入的配置字典完成工厂的初始化加载。

        Args:
            configs: MCP代理tool的配置字典，key为tool_proxy_instance_id，
                    value为对应的ToolProxyConfig配置对象

        Raises:
            ParamException: 配置参数错误时抛出
            BusinessException: 初始化失败时抛出

        Example:
            >>> configs = {
            ...     "neo4j_tool": ToolProxyConfig(
            ...         proxy_type=ProxyType.FAKE,
            ...         connection_info={"host": "localhost", "port": 7687}
            ...     )
            ... }
            >>> factory._init_factory(configs)
        """
        if not configs:
            raise ValueError("配置字典不能为空")

        logger.info(f"[MCPProxyFactory._init_factory] 开始初始化MCP代理工厂: config_count={len(configs)}")

        for tool_proxy_instance_id, config in configs.items():
            if not tool_proxy_instance_id:
                raise ValueError("tool_proxy_instance_id不能为空")
            if not isinstance(config, ToolProxyConfig):
                raise ValueError(f"配置必须是ToolProxyConfig类型，当前类型为: {type(config)}")

        self._configs = configs.copy()
        log_arch_event(logger, component="MCPProxyFactory", stage="MCP", event="init_factory", status="success", design_id="ARCH-4.1", config_count=len(configs))
        logger.info(f"[MCPProxyFactory._init_factory] MCP代理工厂初始化完成: config_ids={list(self._configs.keys())}")

    def get_tool_proxy_instance(self, tool_proxy_instance_id: str) -> 'MCPTool':
        """
        获取MCP代理tool的实例

        从缓存中获取指定ID的MCP代理tool实例。
        如果实例不存在，则创建新实例。

        Args:
            tool_proxy_instance_id: MCP代理tool实例的ID

        Returns:
            MCPTool: MCP代理tool实例

        Raises:
            ParamException: 参数错误时抛出
            ResourceException: 资源访问错误时抛出

        Example:
            >>> tool_proxy = factory.get_tool_proxy_instance("neo4j_tool")
        """
        if not tool_proxy_instance_id:
            raise ValueError("tool_proxy_instance_id不能为空")

        if tool_proxy_instance_id in self._proxy_cache:
            log_arch_event(logger, component="MCPProxyFactory", stage="MCP", event="get_proxy_cache_hit", status="success", design_id="ARCH-4.1", instance_id=tool_proxy_instance_id)
            logger.debug(f"[MCPProxyFactory.get_tool_proxy_instance] 从缓存获取代理实例: id={tool_proxy_instance_id}")
            logger.info(f"[MCP_FACTORY_GET] instance_id={tool_proxy_instance_id}, cache_hit=True")
            return self._proxy_cache[tool_proxy_instance_id]

        logger.info(f"[MCPProxyFactory.get_tool_proxy_instance] 缓存未命中，创建新代理实例: id={tool_proxy_instance_id}")
        logger.info(f"[MCP_FACTORY_GET] instance_id={tool_proxy_instance_id}, cache_hit=False")
        return self.create_tool_proxy_instance(tool_proxy_instance_id)

    def create_tool_proxy_instance(self, tool_proxy_name: str) -> 'MCPTool':
        """
        创建一个MCP代理tool实例

        根据配置创建新的MCP代理tool实例，并添加到缓存中。

        Args:
            tool_proxy_name: MCP代理tool的名称（用于获取配置）

        Returns:
            MCPTool: 新创建的MCP代理tool实例

        Raises:
            ParamException: 参数错误或配置不存在时抛出
            ResourceException: 资源创建失败时抛出
            BusinessException: 业务逻辑错误时抛出

        注意：
            该方法会创建新实例并添加到缓存中。
            如果缓存中已存在同名实例，会抛出异常。

        Example:
            >>> tool_proxy = factory.create_tool_proxy_instance("neo4j_tool")
        """
        if not tool_proxy_name:
            raise ValueError("tool_proxy_name不能为空")

        if tool_proxy_name in self._proxy_cache:
            raise ValueError(f"MCP代理tool实例 '{tool_proxy_name}' 已存在")

        config = self._get_config(tool_proxy_name)
        logger.info(f"[PROXY_TYPE] tool_name={tool_proxy_name}, proxy_type={config.proxy_type}")
        logger.info(f"[MCPProxyFactory.create_tool_proxy_instance] 创建代理实例: name={tool_proxy_name}, proxy_type={config.proxy_type}")

        proxy_instance = None

        if config.proxy_type == ProxyType.FAKE:
            from src.mcp.proxy.Impl.neo4j_medical_proxy import Neo4jMedicalProxy
            from src.mcp.proxy.Impl.vector_retrieval_proxy import VectorRetrievalProxy
            from src.mcp.proxy.Impl.intent_classification_proxy import IntentClassificationProxy
            from src.mcp.proxy.Impl.ner_model_proxy import NerModelProxy

            tool_name = config.connection_info.get("tool_name", tool_proxy_name)

            if tool_name == "neo4j_medical":
                proxy_instance = Neo4jMedicalProxy(config.connection_info)
                logger.debug(f"[MCPProxyFactory.create_tool_proxy_instance] Neo4jMedicalProxy创建: tool_name={tool_name}")
            elif tool_name == "vector_retrieval":
                proxy_instance = VectorRetrievalProxy(config.connection_info)
                logger.debug(f"[MCPProxyFactory.create_tool_proxy_instance] VectorRetrievalProxy创建: tool_name={tool_name}")
            elif tool_name == "intent_classification":
                proxy_instance = IntentClassificationProxy(config.connection_info)
                logger.debug(f"[MCPProxyFactory.create_tool_proxy_instance] IntentClassificationProxy创建: tool_name={tool_name}")
            elif tool_name == "ner_model":
                proxy_instance = NerModelProxy(config.connection_info)
                logger.debug(f"[MCPProxyFactory.create_tool_proxy_instance] NerModelProxy创建: tool_name={tool_name}")
            else:
                logger.error(f"[MCPProxyFactory.create_tool_proxy_instance] 未知的FAKE代理工具名称: {tool_name}")
                raise ValueError(
                    f"未知的FAKE代理工具名称: {tool_name}。"
                    f"支持的工具名称: neo4j_medical, vector_retrieval, intent_classification, ner_model"
                )

            proxy_instance._init_tool()

        elif config.proxy_type == ProxyType.STANDARD:
            from src.mcp.proxy.Impl.mcp_standard_proxy import MCPStandardProxy
            proxy_instance = MCPStandardProxy(config.connection_info)
            logger.debug(f"[MCPProxyFactory.create_tool_proxy_instance] MCPStandardProxy创建: tool_name={config.connection_info.get('tool_name', tool_proxy_name)}")
            proxy_instance._init_tool()
        else:
            logger.error(f"[MCPProxyFactory.create_tool_proxy_instance] 未知的代理类型: {config.proxy_type}")
            raise ValueError(
                f"未知的代理类型: {config.proxy_type}。"
                f"支持的代理类型: STANDARD, FAKE"
            )

        self._proxy_cache[tool_proxy_name] = proxy_instance
        log_arch_event(logger, component="MCPProxyFactory", stage="MCP", event="create_proxy_instance", status="success", design_id="ARCH-4.1", name=tool_proxy_name, proxy_type=str(config.proxy_type))
        logger.info(f"[MCPProxyFactory.create_tool_proxy_instance] 代理实例创建并缓存成功: name={tool_proxy_name}, proxy_class={type(proxy_instance).__name__}")
        return proxy_instance

    def delete_tool_proxy_instance(self, tool_proxy_instance_id: str) -> None:
        """
        删除MCP代理tool实例

        从缓存中删除指定ID的MCP代理tool实例，并释放相关资源。

        Args:
            tool_proxy_instance_id: MCP代理tool实例的ID

        Raises:
            ParamException: 参数错误或实例不存在时抛出
            ResourceException: 资源释放失败时抛出

        Example:
            >>> factory.delete_tool_proxy_instance("neo4j_tool")
        """
        if not tool_proxy_instance_id:
            raise ValueError("tool_proxy_instance_id不能为空")

        if tool_proxy_instance_id not in self._proxy_cache:
            raise ValueError(f"MCP代理tool实例 '{tool_proxy_instance_id}' 不存在")

        proxy_instance = self._proxy_cache[tool_proxy_instance_id]
        logger.info(f"[MCPProxyFactory.delete_tool_proxy_instance] 删除代理实例: id={tool_proxy_instance_id}")

        try:
            proxy_instance.release_tool(None)
        except Exception as e:
            logger.warning(f"[MCPProxyFactory.delete_tool_proxy_instance] 释放代理实例资源时出错: id={tool_proxy_instance_id}, error={e}")

        del self._proxy_cache[tool_proxy_instance_id]
        logger.info(f"[MCPProxyFactory.delete_tool_proxy_instance] 代理实例已删除: id={tool_proxy_instance_id}")

    def _get_config(self, tool_proxy_instance_id: str) -> 'ToolProxyConfig':
        """
        获取MCP代理tool实例的配置

        从配置缓存中获取指定ID的MCP代理tool实例的配置。

        Args:
            tool_proxy_instance_id: MCP代理tool实例的ID

        Returns:
            ToolProxyConfig: MCP代理tool实例的配置

        Raises:
            ParamException: 参数错误或配置不存在时抛出

        Example:
            >>> config = factory._get_config("neo4j_tool")
        """
        if not tool_proxy_instance_id:
            raise ValueError("tool_proxy_instance_id不能为空")

        if tool_proxy_instance_id not in self._configs:
            raise ValueError(f"MCP代理tool实例 '{tool_proxy_instance_id}' 的配置不存在")

        return self._configs[tool_proxy_instance_id]

    def has_proxy_instance(self, tool_proxy_instance_id: str) -> bool:
        """
        检查是否存在指定ID的MCP代理tool实例

        Args:
            tool_proxy_instance_id: MCP代理tool实例的ID

        Returns:
            bool: 是否存在该实例
        """
        return tool_proxy_instance_id in self._proxy_cache

    def has_config(self, tool_proxy_instance_id: str) -> bool:
        """
        检查是否存在指定ID的MCP代理tool配置

        Args:
            tool_proxy_instance_id: MCP代理tool实例的ID

        Returns:
            bool: 是否存在该配置
        """
        return tool_proxy_instance_id in self._configs

    def get_all_proxy_instance_ids(self) -> list:
        """
        获取所有已创建的MCP代理tool实例ID列表

        Returns:
            list: 实例ID列表
        """
        return list(self._proxy_cache.keys())

    def get_all_config_ids(self) -> list:
        """
        获取所有已配置的MCP代理tool配置ID列表

        Returns:
            list: 配置ID列表
        """
        return list(self._configs.keys())

    def clear_all(self) -> None:
        """
        清空所有缓存的代理实例和配置

        释放所有代理实例的资源，并清空缓存。
        """
        # 释放所有代理实例
        for proxy_instance in self._proxy_cache.values():
            try:
                proxy_instance.release_tool(None)
            except Exception as e:
                logger.debug(f"[MCPProxyFactory] 释放代理实例资源失败: {e}")

        # 清空缓存
        self._proxy_cache.clear()
        self._configs.clear()

    def __repr__(self) -> str:
        """返回工厂的字符串表示"""
        return (
            f"MCPProxyFactory("
            f"proxy_count={len(self._proxy_cache)}, "
            f"config_count={len(self._configs)})"
        )
