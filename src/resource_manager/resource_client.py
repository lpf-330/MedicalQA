# AI辅助生成：GLM-5，2026-04-15
"""
资源客户端接口模块

定义资源客户端的基本行为，包括获取资源类型和业务操作能力。
同时定义模型资源客户端子接口，为模型推理资源提供统一的客户端接口。
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AsyncIterator, Iterator, List, Optional

if TYPE_CHECKING:
    from src.resource_manager.resource import Resource


class ResourceClient(ABC):
    """
    资源客户端接口
    
    定义资源客户端的基本行为，所有资源客户端类必须实现此接口。
    资源客户端是对资源的封装，为业务层提供统一的资源访问接口，
    隐藏资源的底层实现细节。
    
    核心职责：
    - 提供资源类型的唯一标识
    - 为业务层提供统一的资源操作接口
    
    设计说明：
    - ResourceClient是业务层访问资源的统一入口
    - ResourceClient封装了Resource实例，提供业务友好的接口
    - ResourceClient由ResourceHandle持有，通过ResourceHandle进行生命周期管理
    """
    
    @abstractmethod
    def get_resource_type(self) -> str:
        """
        获取当前资源客户端对应的资源类型唯一标识

        返回的字符串用于GlobalResourceManager的资源类型匹配、注册校验与生命周期调度，
        是资源分类管理的核心标识。

        Returns:
            str: 资源类型的唯一标识字符串

        Example:
            >>> client.get_resource_type()
            'neo4j_database'
        """
        pass

    @abstractmethod
    def get_raw_resource(self) -> 'Resource':
        """
        获取原始资源实例

        返回客户端封装的底层Resource实例，用于需要直接访问资源的场景。

        Returns:
            Resource: 原始资源实例
        """
        pass
    

class ModelResourceClient(ResourceClient):
    """
    模型资源客户端子接口
    
    定义模型推理资源客户端的标准行为，继承ResourceClient基础接口。
    为业务层提供统一的模型推理操作接口，包括文本生成、流式生成、批量生成等。
    
    设计说明：
    - ModelResourceClient是ResourceClient的子接口，遵循接口隔离原则
    - 专门为模型推理资源提供客户端接口
    - 业务层通过ModelResourceClient接口依赖模型资源，而非具体实现类
    - ReasoningModelClient等具体客户端类实现此接口
    """
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        生成文本

        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数，默认从配置读取
            temperature: 温度参数，默认从配置读取
            top_p: top_p参数，默认从配置读取
            timeout: 超时时间（秒），默认从配置读取
            **kwargs: 其他生成参数

        Returns:
            生成的文本
        """
        pass
    
    @abstractmethod
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> List[str]:
        """
        批量生成文本

        Args:
            prompts: 输入提示列表
            max_tokens: 最大生成token数，默认从配置读取
            temperature: 温度参数，默认从配置读取
            top_p: top_p参数，默认从配置读取
            timeout: 单个prompt超时时间（秒），默认从配置读取
            **kwargs: 其他生成参数

        Returns:
            生成的文本列表
        """
        pass
    
    @abstractmethod
    def stream_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        流式生成文本

        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数，默认从配置读取
            temperature: 温度参数，默认从配置读取
            top_p: top_p参数，默认从配置读取
            **kwargs: 其他生成参数

        Yields:
            生成的文本片段
        """
        pass
    
    @abstractmethod
    async def async_stream_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        异步流式生成文本

        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数，默认从配置读取
            temperature: 温度参数，默认从配置读取
            top_p: top_p参数，默认从配置读取
            **kwargs: 其他生成参数

        Yields:
            生成的文本片段
        """
        pass
    
    @abstractmethod
    def is_model_loaded(self) -> bool:
        """
        检查模型是否已加载
        
        Returns:
            bool: 模型是否已加载
        """
        pass

    @abstractmethod
    def mark_engine_dead(self) -> None:
        """
        标记模型引擎为不可用状态

        当检测到引擎崩溃（如EngineDeadError）时调用此方法，
        后续所有模型调用将快速失败，避免反复重试已崩溃的引擎。

        编排层通过此接口方法调用，避免直接依赖适配层实现类。

        Note:
            此方法为类级别操作，标记后所有实例共享该状态。
        """
        pass
