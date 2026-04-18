"""
资源接口模块

定义资源的基本行为，包括获取类型、获取最后使用时间、激活状态检查、激活、停用、销毁等。
"""

from abc import ABC, abstractmethod
from typing import Any


class Resource(ABC):
    """
    资源接口
    
    定义资源的基本行为，所有资源实例必须实现此接口。
    资源是资源管理层的核心概念，代表系统中可被管理和复用的基础资源。
    
    核心职责：
    - 提供资源的唯一类型标识
    - 管理资源的使用时间戳
    - 管理资源的活跃状态（空闲/活跃）
    - 提供资源的生命周期管理（激活、停用、销毁）
    """
    
    @abstractmethod
    def get_type(self) -> str:
        """
        获取资源的唯一类型标识
        
        返回与全局注册一致的资源类型字符串，用于资源分类、匹配对应资源池与工厂，
        是资源在全局体系中的核心身份标识。
        
        Returns:
            str: 资源的唯一类型标识字符串
            
        Example:
            >>> resource.get_type()
            'neo4j_database'
        """
        pass
    
    @abstractmethod
    def get_last_used_time(self) -> int:
        """
        获取资源的最后使用时间戳
        
        返回资源最后一次被使用的时间戳（单位为毫秒），用于资源池计算资源闲置时长，
        支撑空闲资源超时驱逐逻辑，保障资源池健康运行。
        
        Returns:
            int: 最后使用时间戳（毫秒级）
            
        Example:
            >>> resource.get_last_used_time()
            1713081600000
        """
        pass
    
    @abstractmethod
    def is_activate(self) -> bool:
        """
        校验资源当前的活跃状态
        
        返回布尔值：True表示资源正被业务使用（活跃），False表示资源处于空闲待复用状态。
        用于资源状态合法性校验，防止重复激活、非法访问已释放资源。
        
        Returns:
            bool: 资源是否处于活跃状态
                - True: 资源正在被使用（活跃状态）
                - False: 资源处于空闲状态（可被激活）
                
        Example:
            >>> resource.is_activate()
            False
        """
        pass
    
    @abstractmethod
    def activate(self) -> None:
        """
        激活资源
        
        将资源从空闲状态切换为活跃状态，标记资源为业务占用中，
        同步更新最后使用时间，完成资源从待用到在用的状态流转。
        
        注意：
        - 激活前应确保资源处于空闲状态
        - 激活后会更新最后使用时间戳
        - 激活后is_activate()应返回True
        
        Raises:
            ResourceException: 如果资源已被激活或资源状态异常
            
        Example:
            >>> resource.is_activate()
            False
            >>> resource.activate()
            >>> resource.is_activate()
            True
        """
        pass
    
    @abstractmethod
    def deactivate(self) -> None:
        """
        停用资源（释放回池，保持连接）
        
        将资源从活跃状态切换为空闲状态，为资源归还至资源池、等待复用做准备。
        
        语义：资源从活跃状态变为空闲状态，归还到资源池
        行为：仅标记状态，不断开连接
        场景：资源使用完毕，释放回资源池复用
        
        注意：
        - 停用前应确保资源处于活跃状态
        - 停用后is_activate()应返回False
        - **停用时不应断开连接，保持连接以便下次复用**
        
        Raises:
            ResourceException: 如果资源未被激活或资源状态异常
            
        Example:
            >>> resource.is_activate()
            True
            >>> resource.deactivate()
            >>> resource.is_activate()
            False
        """
        pass
    
    @abstractmethod
    def destroy(self) -> None:
        """
        销毁资源（彻底释放）
        
        执行资源的彻底释放操作，包括关闭连接、清理内存、释放系统资源等。
        用于资源池驱逐冗余/超时资源、系统停机时的全量资源回收，避免资源泄漏。
        
        语义：资源彻底销毁，从资源池移除
        行为：断开连接，释放所有资源
        场景：资源池关闭、资源过期、资源异常需销毁
        
        注意：
        - 销毁操作不可逆，销毁后资源实例不可再使用
        - 销毁时会断开连接，释放所有底层资源
        
        Raises:
            ResourceException: 如果资源销毁失败
            
        Example:
            >>> resource.destroy()
            >>> # 资源已被彻底销毁，不可再使用
        """
        pass
