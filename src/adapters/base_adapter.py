# -*- coding: utf-8 -*-
"""
适配器基类

定义适配器的公共接口，所有适配器接口应继承此类。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

logger = logging.getLogger(__name__)


class BaseAdapter(ABC):
    """
    适配器基类
    
    定义适配器的公共接口，所有适配器接口应继承此类。
    提供适配器的通用方法和状态检查。
    """
    
    def __init__(self):
        """初始化适配器基类"""
        self._initialized: bool = False
        logger.debug(f"[BaseAdapter.__init__] 适配器基类初始化: adapter_class={self.__class__.__name__}")
    
    @abstractmethod
    def is_initialized(self) -> bool:
        """
        检查适配器是否已初始化
        
        Returns:
            bool: 是否已初始化
        """
        pass
    
    def get_adapter_info(self) -> Dict[str, Any]:
        """
        获取适配器信息
        
        Returns:
            Dict[str, Any]: 适配器信息字典
        """
        return {
            "adapter_type": self.__class__.__name__,
            "initialized": self._initialized
        }
    
    def _set_initialized(self, status: bool) -> None:
        """
        设置初始化状态
        
        Args:
            status: 初始化状态
        """
        self._initialized = status
        logger.debug(f"[BaseAdapter._set_initialized] 适配器初始化状态变更: adapter_class={self.__class__.__name__}, initialized={status}")
