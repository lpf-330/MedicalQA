"""
编排层Chain模式包

该包提供chain模式的设计规范与组合包装，包括：
- ChainContext: chain输入数据容器类
- ChainResult: chain输出数据容器类
- Chain: chain策略接口
"""

from src.orchestration.chain.data_classes import ChainContext, ChainResult
from src.orchestration.chain.chain import Chain

__all__ = ['ChainContext', 'ChainResult', 'Chain']
