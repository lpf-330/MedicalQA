# -*- coding: utf-8 -*-
"""
编排层Chain模式数据类（兼容层）

该模块从独立文件re-export ChainContext和ChainResult，保持向后兼容。
新代码应直接从chain_context和chain_result导入。
"""

from src.orchestration.chain.chain_context import ChainContext
from src.orchestration.chain.chain_result import ChainResult

__all__ = ['ChainContext', 'ChainResult']
