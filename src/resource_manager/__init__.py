"""
资源管理层包

本包负责所有基础资源的统一管理，包括：
- 资源包装、统一生命周期管理和池化复用
- 对重要的外部框架或依赖进行适配对接
- 资源连接的创建、复用、销毁和监控
- 资源配置的统一管理和热更新
"""

from .resource import Resource
from .resource_config import ResourceConfig
from .resource_client import ResourceClient
from .resource_factory import ResourceFactory
from .resource_registry import ResourceRegistry
from .resource_pool import ResourcePool
from .pool_manager import ResourcePoolManager
from .resource_handle import ResourceHandle
from .global_resource_manager import GlobalResourceManager

__all__ = [
    'Resource',
    'ResourceConfig',
    'ResourceClient',
    'ResourceFactory',
    'ResourceRegistry',
    'ResourcePool',
    'ResourcePoolManager',
    'ResourceHandle',
    'GlobalResourceManager'
]
