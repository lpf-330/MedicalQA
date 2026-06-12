# -*- coding: utf-8 -*-
"""
资源配置加载器

从配置文件加载全局资源配置。
统一通过 ConfigManager 加载配置。
"""

import logging

from .global_config import GlobalConfig
from src.utils.logger import log_arch_event

logger = logging.getLogger(__name__)


def load_global_config() -> GlobalConfig:
    """
    从配置文件加载全局资源配置

    使用统一配置管理器(ConfigManager)加载所有配置，并转换为GlobalConfig实例。

    Returns:
        GlobalConfig: 全局资源配置实例
    """
    from src.config.config_manager import get_config_manager

    logger.info("[load_global_config] 使用统一配置管理器加载全局配置")
    config_manager = get_config_manager()
    config = config_manager.to_global_config()
    log_arch_event(logger, component="ResourceConfigLoader", stage="CONFIG_LOAD", event="global_config_loaded", status="success", design_id="ARCH-0.1", resource_count=len(config.resource_configs), pool_count=len(config.pool_configs))
    logger.info(f"[load_global_config] 全局配置加载完成: resource_count={len(config.resource_configs)}, pool_count={len(config.pool_configs)}")
    return config
