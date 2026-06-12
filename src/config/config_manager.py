# AI辅助生成：GLM-5，2026-04-15
# -*- coding: utf-8 -*-
"""
统一配置管理器

管理业务配置和资源配置的加载、验证、导出。
"""

import copy
import logging
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Dict, Optional, Set, Union, get_args, get_origin

from src.config.base_config import BaseResourceConfig, BusinessConfig
from src.config.config_loader import ConfigLoader
from src.config.global_config import GlobalConfig
from src.config.pool_config import PoolConfig
from src.utils.logger import log_arch_event

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    统一配置管理器

    负责管理业务配置和资源配置的加载、验证、导出。

    属性：
        business_configs: 业务配置字典
        resource_configs: 资源配置字典
        pool_configs: 资源池配置字典
    """

    _instance: Optional['ConfigManager'] = None

    def __new__(cls) -> 'ConfigManager':
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化配置管理器"""
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._business_configs: Dict[str, BusinessConfig] = {}
        self._resource_configs: Dict[str, BaseResourceConfig] = {}
        self._pool_configs: Dict[str, PoolConfig] = {}
        self._clinical_standards: Dict[str, Any] = {}
        self._load_errors: list[str] = []
        self._runtime_config: Dict[str, Any] = {}
        self._runtime_config_loaded = False

        self._initialized = True
        logger.info("[ConfigManager.__init__] 配置管理器初始化完成，单例模式")

    @property
    def business_configs(self) -> Dict[str, BusinessConfig]:
        """获取所有业务配置"""
        return self._business_configs

    @property
    def resource_configs(self) -> Dict[str, BaseResourceConfig]:
        """获取所有资源配置"""
        return self._resource_configs

    @property
    def pool_configs(self) -> Dict[str, PoolConfig]:
        """获取所有资源池配置"""
        return self._pool_configs

    @property
    def clinical_standards(self) -> Dict[str, Any]:
        """获取临床标准值配置"""
        return self._clinical_standards

    def load_all_configs(self) -> None:
        """
        加载所有配置

        按以下8步顺序加载：
        Step 1: 加载统一运行期配置文件
        Step 2: 扫描业务配置目录
        Step 3: 解析业务配置，收集所需的资源配置文件名
        Step 4: 资源配置去重
        Step 5: 加载并合并所需的资源配置
        Step 6: 确认资源池配置
        Step 7: 验证配置
        Step 8: 导出GlobalConfig
        """
        logger.info("[ConfigManager.load_all_configs] 开始加载所有配置（8步流程）...")
        log_arch_event(logger, component="ConfigManager", stage="CONFIG_LOAD", event="load_all_configs_start", status="start", design_id="ARCH-0.1")
        self._load_errors = []
        self._runtime_config = {}
        self._runtime_config_loaded = False

        logger.info("[CONFIG_STEP] step=1/8, 加载统一运行期配置文件")
        self._load_runtime_config()
        self._load_clinical_standards()
        logger.info("[ConfigManager.load_all_configs] Step 1 完成: 加载统一运行期配置文件")

        logger.info("[CONFIG_STEP] step=2/8, 扫描业务配置目录")
        self._load_business_configs()
        logger.info("[ConfigManager.load_all_configs] Step 2 完成: 扫描业务配置目录")

        logger.info("[CONFIG_STEP] step=3/8, 解析业务配置，收集所需的资源配置文件名")
        required_resources = self._get_required_resource_configs()
        logger.info(f"[ConfigManager.load_all_configs] Step 3 完成: 所需资源配置={required_resources}")

        logger.info("[CONFIG_STEP] step=4/8, 资源配置去重")
        logger.info(f"[ConfigManager.load_all_configs] Step 4 完成: 去重后资源配置数量={len(required_resources)}")

        logger.info("[CONFIG_STEP] step=5/8, 加载并合并所需的资源配置")
        self._load_resource_configs(required_resources)
        logger.info("[ConfigManager.load_all_configs] Step 5 完成: 加载并合并所需的资源配置")

        logger.info("[CONFIG_STEP] step=6/8, 确认资源池配置")
        logger.info("[ConfigManager.load_all_configs] Step 6: 确认资源池配置并执行完整性校验")
        for config_id, pool_config in self._pool_configs.items():
            resource_type = self._resource_configs.get(config_id).resource_type if config_id in self._resource_configs else "unknown"
            if config_id not in self._resource_configs:
                logger.warning(f"[PoolConfig Integrity] config_id={config_id} 缺少对应的资源配置")
            logger.info(f"[PoolConfig Summary] config_id={config_id}, resource_type={resource_type}, max_size={pool_config.max_size}, min_idle={pool_config.min_idle}, max_wait_time={pool_config.max_wait_time}, max_pending_requests={pool_config.max_pending_requests}, pre_create_check_enabled={pool_config.pre_create_check_enabled}, min_memory_mb={pool_config.min_memory_mb}, min_vram_mb={pool_config.min_vram_mb}, allow_dynamic_creation={pool_config.allow_dynamic_creation}, creation_timeout={pool_config.creation_timeout}")
        for config_name, business_config in self._business_configs.items():
            if hasattr(business_config, "resource_configs"):
                for res_config_id in business_config.resource_configs:
                    if res_config_id not in self._pool_configs:
                        logger.warning(f"[PoolConfig Integrity] 业务配置={config_name} 所需的资源 {res_config_id} 缺少资源池配置")
        logger.info(f"[ConfigManager.load_all_configs] Step 6 完成: 资源池配置数量={len(self._pool_configs)}, 完整性校验完成")

        logger.info("[CONFIG_STEP] step=7/8, 验证配置")
        is_valid = self.validate()
        if not is_valid:
            logger.error("[ConfigManager.load_all_configs] Step 7 完成: 配置验证失败")
            raise RuntimeError("配置验证失败")
        logger.info("[ConfigManager.load_all_configs] Step 7 完成: 配置验证通过")

        logger.info("[CONFIG_STEP] step=8/8, 导出GlobalConfig")
        global_config = self.to_global_config()
        logger.info(f"[ConfigManager.load_all_configs] Step 8 完成: GlobalConfig导出成功, resource_count={len(global_config.resource_configs)}, pool_count={len(global_config.pool_configs)}")
        log_arch_event(logger, component="ConfigManager", stage="CONFIG_LOAD", event="load_all_configs_end", status="success", design_id="ARCH-0.1", business_count=len(self._business_configs), resource_count=len(self._resource_configs), pool_count=len(self._pool_configs))
        logger.info("[ConfigManager.load_all_configs] 所有配置加载完成（8步流程结束）")

    def _load_runtime_config(self) -> None:
        """加载统一运行期配置文件。"""
        config_path = self._get_config_file_path()
        if not config_path.exists():
            self._runtime_config = {}
            self._runtime_config_loaded = True
            self._load_errors.append("runtime_config:missing")
            logger.error("[ConfigManager._load_runtime_config] 未找到统一运行期配置文件: config/application.yaml")
            return

        self._runtime_config = self._load_config_file(config_path)
        self._runtime_config_loaded = True
        log_arch_event(logger, component="ConfigManager", stage="CONFIG_LOAD", event="runtime_config_loaded", status="success", design_id="ARCH-0.1", config_path="config/application.yaml")
        logger.info("[ConfigManager._load_runtime_config] 已加载统一运行期配置文件: config/application.yaml")

    def _get_config_file_path(self) -> Path:
        project_root = Path(__file__).resolve().parents[2]
        return project_root / "config" / "application.yaml"

    def _load_clinical_standards(self) -> None:
        """加载临床标准值配置文件。"""
        project_root = Path(__file__).resolve().parents[2]
        clinical_path = project_root / "config" / "clinical_standards.yaml"
        if not clinical_path.exists():
            logger.warning("[ConfigManager._load_clinical_standards] 未找到临床标准值文件: config/clinical_standards.yaml，使用空配置")
            return

        config = ConfigLoader.load_from_yaml(clinical_path)
        if not isinstance(config, dict):
            logger.error("[ConfigManager._load_clinical_standards] 临床标准值文件格式错误，根节点必须为字典")
            return

        self._clinical_standards = config
        logger.info(f"[ConfigManager._load_clinical_standards] 已加载临床标准值配置: sections={list(config.keys())}")

    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        config = ConfigLoader.load_from_yaml(config_path)
        if not isinstance(config, dict):
            raise ValueError("运行期配置文件根节点必须为字典")
        return config

    def _load_business_configs(self) -> None:
        """加载所有业务配置"""
        from src.config.business import get_all_business_configs, load_business_config

        business_config_files = get_all_business_configs()
        logger.info(f"[ConfigManager._load_business_configs] 发现业务配置文件: {list(business_config_files.keys())}")
        logger.info(f"[CONFIG_SCAN] discovered_files={list(business_config_files.keys())}, file_count={len(business_config_files)}")

        for config_name in business_config_files:
            try:
                business_config = load_business_config(config_name)
                business_config = self._apply_runtime_config_overrides(
                    config_name,
                    business_config,
                    "business",
                    required=True,
                )
                self._business_configs[config_name] = business_config
                log_arch_event(logger, component="ConfigManager", stage="CONFIG_LOAD", event="business_config_loaded", status="success", design_id="ARCH-0.1", config_name=config_name)
                logger.info(f"[ConfigManager._load_business_configs] 业务配置加载成功: {config_name}, resource_configs={business_config.resource_configs if hasattr(business_config, 'resource_configs') else 'N/A'}")
                logger.info(f"[CONFIG_LOAD] business_config={config_name}, resource_configs={business_config.resource_configs if hasattr(business_config, 'resource_configs') else []}")
                logger.info(f"[CONFIG_SCAN] business_config={config_name}, parsed_resource_refs={business_config.resource_configs if hasattr(business_config, 'resource_configs') else []}")
            except Exception as e:
                self._load_errors.append(f"business_config:{config_name}:{type(e).__name__}")
                logger.error(f"[ConfigManager._load_business_configs] 业务配置加载失败: {config_name}, error_type={type(e).__name__}")

    def _get_required_resource_configs(self) -> Set[str]:
        """
        获取所有业务配置所需的资源配置文件名（去重）

        Returns:
            Set[str]: 资源配置文件名集合
        """
        required_resources = set()

        for config_name, business_config in self._business_configs.items():
            if hasattr(business_config, "resource_configs"):
                before_count = len(required_resources)
                required_resources.update(business_config.resource_configs)
                new_count = len(required_resources) - before_count
                logger.info(f"[CONFIG_REF] business_config={config_name}, references={business_config.resource_configs}, new_count={new_count}")
                logger.debug(f"[ConfigManager._get_required_resource_configs] 业务配置 {config_name} 贡献 {new_count} 个新资源配置")

        total_before = sum(len(business_config.resource_configs) for business_config in self._business_configs.values() if hasattr(business_config, "resource_configs"))
        total_after = len(required_resources)
        dedup_count = total_before - total_after
        logger.info(f"[CONFIG_DEDUP] 去重前总数={total_before}, 去重后总数={total_after}, 去重数量={dedup_count}, 去重后列表={required_resources}")

        resource_to_businesses = {}
        for b_name, b_config in self._business_configs.items():
            if hasattr(b_config, "resource_configs"):
                for res in b_config.resource_configs:
                    if res not in resource_to_businesses:
                        resource_to_businesses[res] = []
                    resource_to_businesses[res].append(b_name)
        for res, businesses in resource_to_businesses.items():
            if len(businesses) > 1:
                logger.info(f"[CONFIG_SHARE] resource_type={res}, shared_by={businesses}")

        logger.info(f"[ConfigManager._get_required_resource_configs] 去重后所需资源配置: {required_resources}")
        return required_resources

    def _load_resource_configs(self, required_resources: Set[str]) -> None:
        """
        加载所需的资源配置

        Args:
            required_resources: 所需的资源配置文件名集合
        """
        from src.config.resources import load_resource_config

        for config_name in required_resources:
            try:
                config_data = load_resource_config(config_name)

                if config_data.get("resource_config"):
                    resource_config = self._apply_runtime_config_overrides(
                        config_name,
                        config_data["resource_config"],
                        "resources",
                        required=True,
                    )
                    self._resource_configs[config_name] = resource_config
                    logger.debug(f"[ConfigManager._load_resource_configs] 资源配置已加载: {config_name}, resource_type={resource_config.resource_type}")

                if config_data.get("pool_config"):
                    pool_config = self._apply_runtime_config_overrides(
                        config_name,
                        config_data["pool_config"],
                        "resource_pools",
                        required=True,
                    )
                    self._pool_configs[config_name] = pool_config
                    logger.debug(f"[ConfigManager._load_resource_configs] 资源池配置已加载: {config_name}, max_size={pool_config.max_size}")

                log_arch_event(logger, component="ConfigManager", stage="CONFIG_LOAD", event="resource_config_loaded", status="success", design_id="ARCH-0.1", config_name=config_name)
                logger.info(f"[ConfigManager._load_resource_configs] 资源配置加载成功: {config_name}")
            except Exception as e:
                self._load_errors.append(f"resource_config:{config_name}:{type(e).__name__}")
                logger.error(f"[ConfigManager._load_resource_configs] 资源配置加载失败: {config_name}, error_type={type(e).__name__}")

    def _apply_runtime_config_overrides(self, config_name: str, config: Any, section_name: str, required: bool) -> Any:
        section = self._runtime_config.get(section_name, {})
        if section is None:
            section = {}
        if not isinstance(section, dict):
            self._load_errors.append(f"{section_name}:{config_name}:InvalidRuntimeConfig")
            logger.error(f"[ConfigManager._apply_runtime_config_overrides] 配置段格式错误: section={section_name}, config_name={config_name}")
            return config

        overrides = section.get(config_name)
        if overrides is None:
            if required and self._runtime_config_loaded:
                self._load_errors.append(f"{self._get_config_error_prefix(section_name)}:{config_name}:MissingRuntimeConfig")
                logger.error(f"[ConfigManager._apply_runtime_config_overrides] 缺少运行期配置: section={section_name}, config_name={config_name}")
            return config
        if not isinstance(overrides, dict):
            self._load_errors.append(f"{self._get_config_error_prefix(section_name)}:{config_name}:InvalidRuntimeConfig")
            logger.error(f"[ConfigManager._apply_runtime_config_overrides] 运行期配置格式错误: section={section_name}, config_name={config_name}")
            return config

        missing_fields = self._get_missing_runtime_fields(config, overrides)
        if required and missing_fields:
            self._load_errors.append(f"{self._get_config_error_prefix(section_name)}:{config_name}:MissingRuntimeFields")
            logger.error(f"[ConfigManager._apply_runtime_config_overrides] 运行期配置字段缺失: section={section_name}, config_name={config_name}, fields={missing_fields}")

        unknown_fields = self._get_unknown_runtime_fields(config, overrides)
        if unknown_fields:
            self._load_errors.append(f"{self._get_config_error_prefix(section_name)}:{config_name}:UnknownRuntimeFields")
            logger.error(f"[ConfigManager._apply_runtime_config_overrides] 运行期配置字段不存在: section={section_name}, config_name={config_name}, fields={unknown_fields}")

        invalid_type_fields = self._get_invalid_runtime_field_types(config, overrides)
        if invalid_type_fields:
            self._load_errors.append(f"{self._get_config_error_prefix(section_name)}:{config_name}:InvalidRuntimeFieldTypes")
            logger.error(f"[ConfigManager._apply_runtime_config_overrides] 运行期配置字段类型错误: section={section_name}, config_name={config_name}, fields={invalid_type_fields}")

        overridden_config = copy.copy(config)
        applied_fields = []
        for attr_name, value in overrides.items():
            if attr_name in unknown_fields or attr_name in invalid_type_fields:
                continue
            setattr(overridden_config, attr_name, value)
            applied_fields.append(attr_name)

        logger.info(f"[ConfigManager._apply_runtime_config_overrides] 已应用运行期配置: section={section_name}, config_name={config_name}, fields={applied_fields}")
        return overridden_config

    def _get_missing_runtime_fields(self, config: Any, overrides: Dict[str, Any]) -> list[str]:
        config_fields = self._get_runtime_config_fields(config)
        identity_fields = {"config_id", "resource_type", "business_id"}
        return [
            field_name
            for field_name in config_fields
            if field_name not in identity_fields and field_name not in overrides
        ]

    def _get_unknown_runtime_fields(self, config: Any, overrides: Dict[str, Any]) -> list[str]:
        config_fields = self._get_runtime_config_fields(config)
        if not config_fields:
            return []
        return [field_name for field_name in overrides if field_name not in config_fields]

    def _get_invalid_runtime_field_types(self, config: Any, overrides: Dict[str, Any]) -> list[str]:
        config_fields = self._get_runtime_config_fields(config)
        return [
            field_name
            for field_name, value in overrides.items()
            if field_name in config_fields and not self._is_runtime_value_type_valid(config_fields[field_name].type, value)
        ]

    def _get_runtime_config_fields(self, config: Any) -> Dict[str, Any]:
        if not is_dataclass(config):
            return {}
        return {field.name: field for field in fields(config)}

    def _is_runtime_value_type_valid(self, expected_type: Any, value: Any) -> bool:
        if expected_type is Any:
            return True
        origin = get_origin(expected_type)
        if origin in (Union, UnionType):
            return any(self._is_runtime_value_type_valid(arg, value) for arg in get_args(expected_type))
        if origin is list:
            return isinstance(value, list) and self._are_sequence_items_type_valid(expected_type, value)
        if origin is dict:
            return isinstance(value, dict) and self._are_mapping_items_type_valid(expected_type, value)
        if isinstance(expected_type, type):
            return isinstance(value, expected_type)
        return True

    def _are_sequence_items_type_valid(self, expected_type: Any, value: list[Any]) -> bool:
        item_types = get_args(expected_type)
        if not item_types:
            return True
        item_type = item_types[0]
        return all(self._is_runtime_value_type_valid(item_type, item) for item in value)

    def _are_mapping_items_type_valid(self, expected_type: Any, value: dict[Any, Any]) -> bool:
        key_type, value_type = (get_args(expected_type) + (Any, Any))[:2]
        return all(
            self._is_runtime_value_type_valid(key_type, item_key)
            and self._is_runtime_value_type_valid(value_type, item_value)
            for item_key, item_value in value.items()
        )

    def _get_config_error_prefix(self, section_name: str) -> str:
        if section_name == "resources":
            return "resource_config"
        if section_name == "resource_pools":
            return "pool_config"
        if section_name == "business":
            return "business_config"
        return section_name

    def get_business_config(self, business_id: str) -> Optional[BusinessConfig]:
        """
        获取指定业务配置

        Args:
            business_id: 业务ID

        Returns:
            BusinessConfig: 业务配置实例
        """
        config = self._business_configs.get(business_id)
        if config is None:
            if self._runtime_config_loaded:
                logger.warning(f"[ConfigManager.get_business_config] 业务配置不存在: business_id={business_id}")
        else:
            logger.debug(f"[ConfigManager.get_business_config] 获取业务配置: business_id={business_id}")
        return config

    def get_resource_config(self, config_id: str) -> Optional[BaseResourceConfig]:
        """
        获取指定资源配置

        Args:
            config_id: 资源配置ID

        Returns:
            BaseResourceConfig: 资源配置实例
        """
        config = self._resource_configs.get(config_id)
        if config is None:
            logger.warning(f"[ConfigManager.get_resource_config] 资源配置不存在: config_id={config_id}")
        else:
            logger.debug(f"[ConfigManager.get_resource_config] 获取资源配置: config_id={config_id}, resource_type={config.resource_type}")
        return config

    def get_pool_config(self, config_id: str) -> Optional[PoolConfig]:
        """
        获取指定资源池配置

        Args:
            config_id: 资源配置ID

        Returns:
            PoolConfig: 资源池配置实例
        """
        config = self._pool_configs.get(config_id)
        if config is None:
            logger.warning(f"[ConfigManager.get_pool_config] 资源池配置不存在: config_id={config_id}")
        else:
            logger.debug(f"[ConfigManager.get_pool_config] 获取资源池配置: config_id={config_id}, max_size={config.max_size}")
        return config

    def validate(self) -> bool:
        """
        验证所有配置有效性

        Returns:
            bool: 所有配置是否有效
        """
        errors = []
        for load_error in self._load_errors:
            errors.append(f"配置加载失败: {load_error}")
            logger.error(f"[ConfigManager.validate] 配置加载失败: error_ref={load_error}")

        for config_id, resource_config in self._resource_configs.items():
            if not resource_config.validate():
                errors.append(f"资源配置 {config_id} 验证失败")
                logger.error(f"[ConfigManager.validate] 资源配置验证失败: config_id={config_id}")
            else:
                logger.debug(f"[ConfigManager.validate] 资源配置验证通过: config_id={config_id}")

        for business_id, business_config in self._business_configs.items():
            if not business_config.validate():
                errors.append(f"业务配置 {business_id} 验证失败")
                logger.error(f"[ConfigManager.validate] 业务配置验证失败: business_id={business_id}")
            else:
                logger.debug(f"[ConfigManager.validate] 业务配置验证通过: business_id={business_id}")

        if errors:
            for error in errors:
                logger.error(f"[ConfigManager.validate] 配置验证失败: {error}")
            return False

        log_arch_event(logger, component="ConfigManager", stage="CONFIG_VALIDATE", event="validation_pass", status="success", design_id="ARCH-0.2", resource_count=len(self._resource_configs), business_count=len(self._business_configs))
        logger.info("[ConfigManager.validate] 所有配置验证通过")
        return True

    def to_global_config(self) -> GlobalConfig:
        """
        转换为GlobalConfig实例

        仅传递BaseResourceConfig原始数据，具体Config类的转换由
        GlobalResourceManager._convert_resource_config()完成，
        避免Config层反向依赖ResourceManager层。

        Returns:
            GlobalConfig: 全局资源配置实例
        """
        global_config = GlobalConfig()
        logger.info("[ConfigManager.to_global_config] 开始转换为GlobalConfig...")

        # 从运行期配置加载服务端参数
        server_config = self._runtime_config.get("server", {})
        if server_config:
            if "port" in server_config:
                global_config._server_port = int(server_config["port"])
            if "vram_sufficient_gb" in server_config:
                global_config._vram_sufficient_gb = float(server_config["vram_sufficient_gb"])
            if "warmup_timeout" in server_config:
                global_config._warmup_timeout = float(server_config["warmup_timeout"])
            if "timeout_keep_alive" in server_config:
                global_config._timeout_keep_alive = int(server_config["timeout_keep_alive"])

        for config_id, resource_config in self._resource_configs.items():
            resource_type = resource_config.resource_type
            global_config.add_resource_config(config_id, resource_config)
            logger.debug(f"[ConfigManager.to_global_config] 资源配置已添加: config_id={config_id}, resource_type={resource_type}")

        for config_id, pool_config in self._pool_configs.items():
            global_config.add_pool_config(config_id, pool_config)
            resource_type = self._resource_configs.get(config_id).resource_type if config_id in self._resource_configs else "unknown"
            logger.info(f"[PoolConfig] resource_type={resource_type}, max_size={pool_config.max_size}, min_idle={pool_config.min_idle}, max_wait_time={pool_config.max_wait_time}, max_pending_requests={pool_config.max_pending_requests}, pre_create_check_enabled={pool_config.pre_create_check_enabled}, min_memory_mb={pool_config.min_memory_mb}, min_vram_mb={pool_config.min_vram_mb}, allow_dynamic_creation={pool_config.allow_dynamic_creation}, creation_timeout={pool_config.creation_timeout}")
            logger.debug(f"[ConfigManager.to_global_config] 资源池配置已添加: config_id={config_id}, max_size={pool_config.max_size}")

        log_arch_event(logger, component="ConfigManager", stage="CONFIG_LOAD", event="to_global_config", status="success", design_id="ARCH-0.1", resource_count=len(global_config.resource_configs), pool_count=len(global_config.pool_configs))
        logger.info(f"[ConfigManager.to_global_config] GlobalConfig转换完成: resource_count={len(global_config.resource_configs)}, pool_count={len(global_config.pool_configs)}")
        return global_config

    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"ConfigManager("
            f"business_configs={len(self._business_configs)}, "
            f"resource_configs={len(self._resource_configs)}, "
            f"pool_configs={len(self._pool_configs)})"
        )


_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """
    获取配置管理器实例（单例模式）

    Returns:
        ConfigManager: 配置管理器实例
    """
    global _config_manager
    if _config_manager is None:
        logger.info("[get_config_manager] 创建ConfigManager实例并加载配置")
        config_manager = ConfigManager()
        config_manager.load_all_configs()
        _config_manager = config_manager
    return _config_manager
