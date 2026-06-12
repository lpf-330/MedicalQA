"""
配置加载工具类

提供从多种格式（YAML、JSON、环境变量等）加载配置的功能。
"""

from typing import Any, Dict, Optional, Type, TypeVar, Union
from pathlib import Path
import json
import logging
import os

from .base_config import BaseConfig
from .logging_config import LoggingConfig
from src.utils.logger import log_arch_event

T = TypeVar('T', bound=BaseConfig)

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    配置加载工具类
    
    设计思想：
    --------
    提供统一的配置加载接口，支持：
    1. 从YAML文件加载配置
    2. 从JSON文件加载配置
    3. 从环境变量加载配置
    4. 从字典加载配置
    5. 配置合并与覆盖
    
    配置加载优先级（从低到高）：
    1. 默认值
    2. 配置文件
    3. 环境变量
    4. 运行时参数
    """
    
    # 支持的配置文件格式
    SUPPORTED_FORMATS = ['yaml', 'yml', 'json']
    
    @staticmethod
    def load_from_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        从YAML文件加载配置
        
        Args:
            file_path: YAML文件路径
            
        Returns:
            Dict[str, Any]: 配置字典
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不支持
            ImportError: 缺少PyYAML依赖
        """
        file_path = Path(file_path)
        logger.debug(f"[ConfigLoader.load_from_yaml] 扫描YAML配置文件: {file_path}")
        
        if not file_path.exists():
            logger.error(f"[ConfigLoader.load_from_yaml] 配置文件不存在: {file_path}")
            raise FileNotFoundError(f"配置文件不存在: {file_path}")
        
        if file_path.suffix.lower() not in ['.yaml', '.yml']:
            logger.error(f"[ConfigLoader.load_from_yaml] 不支持的YAML文件格式: {file_path.suffix}")
            raise ValueError(f"不支持的YAML文件格式: {file_path.suffix}")
        
        try:
            import yaml
        except ImportError:
            logger.error("[ConfigLoader.load_from_yaml] 缺少PyYAML依赖")
            raise ImportError(
                "缺少PyYAML依赖，请安装: pip install pyyaml"
            )
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        log_arch_event(logger, component="ConfigLoader", stage="CONFIG_LOAD", event="yaml_loaded", status="success", design_id="ARCH-0.1", file_path=str(file_path))
        logger.info(f"[ConfigLoader.load_from_yaml] YAML配置文件加载成功: {file_path}, keys={list((config or {}).keys())}")
        return config or {}
    
    @staticmethod
    def load_from_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        从JSON文件加载配置
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            Dict[str, Any]: 配置字典
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不支持
        """
        file_path = Path(file_path)
        logger.debug(f"[ConfigLoader.load_from_json] 扫描JSON配置文件: {file_path}")
        
        if not file_path.exists():
            logger.error(f"[ConfigLoader.load_from_json] 配置文件不存在: {file_path}")
            raise FileNotFoundError(f"配置文件不存在: {file_path}")
        
        if file_path.suffix.lower() != '.json':
            logger.error(f"[ConfigLoader.load_from_json] 不支持的JSON文件格式: {file_path.suffix}")
            raise ValueError(f"不支持的JSON文件格式: {file_path.suffix}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        log_arch_event(logger, component="ConfigLoader", stage="CONFIG_LOAD", event="json_loaded", status="success", design_id="ARCH-0.1", file_path=str(file_path))
        logger.info(f"[ConfigLoader.load_from_json] JSON配置文件加载成功: {file_path}, keys={list((config or {}).keys())}")
        return config or {}
    
    @staticmethod
    def load_from_env(prefix: str = "") -> Dict[str, Any]:
        """
        从环境变量加载配置
        
        Args:
            prefix: 环境变量前缀
            
        Returns:
            Dict[str, Any]: 配置字典
        """
        config = {}
        matched_count = 0
        
        for key, value in os.environ.items():
            if prefix and not key.startswith(prefix):
                continue
            
            config_key = key[len(prefix):].lower() if prefix else key.lower()
            parsed_value = ConfigLoader._parse_env_value(value)
            config[config_key] = parsed_value
            matched_count += 1
        
        logger.info(f"[ConfigLoader.load_from_env] 环境变量加载完成: prefix='{prefix}', matched_count={matched_count}")
        return config
    
    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """
        解析环境变量值
        
        Args:
            value: 环境变量值字符串
            
        Returns:
            Any: 解析后的值
        """
        # 尝试解析为布尔值
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # 尝试解析为整数
        try:
            return int(value)
        except ValueError:
            pass
        
        # 尝试解析为浮点数
        try:
            return float(value)
        except ValueError:
            pass
        
        # 尝试解析为JSON
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # 返回原始字符串
        return value
    
    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并多个配置字典（后面的配置覆盖前面的）
        
        Args:
            *configs: 配置字典列表
            
        Returns:
            Dict[str, Any]: 合并后的配置字典
        """
        result = {}
        
        for i, config in enumerate(configs):
            if config:
                result = ConfigLoader._deep_merge(result, config)
        
        logger.debug(f"[ConfigLoader.merge_configs] 合并{len(configs)}个配置字典完成, keys={list(result.keys())}")
        return result
    
    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度合并两个字典
        
        Args:
            base: 基础字典
            override: 覆盖字典
            
        Returns:
            Dict[str, Any]: 合并后的字典
        """
        result = base.copy()
        
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                # 递归合并嵌套字典
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                # 直接覆盖
                result[key] = value
        
        return result
    
    @staticmethod
    def load_config(
        config_class: Type[T],
        config_file: Optional[Union[str, Path]] = None,
        env_prefix: Optional[str] = None,
        **kwargs
    ) -> T:
        """
        加载配置并创建配置实例
        
        Args:
            config_class: 配置类（必须继承BaseConfig）
            config_file: 配置文件路径（支持YAML和JSON）
            env_prefix: 环境变量前缀
            **kwargs: 运行时参数
            
        Returns:
            T: 配置实例
            
        Raises:
            ValueError: 配置文件格式不支持
        """
        logger.info(f"[ConfigLoader.load_config] 开始加载配置: config_class={config_class.__name__}, config_file={config_file}, env_prefix={env_prefix}")
        
        file_config = {}
        if config_file:
            config_file = Path(config_file)
            
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                file_config = ConfigLoader.load_from_yaml(config_file)
            elif config_file.suffix.lower() == '.json':
                file_config = ConfigLoader.load_from_json(config_file)
            else:
                logger.error(f"[ConfigLoader.load_config] 不支持的配置文件格式: {config_file.suffix}")
                raise ValueError(
                    f"不支持的配置文件格式: {config_file.suffix}. "
                    f"支持的格式: {ConfigLoader.SUPPORTED_FORMATS}"
                )
        
        env_config = {}
        if env_prefix:
            env_config = ConfigLoader.load_from_env(env_prefix)
        
        merged_config = ConfigLoader.merge_configs(
            file_config,
            env_config,
            kwargs
        )
        
        instance = config_class(**merged_config)
        log_arch_event(logger, component="ConfigLoader", stage="CONFIG_LOAD", event="config_instance_created", status="success", design_id="ARCH-0.1", config_class=config_class.__name__)
        logger.info(f"[ConfigLoader.load_config] 配置实例创建成功: config_class={config_class.__name__}")
        return instance
    
    @staticmethod
    def load_logging_config(
        config_file: Optional[Union[str, Path]] = None,
        env_prefix: str = "LOG_",
        **kwargs
    ) -> LoggingConfig:
        """
        加载日志配置（便捷方法）
        
        Args:
            config_file: 配置文件路径
            env_prefix: 环境变量前缀
            **kwargs: 运行时参数
            
        Returns:
            LoggingConfig: 日志配置实例
        """
        return ConfigLoader.load_config(
            LoggingConfig,
            config_file=config_file,
            env_prefix=env_prefix,
            **kwargs
        )
    
    @staticmethod
    def save_to_yaml(config: Union[BaseConfig, Dict[str, Any]], file_path: Union[str, Path]) -> None:
        """
        保存配置到YAML文件
        
        Args:
            config: 配置对象或配置字典
            file_path: YAML文件路径
            
        Raises:
            ImportError: 缺少PyYAML依赖
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "缺少PyYAML依赖，请安装: pip install pyyaml"
            )
        
        file_path = Path(file_path)
        
        # 确保目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为字典
        if isinstance(config, BaseConfig):
            config_dict = config.to_dict()
        else:
            config_dict = config
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    @staticmethod
    def save_to_json(config: Union[BaseConfig, Dict[str, Any]], file_path: Union[str, Path]) -> None:
        """
        保存配置到JSON文件
        
        Args:
            config: 配置对象或配置字典
            file_path: JSON文件路径
        """
        file_path = Path(file_path)
        
        # 确保目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为字典
        if isinstance(config, BaseConfig):
            config_dict = config.to_dict()
        else:
            config_dict = config
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def validate_config_file(file_path: Union[str, Path]) -> bool:
        """
        验证配置文件是否有效
        
        Args:
            file_path: 配置文件路径
            
        Returns:
            bool: 文件是否有效
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return False
        
        if file_path.suffix.lower() not in ['.yaml', '.yml', '.json']:
            return False
        
        try:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                ConfigLoader.load_from_yaml(file_path)
            else:
                ConfigLoader.load_from_json(file_path)
            return True
        except Exception as e:
            logger.debug(f"[ConfigLoader] 配置文件验证失败: {e}")
            return False
    
    @staticmethod
    def get_config_template(format: str = 'yaml') -> str:
        """
        获取配置文件模板
        
        Args:
            format: 配置文件格式（yaml或json）
            
        Returns:
            str: 配置文件模板字符串
        """
        template = {
            "project_name": "MedicalQA",
            "project_version": "1.0.0",
            "environment": "development",
            "debug": False,
            "log_level": "INFO",
            "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "log_date_format": "%Y-%m-%d %H:%M:%S",
            "log_file_path": "logs/app.log",
            "log_file_max_size": 10485760,
            "log_file_backup_count": 5,
            "log_to_console": True,
            "log_to_file": True,
            "log_encoding": "utf-8"
        }
        
        if format.lower() in ['yaml', 'yml']:
            try:
                import yaml
                return yaml.dump(template, default_flow_style=False, allow_unicode=True)
            except ImportError:
                raise ImportError(
                    "缺少PyYAML依赖，请安装: pip install pyyaml"
                )
        elif format.lower() == 'json':
            return json.dumps(template, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的格式: {format}")
