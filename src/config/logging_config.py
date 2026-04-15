"""
日志配置类

提供日志相关的配置管理。
"""

from typing import Any, Dict, Optional
from pathlib import Path
from enum import Enum
import os
from datetime import datetime

from .base_config import BaseConfig


class LogLevel(Enum):
    """
    日志级别枚举
    
    定义标准的日志级别，用于控制日志输出的详细程度。
    """
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LoggingConfig(BaseConfig):
    """
    日志配置类
    
    设计思想：
    --------
    继承BaseConfig，提供日志相关的配置管理：
    1. 日志级别配置
    2. 日志格式配置
    3. 日志文件路径配置
    4. 日志轮转配置
    5. 日志输出目标配置（控制台、文件、远程）
    6. 会话日志文件生成（每次启动创建独立日志文件）
    
    支持从环境变量、配置文件、字典等多种方式加载配置。
    """
    
    def __init__(
        self,
        project_name: str = "MedicalQA",
        project_version: str = "1.0.0",
        environment: str = "development",
        debug: bool = False,
        config_path: Optional[str] = None,
        log_level: str = "INFO",
        log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        log_date_format: str = "%Y-%m-%d %H:%M:%S",
        log_file_path: Optional[str] = None,
        log_file_max_size: int = 10 * 1024 * 1024,
        log_file_backup_count: int = 5,
        log_to_console: bool = True,
        log_to_file: bool = True,
        log_encoding: str = "utf-8",
        session_log: bool = True
    ):
        """
        初始化日志配置
        
        Args:
            project_name: 项目名称
            project_version: 项目版本
            environment: 运行环境
            debug: 是否开启调试模式
            config_path: 配置文件路径
            log_level: 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
            log_format: 日志格式字符串
            log_date_format: 日志日期格式字符串
            log_file_path: 日志文件路径
            log_file_max_size: 日志文件最大大小（字节）
            log_file_backup_count: 日志文件备份数量
            log_to_console: 是否输出到控制台
            log_to_file: 是否输出到文件
            log_encoding: 日志文件编码
            session_log: 是否启用会话日志（每次启动创建独立日志文件）
        """
        super().__init__(
            project_name=project_name,
            project_version=project_version,
            environment=environment,
            debug=debug,
            config_path=config_path
        )
        
        self._log_level = self._parse_log_level(log_level)
        self._log_format = log_format
        self._log_date_format = log_date_format
        self._log_file_path = self._resolve_log_file_path(log_file_path)
        self._log_file_max_size = log_file_max_size
        self._log_file_backup_count = log_file_backup_count
        self._log_encoding = log_encoding
        self._log_to_console = log_to_console
        self._log_to_file = log_to_file
        self._session_log = session_log
        self._session_log_path: Optional[Path] = None
    
    @property
    def log_level(self) -> LogLevel:
        """获取日志级别"""
        return self._log_level
    
    @property
    def log_level_str(self) -> str:
        """获取日志级别字符串"""
        return self._log_level.value
    
    @property
    def log_format(self) -> str:
        """获取日志格式"""
        return self._log_format
    
    @property
    def log_date_format(self) -> str:
        """获取日志日期格式"""
        return self._log_date_format
    
    @property
    def log_file_path(self) -> Optional[Path]:
        """获取日志文件路径"""
        return self._log_file_path
    
    @property
    def session_log_path(self) -> Optional[Path]:
        """获取会话日志文件路径"""
        return self._session_log_path
    
    @property
    def log_file_max_size(self) -> int:
        """获取日志文件最大大小"""
        return self._log_file_max_size
    
    @property
    def log_file_backup_count(self) -> int:
        """获取日志文件备份数量"""
        return self._log_file_backup_count
    
    @property
    def log_to_console(self) -> bool:
        """获取是否输出到控制台"""
        return self._log_to_console
    
    @property
    def log_to_file(self) -> bool:
        """获取是否输出到文件"""
        return self._log_to_file
    
    @property
    def log_encoding(self) -> str:
        """获取日志文件编码"""
        return self._log_encoding
    
    @property
    def session_log(self) -> bool:
        """获取是否启用会话日志"""
        return self._session_log
    
    def create_session_log_file(self) -> Path:
        """
        创建会话日志文件
        
        每次启动时创建一个独立的日志文件，文件名包含启动时间戳。
        
        Returns:
            Path: 会话日志文件路径
        """
        log_dir = self.project_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_log_path = log_dir / f"medical_qa_{timestamp}.log"
        
        return self._session_log_path
    
    def _parse_log_level(self, level: str) -> LogLevel:
        """
        解析日志级别
        
        Args:
            level: 日志级别字符串
            
        Returns:
            LogLevel: 日志级别枚举值
            
        Raises:
            ValueError: 当日志级别无效时抛出
        """
        try:
            return LogLevel[level.upper()]
        except KeyError:
            raise ValueError(
                f"无效的日志级别: {level}. "
                f"有效级别: {[lvl.name for lvl in LogLevel]}"
            )
    
    def _resolve_log_file_path(self, log_file_path: Optional[str]) -> Optional[Path]:
        """
        解析日志文件路径
        
        Args:
            log_file_path: 日志文件路径字符串
            
        Returns:
            Optional[Path]: 解析后的日志文件路径
        """
        if log_file_path is None:
            return self.project_root / "logs" / "app.log"
        
        path = Path(log_file_path)
        
        if not path.is_absolute():
            path = self.project_root / path
        
        return path
    
    def set_log_level(self, level: str) -> None:
        """
        设置日志级别
        
        Args:
            level: 日志级别字符串
        """
        self._log_level = self._parse_log_level(level)
    
    def set_log_format(self, log_format: str) -> None:
        """
        设置日志格式
        
        Args:
            log_format: 日志格式字符串
        """
        self._log_format = log_format
    
    def set_log_file_path(self, log_file_path: str) -> None:
        """
        设置日志文件路径
        
        Args:
            log_file_path: 日志文件路径字符串
        """
        self._log_file_path = self._resolve_log_file_path(log_file_path)
    
    def update_from_env(self, prefix: str = "LOG_") -> None:
        """
        从环境变量更新日志配置
        
        Args:
            prefix: 环境变量前缀
        """
        super().update_from_env(prefix)
        
        env_mapping = {
            f"{prefix}LEVEL": ("_log_level", self._parse_log_level),
            f"{prefix}FORMAT": ("_log_format", str),
            f"{prefix}DATE_FORMAT": ("_log_date_format", str),
            f"{prefix}FILE_PATH": ("_log_file_path", self._resolve_log_file_path),
            f"{prefix}FILE_MAX_SIZE": ("_log_file_max_size", int),
            f"{prefix}FILE_BACKUP_COUNT": ("_log_file_backup_count", int),
            f"{prefix}TO_CONSOLE": ("_log_to_console", lambda x: x.lower() in ("true", "1", "yes")),
            f"{prefix}TO_FILE": ("_log_to_file", lambda x: x.lower() in ("true", "1", "yes")),
            f"{prefix}ENCODING": ("_log_encoding", str),
            f"{prefix}SESSION_LOG": ("_session_log", lambda x: x.lower() in ("true", "1", "yes")),
        }
        
        for env_key, (attr_name, converter) in env_mapping.items():
            env_value = os.getenv(env_key)
            if env_value is not None:
                try:
                    converted_value = converter(env_value)
                    setattr(self, attr_name, converted_value)
                except (ValueError, TypeError) as e:
                    print(f"警告: 无法从环境变量 {env_key} 更新配置: {e}")
    
    def validate(self) -> bool:
        """
        验证日志配置有效性
        
        Returns:
            bool: 配置是否有效
        """
        errors = []
        
        if not isinstance(self._log_level, LogLevel):
            errors.append(f"无效的日志级别类型: {type(self._log_level)}")
        
        if not self._log_format or not isinstance(self._log_format, str):
            errors.append("日志格式必须是非空字符串")
        
        if self._log_to_file:
            if self._log_file_path is None:
                errors.append("启用文件日志但未指定日志文件路径")
            else:
                log_dir = self._log_file_path.parent
                if not log_dir.exists():
                    try:
                        log_dir.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        errors.append(f"无法创建日志目录 {log_dir}: {e}")
            
            if self._log_file_max_size <= 0:
                errors.append(f"日志文件最大大小必须大于0: {self._log_file_max_size}")
            
            if self._log_file_backup_count < 0:
                errors.append(f"日志文件备份数量不能为负数: {self._log_file_backup_count}")
        
        if not self._log_to_console and not self._log_to_file:
            errors.append("至少需要启用一个日志输出目标（控制台或文件）")
        
        if errors:
            for error in errors:
                print(f"⚠️ 日志配置验证失败: {error}")
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        导出日志配置为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        base_dict = {
            "project_name": self._project_name,
            "project_version": self._project_version,
            "environment": self._environment,
            "debug": self._debug,
            "config_path": self._config_path,
        }
        
        log_dict = {
            "log_level": self._log_level.value,
            "log_format": self._log_format,
            "log_date_format": self._log_date_format,
            "log_file_path": str(self._log_file_path) if self._log_file_path else None,
            "log_file_max_size": self._log_file_max_size,
            "log_file_backup_count": self._log_file_backup_count,
            "log_to_console": self._log_to_console,
            "log_to_file": self._log_to_file,
            "log_encoding": self._log_encoding,
            "session_log": self._session_log,
            "session_log_path": str(self._session_log_path) if self._session_log_path else None,
        }
        
        return {**base_dict, **log_dict, **self._extra_config}
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        获取Python logging模块兼容的配置字典
        
        Returns:
            Dict[str, Any]: logging配置字典
        """
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": self._log_format,
                    "datefmt": self._log_date_format,
                }
            },
            "handlers": {},
            "root": {
                "level": self._log_level.value,
                "handlers": [],
            },
        }
        
        handlers = []
        
        if self._log_to_console:
            config["handlers"]["console"] = {
                "class": "logging.StreamHandler",
                "level": self._log_level.value,
                "formatter": "default",
                "stream": "ext://sys.stdout",
            }
            handlers.append("console")
        
        if self._log_to_file and self._log_file_path:
            config["handlers"]["file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "level": self._log_level.value,
                "formatter": "default",
                "filename": str(self._log_file_path),
                "maxBytes": self._log_file_max_size,
                "backupCount": self._log_file_backup_count,
                "encoding": self._log_encoding,
            }
            handlers.append("file")
        
        config["root"]["handlers"] = handlers
        
        return config
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"project_name='{self._project_name}', "
            f"log_level='{self._log_level.value}', "
            f"log_to_console={self._log_to_console}, "
            f"log_to_file={self._log_to_file}, "
            f"session_log={self._session_log})"
        )
