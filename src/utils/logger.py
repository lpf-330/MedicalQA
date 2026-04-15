"""
日志工具类

提供统一的日志记录功能，支持不同日志级别、日志格式化等。
"""

import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime


class Logger:
    """
    日志工具类
    
    提供统一的日志记录功能，支持不同日志级别、日志格式化等。
    使用Python标准库logging模块实现。
    
    Attributes:
        name: 日志记录器名称
        logger: logging.Logger实例
        level: 日志级别
        formatter: 日志格式化器
        handlers: 日志处理器列表
    """
    
    # 日志级别映射
    LEVEL_MAP = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    # 默认日志格式
    DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 详细日志格式
    DETAILED_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    
    def __init__(
        self,
        name: str,
        level: str = 'INFO',
        log_file: Optional[str] = None,
        log_format: Optional[str] = None,
        console_output: bool = True
    ):
        """
        初始化日志记录器
        
        Args:
            name: 日志记录器名称
            level: 日志级别，可选值：DEBUG, INFO, WARNING, ERROR, CRITICAL
            log_file: 日志文件路径，如果为None则不输出到文件
            log_format: 日志格式，如果为None则使用默认格式
            console_output: 是否输出到控制台
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.level = self.LEVEL_MAP.get(level.upper(), logging.INFO)
        self.logger.setLevel(self.level)
        
        # 设置日志格式
        self.formatter = logging.Formatter(
            log_format if log_format else self.DEFAULT_FORMAT
        )
        
        # 清除已有的处理器
        self.logger.handlers.clear()
        
        # 添加控制台处理器
        if console_output:
            self._add_console_handler()
        
        # 添加文件处理器
        if log_file:
            self._add_file_handler(log_file)
    
    def _add_console_handler(self) -> None:
        """添加控制台日志处理器"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)
    
    def _add_file_handler(self, log_file: str) -> None:
        """
        添加文件日志处理器
        
        Args:
            log_file: 日志文件路径
        """
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(self.level)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """
        记录DEBUG级别日志
        
        Args:
            message: 日志消息
            **kwargs: 额外的上下文信息
        """
        if kwargs:
            message = f"{message} | Context: {kwargs}"
        self.logger.debug(message)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """
        记录INFO级别日志
        
        Args:
            message: 日志消息
            **kwargs: 额外的上下文信息
        """
        if kwargs:
            message = f"{message} | Context: {kwargs}"
        self.logger.info(message)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """
        记录WARNING级别日志
        
        Args:
            message: 日志消息
            **kwargs: 额外的上下文信息
        """
        if kwargs:
            message = f"{message} | Context: {kwargs}"
        self.logger.warning(message)
    
    def error(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """
        记录ERROR级别日志
        
        Args:
            message: 日志消息
            exc_info: 是否包含异常信息
            **kwargs: 额外的上下文信息
        """
        if kwargs:
            message = f"{message} | Context: {kwargs}"
        self.logger.error(message, exc_info=exc_info)
    
    def critical(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """
        记录CRITICAL级别日志
        
        Args:
            message: 日志消息
            exc_info: 是否包含异常信息
            **kwargs: 额外的上下文信息
        """
        if kwargs:
            message = f"{message} | Context: {kwargs}"
        self.logger.critical(message, exc_info=exc_info)
    
    def exception(self, message: str, **kwargs: Any) -> None:
        """
        记录异常日志（自动包含异常堆栈信息）
        
        Args:
            message: 日志消息
            **kwargs: 额外的上下文信息
        """
        if kwargs:
            message = f"{message} | Context: {kwargs}"
        self.logger.exception(message)
    
    def set_level(self, level: str) -> None:
        """
        设置日志级别
        
        Args:
            level: 日志级别，可选值：DEBUG, INFO, WARNING, ERROR, CRITICAL
        """
        self.level = self.LEVEL_MAP.get(level.upper(), logging.INFO)
        self.logger.setLevel(self.level)
        for handler in self.logger.handlers:
            handler.setLevel(self.level)
    
    def get_logger(self) -> logging.Logger:
        """
        获取底层的logging.Logger实例
        
        Returns:
            logging.Logger实例
        """
        return self.logger


# 全局日志记录器缓存
_loggers: Dict[str, Logger] = {}


def get_logger(
    name: str,
    level: str = 'INFO',
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    console_output: bool = True
) -> Logger:
    """
    获取或创建日志记录器（单例模式）
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径
        log_format: 日志格式
        console_output: 是否输出到控制台
    
    Returns:
        Logger实例
    """
    if name not in _loggers:
        _loggers[name] = Logger(
            name=name,
            level=level,
            log_file=log_file,
            log_format=log_format,
            console_output=console_output
        )
    return _loggers[name]
