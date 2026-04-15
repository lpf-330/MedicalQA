"""
通用工具类

提供常用的工具方法，如时间处理、字符串处理、文件操作等。
"""

import os
import json
import uuid
import hashlib
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path


class CommonUtils:
    """
    通用工具类
    
    提供常用的工具方法，如时间处理、字符串处理、文件操作等。
    所有方法都是静态方法，可以直接通过类名调用。
    """
    
    # ==================== 时间处理相关方法 ====================
    
    @staticmethod
    def get_current_timestamp(fmt: str = '%Y-%m-%d %H:%M:%S') -> str:
        """
        获取当前时间戳字符串
        
        Args:
            fmt: 时间格式，默认为 '%Y-%m-%d %H:%M:%S'
        
        Returns:
            格式化的时间戳字符串
        """
        return datetime.now().strftime(fmt)
    
    @staticmethod
    def get_current_timestamp_ms() -> int:
        """
        获取当前时间戳（毫秒）
        
        Returns:
            当前时间戳（毫秒）
        """
        return int(datetime.now().timestamp() * 1000)
    
    @staticmethod
    def parse_datetime(
        datetime_str: str,
        fmt: str = '%Y-%m-%d %H:%M:%S'
    ) -> datetime:
        """
        解析时间字符串为datetime对象
        
        Args:
            datetime_str: 时间字符串
            fmt: 时间格式
        
        Returns:
            datetime对象
        
        Raises:
            ValueError: 时间格式不正确时抛出
        """
        return datetime.strptime(datetime_str, fmt)
    
    @staticmethod
    def format_datetime(
        dt: datetime,
        fmt: str = '%Y-%m-%d %H:%M:%S'
    ) -> str:
        """
        格式化datetime对象为字符串
        
        Args:
            dt: datetime对象
            fmt: 时间格式
        
        Returns:
            格式化的时间字符串
        """
        return dt.strftime(fmt)
    
    @staticmethod
    def get_time_delta(
        start_time: datetime,
        end_time: datetime,
        unit: str = 'seconds'
    ) -> Union[int, float]:
        """
        计算时间差
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            unit: 时间单位，可选值：seconds, minutes, hours, days
        
        Returns:
            时间差值
        
        Raises:
            ValueError: 不支持的时间单位时抛出
        """
        delta = end_time - start_time
        
        if unit == 'seconds':
            return delta.total_seconds()
        elif unit == 'minutes':
            return delta.total_seconds() / 60
        elif unit == 'hours':
            return delta.total_seconds() / 3600
        elif unit == 'days':
            return delta.total_seconds() / 86400
        else:
            raise ValueError(f"不支持的时间单位: {unit}")
    
    # ==================== 字符串处理相关方法 ====================
    
    @staticmethod
    def generate_uuid() -> str:
        """
        生成UUID字符串
        
        Returns:
            UUID字符串
        """
        return str(uuid.uuid4())
    
    @staticmethod
    def generate_short_uuid() -> str:
        """
        生成短UUID字符串（去掉连字符）
        
        Returns:
            短UUID字符串
        """
        return uuid.uuid4().hex
    
    @staticmethod
    def md5_hash(text: str) -> str:
        """
        计算字符串的MD5哈希值
        
        Args:
            text: 输入字符串
        
        Returns:
            MD5哈希值（32位小写）
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    @staticmethod
    def sha256_hash(text: str) -> str:
        """
        计算字符串的SHA256哈希值
        
        Args:
            text: 输入字符串
        
        Returns:
            SHA256哈希值（64位小写）
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    @staticmethod
    def truncate_string(
        text: str,
        max_length: int,
        suffix: str = '...'
    ) -> str:
        """
        截断字符串
        
        Args:
            text: 输入字符串
            max_length: 最大长度
            suffix: 截断后的后缀
        
        Returns:
            截断后的字符串
        """
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def is_empty(text: Optional[str]) -> bool:
        """
        判断字符串是否为空
        
        Args:
            text: 输入字符串
        
        Returns:
            如果字符串为None或空字符串或只包含空白字符，返回True
        """
        return text is None or text.strip() == ''
    
    @staticmethod
    def is_not_empty(text: Optional[str]) -> bool:
        """
        判断字符串是否不为空
        
        Args:
            text: 输入字符串
        
        Returns:
            如果字符串不为空，返回True
        """
        return not CommonUtils.is_empty(text)
    
    # ==================== 文件操作相关方法 ====================
    
    @staticmethod
    def ensure_dir(dir_path: Union[str, Path]) -> Path:
        """
        确保目录存在，如果不存在则创建
        
        Args:
            dir_path: 目录路径
        
        Returns:
            Path对象
        """
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def read_file(
        file_path: Union[str, Path],
        encoding: str = 'utf-8'
    ) -> str:
        """
        读取文件内容
        
        Args:
            file_path: 文件路径
            encoding: 文件编码
        
        Returns:
            文件内容
        
        Raises:
            FileNotFoundError: 文件不存在时抛出
            IOError: 读取文件失败时抛出
        """
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    
    @staticmethod
    def write_file(
        file_path: Union[str, Path],
        content: str,
        encoding: str = 'utf-8',
        mode: str = 'w'
    ) -> None:
        """
        写入文件内容
        
        Args:
            file_path: 文件路径
            content: 文件内容
            encoding: 文件编码
            mode: 写入模式，'w'为覆盖，'a'为追加
        
        Raises:
            IOError: 写入文件失败时抛出
        """
        # 确保目录存在
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, mode, encoding=encoding) as f:
            f.write(content)
    
    @staticmethod
    def read_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        读取JSON文件
        
        Args:
            file_path: JSON文件路径
        
        Returns:
            JSON数据（字典）
        
        Raises:
            FileNotFoundError: 文件不存在时抛出
            json.JSONDecodeError: JSON格式不正确时抛出
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def write_json(
        file_path: Union[str, Path],
        data: Dict[str, Any],
        indent: int = 2,
        ensure_ascii: bool = False
    ) -> None:
        """
        写入JSON文件
        
        Args:
            file_path: JSON文件路径
            data: 要写入的数据
            indent: 缩进空格数
            ensure_ascii: 是否确保ASCII编码
        
        Raises:
            IOError: 写入文件失败时抛出
        """
        # 确保目录存在
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
    
    @staticmethod
    def file_exists(file_path: Union[str, Path]) -> bool:
        """
        判断文件是否存在
        
        Args:
            file_path: 文件路径
        
        Returns:
            文件存在返回True，否则返回False
        """
        return Path(file_path).exists()
    
    @staticmethod
    def delete_file(file_path: Union[str, Path]) -> bool:
        """
        删除文件
        
        Args:
            file_path: 文件路径
        
        Returns:
            删除成功返回True，文件不存在返回False
        """
        path = Path(file_path)
        if path.exists():
            path.unlink()
            return True
        return False
    
    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> int:
        """
        获取文件大小（字节）
        
        Args:
            file_path: 文件路径
        
        Returns:
            文件大小（字节）
        
        Raises:
            FileNotFoundError: 文件不存在时抛出
        """
        return Path(file_path).stat().st_size
    
    @staticmethod
    def list_files(
        dir_path: Union[str, Path],
        pattern: str = '*',
        recursive: bool = False
    ) -> List[Path]:
        """
        列出目录下的文件
        
        Args:
            dir_path: 目录路径
            pattern: 文件匹配模式
            recursive: 是否递归查找
        
        Returns:
            文件路径列表
        """
        path = Path(dir_path)
        if recursive:
            return list(path.rglob(pattern))
        else:
            return list(path.glob(pattern))
    
    # ==================== 数据处理相关方法 ====================
    
    @staticmethod
    def deep_copy_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        深拷贝字典
        
        Args:
            data: 原始字典
        
        Returns:
            深拷贝后的字典
        """
        return json.loads(json.dumps(data))
    
    @staticmethod
    def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并多个字典
        
        Args:
            *dicts: 要合并的字典
        
        Returns:
            合并后的字典
        """
        result = {}
        for d in dicts:
            result.update(d)
        return result
    
    @staticmethod
    def get_nested_value(
        data: Dict[str, Any],
        keys: str,
        default: Any = None
    ) -> Any:
        """
        获取嵌套字典中的值
        
        Args:
            data: 字典数据
            keys: 键路径，用点号分隔，如 'a.b.c'
            default: 默认值
        
        Returns:
            找到的值或默认值
        """
        keys_list = keys.split('.')
        value = data
        for key in keys_list:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
