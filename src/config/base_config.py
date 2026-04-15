"""
基础配置类

提供项目通用配置属性的基类，所有配置类都应继承此类。
"""

from typing import Any, Dict, Optional, List
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os


class BaseConfig(ABC):
    """
    基础配置类
    
    设计思想：
    --------
    作为所有配置类的基类，提供：
    1. 项目通用配置属性（项目名称、版本、环境等）
    2. 配置验证接口
    3. 配置导出接口
    4. 配置更新接口
    
    子类应实现：
    - validate(): 验证配置有效性
    - to_dict(): 导出配置为字典
    """
    
    def __init__(
        self,
        project_name: str = "MedicalQA",
        project_version: str = "1.0.0",
        environment: str = "development",
        debug: bool = False,
        config_path: Optional[str] = None
    ):
        """
        初始化基础配置
        
        Args:
            project_name: 项目名称
            project_version: 项目版本
            environment: 运行环境（development, testing, production）
            debug: 是否开启调试模式
            config_path: 配置文件路径
        """
        self._project_name = project_name
        self._project_version = project_version
        self._environment = environment
        self._debug = debug
        self._config_path = config_path
        self._project_root = self._get_project_root()
        self._extra_config: Dict[str, Any] = {}
    
    @property
    def project_name(self) -> str:
        """获取项目名称"""
        return self._project_name
    
    @property
    def project_version(self) -> str:
        """获取项目版本"""
        return self._project_version
    
    @property
    def environment(self) -> str:
        """获取运行环境"""
        return self._environment
    
    @property
    def debug(self) -> bool:
        """获取调试模式状态"""
        return self._debug
    
    @property
    def config_path(self) -> Optional[str]:
        """获取配置文件路径"""
        return self._config_path
    
    @property
    def project_root(self) -> Path:
        """获取项目根目录"""
        return self._project_root
    
    @property
    def is_production(self) -> bool:
        """判断是否为生产环境"""
        return self._environment == "production"
    
    @property
    def is_development(self) -> bool:
        """判断是否为开发环境"""
        return self._environment == "development"
    
    @property
    def is_testing(self) -> bool:
        """判断是否为测试环境"""
        return self._environment == "testing"
    
    def _get_project_root(self) -> Path:
        """
        获取项目根目录
        
        Returns:
            Path: 项目根目录路径
        """
        # 从当前文件向上查找项目根目录（包含src目录的目录）
        current_path = Path(__file__).resolve()
        while current_path.parent != current_path:
            if (current_path / "src").exists():
                return current_path
            current_path = current_path.parent
        # 如果找不到，返回当前工作目录
        return Path.cwd()
    
    def set_extra_config(self, key: str, value: Any) -> None:
        """
        设置额外配置项
        
        Args:
            key: 配置项键名
            value: 配置项值
        """
        self._extra_config[key] = value
    
    def get_extra_config(self, key: str, default: Any = None) -> Any:
        """
        获取额外配置项
        
        Args:
            key: 配置项键名
            default: 默认值
            
        Returns:
            Any: 配置项值
        """
        return self._extra_config.get(key, default)
    
    def update_from_env(self, prefix: str = "") -> None:
        """
        从环境变量更新配置
        
        Args:
            prefix: 环境变量前缀
        """
        env_mapping = {
            f"{prefix}PROJECT_NAME": "_project_name",
            f"{prefix}PROJECT_VERSION": "_project_version",
            f"{prefix}ENVIRONMENT": "_environment",
            f"{prefix}DEBUG": "_debug",
        }
        
        for env_key, attr_name in env_mapping.items():
            env_value = os.getenv(env_key)
            if env_value is not None:
                if attr_name == "_debug":
                    # 处理布尔值
                    env_value = env_value.lower() in ("true", "1", "yes")
                setattr(self, attr_name, env_value)
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        从字典更新配置
        
        Args:
            config_dict: 配置字典
        """
        for key, value in config_dict.items():
            if hasattr(self, f"_{key}"):
                setattr(self, f"_{key}", value)
            else:
                self.set_extra_config(key, value)
    
    @abstractmethod
    def validate(self) -> bool:
        """
        验证配置有效性（抽象方法，子类必须实现）
        
        Returns:
            bool: 配置是否有效
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        导出配置为字典（抽象方法，子类必须实现）
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        pass
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"project_name='{self._project_name}', "
            f"version='{self._project_version}', "
            f"environment='{self._environment}', "
            f"debug={self._debug})"
        )
    
    def __str__(self) -> str:
        """字符串表示"""
        return self.__repr__()


@dataclass
class BaseResourceConfig:
    """
    资源配置基类
    
    设计思想：
    --------
    作为所有资源配置类的基类，提供：
    1. 资源配置ID（文件名作为唯一标识）
    2. 资源类型标识
    3. 资源连接参数
    4. 配置验证接口
    
    子类应实现：
    - validate(): 验证配置有效性
    - to_dict(): 导出配置为字典
    """
    
    config_id: str = ""
    resource_type: str = ""
    
    def validate(self) -> bool:
        """
        验证配置有效性
        
        Returns:
            bool: 配置是否有效
        """
        if not self.config_id:
            print("警告: config_id 不能为空")
            return False
        if not self.resource_type:
            print("警告: resource_type 不能为空")
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        导出配置为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        return {
            "config_id": self.config_id,
            "resource_type": self.resource_type,
        }
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"config_id='{self.config_id}', "
            f"resource_type='{self.resource_type}')"
        )


@dataclass
class BusinessConfig:
    """
    业务配置基类
    
    设计思想：
    --------
    作为所有业务配置类的基类，提供：
    1. 业务ID（文件名作为唯一标识）
    2. 所需的资源配置文件名列表
    3. 业务参数
    4. 配置验证接口
    
    子类应实现：
    - resource_configs: 指定所需的资源配置文件名列表
    - validate(): 验证配置有效性
    - to_dict(): 导出配置为字典
    
    注意：子类在定义字段时，必须确保business_id和resource_configs字段有默认值
    """
    
    business_id: str = field(default="")
    resource_configs: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """初始化后处理"""
        pass
    
    def validate(self) -> bool:
        """
        验证配置有效性
        
        Returns:
            bool: 配置是否有效
        """
        if not self.business_id:
            print("警告: business_id 不能为空")
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        导出配置为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        return {
            "business_id": self.business_id,
            "resource_configs": self.resource_configs,
        }
    
    def get_required_resources(self) -> List[str]:
        """
        获取所需的资源配置文件名列表
        
        Returns:
            List[str]: 资源配置文件名列表
        """
        return self.resource_configs
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"business_id='{self.business_id}', "
            f"resource_configs={self.resource_configs})"
        )
