# AI辅助生成：GLM-5，2026-04-15
"""
MCP代理层工具方法数据类模块

本模块定义了tool方法属性类ToolMethod。
"""

from dataclasses import dataclass, field
from typing import List, Any

from .method_param import MethodParam


@dataclass
class ToolMethod:
    """
    tool方法属性类
    
    存放tool方法的方法名、方法描述、参数列表、返回类型等属性。
    
    Attributes:
        name: tool方法名称
        description: tool方法描述
        params: tool方法参数列表
        return_type: tool方法返回类型
    """
    
    name: str
    description: str
    params: List[MethodParam] = field(default_factory=list)
    return_type: Any = None  # 使用Any代替Class<?>，因为Python中类型可以是任何类型
    
    def __repr__(self) -> str:
        """返回方法对象的字符串表示"""
        return (f"ToolMethod(name='{self.name}', description='{self.description}', "
                f"params_count={len(self.params)}, return_type={self.return_type})")
