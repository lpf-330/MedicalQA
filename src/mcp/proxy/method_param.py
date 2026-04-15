"""
MCP代理层方法参数数据类模块

本模块定义了tool方法参数的属性类MethodParam。
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class MethodParam:
    """
    tool方法参数的属性类
    
    存放tool方法参数的参数名、参数描述、参数类型、参数必要性等属性。
    
    Attributes:
        name: tool方法参数名称
        description: tool方法参数描述
        type: tool方法参数类型
        required: tool方法参数是否为必需
    """
    
    name: str
    description: str
    type: Any  # 使用Any代替Class<?>，因为Python中类型可以是任何类型
    required: bool
    
    def __repr__(self) -> str:
        """返回参数对象的字符串表示"""
        return (f"MethodParam(name='{self.name}', description='{self.description}', "
                f"type={self.type}, required={self.required})")
