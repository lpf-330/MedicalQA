# -*- coding: utf-8 -*-
"""
Neo4j医疗知识图谱调用处理器

封装Neo4j医疗知识图谱的调用逻辑，实现ToolCallHandler接口。
"""

from typing import Any, Dict, List, Optional

from src.orchestration.tool_call_handler.tool_call_handler import ToolCallHandler
from src.mcp.proxy.interfaces import MCPTool


class Neo4jMedicalHandler(ToolCallHandler[str, Dict[str, Any]]):
    """
    Neo4j医疗知识图谱调用处理器类
    
    实现ToolCallHandler接口，为agent策略和chain策略提供Neo4j医疗知识图谱调用服务。
    
    属性：
        _tool: MCP代理tool实例
    """
    
    def __init__(self):
        """初始化Neo4j医疗知识图谱调用处理器"""
        self._tool: Optional[MCPTool] = None
    
    def _init_tool(self, tool: MCPTool) -> None:
        """
        初始化MCP代理tool
        
        Args:
            tool: MCP代理tool实例
        """
        self._tool = tool
        tool._init_tool()
    
    def call_tool(self, context: str) -> Dict[str, Any]:
        """
        调用tool服务
        
        Args:
            context: 查询内容（疾病名称或症状）
            
        Returns:
            医疗知识查询结果
        """
        if self._tool is None:
            raise RuntimeError("Tool not initialized, call _init_tool first")
        
        result = self._tool.call("get_disease_info", {"disease_name": context})
        return result
    
    def get_disease_info(self, disease_name: str) -> Optional[Dict[str, Any]]:
        """
        获取疾病信息
        
        Args:
            disease_name: 疾病名称
            
        Returns:
            疾病信息字典
        """
        if self._tool is None:
            raise RuntimeError("Tool not initialized")
        return self._tool.call("get_disease_info", {"disease_name": disease_name})
    
    def get_symptoms_by_disease(self, disease_name: str) -> List[str]:
        """
        获取疾病的症状列表
        
        Args:
            disease_name: 疾病名称
            
        Returns:
            症状名称列表
        """
        if self._tool is None:
            raise RuntimeError("Tool not initialized")
        return self._tool.call("get_symptoms_by_disease", {"disease_name": disease_name})
    
    def get_drugs_by_disease(self, disease_name: str) -> Dict[str, List[str]]:
        """
        获取疾病的药物信息
        
        Args:
            disease_name: 疾病名称
            
        Returns:
            药物信息字典
        """
        if self._tool is None:
            raise RuntimeError("Tool not initialized")
        return self._tool.call("get_drugs_by_disease", {"disease_name": disease_name})
    
    def get_foods_by_disease(self, disease_name: str) -> Dict[str, List[str]]:
        """
        获取疾病的饮食建议
        
        Args:
            disease_name: 疾病名称
            
        Returns:
            饮食建议字典
        """
        if self._tool is None:
            raise RuntimeError("Tool not initialized")
        return self._tool.call("get_foods_by_disease", {"disease_name": disease_name})
    
    def search_diseases_by_symptom(self, symptom_name: str) -> List[str]:
        """
        根据症状搜索可能的疾病
        
        Args:
            symptom_name: 症状名称
            
        Returns:
            疾病名称列表
        """
        if self._tool is None:
            raise RuntimeError("Tool not initialized")
        return self._tool.call("search_diseases_by_symptom", {"symptom_name": symptom_name})
    
    def release(self) -> None:
        """释放tool功能实例"""
        if self._tool is not None:
            self._tool.release_tool(None)
            self._tool = None
