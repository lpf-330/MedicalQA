# -*- coding: utf-8 -*-
"""
Neo4j医疗知识图谱代理

封装Neo4j医疗知识图谱工具的MCP代理，提供统一的代理接口。
"""

import time
from typing import Any, Dict, Optional

from src.mcp.proxy.interfaces import MCPFakeProxy
from src.mcp.proxy.data_classes import ToolInfo, DirectConnectionInfo
from src.tools.neo4j_medical_tool import Neo4jMedicalTool


class Neo4jMedicalProxy(MCPFakeProxy):
    """
    Neo4j医疗知识图谱代理类
    
    封装Neo4j医疗知识图谱工具，实现MCPFakeProxy接口。
    提供高效的直连调用方式。
    
    属性：
        _config: 代理配置
        _tool: Neo4j医疗知识图谱工具实例
        _call_count: 调用次数
        _total_time: 总调用时间
        _error_count: 错误次数
        _mock_responses: 模拟响应字典
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Neo4j医疗知识图谱代理
        
        Args:
            config: 代理配置，包含：
                - uri: Neo4j连接URI
                - user: 用户名
                - password: 密码
        """
        self._config = config
        self._tool: Optional[Neo4jMedicalTool] = None
        self._call_count = 0
        self._total_time = 0.0
        self._error_count = 0
        self._mock_responses: Dict[str, Any] = {}
    
    def _init_tool(self) -> None:
        """初始化tool功能实例"""
        if self._tool is not None:
            return
        
        self._tool = Neo4jMedicalTool(
            uri=self._config.get("uri"),
            user=self._config.get("user"),
            password=self._config.get("password")
        )
        self._tool._init_resource()
    
    def release_tool(self, tool=None) -> None:
        """释放tool功能实例"""
        if self._tool is not None:
            self._tool.release_source()
            self._tool = None
    
    def call(self, method_name: str, params: Dict[str, Any]) -> Any:
        """
        调用tool功能实例的方法
        
        Args:
            method_name: 要调用的方法名称
            params: 方法参数字典
            
        Returns:
            方法调用的返回值
        """
        if self._tool is None:
            raise RuntimeError("Tool not initialized, call _init_tool first")
        
        if method_name in self._mock_responses:
            return self._mock_responses[method_name]
        
        start_time = time.time()
        try:
            method = getattr(self._tool, method_name, None)
            if method is None:
                raise AttributeError(f"Method {method_name} not found")
            
            result = method(**params)
            self._call_count += 1
            return result
        except Exception as e:
            self._error_count += 1
            raise e
        finally:
            self._total_time += time.time() - start_time
    
    def get_tool_info(self) -> ToolInfo:
        """获取tool功能实例的信息"""
        return ToolInfo(
            name="neo4j_medical_tool",
            description="Neo4j医疗知识图谱工具，提供疾病、症状、药物等医疗知识的查询功能",
            methods=[
                "get_disease_info",
                "get_symptoms_by_disease",
                "get_drugs_by_disease",
                "get_foods_by_disease",
                "get_checks_by_disease",
                "get_department_by_disease",
                "get_complications_by_disease",
                "get_cure_methods_by_disease",
                "search_diseases_by_symptom",
                "query_medical_knowledge",
                "query_with_params"
            ]
        )
    
    def get_direct_connection_info(self) -> DirectConnectionInfo:
        """获取直连信息"""
        return DirectConnectionInfo(
            connection_type="local",
            endpoint="local_tool_instance"
        )
    
    def set_mock_response(self, method_name: str, response: Any) -> None:
        """设置模拟返回数据"""
        self._mock_responses[method_name] = response
    
    def is_available(self) -> bool:
        """检查代理是否可用"""
        return self._tool is not None
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取运行指标"""
        avg_time = self._total_time / self._call_count if self._call_count > 0 else 0
        error_rate = self._error_count / self._call_count if self._call_count > 0 else 0
        
        return {
            "call_count": self._call_count,
            "error_count": self._error_count,
            "average_response_time": avg_time,
            "error_rate": error_rate,
            "total_time": self._total_time
        }
