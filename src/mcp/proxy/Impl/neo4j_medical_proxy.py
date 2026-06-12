# -*- coding: utf-8 -*-
"""
Neo4j医疗知识图谱代理

封装Neo4j医疗知识图谱工具的MCP代理，提供统一的代理接口。
"""

import logging
import time
from typing import Any, Dict, Optional

from src.mcp.proxy.interfaces import MCPFakeProxy
from src.mcp.proxy.data_classes import ToolInfo, DirectConnectionInfo
from src.tools.neo4j_medical_tool import Neo4jMedicalTool
from src.utils.logger import log_arch_event

logger = logging.getLogger(__name__)


class Neo4jMedicalProxy(MCPFakeProxy):
    """
    Neo4j医疗知识图谱代理类
    
    封装Neo4j医疗知识图谱工具，实现MCPFakeProxy接口。
    提供高效的直连调用方式。
    
    注意：Neo4jMedicalTool现在使用资源池管理连接，
    不再需要通过Proxy传递连接参数。
    
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
            config: 代理配置（保留兼容性，不再使用连接参数）
        """
        self._config = {}
        logger.info("[PROXY_INIT] Neo4jMedicalProxy初始化: resource_config=managed_by_resource_pool")
        self._tool: Optional[Neo4jMedicalTool] = None
        self._call_count = 0
        self._total_time = 0.0
        self._error_count = 0
        self._mock_responses: Dict[str, Any] = {}
    
    def _init_tool(self) -> None:
        """初始化tool功能实例"""
        if self._tool is not None:
            logger.debug("[Neo4jMedicalProxy._init_tool] Tool已初始化，跳过")
            return
        
        logger.info("[Neo4jMedicalProxy._init_tool] 开始初始化Neo4jMedicalTool")
        start_time = time.time()
        tool = None
        try:
            tool = Neo4jMedicalTool()
            tool._init_resource()
            self._tool = tool
            log_arch_event(logger, component="Neo4jMedicalProxy", stage="MCP", event="init_tool", status="success", design_id="ARCH-4.3", tool="Neo4jMedicalTool")
            logger.info("[MCP_PROXY_INIT] type=FAKE, tool=Neo4jMedicalTool")
            elapsed = time.time() - start_time
            logger.info(f"[Neo4jMedicalProxy._init_tool] Neo4jMedicalTool初始化完成: elapsed={elapsed:.3f}s")
        except Exception as e:
            if tool is not None:
                try:
                    tool.release_source()
                except Exception as cleanup_error:
                    logger.warning(f"[Neo4jMedicalProxy._init_tool] cleanup failed: {cleanup_error}")
            self._tool = None
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalProxy._init_tool] Neo4jMedicalTool初始化失败: elapsed={elapsed:.3f}s, error={str(e)}")
            raise
    
    def release_tool(self, tool=None) -> None:
        """释放tool功能实例"""
        logger.info("[Neo4jMedicalProxy.release_tool] 开始释放Neo4jMedicalTool")
        if self._tool is not None:
            self._tool.release_source()
            self._tool = None
        log_arch_event(logger, component="Neo4jMedicalProxy", stage="MCP", event="release_tool", status="success", design_id="ARCH-4.3", tool="Neo4jMedicalTool")
        logger.info("[Neo4jMedicalProxy.release_tool] Neo4jMedicalTool释放完成")
    
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
            logger.error("[Neo4jMedicalProxy.call] Tool未初始化")
            raise RuntimeError("Tool not initialized, call _init_tool first")
        
        if method_name in self._mock_responses:
            logger.debug(f"[Neo4jMedicalProxy.call] 使用mock响应: method_name={method_name}")
            return self._mock_responses[method_name]
        
        logger.debug(f"[Neo4jMedicalProxy.call] 代理方法调用开始: method_name={method_name}, params_keys={list(params.keys())}")
        start_time = time.time()
        try:
            method = getattr(self._tool, method_name, None)
            if method is None:
                logger.error(f"[Neo4jMedicalProxy.call] 方法不存在: method_name={method_name}")
                raise AttributeError(f"Method {method_name} not found")
            
            result = method(**params)
            self._call_count += 1
            elapsed = time.time() - start_time
            log_arch_event(logger, component="Neo4jMedicalProxy", stage="MCP", event="proxy_call", status="success", design_id="ARCH-4.3", method_name=method_name, elapsed=f"{elapsed:.3f}s")
            logger.info(f"[Neo4jMedicalProxy.call] 代理方法调用完成: method_name={method_name}, elapsed={elapsed:.3f}s")
            return result
        except Exception as e:
            self._error_count += 1
            elapsed = time.time() - start_time
            logger.error(f"[Neo4jMedicalProxy.call] 代理方法调用失败: method_name={method_name}, elapsed={elapsed:.3f}s, error={str(e)}")
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
            ]
        )
    
    def get_direct_connection_info(self) -> DirectConnectionInfo:
        """获取直连信息"""
        return DirectConnectionInfo(
            type="local",
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
