# -*- coding: utf-8 -*-
"""
MedicalQA项目启动文件

基于7层架构设计的健康咨询服务启动入口。
按照《项目架构设计v2.1》和《项目架构原则与使用规范v1》的规范流程启动系统。
"""

import os
import sys
import logging
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"medical_qa_{timestamp}.log")
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(log_format))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(log_format))

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logging.getLogger('vllm').setLevel(logging.WARNING)
logging.getLogger('neo4j').setLevel(logging.WARNING)

from src.utils.logger import get_logger

logger = get_logger(
    name="Main",
    level="INFO",
    log_file=log_file,
    console_output=True
)
logger.info(f"日志文件创建: {log_file}")
from src.config.config_manager import get_config_manager, ConfigManager
from src.config import load_global_config
from src.resource_manager import GlobalResourceManager
from src.resource_manager.neo4j_connection import Neo4jConnectionFactory
from src.resource_manager.vllm_model import VLLMModelFactory
from src.controller.consult_controller import ConsultController
from src.service.consult_service import ConsultService
from src.orchestration.agent.consult_strategy import ConsultStrategy
from src.orchestration.chain.consult_with_knowledge_chain import (
    ConsultWithKnowledgeChain,
    ConsultWithKnowledgeResource
)
from src.orchestration.tool_call_handler.Impl.neo4j_medical_handler import Neo4jMedicalHandler
from src.orchestration.agent.agent_resource import AgentResource
from src.orchestration.agent.agent import Agent
from src.mcp.proxy.Impl.neo4j_medical_proxy import Neo4jMedicalProxy
from src.orchestration.model_business_service.Impl.consult_model_service import ConsultModelService
from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn


def _load_configs() -> ConfigManager:
    """
    加载所有配置
    
    按以下顺序加载：
    1. 扫描业务配置目录
    2. 解析业务配置，收集所需的资源配置文件名
    3. 资源配置去重
    4. 加载所需的资源配置
    
    Returns:
        ConfigManager: 配置管理器实例
    """
    logger.info("步骤1: 加载配置...")
    config_manager = get_config_manager()
    
    logger.info(f"  业务配置: {list(config_manager.business_configs.keys())}")
    logger.info(f"  资源配置: {list(config_manager.resource_configs.keys())}")
    logger.info(f"  资源池配置: {list(config_manager.pool_configs.keys())}")
    
    if not config_manager.validate():
        raise RuntimeError("配置验证失败")
    
    logger.info("配置加载完成")
    return config_manager


def _register_resource_factories():
    """注册所有资源工厂"""
    GlobalResourceManager.INSTANCE.register_factory(
        "neo4j_connection", 
        Neo4jConnectionFactory()
    )
    GlobalResourceManager.INSTANCE.register_factory(
        "vllm_model", 
        VLLMModelFactory()
    )
    logger.info("资源工厂注册完成")


def _create_initial_resources(config_manager: ConfigManager):
    """
    创建初始资源实例
    
    根据业务配置确定需要哪些资源配置，然后创建对应的资源池
    """
    global_config = config_manager.to_global_config()
    GlobalResourceManager.INSTANCE._init_global_resource_manager(global_config)
    stats = GlobalResourceManager.INSTANCE.get_stats()
    logger.info(f"初始资源创建完成: {stats}")


def _init_business_components(config_manager: ConfigManager):
    """初始化业务组件"""
    neo4j_resource_config = config_manager.get_resource_config("neo4j_config")
    vllm_resource_config = config_manager.get_resource_config("vllm_config")
    
    neo4j_proxy = Neo4jMedicalProxy({
        "uri": neo4j_resource_config.uri,
        "user": neo4j_resource_config.user,
        "password": neo4j_resource_config.password
    })
    
    neo4j_handler = Neo4jMedicalHandler()
    neo4j_handler._init_tool(neo4j_proxy)
    
    model_path = vllm_resource_config.model_path
    logger.info(f"初始化模型服务，模型路径: {model_path}")
    model_service = ConsultModelService(model_path=model_path)
    model_service._init_model()
    logger.info("模型服务初始化完成")
    
    chain_resource = ConsultWithKnowledgeResource()
    chain_resource.neo4j_handler = neo4j_handler
    chain_resource.model_service = model_service
    
    knowledge_chain = ConsultWithKnowledgeChain(chain_resource)
    
    agent_resource = AgentResource()
    agent_resource.register_chain("knowledge_chain", knowledge_chain)
    
    consult_strategy = ConsultStrategy()
    
    agent = Agent(
        strategy=consult_strategy,
        resources=agent_resource
    )
    
    consult_service = ConsultService(agent=agent)
    consult_controller = ConsultController(consult_service=consult_service)
    
    return consult_controller, neo4j_handler, model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("=" * 60)
    logger.info("初始化健康咨询服务...")
    logger.info("=" * 60)
    
    try:
        config_manager = _load_configs()
        
        logger.info("步骤2: 注册资源工厂...")
        _register_resource_factories()
        
        logger.info("步骤3: 创建初始资源实例...")
        _create_initial_resources(config_manager)
        
        logger.info("步骤4: 初始化业务组件...")
        consult_controller, neo4j_handler, model_service = _init_business_components(config_manager)
        app.state.consult_controller = consult_controller
        app.state.neo4j_handler = neo4j_handler
        app.state.model_service = model_service
        
        logger.info("=" * 60)
        logger.info("健康咨询服务初始化完成")
        logger.info("=" * 60)
        
        yield
        
        logger.info("=" * 60)
        logger.info("正在关闭服务...")
        logger.info("=" * 60)
        
        model_service = getattr(app.state, 'model_service', None)
        if model_service:
            model_service.release()
        neo4j_handler.release()
        GlobalResourceManager.INSTANCE.shutdown()
        
        logger.info("服务已关闭")
        
    except Exception as e:
        logger.error(f"初始化服务失败: {str(e)}", exc_info=True)
        raise


app = FastAPI(
    title="MedicalQA API Server",
    description="基于7层架构设计的健康咨询API服务",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """根路径"""
    return {"message": "MedicalQA API Server is running", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """健康检查"""
    stats = GlobalResourceManager.INSTANCE.get_stats()
    return {
        "status": "healthy",
        "resource_stats": stats
    }


@app.post("/api/v1/consult")
async def consult(request: dict):
    """
    健康咨询API
    
    Args:
        request: 包含question字段的请求体
        
    Returns:
        咨询结果
    """
    from src.schemas.consult_request import ConsultRequest, ConsultRequestBody
    
    consult_request = ConsultRequest(
        request_id=request.get("request_id", "default"),
        body=ConsultRequestBody(
            question=request.get("question", "")
        )
    )
    
    controller = app.state.consult_controller
    response = controller.consult(consult_request)
    
    return response.to_dict()


def main():
    """启动服务"""
    logger.info("启动MedicalQA服务...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info",
        timeout_keep_alive=60
    )


if __name__ == "__main__":
    main()
