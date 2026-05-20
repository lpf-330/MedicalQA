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

# 防止vllm子进程重复创建日志文件
_LOG_INITIALIZED = os.environ.get('MEDICALQA_LOG_INITIALIZED', '')

log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

if not _LOG_INITIALIZED:
    os.environ['MEDICALQA_LOG_INITIALIZED'] = 'true'
    
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"medical_qa_{timestamp}.log")

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter(log_format))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.getLogger('vllm').setLevel(logging.WARNING)
    logging.getLogger('neo4j').setLevel(logging.WARNING)

from src.utils.logger import get_logger

business_logger = logging.getLogger('Business')
business_console_handler = logging.StreamHandler(sys.stdout)
business_console_handler.setLevel(logging.INFO)
business_console_handler.setFormatter(logging.Formatter(log_format))
business_logger.addHandler(business_console_handler)
business_logger.setLevel(logging.INFO)
business_logger.propagate = False

logger = get_logger(
    name="Main",
    level="INFO",
    log_file=None,
    console_output=False
)
# logger.info(f"日志文件创建: {log_file}")
from src.config.config_manager import get_config_manager, ConfigManager
from src.config import load_global_config
from src.resource_manager import GlobalResourceManager
from src.controller.consult_controller import ConsultController
from src.service.consult_service import ConsultService
from src.schemas.consult_request import ConsultRequest, ConsultRequestBody
from src.orchestration.agent.consult_strategy import ConsultStrategy
from src.orchestration.chain.knowledge_retrieval_chain import KnowledgeRetrievalChain, KnowledgeRetrievalResource
from src.orchestration.chain.answer_generation_chain import AnswerGenerationChain, AnswerGenerationResource
from src.orchestration.tool_call_handler.Impl.neo4j_medical_handler import Neo4jMedicalHandler
from src.orchestration.tool_call_handler.Impl.vector_retrieval_handler import VectorRetrievalHandler
from src.orchestration.model_business_service.Impl.consult_model_service import ConsultModelService
from src.orchestration.agent.agent_resource import AgentResource
from src.orchestration.agent.agent import Agent
from src.orchestration.state_machine.state_machine import StateMachine
from src.mcp.proxy.Impl.neo4j_medical_proxy import Neo4jMedicalProxy
from src.mcp.proxy.Impl.milvus_medical_proxy import MilvusMedicalProxy
from src.mcp.proxy.Impl.intent_classification_proxy import IntentClassificationProxy
from src.mcp.factory.mcp_proxy_factory import MCPProxyFactory
from src.mcp.factory.config import ProxyType, ToolProxyConfig
from src.orchestration.tool_call_handler.Impl.intent_classification_handler import IntentClassificationHandler

from src.controller.report_controller import ReportController
from src.service.report_service import ReportService
from src.schemas.report_request import ReportRequest, ReportRequestBody
from src.orchestration.agent.report_strategy.report_strategy import ReportStrategy
from src.orchestration.chain.data_prepare_chain.data_prepare_chain import DataPrepareChain, DataPrepareResource
from src.orchestration.chain.multi_analysis_chain.multi_analysis_chain import MultiAnalysisChain, MultiAnalysisResource
from src.orchestration.chain.dimension_evaluation_chain.dimension_evaluation_chain import DimensionEvaluationChain, DimensionEvaluationResource
from src.orchestration.chain.report_knowledge_retrieval_chain.report_knowledge_retrieval_chain import ReportKnowledgeRetrievalChain, ReportKnowledgeRetrievalResource
from src.orchestration.chain.integration_chain.integration_chain import IntegrationChain, IntegrationResource
from src.orchestration.chain.report_generation_chain.report_generation_chain import ReportGenerationChain, ReportGenerationResource
from src.orchestration.model_business_service.Impl.report_model_service import ReportModelService
from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn


def _init_business_components(config_manager: ConfigManager):
    neo4j_resource_config = config_manager.get_resource_config("neo4j_config")
    vllm_resource_config = config_manager.get_resource_config("vllm_config")
    milvus_resource_config = config_manager.get_resource_config("milvus_config")
    vector_model_resource_config = config_manager.get_resource_config("vector_model_config")

    neo4j_proxy = Neo4jMedicalProxy({
        "uri": neo4j_resource_config.uri,
        "user": neo4j_resource_config.user,
        "password": neo4j_resource_config.password
    })

    milvus_proxy = MilvusMedicalProxy({
        "milvus_uri": milvus_resource_config.uri,
        "milvus_user": milvus_resource_config.user,
        "milvus_password": milvus_resource_config.password,
        "milvus_token": milvus_resource_config.token,
        "vector_model_path": vector_model_resource_config.model_path,
        "vector_device": vector_model_resource_config.device,
        "vector_dimension": vector_model_resource_config.dimension
    })

    neo4j_handler = Neo4jMedicalHandler()
    neo4j_handler._init_tool(neo4j_proxy)

    vector_handler = VectorRetrievalHandler()
    vector_handler._init_tool(milvus_proxy)

    consult_model_service = ConsultModelService()
    logger.info("咨询模型服务创建完成")

    knowledge_retrieval_resource = KnowledgeRetrievalResource(
        vector_handler=vector_handler,
        neo4j_handler=neo4j_handler
    )
    knowledge_retrieval_chain = KnowledgeRetrievalChain(knowledge_retrieval_resource)

    answer_generation_resource = AnswerGenerationResource(
        model_service=consult_model_service
    )
    answer_generation_chain = AnswerGenerationChain(answer_generation_resource)

    agent_resource = AgentResource()
    agent_resource.register_chain("knowledge_retrieval_chain", knowledge_retrieval_chain)
    agent_resource.register_chain("answer_generation_chain", answer_generation_chain)
    agent_resource.register_tool_handler("neo4j_medical", neo4j_handler)
    agent_resource.register_tool_handler("vector_retrieval", vector_handler)
    agent_resource.model_service = consult_model_service

    consult_strategy = ConsultStrategy()

    agent = Agent(
        strategy=consult_strategy,
        resources=agent_resource
    )

    consult_service = ConsultService(agent=agent)
    consult_controller = ConsultController(consult_service=consult_service)

    report_model_service = ReportModelService()
    logger.info("报告模型服务创建完成")

    intent_handler = None
    intent_model_resource_config = config_manager.get_resource_config("intent_model_config")
    if intent_model_resource_config is not None:
        try:
            intent_classification_proxy = IntentClassificationProxy({
                "model_path": intent_model_resource_config.model_path,
                "device": intent_model_resource_config.device,
                "max_length": intent_model_resource_config.max_length
            })
            intent_classification_proxy._init_tool()
            logger.info("意图分类代理创建完成")

            intent_handler = IntentClassificationHandler()
            intent_handler._init_tool(intent_classification_proxy)
            logger.info("意图分类Handler创建完成")
        except Exception as e:
            logger.warning(f"意图分类Handler创建失败，将使用规则引擎降级: {str(e)}")
            intent_handler = None
    else:
        logger.info("未配置intent_model_config，MultiAnalysisChain将使用规则引擎降级")

    data_prepare_resource = DataPrepareResource()
    data_prepare_chain = DataPrepareChain(resource=data_prepare_resource)

    multi_analysis_resource = MultiAnalysisResource(intent_handler=intent_handler)
    multi_analysis_chain = MultiAnalysisChain(resource=multi_analysis_resource)

    dimension_evaluation_resource = DimensionEvaluationResource(
        vector_handler=vector_handler,
        neo4j_handler=neo4j_handler
    )
    dimension_evaluation_chain = DimensionEvaluationChain(resource=dimension_evaluation_resource)

    report_knowledge_retrieval_resource = ReportKnowledgeRetrievalResource(
        vector_handler=vector_handler,
        neo4j_handler=neo4j_handler
    )
    report_knowledge_retrieval_chain = ReportKnowledgeRetrievalChain(resource=report_knowledge_retrieval_resource)

    integration_resource = IntegrationResource()
    integration_chain = IntegrationChain(resource=integration_resource)

    report_generation_resource = ReportGenerationResource(
        model_service=report_model_service
    )
    report_generation_chain = ReportGenerationChain(resource=report_generation_resource)

    report_strategy = ReportStrategy()

    report_agent_resource = AgentResource(
        model_service=report_model_service,
        chain_registry={
            "data_prepare_chain": data_prepare_chain,
            "multi_analysis_chain": multi_analysis_chain,
            "dimension_evaluation_chain": dimension_evaluation_chain,
            "report_knowledge_retrieval_chain": report_knowledge_retrieval_chain,
            "integration_chain": integration_chain,
            "report_generation_chain": report_generation_chain
        },
        tool_handlers={
            "neo4j_tool": neo4j_handler,
            "vector_tool": vector_handler
        }
    )

    report_agent = Agent(strategy=report_strategy, resources=report_agent_resource)

    report_service = ReportService(agent=report_agent)
    report_controller = ReportController(report_service=report_service)

    return consult_controller, neo4j_handler, vector_handler, consult_model_service, report_controller, report_model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("=" * 60)
    logger.info("初始化健康咨询服务...")
    logger.info("=" * 60)
    
    try:
        config_manager = GlobalResourceManager.initialize()
        
        logger.info("步骤4: 初始化业务组件...")
        consult_controller, neo4j_handler, vector_handler, consult_model_service, report_controller, report_model_service = _init_business_components(config_manager)
        app.state.consult_controller = consult_controller
        app.state.neo4j_handler = neo4j_handler
        app.state.vector_handler = vector_handler
        app.state.model_service = consult_model_service
        app.state.report_controller = report_controller
        app.state.report_model_service = report_model_service
        
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
        report_model_service = getattr(app.state, 'report_model_service', None)
        if report_model_service:
            report_model_service.release()
        vector_handler = getattr(app.state, 'vector_handler', None)
        if vector_handler:
            vector_handler.release()
        neo4j_handler = getattr(app.state, 'neo4j_handler', None)
        if neo4j_handler:
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
async def consult(body: ConsultRequestBody, request_id: str = "default", user_id: str = None):
    """
    健康咨询API
    
    Args:
        body: 咨询请求体（包含task_id, question, chat_history等）
        request_id: 请求ID（可选）
        user_id: 用户ID（可选）
        
    Returns:
        咨询结果
    """
    consult_request = ConsultRequest(
        request_id=request_id,
        user_id=user_id,
        body=body
    )
    controller = app.state.consult_controller
    return controller.consult(consult_request)


@app.post("/api/v1/report")
async def generate_report(body: ReportRequestBody, request_id: str = "default", user_id: str = None):
    """
    健康报告生成API
    
    Args:
        body: 报告请求体（包含task_id, monitoring_data, user_profile等）
        request_id: 请求ID（可选）
        user_id: 用户ID（可选）
        
    Returns:
        健康报告生成结果
    """
    report_request = ReportRequest(
        request_id=request_id,
        user_id=user_id,
        body=body
    )
    controller = app.state.report_controller
    return controller.generate_report(report_request)


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
