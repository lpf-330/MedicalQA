"""
服务层健康报告生成服务模块

该模块定义了ReportService类，是健康报告生成业务的服务类。
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, TypeVar, TypeAlias, Any, AsyncGenerator, Optional

from src.utils.async_helpers import run_with_context

if TYPE_CHECKING:
    from src.orchestration.agent.agent import Agent
    from src.orchestration.agent.data_classes import AgentContext, AgentResult
    from src.config.config_manager import ConfigManager

from src.orchestration.agent.data_classes import AgentContext
from src.orchestration.agent.report_strategy.report_context import ReportContextBody
from src.schemas.report_request import ReportRequest
from src.utils.logger import log_arch_event

from src.orchestration.tool_call_handler.Impl.neo4j_medical_handler import Neo4jMedicalHandler
from src.orchestration.tool_call_handler.Impl.vector_retrieval_handler import VectorRetrievalHandler
from src.orchestration.tool_call_handler.Impl.intent_classification_handler import IntentClassificationHandler
from src.orchestration.tool_call_handler.Impl.ner_model_handler import NerModelHandler
from src.orchestration.model_business_service.Impl.report_model_service import ReportModelService
from src.orchestration.model_business_service.Impl.health_assessment_model_service import HealthAssessmentModelService
from src.orchestration.chain.data_prepare_chain.data_prepare_chain import DataPrepareChain, DataPrepareResource
from src.orchestration.chain.multi_analysis_chain.multi_analysis_chain import MultiAnalysisChain, MultiAnalysisResource
from src.orchestration.chain.report_knowledge_retrieval_chain.report_knowledge_retrieval_chain import ReportKnowledgeRetrievalChain, ReportKnowledgeRetrievalResource
from src.orchestration.chain.report_generation_chain.report_generation_chain import ReportGenerationChain, ReportGenerationResource
from src.orchestration.chain.health_assessment_chain import HealthAssessmentChain, HealthAssessmentResource
from src.orchestration.agent.comprehensive_health_analysis_strategy import ComprehensiveHealthAnalysisStrategy
from src.orchestration.agent.report_strategy.report_strategy import ReportStrategy
from src.orchestration.agent.agent_resource import AgentResource
from src.orchestration.agent.agent import Agent

logger = logging.getLogger(__name__)

T = TypeVar('T')

ReportContext: TypeAlias = 'AgentContext[Any]'
ReportResult: TypeAlias = 'AgentResult[Any]'


class ReportService:
    """
    报告服务类

    健康报告生成业务的服务类，负责处理报告生成请求。
    依赖Agent组合容器，通过Agent执行报告生成策略。

    Attributes:
        _agent: Agent组合容器实例
    """

    def __init__(self, config_manager: 'ConfigManager') -> None:
        """
        初始化ReportService实例

        Args:
            config_manager: 配置管理器实例

        Raises:
            ValueError: config_manager为None时抛出
        """
        if config_manager is None:
            raise ValueError("config_manager不能为None")

        self._config_manager = config_manager
        self._agent: 'Agent[Any, Any]' = self._build_agent()
        logger.info("[ReportService] 服务初始化完成")

    @property
    def agent(self) -> 'Agent[Any, Any]':
        """
        获取Agent实例（只读属性）

        Returns:
            Agent[Any, Any]: Agent组合容器实例
        """
        return self._agent

    def _build_agent(self) -> 'Agent[Any, Any]':
        """自行组装报告业务的Agent树

        设计依据：2.3.3节 ToolCallHandler生命周期管理规范
        - Handler在首次call_tool()时通过_ensure_initialized()从MCPProxyFactory获取缓存实例
        - 服务层无需直接创建Proxy，也无需立即初始化Handler
        - 可选工具调用_ensure_initialized()验证可用性
        """
        # 必选工具：只创建Handler，不立即初始化
        # 首次 call_tool() 时 _ensure_initialized() 通过 MCPProxyFactory 获取缓存实例
        neo4j_handler = Neo4jMedicalHandler(tool_proxy_instance_id="neo4j_medical")
        logger.info("[COMPONENT_CREATE] 创建组件: Neo4jMedicalHandler, type=Neo4jMedicalHandler")

        vector_handler = VectorRetrievalHandler(tool_proxy_instance_id="vector_retrieval")
        logger.info("[COMPONENT_CREATE] 创建组件: VectorRetrievalHandler, type=VectorRetrievalHandler")

        # 意图分类Handler（可选，配置驱动降级）
        intent_handler: Optional[IntentClassificationHandler] = None
        if self._config_manager.resource_configs.get("intent_model_config") is not None:
            try:
                intent_handler = IntentClassificationHandler(tool_proxy_instance_id="intent_classification")
                intent_handler._ensure_initialized()
                logger.info("[COMPONENT_CREATE] 创建组件: IntentClassificationHandler, type=IntentClassificationHandler")
            except Exception as e:
                logger.warning(f"意图分类Handler初始化失败，将使用规则引擎降级: {str(e)}")
                intent_handler = None
        else:
            logger.info("未配置intent_model_config，报告业务将使用规则引擎降级")

        # NER模型Handler（可选，配置驱动降级）
        ner_handler: Optional[NerModelHandler] = None
        if self._config_manager.resource_configs.get("ner_model_config") is not None:
            try:
                ner_handler = NerModelHandler(tool_proxy_instance_id="ner_model")
                ner_handler._ensure_initialized()
                logger.info("[COMPONENT_CREATE] 创建组件: NerModelHandler, type=NerModelHandler")
            except Exception as e:
                logger.warning(f"NER模型Handler初始化失败，将使用规则引擎降级: {str(e)}")
                ner_handler = None
        else:
            logger.info("未配置ner_model_config，实体提取将使用规则引擎降级")

        # 报告业务独有的 ModelService
        report_model_service = ReportModelService()
        logger.info("[COMPONENT_CREATE] 创建组件: 报告模型服务, type=ReportModelService")

        health_assessment_model_service = HealthAssessmentModelService()
        logger.info("[COMPONENT_CREATE] 创建组件: 健康评估模型服务, type=HealthAssessmentModelService")

        # Chain 组装
        data_prepare_resource = DataPrepareResource()
        data_prepare_chain = DataPrepareChain(resource=data_prepare_resource)
        logger.info("[COMPONENT_CREATE] 创建组件: DataPrepareChain, type=DataPrepareChain")

        multi_analysis_resource = MultiAnalysisResource(intent_handler=intent_handler, ner_handler=ner_handler)
        multi_analysis_chain = MultiAnalysisChain(resource=multi_analysis_resource)
        logger.info("[COMPONENT_CREATE] 创建组件: MultiAnalysisChain, type=MultiAnalysisChain")

        report_knowledge_retrieval_resource = ReportKnowledgeRetrievalResource(
            vector_handler=vector_handler,
            neo4j_handler=neo4j_handler
        )
        report_knowledge_retrieval_chain = ReportKnowledgeRetrievalChain(resource=report_knowledge_retrieval_resource)
        logger.info("[COMPONENT_CREATE] 创建组件: ReportKnowledgeRetrievalChain, type=ReportKnowledgeRetrievalChain")

        health_assessment_resource = HealthAssessmentResource(health_assessment_model=health_assessment_model_service)
        health_assessment_chain = HealthAssessmentChain(resource=health_assessment_resource)
        logger.info("[COMPONENT_CREATE] 创建组件: HealthAssessmentChain, type=HealthAssessmentChain")

        report_generation_resource = ReportGenerationResource(
            model_service=report_model_service
        )
        report_generation_chain = ReportGenerationChain(resource=report_generation_resource)
        logger.info("[COMPONENT_CREATE] 创建组件: ReportGenerationChain, type=ReportGenerationChain")

        # AgentResource 组装
        report_agent_resource = AgentResource(
            model_service=report_model_service,
            chain_registry={
                "data_prepare_chain": data_prepare_chain,
                "multi_analysis_chain": multi_analysis_chain,
                "report_knowledge_retrieval_chain": report_knowledge_retrieval_chain,
                "health_assessment_chain": health_assessment_chain,
                "report_generation_chain": report_generation_chain
            },
            tool_handlers={
                "neo4j_medical_tool": neo4j_handler,
                "vector_retrieval_tool": vector_handler,
                **({"intent_classification_tool": intent_handler} if intent_handler is not None else {})
            }
        )
        logger.info("[COMPONENT_CREATE] 创建组件: AgentResource(报告), type=AgentResource")

        # Strategy 组装
        comprehensive_health_analysis_strategy = ComprehensiveHealthAnalysisStrategy()
        logger.info("[COMPONENT_CREATE] 创建组件: ComprehensiveHealthAnalysisStrategy, type=ComprehensiveHealthAnalysisStrategy")

        report_strategy = ReportStrategy(comprehensive_health_analysis_strategy=comprehensive_health_analysis_strategy)
        logger.info("[COMPONENT_CREATE] 创建组件: ReportStrategy, type=ReportStrategy")

        # Agent 容器
        agent = Agent(strategy=report_strategy, resources=report_agent_resource)
        logger.info("[COMPONENT_CREATE] 创建组件: Agent(报告), type=Agent")

        return agent

    def process_report(self, context: ReportContext) -> ReportResult:
        """
        处理报告请求

        验证输入参数，调用Agent执行报告生成策略，并释放资源。

        Args:
            context: Agent输入数据容器，包含报告生成所需的上下文数据

        Returns:
            ReportResult: Agent输出数据容器，包含报告生成结果

        Raises:
            ValueError: context为None或session_id为空时抛出

        Example:
            >>> result = report_service.process_report(context)
            >>> print(result.data.report)
        """
        start_time = time.time()

        # 验证context不为None
        if context is None:
            logger.error("[ReportService] context为None")
            raise ValueError("context不能为None")

        # 验证session_id不为空
        if not hasattr(context, 'session_id') or not context.session_id:
            logger.error("[ReportService] session_id为空")
            raise ValueError("context.session_id不能为空")

        logger.info(f"[ReportService.process_report] 健康报告业务8环节流程启动: session_id={context.session_id}")
        stage_start = time.time()
        logger.info("[STAGE_ENTER] stage=INITIAL")
        logger.info(f"[ReportService.process_report] INITIAL环节: 接收报告请求, session_id={context.session_id}")
        logger.info(f"[STAGE_EXIT] stage=INITIAL, duration={time.time() - stage_start:.2f}s")
        logger.info("[STATE_TRANSITION] from=INITIAL to=DATA_PREPARE")

        try:
            stage_start = time.time()
            logger.info("[STAGE_ENTER] stage=DATA_PREPARE")
            logger.info(f"[STAGE_EXIT] stage=DATA_PREPARE, duration={time.time() - stage_start:.2f}s")
            logger.info("[STATE_TRANSITION] from=DATA_PREPARE to=DATA_PARSE")

            stage_start = time.time()
            logger.info("[STAGE_ENTER] stage=DATA_PARSE")
            logger.info(f"[STAGE_EXIT] stage=DATA_PARSE, duration={time.time() - stage_start:.2f}s")
            logger.info("[STATE_TRANSITION] from=DATA_PARSE to=COMPREHENSIVE_HEALTH_ANALYSIS")

            stage_start = time.time()
            logger.info("[STAGE_ENTER] stage=COMPREHENSIVE_HEALTH_ANALYSIS")
            logger.info(f"[STAGE_EXIT] stage=COMPREHENSIVE_HEALTH_ANALYSIS, duration={time.time() - stage_start:.2f}s")
            logger.info("[STATE_TRANSITION] from=COMPREHENSIVE_HEALTH_ANALYSIS to=REPORT_GENERATION")

            stage_start = time.time()
            logger.info("[STAGE_ENTER] stage=REPORT_GENERATION")
            logger.info(f"[STAGE_EXIT] stage=REPORT_GENERATION, duration={time.time() - stage_start:.2f}s")
            logger.info("[STATE_TRANSITION] from=REPORT_GENERATION to=STREAMING")

            stage_start = time.time()
            logger.info("[STAGE_ENTER] stage=STREAMING")
            logger.info(f"[STAGE_EXIT] stage=STREAMING, duration={time.time() - stage_start:.2f}s")
            logger.info("[STATE_TRANSITION] from=STREAMING to=ASSEMBLY")

            stage_start = time.time()
            logger.info("[STAGE_ENTER] stage=ASSEMBLY")
            logger.info(f"[STAGE_EXIT] stage=ASSEMBLY, duration={time.time() - stage_start:.2f}s")
            logger.info("[STATE_TRANSITION] from=ASSEMBLY to=FINISHED")

            logger.info("[ReportService.process_report] 调用Agent执行8环节流程(INITIAL→DATA_PREPARE→DATA_PARSE→COMPREHENSIVE_HEALTH_ANALYSIS→REPORT_GENERATION→STREAMING→ASSEMBLY→FINISHED)")
            result = self._agent.run(context)
            logger.info(f"[ReportService.process_report] Agent执行完成, session_id={context.session_id}")
            return result
        finally:
            self._release_resources()
            stage_start = time.time()
            logger.info("[STAGE_ENTER] stage=FINISHED")
            elapsed = time.time() - start_time
            logger.info(f"[ReportService.process_report] FINISHED环节: 报告处理完成, session_id={context.session_id}, elapsed={elapsed:.2f}s")
            logger.info(f"[STAGE_EXIT] stage=FINISHED, duration={time.time() - stage_start:.2f}s")

    async def process_report_stream(self, context: ReportContext) -> AsyncGenerator[str, None]:
        import json

        if context is None:
            yield f"event: error\ndata: {json.dumps({'error_code': 400, 'error_message': 'context不能为None'})}\n\n"
            return

        if not hasattr(context, 'session_id') or not context.session_id:
            yield f"event: error\ndata: {json.dumps({'error_code': 400, 'error_message': 'session_id不能为空'})}\n\n"
            return

        logger.info(f"[ReportService.process_report_stream] 健康报告业务8环节流式流程启动: session_id={context.session_id}")
        log_arch_event(
            logger,
            component="ReportService",
            stage="SERVICE",
            event="process_report_stream",
            status="start",
            design_id="ARCH-2.1",
        )
        stage_start = time.time()
        logger.info("[STAGE_ENTER] stage=INITIAL")
        logger.info(f"[ReportService.process_report_stream] INITIAL环节: 接收流式报告请求, session_id={context.session_id}")
        logger.info(f"[STAGE_EXIT] stage=INITIAL, duration={time.time() - stage_start:.2f}s")
        logger.info("[STATE_TRANSITION] from=INITIAL to=DATA_PREPARE")

        try:
            stage_start = time.time()
            logger.info("[STAGE_ENTER] stage=DATA_PREPARE")
            logger.info(f"[STAGE_EXIT] stage=DATA_PREPARE, duration={time.time() - stage_start:.2f}s")
            logger.info("[STATE_TRANSITION] from=DATA_PREPARE to=DATA_PARSE")

            stage_start = time.time()
            logger.info("[STAGE_ENTER] stage=DATA_PARSE")
            logger.info(f"[STAGE_EXIT] stage=DATA_PARSE, duration={time.time() - stage_start:.2f}s")
            logger.info("[STATE_TRANSITION] from=DATA_PARSE to=COMPREHENSIVE_HEALTH_ANALYSIS")

            stage_start = time.time()
            logger.info("[STAGE_ENTER] stage=COMPREHENSIVE_HEALTH_ANALYSIS")
            logger.info(f"[STAGE_EXIT] stage=COMPREHENSIVE_HEALTH_ANALYSIS, duration={time.time() - stage_start:.2f}s")
            logger.info("[STATE_TRANSITION] from=COMPREHENSIVE_HEALTH_ANALYSIS to=REPORT_GENERATION")

            logger.info("[ReportService.process_report_stream] 调用Agent执行8环节流程(INITIAL→DATA_PREPARE→DATA_PARSE→COMPREHENSIVE_HEALTH_ANALYSIS→REPORT_GENERATION→STREAMING→ASSEMBLY→FINISHED)")
            result = await asyncio.to_thread(run_with_context, self._agent.run, context)

            if result is not None and result.data is not None:
                body = context.body

                if hasattr(body, 'stream_generator') and body.stream_generator is not None:
                    # REPORT_GENERATION+STREAMING阶段完整计时：从流式生成开始到流式输出完成
                    report_stream_start_time = time.time()
                    logger.info("[STATE_TRANSITION] from=REPORT_GENERATION to=STREAMING")
                    logger.info("[STAGE_ENTER] stage=REPORT_GENERATION(stream)")
                    logger.info(f"[ReportService.process_report_stream] STREAMING环节: 开始流式返回, session_id={context.session_id}")
                    async for token in body.stream_generator:
                        payload = json.dumps({"content": token}, ensure_ascii=False)
                        yield f"event: message\ndata: {payload}\n\n"

                    context.is_streaming = False
                    report_stream_elapsed = time.time() - report_stream_start_time
                    logger.info(f"[STAGE_EXIT] stage=REPORT_GENERATION(stream), duration={report_stream_elapsed:.2f}s")
                    logger.info(f"[ReportService.process_report_stream] STREAMING环节: 流式返回完成, session_id={context.session_id}, duration={report_stream_elapsed:.2f}s")

                    logger.info("[STATE_TRANSITION] from=STREAMING to=ASSEMBLY")
                    stage_start = time.time()
                    logger.info("[STAGE_ENTER] stage=ASSEMBLY")
                    logger.info(f"[ReportService.process_report_stream] ASSEMBLY环节: 组装结束响应, session_id={context.session_id}")
                    end_data = {
                        "session_id": result.session_id,
                        "health_score": getattr(body, 'health_score', 0),
                        "health_level": getattr(body, 'health_level', ''),
                        "risk_level": getattr(body, 'risk_level', ''),
                        "sources": getattr(body, 'sources', []),
                        "error_code": getattr(body, 'error_code', 0),
                    }
                    yield f"event: end\ndata: {json.dumps(end_data, ensure_ascii=False)}\n\n"
                    logger.info(f"[STAGE_EXIT] stage=ASSEMBLY, duration={time.time() - stage_start:.2f}s")
                    logger.info("[STATE_TRANSITION] from=ASSEMBLY to=FINISHED")
                    stage_start = time.time()
                    logger.info("[STAGE_ENTER] stage=FINISHED")
                    logger.info(f"[STAGE_EXIT] stage=FINISHED, duration={time.time() - stage_start:.2f}s")
                else:
                    report = result.data.report if hasattr(result.data, 'report') else str(result.data)
                    payload = json.dumps({"content": report}, ensure_ascii=False)
                    yield f"event: message\ndata: {payload}\n\n"

                    end_data = {
                        "session_id": result.session_id,
                        "health_score": result.data.health_score if hasattr(result.data, 'health_score') else 0,
                        "health_level": result.data.health_level if hasattr(result.data, 'health_level') else '',
                        "risk_level": result.data.risk_level if hasattr(result.data, 'risk_level') else '',
                    }
                    yield f"event: end\ndata: {json.dumps(end_data, ensure_ascii=False)}\n\n"
            else:
                logger.error(f"[ReportService.process_report_stream] Agent返回空结果, session_id={context.session_id}")
                yield f"event: error\ndata: {json.dumps({'error_code': 500, 'error_message': '处理结果为空'})}\n\n"

        except Exception as e:
            logger.error(f"[ReportService.process_report_stream] 流式处理异常: session_id={context.session_id}, error={str(e)}")
            yield f"event: error\ndata: {json.dumps({'error_code': 500, 'error_message': '报告生成服务异常，请稍后重试'})}\n\n"

        finally:
            self._release_resources()

    def build_agent_context(self, request: ReportRequest) -> AgentContext:
        """
        从ReportRequest构建AgentContext（公开方法）

        将API请求数据转换为Agent执行所需的上下文数据。

        Args:
            request: 健康报告生成请求数据

        Returns:
            AgentContext: Agent输入数据容器

        Example:
            >>> context = report_service.build_agent_context(request)
            >>> print(context.session_id)
        """
        # 获取session_id，优先使用body中的session_id，否则使用task_id
        session_id = request.get_session_id() or request.body.task_id

        # 获取监测数据
        monitoring_data = {}
        if request.body.monitoring_data:
            monitoring_data = request.body.monitoring_data.model_dump()

        # 获取用户档案
        user_profile = {}
        if request.body.user_profile:
            user_profile = request.body.user_profile.model_dump()

        # 创建ReportContextBody
        context_body = ReportContextBody(
            task_id=request.body.task_id,
            monitoring_data=monitoring_data,
            user_profile=user_profile,
            session_id=session_id,
            current_state="INITIAL"
        )

        # 创建AgentContext
        agent_context = AgentContext(
            session_id=session_id,
            current_state="INITIAL",
            body=context_body
        )

        return agent_context

    def _release_resources(self) -> None:
        """
        释放Agent资源

        释放model service资源。
        注意：tool handlers使用自动重新初始化机制，不需要显式释放。
        """
        agent_resource = self._agent.resources
        if agent_resource is not None and agent_resource.model_service is not None:
            try:
                agent_resource.model_service.release_model()
            except Exception as e:
                logger.error(f"[ReportService] 释放model service失败: {e}")

    def __repr__(self) -> str:
        """
        返回ReportService的字符串表示

        Returns:
            str: ReportService的字符串表示
        """
        return f"ReportService(agent={self._agent})"
