"""
服务层健康咨询服务模块

该模块定义了ConsultService类，是健康咨询业务的服务类。
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
from src.orchestration.agent.consult_strategy.consult_context import ConsultContextBody
from src.config.business.consult_service_config import get_runtime_config
from src.schemas.error_codes import ErrorCode
from src.schemas.consult_request import ConsultRequest
from src.utils.logger import log_arch_event

from src.orchestration.tool_call_handler.Impl.neo4j_medical_handler import Neo4jMedicalHandler
from src.orchestration.tool_call_handler.Impl.vector_retrieval_handler import VectorRetrievalHandler
from src.orchestration.tool_call_handler.Impl.intent_classification_handler import IntentClassificationHandler
from src.orchestration.tool_call_handler.Impl.ner_model_handler import NerModelHandler
from src.orchestration.model_business_service.Impl.consult_model_service import ConsultModelService
from src.orchestration.chain.knowledge_retrieval_chain import KnowledgeRetrievalChain, KnowledgeRetrievalResource
from src.orchestration.chain.answer_generation_chain import AnswerGenerationChain, AnswerGenerationResource
from src.orchestration.agent.knowledge_retrieval_strategy import KnowledgeRetrievalStrategy
from src.orchestration.agent.consult_strategy import ConsultStrategy
from src.orchestration.agent.agent_resource import AgentResource
from src.orchestration.agent.agent import Agent

logger = logging.getLogger(__name__)

T = TypeVar('T')

ConsultContext: TypeAlias = 'AgentContext[Any]'
ConsultResult: TypeAlias = 'AgentResult[Any]'


class ConsultService:

    def __init__(self, config_manager: 'ConfigManager') -> None:
        if config_manager is None:
            raise ValueError("config_manager不能为None")

        self._config_manager = config_manager
        self._consult_model_service: ConsultModelService = ConsultModelService()
        logger.info("[COMPONENT_CREATE] 创建组件: 咨询模型服务, type=ConsultModelService")
        self._agent: 'Agent[Any, Any]' = self._build_agent()
        logger.info("[ConsultService] 服务初始化完成")

    @property
    def agent(self) -> 'Agent[Any, Any]':
        return self._agent

    @property
    def consult_model_service(self) -> ConsultModelService:
        return self._consult_model_service

    def _build_agent(self) -> 'Agent[Any, Any]':
        """自行组装咨询业务的Agent树

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
            logger.info("未配置intent_model_config，咨询业务将使用规则引擎降级")

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

        # Chain 组装
        knowledge_retrieval_resource = KnowledgeRetrievalResource(
            vector_handler=vector_handler,
            neo4j_handler=neo4j_handler
        )
        knowledge_retrieval_chain = KnowledgeRetrievalChain(knowledge_retrieval_resource)
        logger.info("[COMPONENT_CREATE] 创建组件: KnowledgeRetrievalChain, type=KnowledgeRetrievalChain")

        answer_generation_resource = AnswerGenerationResource(
            model_service=self._consult_model_service
        )
        answer_generation_chain = AnswerGenerationChain(answer_generation_resource)
        logger.info("[COMPONENT_CREATE] 创建组件: AnswerGenerationChain, type=AnswerGenerationChain")

        # AgentResource 组装
        agent_resource = AgentResource()
        agent_resource.register_chain("knowledge_retrieval_chain", knowledge_retrieval_chain)
        agent_resource.register_chain("answer_generation_chain", answer_generation_chain)
        agent_resource.register_tool_handler("neo4j_medical_tool", neo4j_handler)
        agent_resource.register_tool_handler("vector_retrieval_tool", vector_handler)
        if intent_handler is not None:
            agent_resource.register_tool_handler("intent_classification_tool", intent_handler)
            logger.info("咨询业务Agent已注册intent_classification_tool")
        if ner_handler is not None:
            agent_resource.register_tool_handler("ner_model_tool", ner_handler)
            logger.info("咨询业务Agent已注册ner_model_tool")
        agent_resource.model_service = self._consult_model_service
        logger.info("[COMPONENT_CREATE] 创建组件: AgentResource(咨询), type=AgentResource")

        # Strategy 组装
        knowledge_retrieval_strategy = KnowledgeRetrievalStrategy()
        logger.info("[COMPONENT_CREATE] 创建组件: KnowledgeRetrievalStrategy, type=KnowledgeRetrievalStrategy")

        consult_strategy = ConsultStrategy(knowledge_retrieval_strategy=knowledge_retrieval_strategy)
        logger.info("[COMPONENT_CREATE] 创建组件: ConsultStrategy, type=ConsultStrategy")

        # Agent 容器
        agent = Agent(
            strategy=consult_strategy,
            resources=agent_resource
        )
        logger.info("[COMPONENT_CREATE] 创建组件: Agent(咨询), type=Agent")

        return agent

    def process_consult(self, context: ConsultContext) -> ConsultResult:
        start_time = time.time()

        if context is None:
            logger.error("[ConsultService.process_consult] context为None")
            raise ValueError("context不能为None")

        if not hasattr(context, 'session_id') or not context.session_id:
            logger.error("[ConsultService.process_consult] session_id为空")
            raise ValueError("context.session_id不能为空")

        logger.info(f"[ConsultService.process_consult] 健康咨询业务7环节流程启动: session_id={context.session_id}")
        stage_start = time.time()
        logger.info("[STAGE_ENTER] stage=INITIAL")
        logger.info(f"[ConsultService.process_consult] INITIAL环节: 接收请求, session_id={context.session_id}")
        logger.info(f"[STAGE_EXIT] stage=INITIAL, duration={time.time() - stage_start:.2f}s")
        logger.info("[STATE_TRANSITION] from=INITIAL to=QUERY_PARSE")

        try:
            stage_start = time.time()
            logger.info("[STAGE_ENTER] stage=QUERY_PARSE")
            logger.info(f"[STAGE_EXIT] stage=QUERY_PARSE, duration={time.time() - stage_start:.2f}s")
            logger.info("[STATE_TRANSITION] from=QUERY_PARSE to=KNOWLEDGE_RETRIEVAL")

            stage_start = time.time()
            logger.info("[STAGE_ENTER] stage=KNOWLEDGE_RETRIEVAL")
            logger.info(f"[STAGE_EXIT] stage=KNOWLEDGE_RETRIEVAL, duration={time.time() - stage_start:.2f}s")
            logger.info("[STATE_TRANSITION] from=KNOWLEDGE_RETRIEVAL to=KNOWLEDGE_INTEGRATION")

            stage_start = time.time()
            logger.info("[STAGE_ENTER] stage=KNOWLEDGE_INTEGRATION")
            logger.info(f"[STAGE_EXIT] stage=KNOWLEDGE_INTEGRATION, duration={time.time() - stage_start:.2f}s")
            logger.info("[STATE_TRANSITION] from=KNOWLEDGE_INTEGRATION to=ANSWER_GENERATION")

            stage_start = time.time()
            logger.info("[STAGE_ENTER] stage=ANSWER_GENERATION")
            logger.info(f"[STAGE_EXIT] stage=ANSWER_GENERATION, duration={time.time() - stage_start:.2f}s")
            logger.info("[STATE_TRANSITION] from=ANSWER_GENERATION to=STREAMING")

            logger.info("[ConsultService.process_consult] 调用Agent执行7环节流程(INITIAL→QUERY_PARSE→KNOWLEDGE_RETRIEVAL→KNOWLEDGE_INTEGRATION→ANSWER_GENERATION→STREAMING→FINISHED)")
            result = self._agent.run(context)
            logger.info(f"[ConsultService.process_consult] Agent执行完成, session_id={context.session_id}")
            return result
        finally:
            agent_resource = self._agent.resources
            if agent_resource is not None and agent_resource.model_service is not None:
                try:
                    agent_resource.model_service.release_model()
                except Exception as e:
                    logger.error(f"[ConsultService.process_consult] 释放model service失败: {e}")
            logger.info("[STATE_TRANSITION] from=STREAMING to=FINISHED")
            stage_start = time.time()
            logger.info("[STAGE_ENTER] stage=FINISHED")
            elapsed = time.time() - start_time
            logger.info(f"[ConsultService.process_consult] FINISHED环节: 咨询处理完成, session_id={context.session_id}, elapsed={elapsed:.2f}s")
            logger.info(f"[STAGE_EXIT] stage=FINISHED, duration={time.time() - stage_start:.2f}s")

    async def process_consult_stream(self, context: ConsultContext) -> AsyncGenerator[str, None]:
        import json

        if context is None:
            yield f"event: error\ndata: {json.dumps({'error_code': 400, 'error_message': 'context不能为None'})}\n\n"
            return

        if not hasattr(context, 'session_id') or not context.session_id:
            yield f"event: error\ndata: {json.dumps({'error_code': 400, 'error_message': 'session_id不能为空'})}\n\n"
            return

        logger.info(f"[ConsultService.process_consult_stream] 健康咨询业务7环节流式流程启动: session_id={context.session_id}")
        log_arch_event(
            logger,
            component="ConsultService",
            stage="SERVICE",
            event="process_consult_stream",
            status="start",
            design_id="ARCH-2.1",
        )
        stage_start = time.time()
        logger.info("[STAGE_ENTER] stage=INITIAL")
        logger.info(f"[ConsultService.process_consult_stream] INITIAL环节: 接收流式请求, session_id={context.session_id}")
        logger.info(f"[STAGE_EXIT] stage=INITIAL, duration={time.time() - stage_start:.2f}s")
        logger.info("[STATE_TRANSITION] from=INITIAL to=QUERY_PARSE")

        try:
            stage_start = time.time()
            logger.info("[STAGE_ENTER] stage=QUERY_PARSE")
            logger.info(f"[STAGE_EXIT] stage=QUERY_PARSE, duration={time.time() - stage_start:.2f}s")
            logger.info("[STATE_TRANSITION] from=QUERY_PARSE to=KNOWLEDGE_RETRIEVAL")

            stage_start = time.time()
            logger.info("[STAGE_ENTER] stage=KNOWLEDGE_RETRIEVAL")
            logger.info(f"[STAGE_EXIT] stage=KNOWLEDGE_RETRIEVAL, duration={time.time() - stage_start:.2f}s")
            logger.info("[STATE_TRANSITION] from=KNOWLEDGE_RETRIEVAL to=KNOWLEDGE_INTEGRATION")

            stage_start = time.time()
            logger.info("[STAGE_ENTER] stage=KNOWLEDGE_INTEGRATION")
            logger.info(f"[STAGE_EXIT] stage=KNOWLEDGE_INTEGRATION, duration={time.time() - stage_start:.2f}s")
            logger.info("[STATE_TRANSITION] from=KNOWLEDGE_INTEGRATION to=ANSWER_GENERATION")

            logger.info("[ConsultService.process_consult_stream] 调用Agent执行7环节流程(INITIAL→QUERY_PARSE→KNOWLEDGE_RETRIEVAL→KNOWLEDGE_INTEGRATION→ANSWER_GENERATION→STREAMING→FINISHED)")
            result = await asyncio.to_thread(run_with_context, self._agent.run, context)

            if result is not None and result.data is not None:
                body = context.body

                if hasattr(body, 'stream_generator') and body.stream_generator is not None:
                    answer_gen_start_time = time.time()
                    logger.info("[STATE_TRANSITION] from=ANSWER_GENERATION to=STREAMING")
                    logger.info("[STAGE_ENTER] stage=ANSWER_GENERATION(stream)")
                    logger.info(f"[ConsultService.process_consult_stream] STREAMING环节: 开始流式返回, session_id={context.session_id}")

                    streaming_timeout = get_runtime_config().state_timeouts.get("STREAMING", 30)
                    stream_failed = False
                    stream_degraded_reason = ""
                    try:
                        generator = body.stream_generator.__aiter__()
                        while True:
                            try:
                                token = await asyncio.wait_for(generator.__anext__(), timeout=streaming_timeout)
                            except StopAsyncIteration:
                                break
                            payload = json.dumps({"content": token}, ensure_ascii=False)
                            yield f"event: message\ndata: {payload}\n\n"
                    except asyncio.TimeoutError:
                        logger.error(f"[ConsultService.process_consult_stream] 流式生成超时: session_id={context.session_id}, timeout={streaming_timeout}s")
                        logger.warning("[DEGRADE_TRIGGER] reason=LLM流式生成超时, level=llm_to_template, from_state=ANSWER_GENERATION")
                        stream_failed = True
                        stream_degraded_reason = "LLM流式生成超时，降级为模板回答"
                    except Exception as stream_err:
                        logger.error(f"[ConsultService.process_consult_stream] 流式生成异常: session_id={context.session_id}, error={str(stream_err)}")
                        logger.warning("[DEGRADE_TRIGGER] reason=LLM流式生成失败, level=llm_to_template, from_state=ANSWER_GENERATION")
                        stream_failed = True
                        stream_degraded_reason = "LLM流式生成失败，降级为模板回答"

                    if stream_failed:
                        body.degraded = True
                        body.degraded_reason = stream_degraded_reason
                        body.error_code = ErrorCode.LLM_FAILURE
                        body.error_message = body.degraded_reason
                        logger.info(f"[DEGRADE_MARK] mark=degraded, propagated_to=context, reason={body.degraded_reason}")
                        template_answer = self._generate_template_answer(body)
                        payload = json.dumps({"content": template_answer}, ensure_ascii=False)
                        yield f"event: message\ndata: {payload}\n\n"
                        logger.info(f"[ConsultService.process_consult_stream] 已降级为模板回答: session_id={context.session_id}")

                    context.is_streaming = False
                    answer_gen_elapsed = time.time() - answer_gen_start_time
                    logger.info(f"[STAGE_EXIT] stage=ANSWER_GENERATION(stream), duration={answer_gen_elapsed:.2f}s")
                    logger.info(f"[ConsultService.process_consult_stream] STREAMING环节: 流式返回完成, session_id={context.session_id}, duration={answer_gen_elapsed:.2f}s")

                    logger.info("[STATE_TRANSITION] from=STREAMING to=FINISHED")
                    stage_start = time.time()
                    logger.info("[STAGE_ENTER] stage=FINISHED")
                    logger.info(f"[ConsultService.process_consult_stream] FINISHED环节: 组装结束响应, session_id={context.session_id}")
                    end_data = {
                        "session_id": result.session_id,
                        "intent_label": getattr(body, 'intent_label', ''),
                        "sources": getattr(body, 'sources', []),
                        "error_code": getattr(body, 'error_code', 0),
                    }
                    yield f"event: end\ndata: {json.dumps(end_data, ensure_ascii=False)}\n\n"
                    logger.info(f"[STAGE_EXIT] stage=FINISHED, duration={time.time() - stage_start:.2f}s")
                else:
                    answer = result.data.answer if hasattr(result.data, 'answer') else str(result.data)
                    payload = json.dumps({"content": answer}, ensure_ascii=False)
                    yield f"event: message\ndata: {payload}\n\n"

                    end_data = {
                        "session_id": result.session_id,
                    }
                    yield f"event: end\ndata: {json.dumps(end_data, ensure_ascii=False)}\n\n"
            else:
                logger.error(f"[ConsultService.process_consult_stream] Agent返回空结果, session_id={context.session_id}")
                yield f"event: error\ndata: {json.dumps({'error_code': 500, 'error_message': '处理结果为空'})}\n\n"

        except Exception as e:
            logger.error(f"[ConsultService.process_consult_stream] 流式处理异常: session_id={context.session_id}, error={str(e)}")
            yield f"event: error\ndata: {json.dumps({'error_code': 500, 'error_message': '服务内部错误，请稍后重试'})}\n\n"

        finally:
            self._release_resources()

    def _generate_template_answer(self, body: ConsultContextBody) -> str:
        """LLM不可用时生成模板回答"""
        template = f"关于您咨询的「{body.question}」：\n\n"
        if body.knowledge_results:
            for item in body.knowledge_results[:3]:
                entity = item.get("entity", "")
                data = item.get("data", {})
                if isinstance(data, dict):
                    desc = data.get("description", "")
                    if desc:
                        template += f"- {entity}：{desc}\n"
        template += "\n以上信息仅供参考，不构成医疗建议。如有健康问题，请及时就医。"
        return template

    def _release_resources(self) -> None:
        agent_resource = self._agent.resources
        if agent_resource is not None and agent_resource.model_service is not None:
            try:
                agent_resource.model_service.release_model()
            except Exception as e:
                logger.error(f"[ConsultService._release_resources] 释放model service失败: {e}")

    def build_agent_context(self, request: ConsultRequest) -> AgentContext:
        """
        从ConsultRequest构建AgentContext（公开方法）
        
        Args:
            request: 健康咨询请求数据
        
        Returns:
            AgentContext: Agent输入数据容器
        """
        session_id = request.get_session_id() or request.body.task_id

        logger.info(f"[ConsultService.build_agent_context] 构建Agent上下文: session_id={session_id}")

        conversation_history = request.get_conversation_history() or []
        if not conversation_history and request.body.chat_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.body.chat_history
            ]

        context_body = ConsultContextBody(
            question=request.get_question(),
            session_id=session_id,
            conversation_history=conversation_history,
            user_profile=request.get_user_profile() or {},
            current_state="INITIAL"
        )

        agent_context = AgentContext(
            session_id=session_id,
            current_state="INITIAL",
            body=context_body
        )

        logger.info(f"[ConsultService.build_agent_context] Agent上下文构建完成: session_id={session_id}, current_state=INITIAL, question={request.get_question()[:100]}")

        return agent_context

    def __repr__(self) -> str:
        return f"ConsultService(agent={self._agent})"
