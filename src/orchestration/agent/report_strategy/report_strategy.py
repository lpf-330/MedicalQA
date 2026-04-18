# -*- coding: utf-8 -*-
"""
健康报告生成策略

实现健康报告生成业务的报告策略类，包含ReportContextBody和ReportResultData数据类。
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.orchestration.agent.agent_strategy import AgentStrategy
from src.orchestration.agent.data_classes import AgentContext, AgentResult
from src.orchestration.agent.agent_resource import AgentResource
from src.orchestration.state_machine.state_machine import StateMachine
from src.orchestration.chain.data_classes import ChainContext
from src.orchestration.chain.data_prepare_chain.data_prepare_chain import DataPrepareContextBody
from src.orchestration.chain.multi_analysis_chain.multi_analysis_chain import MultiAnalysisContextBody
from src.orchestration.chain.integration_chain.integration_chain import IntegrationContextBody
from src.orchestration.chain.report_generation_chain.report_generation_chain import ReportGenerationContextBody
from src.orchestration.chain.dimension_evaluation_chain.dimension_evaluation_chain import DimensionEvaluationContextBody

logger = logging.getLogger(__name__)


@dataclass
class ReportContextBody:
    """
    报告策略上下文数据类

    Attributes:
        task_id: 任务ID
        monitoring_data: 监测数据
        user_profile: 用户档案
        session_id: 会话ID
        current_state: 当前状态，默认"INITIAL"
        validated_data: 校验后的数据
        degradation_level: 降级级别
        anomalies: 异常指标
        risk_factors: 风险因子
        medical_entities: 医疗实体
        dimension_results: 8个维度的评估结果
        knowledge_results: 知识检索结果
        health_score: 健康评分
        health_level: 健康等级
        risk_level: 风险等级
        risk_diseases: 风险疾病
        report_content: 报告内容
        sources: 知识来源
        error_code: 错误码
        error_message: 错误消息
        stream_generator: 流式生成器
        is_streaming: 是否流式输出
    """
    task_id: str = ""
    monitoring_data: Dict = field(default_factory=dict)
    user_profile: Dict = field(default_factory=dict)
    session_id: str = ""
    current_state: str = "INITIAL"
    validated_data: Dict = field(default_factory=dict)
    degradation_level: int = 0
    anomalies: List[Dict] = field(default_factory=list)
    risk_factors: List[Dict] = field(default_factory=list)
    medical_entities: List[Dict] = field(default_factory=list)
    dimension_results: Dict[str, Dict] = field(default_factory=dict)
    knowledge_results: List[Dict] = field(default_factory=list)
    health_score: int = 0
    health_level: str = ""
    risk_level: str = ""
    risk_diseases: List[Dict] = field(default_factory=list)
    report_content: str = ""
    sources: List[str] = field(default_factory=list)
    error_code: int = 0
    error_message: str = ""
    stream_generator: Any = None
    is_streaming: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "monitoring_data": self.monitoring_data,
            "user_profile": self.user_profile,
            "session_id": self.session_id,
            "current_state": self.current_state,
            "validated_data": self.validated_data,
            "degradation_level": self.degradation_level,
            "anomalies": self.anomalies,
            "risk_factors": self.risk_factors,
            "medical_entities": self.medical_entities,
            "dimension_results": self.dimension_results,
            "knowledge_results": self.knowledge_results,
            "health_score": self.health_score,
            "health_level": self.health_level,
            "risk_level": self.risk_level,
            "risk_diseases": self.risk_diseases,
            "report_content": self.report_content,
            "sources": self.sources,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "stream_generator": self.stream_generator,
            "is_streaming": self.is_streaming
        }


@dataclass
class ReportResultData:
    """
    报告策略结果数据类

    Attributes:
        report: 报告内容
        health_score: 健康评分
        health_level: 健康等级
        risk_level: 风险等级
        risk_diseases: 风险疾病
        sources: 知识来源
        word_count: 报告字数
        session_id: 会话ID
        dimension_results: 各维度评估结果
        error_code: 错误码
        error_message: 错误消息
    """
    report: str = ""
    health_score: int = 0
    health_level: str = ""
    risk_level: str = ""
    risk_diseases: List[Dict] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    word_count: int = 0
    session_id: str = ""
    dimension_results: Dict = field(default_factory=dict)
    error_code: int = 0
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "report": self.report,
            "health_score": self.health_score,
            "health_level": self.health_level,
            "risk_level": self.risk_level,
            "risk_diseases": self.risk_diseases,
            "sources": self.sources,
            "word_count": self.word_count,
            "session_id": self.session_id,
            "dimension_results": self.dimension_results,
            "error_code": self.error_code,
            "error_message": self.error_message
        }


class ReportStrategy(AgentStrategy[ReportContextBody, ReportResultData]):
    """
    报告策略类

    继承AgentStrategy[ReportContextBody, ReportResultData]，实现10状态有限状态机(FSM)：
    - INITIAL（初始状态）
    - DATA_PREPARE（数据准备）
    - MULTI_ANALYSIS（模型分析）
    - PARALLEL_PROCESSING（并行处理）
    - INTEGRATION（整合计算）
    - REPORT_GENERATION（报告生成）
    - STREAMING（流式返回）
    - ASSEMBLY（组装结束）
    - FINISHED（完成状态）
    - ERROR（错误状态）
    """

    # 各状态超时时间配置（秒）
    _STATE_TIMEOUTS = {
        "DATA_PREPARE": 15,
        "MULTI_ANALYSIS": 30,
        "PARALLEL_PROCESSING": 60,
        "INTEGRATION": 20,
        "REPORT_GENERATION": 90,
        "STREAMING": 120,
        "ASSEMBLY": 5,
    }

    def execute(
        self,
        context: AgentContext[ReportContextBody],
        resource: AgentResource
    ) -> AgentResult[ReportResultData]:
        """
        执行报告策略

        Args:
            context: Agent输入数据容器
            resource: Agent资源类

        Returns:
            AgentResult: Agent输出数据容器
        """
        start_time = time.time()
        logger.info(f"[ReportStrategy] 开始执行策略: session_id={context.session_id}")

        body = context.body
        if body is None:
            logger.warning(f"[ReportStrategy] 输入数据为空: session_id={context.session_id}")
            return AgentResult(
                session_id=context.session_id,
                data=ReportResultData(
                    report="输入数据为空",
                    error_code=1,
                    error_message="输入数据为空"
                )
            )

        # 创建状态机
        state_machine = StateMachine(context.session_id)
        self._register_state_transitions(state_machine)

        # 获取当前状态
        current_state = body.current_state if body.current_state else "INITIAL"
        body.current_state = current_state

        # 注册状态处理器
        self._state_handlers = {
            "INITIAL": self._handle_initial,
            "DATA_PREPARE": self._handle_data_prepare,
            "MULTI_ANALYSIS": self._handle_multi_analysis,
            "PARALLEL_PROCESSING": self._handle_parallel_processing,
            "INTEGRATION": self._handle_integration,
            "REPORT_GENERATION": self._handle_report_generation,
            "STREAMING": self._handle_streaming,
            "ASSEMBLY": self._handle_assembly,
            "FINISHED": self._handle_finished,
            "ERROR": self._handle_error,
        }

        # 状态循环驱动
        max_iterations = 30
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"[ReportStrategy] 状态转换: current_state={current_state}, iteration={iteration}")

            handler = self._state_handlers.get(current_state)
            if handler is None:
                logger.error(f"[ReportStrategy] 未知状态: {current_state}")
                body.current_state = "ERROR"
                body.error_code = 9999
                body.error_message = f"未知状态: {current_state}"
                break

            # 执行状态处理器（带超时控制）
            timeout = self._STATE_TIMEOUTS.get(current_state)
            try:
                if timeout:
                    next_state = self._execute_with_timeout(handler, body, resource, timeout)
                else:
                    next_state = handler(body, resource)
            except TimeoutError as te:
                logger.error(f"[ReportStrategy] 状态超时: state={current_state}, timeout={timeout}s")
                next_state = self._handle_timeout(body, current_state, te)
            except Exception as e:
                logger.error(f"[ReportStrategy] 状态处理异常: state={current_state}, error={str(e)}")
                next_state = self._handle_error_state(body, e)

            # 状态转换
            state_machine.transition(current_state, next_state)
            current_state = next_state
            body.current_state = current_state

            # 检查是否到达终止状态
            if current_state in ("FINISHED", "ERROR"):
                break

        # 如果最终状态是ERROR，转换为FINISHED
        if current_state == "ERROR":
            current_state = "FINISHED"
            body.current_state = current_state

        # 构建结果
        result_data = self._build_result(body)

        elapsed = time.time() - start_time
        logger.info(f"[ReportStrategy] 策略执行完成: session_id={context.session_id}, "
                    f"health_score={result_data.health_score}, elapsed={elapsed:.2f}s")

        return AgentResult(session_id=context.session_id, data=result_data)

    def _register_state_transitions(self, state_machine: StateMachine):
        """
        注册状态转换规则

        Args:
            state_machine: 状态机实例
        """
        state_machine.add_state_transition("INITIAL", ["DATA_PREPARE"])
        state_machine.add_state_transition("DATA_PREPARE", ["MULTI_ANALYSIS", "ERROR"])
        state_machine.add_state_transition("MULTI_ANALYSIS", ["PARALLEL_PROCESSING"])
        state_machine.add_state_transition("PARALLEL_PROCESSING", ["INTEGRATION"])
        state_machine.add_state_transition("INTEGRATION", ["REPORT_GENERATION"])
        state_machine.add_state_transition("REPORT_GENERATION", ["STREAMING", "ERROR"])
        state_machine.add_state_transition("STREAMING", ["ASSEMBLY", "ERROR"])
        state_machine.add_state_transition("ASSEMBLY", ["FINISHED"])
        state_machine.add_state_transition("ERROR", ["FINISHED"])

    def _execute_with_timeout(self, handler, context, resource, timeout_seconds):
        """
        带超时控制的状态执行

        Args:
            handler: 状态处理器
            context: 上下文数据
            resource: 资源类
            timeout_seconds: 超时时间（秒）

        Returns:
            下一个状态

        Raises:
            TimeoutError: 执行超时时抛出
        """
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(handler, context, resource)
            try:
                return future.result(timeout=timeout_seconds)
            except FuturesTimeoutError:
                raise TimeoutError(f"状态执行超时，超过{timeout_seconds}秒")

    def _handle_initial(self, context: ReportContextBody, resource: AgentResource) -> str:
        """
        INITIAL状态处理

        Args:
            context: 上下文数据
            resource: 资源类

        Returns:
            下一个状态
        """
        logger.info(f"[ReportStrategy] INITIAL: task_id={context.task_id}")
        return "DATA_PREPARE"

    def _handle_data_prepare(self, context: ReportContextBody, resource: AgentResource) -> str:
        """
        DATA_PREPARE状态处理，调用DataPrepareChain

        Args:
            context: 上下文数据
            resource: 资源类

        Returns:
            下一个状态
        """
        logger.info(f"[ReportStrategy] DATA_PREPARE: 开始数据准备")

        # 获取DataPrepareChain
        data_prepare_chain = resource.get_chain("data_prepare_chain")
        if data_prepare_chain is None:
            logger.error("[ReportStrategy] DataPrepareChain未注册")
            context.error_code = 2001
            context.error_message = "DataPrepareChain未注册"
            return "ERROR"

        # 构建ChainContext
        chain_context = ChainContext(
            session_id=context.session_id,
            body=DataPrepareContextBody(
                monitoring_data=context.monitoring_data,
                user_profile=context.user_profile,
                task_id=context.task_id
            )
        )

        # 执行Chain
        chain_result = data_prepare_chain.execute(chain_context)
        if chain_result.data is None:
            logger.error("[ReportStrategy] DataPrepareChain返回空结果")
            context.error_code = 2002
            context.error_message = "DataPrepareChain返回空结果"
            return "ERROR"

        # 更新上下文
        context.validated_data = chain_result.data.validated_data
        context.degradation_level = chain_result.data.degradation_level

        logger.info(f"[ReportStrategy] DATA_PREPARE完成: degradation_level={context.degradation_level}")

        return "MULTI_ANALYSIS"

    def _handle_multi_analysis(self, context: ReportContextBody, resource: AgentResource) -> str:
        """
        MULTI_ANALYSIS状态处理，调用MultiAnalysisChain

        Args:
            context: 上下文数据
            resource: 资源类

        Returns:
            下一个状态
        """
        logger.info(f"[ReportStrategy] MULTI_ANALYSIS: 开始多维度分析")

        # 获取MultiAnalysisChain
        multi_analysis_chain = resource.get_chain("multi_analysis_chain")
        if multi_analysis_chain is None:
            logger.error("[ReportStrategy] MultiAnalysisChain未注册")
            context.error_code = 2003
            context.error_message = "MultiAnalysisChain未注册"
            return "ERROR"

        # 构建ChainContext
        chain_context = ChainContext(
            session_id=context.session_id,
            body=MultiAnalysisContextBody(
                validated_data=context.validated_data,
                degradation_level=context.degradation_level
            )
        )

        # 执行Chain
        chain_result = multi_analysis_chain.execute(chain_context)
        if chain_result.data is None:
            logger.error("[ReportStrategy] MultiAnalysisChain返回空结果")
            context.error_code = 2004
            context.error_message = "MultiAnalysisChain返回空结果"
            return "ERROR"

        # 更新上下文
        context.anomalies = chain_result.data.anomalies
        context.risk_factors = chain_result.data.risk_factors
        context.medical_entities = chain_result.data.medical_entities

        logger.info(f"[ReportStrategy] MULTI_ANALYSIS完成: "
                    f"anomalies={len(context.anomalies)}, "
                    f"risk_factors={len(context.risk_factors)}, "
                    f"medical_entities={len(context.medical_entities)}")

        return "PARALLEL_PROCESSING"

    def _handle_parallel_processing(self, context: ReportContextBody, resource: AgentResource) -> str:
        """
        PARALLEL_PROCESSING状态处理，双路并发

        双路并发：
        - 路径A：并发执行8个维度评估
        - 路径B：顺序检索任务

        Args:
            context: 上下文数据
            resource: 资源类

        Returns:
            下一个状态
        """
        logger.info(f"[ReportStrategy] PARALLEL_PROCESSING: 开始双路并发处理")

        # 使用线程池并发执行双路处理
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 路径A：并发执行8个维度评估
            future_dimensions = executor.submit(
                self._execute_parallel_dimensions, context, resource
            )

            # 路径B：顺序检索任务
            future_retrieval = executor.submit(
                self._execute_sequential_retrieval, context, resource
            )

            # 等待双路完成
            try:
                dimension_results = future_dimensions.result(timeout=50)
                context.dimension_results = dimension_results
                logger.info(f"[ReportStrategy] 维度评估完成: dimension_count={len(dimension_results)}")
            except Exception as e:
                logger.error(f"[ReportStrategy] 维度评估失败: {str(e)}")
                context.dimension_results = {}

            try:
                knowledge_results = future_retrieval.result(timeout=50)
                context.knowledge_results = knowledge_results
                logger.info(f"[ReportStrategy] 知识检索完成: knowledge_count={len(knowledge_results)}")
            except Exception as e:
                logger.error(f"[ReportStrategy] 知识检索失败: {str(e)}")
                context.knowledge_results = []

        logger.info(f"[ReportStrategy] PARALLEL_PROCESSING完成")

        return "INTEGRATION"

    def _handle_integration(self, context: ReportContextBody, resource: AgentResource) -> str:
        """
        INTEGRATION状态处理，调用IntegrationChain

        Args:
            context: 上下文数据
            resource: 资源类

        Returns:
            下一个状态
        """
        logger.info(f"[ReportStrategy] INTEGRATION: 开始整合计算")

        # 获取IntegrationChain
        integration_chain = resource.get_chain("integration_chain")
        if integration_chain is None:
            logger.error("[ReportStrategy] IntegrationChain未注册")
            context.error_code = 2005
            context.error_message = "IntegrationChain未注册"
            return "ERROR"

        # 构建ChainContext
        chain_context = ChainContext(
            session_id=context.session_id,
            body=IntegrationContextBody(
                dimension_results=context.dimension_results,
                knowledge_results=context.knowledge_results,
                anomalies=context.anomalies,
                risk_factors=context.risk_factors
            )
        )

        # 执行Chain
        chain_result = integration_chain.execute(chain_context)
        if chain_result.data is None:
            logger.error("[ReportStrategy] IntegrationChain返回空结果")
            context.error_code = 2006
            context.error_message = "IntegrationChain返回空结果"
            return "ERROR"

        # 更新上下文
        context.health_score = chain_result.data.health_score
        context.health_level = chain_result.data.health_level
        context.risk_level = chain_result.data.risk_level
        context.risk_diseases = chain_result.data.risk_diseases

        logger.info(f"[ReportStrategy] INTEGRATION完成: "
                    f"health_score={context.health_score}, "
                    f"health_level={context.health_level}, "
                    f"risk_level={context.risk_level}")

        return "REPORT_GENERATION"

    def _handle_report_generation(self, context: ReportContextBody, resource: AgentResource) -> str:
        """
        REPORT_GENERATION状态处理，调用ReportGenerationChain

        Args:
            context: 上下文数据
            resource: 资源类

        Returns:
            下一个状态
        """
        logger.info(f"[ReportStrategy] REPORT_GENERATION: 开始报告生成")

        # 获取ReportGenerationChain
        report_generation_chain = resource.get_chain("report_generation_chain")
        if report_generation_chain is None:
            logger.error("[ReportStrategy] ReportGenerationChain未注册")
            context.error_code = 2007
            context.error_message = "ReportGenerationChain未注册"
            return "ERROR"

        # 构建报告素材
        report_materials = {
            "monitoring_data": context.monitoring_data,
            "merged_results": context.knowledge_results,
            "dimension_results": context.dimension_results,
            "anomalies": context.anomalies,
            "risk_factors": context.risk_factors
        }

        # 构建ChainContext
        chain_context = ChainContext(
            session_id=context.session_id,
            body=ReportGenerationContextBody(
                report_materials=report_materials,
                health_score=context.health_score,
                health_level=context.health_level,
                risk_level=context.risk_level,
                risk_diseases=context.risk_diseases,
                user_profile=context.user_profile,
                monitoring_data=context.monitoring_data
            )
        )

        # 执行流式生成
        try:
            context.stream_generator = report_generation_chain.execute_stream(chain_context)
            context.is_streaming = True
            logger.info(f"[ReportStrategy] REPORT_GENERATION: 流式生成器已创建")
        except Exception as e:
            logger.error(f"[ReportStrategy] 流式生成失败: {str(e)}")
            # 降级为普通生成
            try:
                chain_result = report_generation_chain.execute(chain_context)
                if chain_result.data:
                    context.report_content = chain_result.data.report_content
                    context.sources = chain_result.data.sources
                    logger.info(f"[ReportStrategy] REPORT_GENERATION: 普通生成完成")
                    return "ASSEMBLY"
                else:
                    # 使用模板报告降级
                    context.report_content = self._generate_template_report(context)
                    logger.info(f"[ReportStrategy] REPORT_GENERATION: 模板报告生成完成")
                    return "ASSEMBLY"
            except Exception as e2:
                logger.error(f"[ReportStrategy] 普通生成也失败: {str(e2)}")
                context.report_content = self._generate_template_report(context)
                return "ASSEMBLY"

        return "STREAMING"

    def _handle_streaming(self, context: ReportContextBody, resource: AgentResource) -> str:
        """
        STREAMING状态处理

        Args:
            context: 上下文数据
            resource: 资源类

        Returns:
            下一个状态
        """
        logger.info(f"[ReportStrategy] STREAMING: 流式输出状态")

        # 如果有流式生成器，收集完整内容
        if context.stream_generator is not None:
            try:
                full_content = []
                for chunk in context.stream_generator:
                    if isinstance(chunk, str):
                        full_content.append(chunk)
                context.report_content = "".join(full_content)
                logger.info(f"[ReportStrategy] STREAMING: 流式内容收集完成, length={len(context.report_content)}")
            except Exception as e:
                logger.error(f"[ReportStrategy] 流式内容收集失败: {str(e)}")

        return "ASSEMBLY"

    def _handle_assembly(self, context: ReportContextBody, resource: AgentResource) -> str:
        """
        ASSEMBLY状态处理，组装结束

        Args:
            context: 上下文数据
            resource: 资源类

        Returns:
            下一个状态
        """
        logger.info(f"[ReportStrategy] ASSEMBLY: 组装结束")

        # 确保报告内容不为空
        if not context.report_content:
            context.report_content = self._generate_template_report(context)
            logger.info(f"[ReportStrategy] ASSEMBLY: 使用模板报告")

        # 提取知识来源
        if not context.sources and context.knowledge_results:
            for item in context.knowledge_results:
                entity = item.get("entity", "")
                if entity and entity not in context.sources:
                    context.sources.append(entity)

        logger.info(f"[ReportStrategy] ASSEMBLY完成: report_length={len(context.report_content)}")

        return "FINISHED"

    def _handle_finished(self, context: ReportContextBody, resource: AgentResource) -> str:
        """
        FINISHED状态处理

        Args:
            context: 上下文数据
            resource: 资源类

        Returns:
            下一个状态
        """
        logger.info(f"[ReportStrategy] FINISHED: 策略执行结束")
        return "FINISHED"

    def _handle_error(self, context: ReportContextBody, resource: AgentResource) -> str:
        """
        ERROR状态处理，降级策略

        Args:
            context: 上下文数据
            resource: 资源类

        Returns:
            下一个状态
        """
        logger.error(f"[ReportStrategy] ERROR: error_code={context.error_code}, error_message={context.error_message}")

        # 根据错误码执行降级策略
        if context.error_code >= 2000:
            # Chain相关错误，生成模板报告
            context.report_content = self._generate_template_report(context)
            logger.info("[ReportStrategy] 降级策略: 使用模板报告")

        return "FINISHED"

    def _handle_error_state(self, context: ReportContextBody, error: Exception) -> str:
        """
        处理错误状态

        Args:
            context: 上下文数据
            error: 异常对象

        Returns:
            下一个状态
        """
        error_message = str(error)
        context.error_message = error_message

        # 根据错误类型设置错误码
        if "DataPrepare" in error_message:
            context.error_code = 3001
        elif "MultiAnalysis" in error_message:
            context.error_code = 3002
        elif "Integration" in error_message:
            context.error_code = 3003
        elif "ReportGeneration" in error_message or "LLM" in error_message or "model" in error_message.lower():
            context.error_code = 3004
            # LLM故障降级：生成模板报告
            context.report_content = self._generate_template_report(context)
            logger.warning("[ReportStrategy] 降级策略: LLM故障，使用模板报告")
            return "ASSEMBLY"
        else:
            context.error_code = 9999

        return "ERROR"

    def _handle_timeout(self, context: ReportContextBody, state: str, error: TimeoutError) -> str:
        """
        处理超时状态

        Args:
            context: 上下文数据
            state: 当前状态
            error: 超时异常

        Returns:
            下一个状态
        """
        logger.warning(f"[ReportStrategy] 超时降级: state={state}")

        if state == "DATA_PREPARE":
            context.error_code = 40001
            context.error_message = "数据准备超时"
            return "ERROR"
        elif state == "MULTI_ANALYSIS":
            context.error_code = 40002
            context.error_message = "多维度分析超时"
            return "ERROR"
        elif state == "PARALLEL_PROCESSING":
            logger.warning("[ReportStrategy] 并行处理超时，使用已有部分结果继续")
            context.error_code = 40003
            context.error_message = "并行处理超时，使用部分结果"
            return "INTEGRATION"
        elif state == "INTEGRATION":
            context.error_code = 40004
            context.error_message = "整合计算超时"
            return "ERROR"
        elif state == "REPORT_GENERATION":
            logger.warning("[ReportStrategy] 报告生成超时，降级为模板报告")
            context.error_code = 40005
            context.error_message = "报告生成超时，使用模板报告"
            context.report_content = self._generate_template_report(context)
            return "ASSEMBLY"
        elif state == "STREAMING":
            logger.warning("[ReportStrategy] 流式输出超时，降级为模板报告")
            context.error_code = 40007
            context.error_message = "流式输出超时，使用模板报告"
            if not context.report_content:
                context.report_content = self._generate_template_report(context)
            return "ASSEMBLY"
        else:
            context.error_code = 40006
            context.error_message = f"状态{state}执行超时"
            return "ERROR"

    def _execute_parallel_dimensions(self, context: ReportContextBody, resource: AgentResource) -> Dict[str, Dict]:
        """
        并发执行8个维度评估

        Args:
            context: 上下文数据
            resource: 资源类

        Returns:
            8个维度的评估结果
        """
        logger.info("[ReportStrategy] 开始并发执行8个维度评估")

        dimension_results = {}

        # 获取DimensionEvaluationChain
        dimension_chain = resource.get_chain("dimension_evaluation_chain")
        if dimension_chain is None:
            logger.warning("[ReportStrategy] DimensionEvaluationChain未注册，跳过维度评估")
            return dimension_results

        # 使用线程池并发执行8个维度评估
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}
            for dim_id in range(1, 9):
                dim_key = f"dimension_{dim_id}"
                future = executor.submit(
                    self._evaluate_single_dimension,
                    dimension_chain,
                    context,
                    str(dim_id),
                    resource
                )
                futures[future] = dim_key

            # 收集结果
            for future in futures:
                dim_key = futures[future]
                try:
                    result = future.result(timeout=10)
                    if result:
                        dimension_results[dim_key] = result
                        logger.info(f"[ReportStrategy] 维度{dim_key}评估完成")
                except Exception as e:
                    logger.error(f"[ReportStrategy] 维度{dim_key}评估失败: {str(e)}")
                    dimension_results[dim_key] = {"error": str(e)}

        logger.info(f"[ReportStrategy] 8个维度评估完成: completed={len(dimension_results)}/8")

        return dimension_results

    def _evaluate_single_dimension(
        self,
        dimension_chain,
        context: ReportContextBody,
        dimension_id: str,
        resource: AgentResource
    ) -> Dict:
        """
        评估单个维度

        Args:
            dimension_chain: 维度评估Chain
            context: 上下文数据
            dimension_id: 维度ID
            resource: 资源类

        Returns:
            维度评估结果
        """
        chain_context = ChainContext(
            session_id=context.session_id,
            body=DimensionEvaluationContextBody(
                anomalies=context.anomalies,
                risk_factors=context.risk_factors,
                medical_entities=context.medical_entities,
                dimension_id=dimension_id
            )
        )

        chain_result = dimension_chain.execute(chain_context)
        if chain_result.data:
            return chain_result.data.to_dict()
        return {}

    def _execute_sequential_retrieval(self, context: ReportContextBody, resource: AgentResource) -> List[Dict]:
        """
        顺序检索任务

        Args:
            context: 上下文数据
            resource: 资源类

        Returns:
            知识检索结果
        """
        logger.info("[ReportStrategy] 开始顺序检索任务")

        knowledge_results = []

        # 获取ReportKnowledgeRetrievalChain
        retrieval_chain = resource.get_chain("report_knowledge_retrieval_chain")
        if retrieval_chain is None:
            logger.warning("[ReportStrategy] ReportKnowledgeRetrievalChain未注册，跳过知识检索")
            return knowledge_results

        # 构建检索查询
        query_parts = []

        # 基于异常指标构建查询
        if context.anomalies:
            anomaly_names = [a.get("indicator_name", "") for a in context.anomalies if a.get("indicator_name")]
            if anomaly_names:
                query_parts.append(f"异常指标: {', '.join(anomaly_names)}")

        # 基于风险因子构建查询
        if context.risk_factors:
            factor_names = [f.get("factor_name", "") for f in context.risk_factors if f.get("factor_name")]
            if factor_names:
                query_parts.append(f"风险因子: {', '.join(factor_names)}")

        # 基于医疗实体构建查询
        if context.medical_entities:
            entity_names = [e.get("entity_name", "") for e in context.medical_entities if e.get("entity_name")]
            if entity_names:
                query_parts.append(f"相关疾病: {', '.join(entity_names)}")

        query_text = " ".join(query_parts) if query_parts else "健康评估"

        logger.info(f"[ReportStrategy] 检索查询: {query_text}")

        # 执行检索（这里简化处理，实际应调用ReportKnowledgeRetrievalChain）
        # 由于ReportKnowledgeRetrievalChain可能还未完全实现，这里返回空列表
        # 后续可以补充完整的检索逻辑

        logger.info(f"[ReportStrategy] 顺序检索完成: result_count={len(knowledge_results)}")

        return knowledge_results

    def _generate_template_report(self, context: ReportContextBody) -> str:
        """
        生成模板报告（LLM故障降级）

        Args:
            context: 上下文数据

        Returns:
            模板报告内容
        """
        logger.info("[ReportStrategy] 生成模板报告")

        # 用户信息
        user_info = ""
        if context.user_profile:
            user_info = f"""
**用户信息**
- 年龄：{context.user_profile.get('age', '未知')}
- 性别：{context.user_profile.get('gender', '未知')}
"""

        # 异常指标
        anomaly_text = "暂无异常指标"
        if context.anomalies:
            anomaly_items = []
            for a in context.anomalies:
                indicator = a.get("indicator_name", "")
                anomaly_type = a.get("anomaly_type", "")
                value = a.get("anomaly_value", "")
                anomaly_items.append(f"- {indicator}：{anomaly_type}（{value}）")
            anomaly_text = "\n".join(anomaly_items)

        # 风险因子
        risk_text = "暂无风险因子"
        if context.risk_factors:
            risk_items = []
            for f in context.risk_factors:
                factor_name = f.get("factor_name", "")
                risk_level = f.get("risk_level", "")
                risk_items.append(f"- {factor_name}（{risk_level}风险）")
            risk_text = "\n".join(risk_items)

        # 风险疾病
        disease_text = "暂无高风险疾病"
        if context.risk_diseases:
            disease_items = []
            for d in context.risk_diseases:
                disease_name = d.get("disease_name", d.get("name", ""))
                risk_score = d.get("risk_score", 0)
                disease_items.append(f"- {disease_name}（风险分：{risk_score}）")
            disease_text = "\n".join(disease_items)

        # 模板报告
        template_report = f"""# 健康评估报告

## 一、健康综合评分
**{context.health_score}分**（{context.health_level}）

{user_info}

## 二、监测数据分析

### 异常指标
{anomaly_text}

## 三、风险评估

### 风险等级
**{context.risk_level}**

### 风险因子
{risk_text}

### 高风险疾病
{disease_text}

## 四、健康建议

根据您的健康评分和风险评估结果，建议：
1. 定期进行健康体检，关注异常指标变化
2. 保持良好的生活习惯，合理饮食、适量运动
3. 如有不适症状，及时就医咨询
4. 遵医嘱用药，不自行调整药物剂量

## 五、免责声明
以上信息仅供参考，不构成医疗建议。如有健康问题，请及时就医。
"""

        return template_report

    def _build_result(self, context: ReportContextBody) -> ReportResultData:
        """
        构建ReportResultData

        Args:
            context: 上下文数据

        Returns:
            ReportResultData实例
        """
        result_data = ReportResultData(
            report=context.report_content,
            health_score=context.health_score,
            health_level=context.health_level,
            risk_level=context.risk_level,
            risk_diseases=context.risk_diseases,
            sources=context.sources,
            session_id=context.session_id,
            dimension_results=context.dimension_results,
            error_code=context.error_code,
            error_message=context.error_message
        )

        # 计算报告字数
        if context.report_content:
            result_data.word_count = len(context.report_content)

        return result_data
