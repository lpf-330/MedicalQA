# -*- coding: utf-8 -*-
"""
健康报告生成策略

实现健康报告生成业务的报告策略类，包含ReportContextBody和ReportResultData数据类。

基于设计文档《项目业务详细设计v5》第3.2节设计实现。

流程环节（8个环节）：
INITIAL → DATA_PREPARE → DATA_PARSE → COMPREHENSIVE_HEALTH_ANALYSIS → 
REPORT_GENERATION → STREAMING → ASSEMBLY → FINISHED
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional

from src.orchestration.agent.agent_strategy import AgentStrategy
from src.orchestration.agent.data_classes import AgentContext, AgentResult
from src.orchestration.agent.agent_resource import AgentResource
from src.orchestration.state_machine.state_machine import StateMachine
from src.orchestration.chain.data_classes import ChainContext
from src.orchestration.chain.data_prepare_chain.data_prepare_chain import DataPrepareContextBody
from src.orchestration.chain.multi_analysis_chain.multi_analysis_chain import MultiAnalysisContextBody
from src.orchestration.chain.report_generation_chain.report_generation_chain import ReportGenerationContextBody
from src.orchestration.agent.comprehensive_health_analysis_strategy.comprehensive_health_analysis_context import (
    ComprehensiveHealthAnalysisContextBody,
)
from src.orchestration.agent.comprehensive_health_analysis_strategy.comprehensive_health_analysis_strategy import (
    ComprehensiveHealthAnalysisStrategy,
)
from src.orchestration.agent.report_strategy.report_context import ReportContextBody
from src.orchestration.agent.report_strategy.report_result import ReportResultData
from src.config.business.report_service_config import get_runtime_config
from src.errors import ErrorCode, DataPrepareError, DataParseError, MultiAnalysisError, ComprehensiveAnalysisError, LLMServiceError
from src.utils.logger import log_arch_event

logger = logging.getLogger(__name__)


class ReportStrategy(AgentStrategy[ReportContextBody, ReportResultData]):
    """
    报告策略类

    继承AgentStrategy[ReportContextBody, ReportResultData]，实现8状态有限状态机(FSM)：
    - INITIAL（初始状态）
    - DATA_PREPARE（数据准备）
    - DATA_PARSE（数据解析）
    - COMPREHENSIVE_HEALTH_ANALYSIS（综合健康分析）
    - REPORT_GENERATION（报告生成）
    - STREAMING（流式返回）
    - ASSEMBLY（组装结束）
    - FINISHED（完成状态）
    - ERROR（错误状态）

    Attributes:
        _comprehensive_health_analysis_strategy: 综合健康分析策略实例（依赖注入）
    """

    def __init__(self, comprehensive_health_analysis_strategy: Optional[ComprehensiveHealthAnalysisStrategy] = None):
        self._comprehensive_health_analysis_strategy = comprehensive_health_analysis_strategy
        self._report_config = get_runtime_config()
        self._STATE_TIMEOUTS = self._report_config.state_timeouts

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
        log_arch_event(
            logger,
            component="ReportStrategy",
            stage="ORCHESTRATION",
            event="strategy_execute",
            status="start",
            design_id="BIZ-3.3",
        )

        body = context.body
        if body is None:
            logger.warning(f"[ReportStrategy] 输入数据为空: session_id={context.session_id}")
            return AgentResult(
                session_id=context.session_id,
                data=ReportResultData(
                    report="输入数据为空",
                    error_code=ErrorCode.UNKNOWN,
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
            "DATA_PARSE": self._handle_data_parse,
            "COMPREHENSIVE_HEALTH_ANALYSIS": self._handle_comprehensive_health_analysis,
            "REPORT_GENERATION": self._handle_report_generation,
            "STREAMING": self._handle_streaming,
            "ASSEMBLY": self._handle_assembly,
            "FINISHED": self._handle_finished,
            "ERROR": self._handle_error,
        }

        # 状态循环驱动
        max_iterations = self._report_config.report_max_iterations
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"[ReportStrategy] 状态转换: current_state={current_state}, iteration={iteration}")

            handler = self._state_handlers.get(current_state)
            if handler is None:
                logger.error(f"[ReportStrategy] 未知状态: {current_state}")
                body.current_state = "ERROR"
                body.error_code = ErrorCode.UNKNOWN
                body.error_message = f"未知状态: {current_state}"
                break

            # 执行状态处理器（带超时控制）
            timeout = self._STATE_TIMEOUTS.get(current_state)
            try:
                if timeout:
                    next_state = self._execute_with_timeout(handler, body, resource, timeout)
                else:
                    next_state = handler(body, resource)
                logger.info(f"[STATE_TRANSITION_REASON] current_state={current_state}, next_state={next_state}, reason=状态处理器正常返回")
            except TimeoutError as te:
                logger.error(f"[ReportStrategy] 状态超时: state={current_state}, timeout={timeout}s")
                next_state = self._handle_timeout(body, current_state, te)
                logger.info(f"[STATE_TRANSITION_REASON] current_state={current_state}, next_state={next_state}, reason=状态超时")
            except Exception as e:
                logger.error(f"[ReportStrategy] 状态处理异常: state={current_state}, error={str(e)}")
                next_state = self._handle_error_state(body, e)
                logger.info(f"[STATE_TRANSITION_REASON] current_state={current_state}, next_state={next_state}, reason=状态处理异常:{str(e)}")

            # 状态转换
            state_machine.transition(current_state, next_state, trigger=_get_report_transition_trigger(current_state, next_state), reason=_get_report_transition_reason(current_state, next_state, body))
            current_state = next_state
            body.current_state = current_state

            # 检查是否到达终止状态
            if current_state in ("FINISHED", "ERROR"):
                break

        # 如果最终状态是ERROR，转换为FINISHED
        if current_state == "ERROR":
            state_machine.transition(current_state, "FINISHED", trigger="error_resolved", reason="error_state_converted_to_finished")
            current_state = "FINISHED"
            body.current_state = current_state

        # 构建结果
        result_data = self._build_result(body)

        elapsed = time.time() - start_time
        logger.info(f"[ReportStrategy] 策略执行完成: session_id={context.session_id}, "
                    f"health_score={result_data.health_score}, elapsed={elapsed:.2f}s, "
                    f"degraded={result_data.degraded}")

        return AgentResult(session_id=context.session_id, data=result_data)

    def _register_state_transitions(self, state_machine: StateMachine):
        """
        注册状态转换规则

        基于设计文档3.2.2节完整转换规则表

        Args:
            state_machine: 状态机实例
        """
        # INITIAL → DATA_PREPARE
        state_machine.add_state_transition("INITIAL", ["DATA_PREPARE", "ERROR"])
        
        # DATA_PREPARE → DATA_PARSE / ERROR
        state_machine.add_state_transition("DATA_PREPARE", ["DATA_PARSE", "ERROR"])
        
        # DATA_PARSE → COMPREHENSIVE_HEALTH_ANALYSIS / ERROR
        state_machine.add_state_transition("DATA_PARSE", ["COMPREHENSIVE_HEALTH_ANALYSIS", "ERROR"])
        
        # COMPREHENSIVE_HEALTH_ANALYSIS → REPORT_GENERATION
        state_machine.add_state_transition("COMPREHENSIVE_HEALTH_ANALYSIS", ["REPORT_GENERATION"])
        
        # REPORT_GENERATION → STREAMING / REPORT_GENERATION(重试) / ERROR
        # 基于设计文档：REPORT_GENERATION校验失败时，状态转换回自身重新生成(最多重试2次)
        state_machine.add_state_transition("REPORT_GENERATION", ["STREAMING", "REPORT_GENERATION", "ERROR"])
        
        # STREAMING → ASSEMBLY / ERROR
        state_machine.add_state_transition("STREAMING", ["ASSEMBLY", "ERROR"])
        
        # ASSEMBLY → FINISHED
        state_machine.add_state_transition("ASSEMBLY", ["FINISHED", "ERROR"])
        
        # ERROR → FINISHED
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
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(handler, context, resource)
        timed_out = False
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            timed_out = True
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError(f"状态执行超时，超过{timeout_seconds}秒")
        finally:
            if not timed_out:
                executor.shutdown(wait=True)

    def _handle_initial(self, context: ReportContextBody, resource: AgentResource) -> str:
        """
        INITIAL状态处理

        Args:
            context: 上下文数据
            resource: 资源类

        Returns:
            下一个状态
        """
        stage_start_time = time.time()
        logger.info("[STAGE_ENTER] INITIAL, task_id=%s, session_id=%s", context.task_id, context.session_id)
        logger.info(f"[ReportStrategy._handle_initial] INITIAL环节: 接收请求, task_id={context.task_id}, session_id={context.session_id}")
        logger.info("[ReportStrategy._handle_initial] INITIAL环节完成, 转入DATA_PREPARE")
        logger.info(f"[STAGE_EXIT] INITIAL, duration={time.time() - stage_start_time:.2f}s, task_id={context.task_id}")
        return "DATA_PREPARE"

    def _handle_data_prepare(self, context: ReportContextBody, resource: AgentResource) -> str:
        """
        DATA_PREPARE状态处理，调用DataPrepareChain

        处理逻辑：
        1. 参数校验
        2. 数据标准化
        3. 空值处理
        4. 完整性判断

        Args:
            context: 上下文数据
            resource: 资源类

        Returns:
            下一个状态
        """
        logger.info("[ReportStrategy._handle_data_prepare] DATA_PREPARE环节: 开始数据准备(参数校验+数据标准化+空值处理+完整性判断)")
        stage_start_time = time.time()
        logger.info("[STAGE_ENTER] DATA_PREPARE, task_id=%s", context.task_id)

        data_prepare_chain = resource.get_chain("data_prepare_chain")
        if data_prepare_chain is None:
            logger.error("[ReportStrategy._handle_data_prepare] DataPrepareChain未注册")
            context.error_code = ErrorCode.REPORT_DATA_PREPARE_TIMEOUT
            context.error_message = "DataPrepareChain未注册"
            logger.info(f"[STAGE_EXIT] DATA_PREPARE, duration={time.time() - stage_start_time:.2f}s, error=chain_not_registered")
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
        try:
            chain_result = data_prepare_chain.execute(chain_context)
            if chain_result.data is None:
                logger.error("[ReportStrategy._handle_data_prepare] DataPrepareChain返回空结果")
                context.error_code = ErrorCode.REPORT_DATA_PREPARE_TIMEOUT
                context.error_message = "DataPrepareChain返回空结果"
                return "ERROR"

            context.validated_data = chain_result.data.validated_data
            context.degradation_level = chain_result.data.degradation_level

            if context.degradation_level > 0:
                logger.warning(f"[ReportStrategy._handle_data_prepare] 降级触发: 数据部分缺失, 降级级别={context.degradation_level}, 降级策略=标记降级级别后继续")
                logger.warning(f"[DEGRADE_TRIGGER] reason=数据部分缺失, level={context.degradation_level}, from_state=DATA_PREPARE")

            logger.info(f"[ReportStrategy._handle_data_prepare] DATA_PREPARE环节完成: degradation_level={context.degradation_level}, 转入DATA_PARSE")
            logger.info(f"[STAGE_EXIT] DATA_PREPARE, duration={time.time() - stage_start_time:.2f}s, degradation_level={context.degradation_level}")

            return "DATA_PARSE"
            
        except Exception as e:
            logger.error(f"[ReportStrategy._handle_data_prepare] DATA_PREPARE执行失败: {str(e)}")
            context.error_code = ErrorCode.REPORT_DATA_PREPARE_TIMEOUT
            context.error_message = f"数据准备失败: {str(e)}"
            logger.info(f"[STAGE_EXIT] DATA_PREPARE, duration={time.time() - stage_start_time:.2f}s, error=chain_execution_failed")
            return "ERROR"

    def _handle_data_parse(self, context: ReportContextBody, resource: AgentResource) -> str:
        """
        DATA_PARSE状态处理，调用MultiAnalysisChain

        处理逻辑：
        1. 异常指标提取（规则引擎）
        2. 风险因子提取（nlp_raner模型，降级为规则引擎）
        3. 医疗实体提取（nlp_raner模型，降级为规则引擎）
        4. 特殊规则应用（规则引擎）
        5. 生成分析摘要

        Args:
            context: 上下文数据
            resource: 资源类

        Returns:
            下一个状态
        """
        logger.info("[ReportStrategy._handle_data_parse] DATA_PARSE环节: 开始数据解析(异常指标提取+风险因子提取+医疗实体提取+特殊规则应用)")
        stage_start_time = time.time()
        logger.info("[STAGE_ENTER] DATA_PARSE, degradation_level=%d", context.degradation_level)

        multi_analysis_chain = resource.get_chain("multi_analysis_chain")
        if multi_analysis_chain is None:
            logger.error("[ReportStrategy._handle_data_parse] MultiAnalysisChain未注册")
            context.error_code = ErrorCode.REPORT_DATA_PARSE_TIMEOUT
            context.error_message = "MultiAnalysisChain未注册"
            logger.info(f"[STAGE_EXIT] DATA_PARSE, duration={time.time() - stage_start_time:.2f}s, error=chain_not_registered")
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
        try:
            chain_result = multi_analysis_chain.execute(chain_context)
            if chain_result.data is None:
                logger.error("[ReportStrategy._handle_data_parse] MultiAnalysisChain返回空结果")
                context.error_code = ErrorCode.REPORT_DATA_PARSE_TIMEOUT
                context.error_message = "MultiAnalysisChain返回空结果"
                return "ERROR"

            context.anomalies = chain_result.data.anomalies
            context.risk_factors = chain_result.data.risk_factors
            context.medical_entities = chain_result.data.medical_entities

            logger.info(f"[ReportStrategy._handle_data_parse] DATA_PARSE环节完成: "
                        f"anomalies={len(context.anomalies)}, "
                        f"risk_factors={len(context.risk_factors)}, "
                        f"medical_entities={len(context.medical_entities)}, "
                        f"转入COMPREHENSIVE_HEALTH_ANALYSIS")
            logger.info(f"[STAGE_EXIT] DATA_PARSE, duration={time.time() - stage_start_time:.2f}s, anomalies_count={len(context.anomalies)}, entities_count={sum(len(v) for v in context.medical_entities.values() if isinstance(v, list)) if isinstance(context.medical_entities, dict) else 0}")

            return "COMPREHENSIVE_HEALTH_ANALYSIS"
            
        except Exception as e:
            logger.error(f"[ReportStrategy._handle_data_parse] DATA_PARSE执行失败: {str(e)}")
            context.error_code = ErrorCode.REPORT_DATA_PARSE_TIMEOUT
            context.error_message = f"数据解析失败: {str(e)}"
            logger.info(f"[STAGE_EXIT] DATA_PARSE, duration={time.time() - stage_start_time:.2f}s, error=chain_execution_failed")
            return "ERROR"

    def _handle_comprehensive_health_analysis(
        self, 
        context: ReportContextBody, 
        resource: AgentResource
    ) -> str:
        """
        COMPREHENSIVE_HEALTH_ANALYSIS状态处理，调用ComprehensiveHealthAnalysisStrategy

        处理逻辑：
        1. 8维度知识检索
        2. 去重
        3. 充分性判断
        4. 知识精炼
        5. 健康评估（HealthAssessmentChain）

        基于设计文档3.3节Agent设计

        Args:
            context: 上下文数据
            resource: 资源类

        Returns:
            下一个状态
        """
        logger.info("[ReportStrategy._handle_comprehensive_health_analysis] COMPREHENSIVE_HEALTH_ANALYSIS环节: 开始综合健康分析(8维度知识检索+去重+充分性判断+精炼+健康评估)")
        stage_start_time = time.time()
        logger.info("[STAGE_ENTER] COMPREHENSIVE_HEALTH_ANALYSIS, anomalies_count=%d, risk_factors_count=%d, entities_count=%d", len(context.anomalies), len(context.risk_factors), sum(len(v) for v in context.medical_entities.values() if isinstance(v, list)) if isinstance(context.medical_entities, dict) else 0)

        try:
            agent = self._comprehensive_health_analysis_strategy or ComprehensiveHealthAnalysisStrategy()
            
            # 使用DataPrepareChain处理后的user_profile（包含计算后的age字段）
            user_profile = context.validated_data.get("user_profile", context.user_profile)
            logger.info(f"[ReportStrategy] COMPREHENSIVE_HEALTH_ANALYSIS使用user_profile, age={user_profile.get('age', '未找到')}")

            # 构建AgentContext
            agent_context = AgentContext(
                session_id=context.session_id,
                body=ComprehensiveHealthAnalysisContextBody(
                    anomalies=context.anomalies,
                    risk_factors=context.risk_factors,
                    medical_entities=context.medical_entities,
                    user_profile=user_profile
                )
            )
            
            # 执行Agent
            agent_result = agent.execute(agent_context, resource)
            
            if agent_result.data is None:
                logger.error("[ReportStrategy._handle_comprehensive_health_analysis] ComprehensiveHealthAnalysisStrategy返回空结果")
                logger.warning("[ReportStrategy._handle_comprehensive_health_analysis] 降级触发: Agent返回空结果, 降级策略=使用规则引擎评估")
                logger.warning("[DEGRADE_TRIGGER] reason=Agent返回空结果, level=agent_to_rule_engine, from_state=COMPREHENSIVE_HEALTH_ANALYSIS")
                context.degraded = context.degraded or True
                context.degraded_reason = (context.degraded_reason + "; " if context.degraded_reason else "") + "ComprehensiveHealthAnalysisStrategy返回空结果"
                logger.info("[DEGRADE_MARK] mark=degraded, propagated_to=context, reason=ComprehensiveHealthAnalysisStrategy返回空结果")
                self._fallback_health_assessment(context)
                logger.info(f"[STAGE_EXIT] COMPREHENSIVE_HEALTH_ANALYSIS, duration={time.time() - stage_start_time:.2f}s, health_score={context.health_score}, degraded=True")
                return "REPORT_GENERATION"
            
            result_data = agent_result.data
            
            context.dimension_summaries = result_data.dimension_summaries
            context.health_assessment = result_data.health_assessment
            # 合并medical_entities而非替换：Agent返回的实体与已有实体合并
            agent_medical_entities = result_data.medical_entities
            if agent_medical_entities and isinstance(agent_medical_entities, dict):
                # 合并：对每个类别的实体列表进行去重合并
                for key, entities in agent_medical_entities.items():
                    if isinstance(entities, list) and entities:
                        existing = context.medical_entities.get(key, [])
                        if existing:
                            # 去重合并：基于entity_name或name字段
                            existing_names = {
                                e.get("entity_name", e.get("name", "")) for e in existing
                            }
                            for entity in entities:
                                entity_name = entity.get("entity_name", entity.get("name", ""))
                                if entity_name not in existing_names:
                                    existing.append(entity)
                                    existing_names.add(entity_name)
                            context.medical_entities[key] = existing
                        else:
                            context.medical_entities[key] = entities
                logger.info(f"[ReportStrategy] medical_entities合并完成: "
                           f"merged_keys={list(context.medical_entities.keys())}, "
                           f"total_entities={sum(len(v) for v in context.medical_entities.values() if isinstance(v, list))}")
            # 如果Agent返回的medical_entities为空，保留已有的medical_entities不做覆盖
            
            # 提取健康评估结果（确保health_assessment是字典类型）
            if result_data.health_assessment and isinstance(result_data.health_assessment, dict):
                health_score = result_data.health_assessment.get("health_score")
                context.health_score = float(health_score) if health_score is not None else 0.0
                context.health_level = result_data.health_assessment.get("health_level", "")
                context.risk_level = result_data.health_assessment.get("risk_level", "")
                context.risk_diseases = result_data.health_assessment.get("disease_risks", [])
                logger.info(f"[ReportStrategy] 健康评估结果: health_score={context.health_score}, "
                            f"来源={'HealthAssessmentChain' if health_score is not None else '规则引擎降级'}")
            elif result_data.health_assessment:
                logger.warning(f"[ReportStrategy] health_assessment类型错误: "
                              f"type={type(result_data.health_assessment)}, expected=dict")
            
            # 标记降级状态
            if result_data.degraded:
                logger.warning(f"[ReportStrategy._handle_comprehensive_health_analysis] 降级触发: Agent部分降级, 降级原因={result_data.degraded_reason}")
                logger.warning("[DEGRADE_TRIGGER] reason=Agent部分降级, level=agent_partial, from_state=COMPREHENSIVE_HEALTH_ANALYSIS")
                context.degraded = context.degraded or True
                context.degraded_reason = (context.degraded_reason + "; " if context.degraded_reason else "") + result_data.degraded_reason
                logger.info(f"[DEGRADE_MARK] mark=degraded, propagated_to=context, reason={result_data.degraded_reason}")
            
            logger.info(f"[ReportStrategy._handle_comprehensive_health_analysis] COMPREHENSIVE_HEALTH_ANALYSIS环节完成: "
                       f"dimension_summaries={len(context.dimension_summaries)}, "
                       f"health_score={context.health_score}, "
                       f"health_level={context.health_level}, "
                       f"risk_level={context.risk_level}, "
                       f"degraded={context.degraded}, 转入REPORT_GENERATION")
            logger.info(f"[STAGE_EXIT] COMPREHENSIVE_HEALTH_ANALYSIS, duration={time.time() - stage_start_time:.2f}s, health_score={context.health_score}, anomalies_count={len(context.anomalies)}, entities_count={sum(len(v) for v in context.medical_entities.values() if isinstance(v, list)) if isinstance(context.medical_entities, dict) else 0}")

            return "REPORT_GENERATION"

        except Exception as e:
            logger.error(f"[ReportStrategy] COMPREHENSIVE_HEALTH_ANALYSIS执行失败: {str(e)}")

            # 降级策略：优先使用已完成的部分结果
            has_partial_results = bool(context.dimension_summaries)
            if has_partial_results:
                logger.warning(f"[ReportStrategy] 降级策略: 保留已有部分结果({len(context.dimension_summaries)}个维度), 仅补充健康评估")
                logger.warning("[DEGRADE_TRIGGER] reason=ComprehensiveHealthAnalysisStrategy部分失败, level=agent_partial_to_rule, from_state=COMPREHENSIVE_HEALTH_ANALYSIS")
            else:
                logger.warning("[ReportStrategy] 降级策略: 使用规则引擎评估")
                logger.warning("[DEGRADE_TRIGGER] reason=ComprehensiveHealthAnalysisStrategy失败, level=agent_to_rule_engine, from_state=COMPREHENSIVE_HEALTH_ANALYSIS")

            context.degraded = context.degraded or True
            context.degraded_reason = (context.degraded_reason + "; " if context.degraded_reason else "") + f"ComprehensiveHealthAnalysisStrategy失败: {str(e)}"
            logger.info(f"[DEGRADE_MARK] mark=degraded, propagated_to=context, reason=ComprehensiveHealthAnalysisStrategy失败: {str(e)}")

            # 仅补充健康评估（已有部分维度结果时保留不覆盖）
            self._fallback_health_assessment(context)
            logger.info(f"[STAGE_EXIT] COMPREHENSIVE_HEALTH_ANALYSIS, duration={time.time() - stage_start_time:.2f}s, health_score={context.health_score}, degraded=True, error=agent_execution_failed")

            return "REPORT_GENERATION"

    def _handle_report_generation(self, context: ReportContextBody, resource: AgentResource) -> str:
        """
        REPORT_GENERATION状态处理，调用ReportGenerationChain

        处理逻辑：
        1. 构建报告生成Prompt
        2. 调用Qwen3-4B模型生成Markdown格式报告
        3. 内容校验
        4. 校验失败时重试（最多2次，基于设计文档状态转换规则）
        5. 重试耗尽后降级为模板报告
        6. 流式输出

        基于设计文档：REPORT_GENERATION | 校验失败 | REPORT_GENERATION | 重新生成(最多重试2次)

        Args:
            context: 上下文数据
            resource: 资源类

        Returns:
            下一个状态
        """
        retry_info = f", retry_count={context.report_generation_retry_count}" if context.report_generation_retry_count > 0 else ""
        logger.info(f"[ReportStrategy] REPORT_GENERATION: 开始报告生成{retry_info}")
        stage_start_time = time.time()
        logger.info("[STAGE_ENTER] REPORT_GENERATION, health_score=%s, health_level=%s, anomalies_count=%d", context.health_score, context.health_level, len(context.anomalies))

        # 获取ReportGenerationChain
        report_generation_chain = resource.get_chain("report_generation_chain")
        if report_generation_chain is None:
            logger.error("[ReportStrategy] ReportGenerationChain未注册")
            context.error_code = ErrorCode.REPORT_REPORT_GENERATION_TIMEOUT
            context.error_message = "ReportGenerationChain未注册"
            # Chain未注册是永久性故障，无需重试，直接降级为模板报告
            context.report_content = self._generate_template_report(context)
            context.degraded = context.degraded or True
            context.degraded_reason = (context.degraded_reason + "; " if context.degraded_reason else "") + "ReportGenerationChain未注册"
            logger.warning("[DEGRADE_TRIGGER] reason=ReportGenerationChain未注册, level=chain_to_template, from_state=REPORT_GENERATION")
            logger.info("[DEGRADE_MARK] mark=degraded, propagated_to=context, reason=ReportGenerationChain未注册")
            logger.info(f"[STAGE_EXIT] REPORT_GENERATION, duration={time.time() - stage_start_time:.2f}s, degraded=True, fallback=template_report")
            return "STREAMING"

        # 构建报告素材
        report_materials = {
            "monitoring_data": context.monitoring_data,
            "merged_results": context.knowledge_results,
            "dimension_results": context.dimension_summaries,
            "anomalies": context.anomalies,
            "risk_factors": context.risk_factors,
            "medical_entities": context.medical_entities,
            "health_assessment": context.health_assessment
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
            logger.info("[ReportStrategy] REPORT_GENERATION: 流式生成器已创建")
            logger.info(f"[STAGE_EXIT] REPORT_GENERATION, duration={time.time() - stage_start_time:.2f}s, is_streaming=True")
            return "STREAMING"
        except Exception as e:
            logger.error(f"[ReportStrategy] 流式生成失败: {str(e)}")
            # 流式生成失败，降级为普通生成

        # 普通生成（流式失败后的降级路径）
        try:
            chain_result = report_generation_chain.execute(chain_context)
            if chain_result.data and chain_result.data.report_content:
                report_content = chain_result.data.report_content
                # 校验报告生成结果
                is_valid, fail_reason = self._validate_report_generation_result(report_content)
                if is_valid:
                    # 校验通过
                    context.report_content = report_content
                    context.sources = chain_result.data.sources
                    logger.info("[ReportStrategy] REPORT_GENERATION: 普通生成完成，校验通过")
                    logger.info(f"[STAGE_EXIT] REPORT_GENERATION, duration={time.time() - stage_start_time:.2f}s, report_length={len(report_content)}, validation=passed")
                    return "STREAMING"
                else:
                    # 校验失败 - 重试或降级
                    logger.warning(f"[ReportStrategy] REPORT_GENERATION: 报告校验失败: {fail_reason}")
                    logger.info(f"[STAGE_EXIT] REPORT_GENERATION, duration={time.time() - stage_start_time:.2f}s, validation=failed, reason={fail_reason}")
                    return self._handle_report_generation_failure(context, fail_reason)
            else:
                # 生成结果为空 - 校验失败，重试或降级
                logger.warning("[ReportStrategy] REPORT_GENERATION: 报告生成返回空结果")
                logger.info(f"[STAGE_EXIT] REPORT_GENERATION, duration={time.time() - stage_start_time:.2f}s, validation=failed, reason=empty_result")
                return self._handle_report_generation_failure(context, "报告生成返回空结果")
        except Exception as e2:
            logger.error(f"[ReportStrategy] 普通生成也失败: {str(e2)}")
            # 生成异常 - 校验失败，重试或降级
            logger.info(f"[STAGE_EXIT] REPORT_GENERATION, duration={time.time() - stage_start_time:.2f}s, error=generation_failed")
            return self._handle_report_generation_failure(context, f"报告生成失败: {str(e2)}")

    def _handle_streaming(self, context: ReportContextBody, resource: AgentResource) -> str:
        """
        STREAMING状态处理

        Args:
            context: 上下文数据
            resource: 资源类

        Returns:
            下一个状态
        """
        logger.info("[ReportStrategy] STREAMING: 流式输出状态")
        stage_start_time = time.time()
        logger.info("[STAGE_ENTER] STREAMING, is_streaming=%s", context.is_streaming)
        logger.info(f"[STAGE_EXIT] STREAMING, duration={time.time() - stage_start_time:.2f}s, report_length={len(context.report_content) if context.report_content else 0}")

        return "ASSEMBLY"

    def _handle_assembly(self, context: ReportContextBody, resource: AgentResource) -> str:
        """
        ASSEMBLY状态处理，组装结束

        处理逻辑：
        1. 组装结束响应
        2. 元数据封装
        3. 日志记录

        Args:
            context: 上下文数据
            resource: 资源类

        Returns:
            下一个状态
        """
        logger.info("[ReportStrategy] ASSEMBLY: 组装结束")
        stage_start_time = time.time()
        logger.info("[STAGE_ENTER] ASSEMBLY, health_score=%s, report_length=%d", context.health_score, len(context.report_content) if context.report_content else 0)

        # 确保报告内容不为空
        if not context.report_content:
            context.report_content = self._generate_template_report(context)
            logger.info("[ReportStrategy] ASSEMBLY: 使用模板报告")

        # 提取知识来源
        if not context.sources and context.knowledge_results:
            for item in context.knowledge_results:
                entity = item.get("entity", "")
                if entity and entity not in context.sources:
                    context.sources.append(entity)

        # 从dimension_summaries中提取知识来源
        if not context.sources and context.dimension_summaries:
            for dim_name, dim_data in context.dimension_summaries.items():
                key_entities = dim_data.get("key_entities", [])
                for entity in key_entities:
                    if entity and entity not in context.sources:
                        context.sources.append(entity)

        logger.info(f"[ReportStrategy] ASSEMBLY完成: report_length={len(context.report_content)}, "
                   f"sources={len(context.sources)}")
        logger.info(f"[STAGE_EXIT] ASSEMBLY, duration={time.time() - stage_start_time:.2f}s, report_length={len(context.report_content)}, health_score={context.health_score}, sources_count={len(context.sources)}")

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
        logger.info("[ReportStrategy] FINISHED: 策略执行结束")
        stage_start_time = time.time()
        logger.info("[STAGE_ENTER] FINISHED, session_id=%s, health_score=%s", context.session_id, context.health_score)
        logger.info(f"[STAGE_EXIT] FINISHED, duration={time.time() - stage_start_time:.2f}s, report_length={len(context.report_content) if context.report_content else 0}")
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
        if context.error_code >= ErrorCode.REPORT_DATA_PREPARE_TIMEOUT:
            # 业务错误，生成模板报告
            context.report_content = self._generate_template_report(context)
            context.degraded = context.degraded or True
            context.degraded_reason = (context.degraded_reason + "; " if context.degraded_reason else "") + context.error_message
            logger.info("[ReportStrategy] 降级策略: 使用模板报告")
            logger.warning(f"[DEGRADE_TRIGGER] reason=业务错误(error_code={context.error_code}), level=error_to_template, from_state=ERROR")
            logger.info(f"[DEGRADE_MARK] mark=degraded, propagated_to=context, reason={context.error_message}")

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

        # 根据异常类型设置错误码
        if isinstance(error, DataPrepareError):
            context.error_code = ErrorCode.REPORT_DATA_PREPARE_TIMEOUT
        elif isinstance(error, (DataParseError, MultiAnalysisError)):
            context.error_code = ErrorCode.REPORT_DATA_PARSE_TIMEOUT
        elif isinstance(error, ComprehensiveAnalysisError):
            context.error_code = ErrorCode.REPORT_COMPREHENSIVE_ANALYSIS_TIMEOUT
        elif isinstance(error, LLMServiceError):
            context.error_code = ErrorCode.REPORT_REPORT_GENERATION_TIMEOUT
            # LLM故障降级：生成模板报告
            context.report_content = self._generate_template_report(context)
            context.degraded = context.degraded or True
            context.degraded_reason = (context.degraded_reason + "; " if context.degraded_reason else "") + f"LLM故障: {error_message}"
            logger.warning("[ReportStrategy] 降级策略: LLM故障，使用模板报告")
            logger.warning("[DEGRADE_TRIGGER] reason=LLM故障, level=llm_to_template, from_state=ERROR")
            logger.info(f"[DEGRADE_MARK] mark=degraded, propagated_to=context, reason=LLM故障: {error_message}")
            return "FINISHED"
        else:
            context.error_code = ErrorCode.UNKNOWN

        return "ERROR"

    def _handle_timeout(self, context: ReportContextBody, state: str, error: TimeoutError) -> str:
        """
        处理超时状态

        基于设计文档3.2.3节超时配置和3.2.4节降级策略矩阵

        Args:
            context: 上下文数据
            state: 当前状态
            error: 超时异常

        Returns:
            下一个状态
        """
        logger.warning(f"[ReportStrategy] 超时降级: state={state}")

        if state == "DATA_PREPARE":
            # DATA_PREPARE超时：返回错误码
            context.error_code = ErrorCode.REPORT_DATA_PREPARE_TIMEOUT
            context.error_message = "数据准备超时"
            return "ERROR"
            
        elif state == "DATA_PARSE":
            # DATA_PARSE超时：nlp_raner降级为规则引擎
            logger.warning("[ReportStrategy] 数据解析超时，降级为规则引擎")
            logger.warning(f"[DEGRADE_TRIGGER] reason=DATA_PARSE超时, level=nlp_to_rule_engine, from_state={state}")
            context.error_code = ErrorCode.REPORT_DATA_PARSE_TIMEOUT
            context.error_message = "数据解析超时，使用规则引擎"
            # 使用规则引擎提取异常指标和风险因子
            self._fallback_data_parse(context)
            return "COMPREHENSIVE_HEALTH_ANALYSIS"
            
        elif state == "COMPREHENSIVE_HEALTH_ANALYSIS":
            # COMPREHENSIVE_HEALTH_ANALYSIS超时：Agent失败使用已完成结果继续，健康评估模型降级为规则计算
            logger.warning("[ReportStrategy] 综合健康分析超时，使用规则引擎评估")
            logger.warning(f"[DEGRADE_TRIGGER] reason=COMPREHENSIVE_HEALTH_ANALYSIS超时, level=agent_to_rule_engine, from_state={state}")
            context.error_code = ErrorCode.REPORT_COMPREHENSIVE_ANALYSIS_TIMEOUT
            context.error_message = "综合健康分析超时，使用规则评估"
            context.degraded = context.degraded or True
            context.degraded_reason = (context.degraded_reason + "; " if context.degraded_reason else "") + "综合健康分析超时"
            logger.info("[DEGRADE_MARK] mark=degraded, propagated_to=context, reason=综合健康分析超时")

            # Ensure dimension_summaries exists and is not None before marking
            if not hasattr(context, 'dimension_summaries') or context.dimension_summaries is None:
                context.dimension_summaries = {}

            # Mark all existing dimension_summaries as degraded
            for dim_name, dim_data in context.dimension_summaries.items():
                if isinstance(dim_data, dict):
                    dim_data['_degraded'] = True
                    dim_data['_degraded_reason'] = '综合健康分析超时，数据可能不完整'

            context.degraded = context.degraded or True

            # Also mark the context itself as degraded for downstream detection
            logger.info(f"[ReportStrategy] 超时降级: dimension_summaries已标记_degraded=True, 维度数={len(context.dimension_summaries)}")

            # 使用规则引擎评估（仅当HealthAssessmentChain未提供评分时）
            self._fallback_health_assessment(context)
            return "REPORT_GENERATION"
            
        elif state == "REPORT_GENERATION":
            # REPORT_GENERATION超时：降级为模板报告
            logger.warning("[ReportStrategy] 报告生成超时，降级为模板报告")
            logger.warning(f"[DEGRADE_TRIGGER] reason=REPORT_GENERATION超时, level=llm_to_template, from_state={state}")
            context.error_code = ErrorCode.REPORT_REPORT_GENERATION_TIMEOUT
            context.error_message = "报告生成超时，使用模板报告"
            context.report_content = self._generate_template_report(context)
            context.degraded = context.degraded or True
            context.degraded_reason = (context.degraded_reason + "; " if context.degraded_reason else "") + "报告生成超时"
            logger.info("[DEGRADE_MARK] mark=degraded, propagated_to=context, reason=报告生成超时")
            return "STREAMING"

        elif state == "STREAMING":
            # STREAMING超时：降级为模板报告
            logger.warning("[ReportStrategy] 流式输出超时，降级为模板报告")
            logger.warning(f"[DEGRADE_TRIGGER] reason=STREAMING超时, level=streaming_to_template, from_state={state}")
            context.error_code = ErrorCode.REPORT_STREAMING_TIMEOUT
            context.error_message = "流式输出超时，使用模板报告"
            if not context.report_content:
                context.report_content = self._generate_template_report(context)
            context.degraded = context.degraded or True
            context.degraded_reason = (context.degraded_reason + "; " if context.degraded_reason else "") + "流式输出超时"
            logger.info("[DEGRADE_MARK] mark=degraded, propagated_to=context, reason=流式输出超时")
            return "ASSEMBLY"
            
        else:
            context.error_code = ErrorCode.REPORT_OTHER_TIMEOUT
            context.error_message = f"状态{state}执行超时"
            return "ERROR"

    def _fallback_data_parse(self, context: ReportContextBody) -> None:
        """
        降级策略：规则引擎数据解析

        当nlp_raner模型不可用时，使用规则引擎提取异常指标和风险因子。
        """
        logger.info("[ReportStrategy] 使用规则引擎进行数据解析")

        validated_data = context.validated_data
        
        # 提取异常指标（规则引擎）
        anomalies = []
        monitoring_data = validated_data.get("monitoring_data", {})
        for indicator_name, indicator_data in monitoring_data.items():
            if isinstance(indicator_data, dict):
                value = indicator_data.get("value")
                reference_range = indicator_data.get("reference_range", [])
                if value and reference_range and len(reference_range) == 2:
                    if value < reference_range[0] or value > reference_range[1]:
                        anomalies.append({
                            "indicator_name": indicator_name,
                            "value": value,
                            "reference_range": reference_range,
                            "severity": "mild" if abs(value - (reference_range[0] + reference_range[1]) / 2) < self._report_config.anomaly_deviation_threshold else "moderate"
                        })
        
        context.anomalies = anomalies
        
        # 提取风险因子（规则引擎）
        risk_factors = []
        user_profile = validated_data.get("user_profile", {})
        
        # 既往病史风险
        past_medical_history = user_profile.get("past_medical_history", "")
        if past_medical_history:
            risk_factors.append({
                "factor_name": "既往病史",
                "risk_level": "中",
                "basis": past_medical_history[:self._report_config.past_medical_history_limit]
            })
        
        # 家族病史风险
        family_history = user_profile.get("family_history", "")
        if family_history:
            risk_factors.append({
                "factor_name": "家族病史",
                "risk_level": "中",
                "basis": family_history[:self._report_config.family_history_limit]
            })
        
        context.risk_factors = risk_factors
        
        # 医疗实体：保留已有实体，仅在未填充时设置空字典
        if not context.medical_entities:
            context.medical_entities = {
                "diseases": [],
                "symptoms": [],
                "medications": [],
                "examinations": []
            }
        
        logger.info(f"[ReportStrategy] 规则引擎数据解析完成: "
                   f"anomalies={len(anomalies)}, risk_factors={len(risk_factors)}")

    def _fallback_health_assessment(self, context: ReportContextBody) -> None:
        """
        降级策略：规则引擎健康评估

        当健康评估模型不可用时，使用规则引擎进行健康评估。
        基于设计文档3.4节降级算法实现。

        注意：如果HealthAssessmentChain已提供评分，则保留其结果，仅补充缺失字段。
        规则引擎仅在Chain评分缺失或Chain标记为降级时使用。
        """
        logger.info("[ReportStrategy] 使用规则引擎进行健康评估")

        # 检查HealthAssessmentChain是否已提供评分
        # Chain评分优先：仅在Chain评分缺失或Chain标记为降级时才使用规则引擎覆盖
        chain_score_available = (
            context.health_assessment is not None
            and isinstance(context.health_assessment, dict)
            and context.health_assessment.get("health_score") is not None
        )
        chain_is_degraded = (
            context.health_assessment is not None
            and isinstance(context.health_assessment, dict)
            and context.health_assessment.get("degraded", False)
        )

        if chain_score_available and not chain_is_degraded:
            # Chain评分有效且未降级，保留Chain评分，规则引擎仅补充缺失字段
            logger.info(f"[ReportStrategy] 健康评分来源=HealthAssessmentChain(有效), "
                        f"health_score={context.health_score}, "
                        f"规则引擎仅补充缺失字段")
            # 仅补充缺失的风险等级等字段 - 通过配置类集中管理
            if not context.risk_level:
                if context.health_score >= self._report_config.risk_level_thresholds["low"]:
                    context.risk_level = "低"
                elif context.health_score >= self._report_config.risk_level_thresholds["mild"]:
                    context.risk_level = "轻"
                elif context.health_score >= self._report_config.risk_level_thresholds["moderate"]:
                    context.risk_level = "中"
                else:
                    context.risk_level = "高"
            if not context.health_level:
                if context.health_score >= self._report_config.health_score_thresholds["excellent"]:
                    context.health_level = "优秀"
                elif context.health_score >= self._report_config.health_score_thresholds["good"]:
                    context.health_level = "良好"
                elif context.health_score >= self._report_config.health_score_thresholds["normal"]:
                    context.health_level = "一般"
                elif context.health_score >= self._report_config.health_score_thresholds["poor"]:
                    context.health_level = "较差"
                else:
                    context.health_level = "差"
            return

        # Chain评分缺失或已降级，使用规则引擎计算
        if chain_is_degraded:
            logger.info(f"[ReportStrategy] 健康评分来源=规则引擎(Chain已降级), "
                        f"chain_degraded_reason={context.health_assessment.get('degraded_reason', '未知')}")
        else:
            logger.info("[ReportStrategy] 健康评分来源=规则引擎(Chain评分缺失)")

        # 基于异常指标数量和严重程度计算健康评分
        base_score = self._report_config.base_health_score

        # 异常指标扣分 - 通过配置类集中管理
        for anomaly in context.anomalies:
            severity = anomaly.get("severity", "normal")
            if severity == "severe":
                base_score -= self._report_config.deduction_severe
            elif severity == "moderate":
                base_score -= self._report_config.deduction_moderate
            elif severity == "mild":
                base_score -= self._report_config.deduction_mild

        # 风险因子扣分
        base_score -= len(context.risk_factors) * self._report_config.deduction_risk_factor

        # 疾病实体扣分
        diseases = context.medical_entities.get("diseases", [])
        base_score -= len(diseases) * self._report_config.deduction_disease

        # 确保分数在0-100范围内
        health_score = float(max(0, min(100, base_score)))

        # 判断健康等级 - 通过配置类集中管理
        if health_score >= self._report_config.health_score_thresholds["excellent"]:
            health_level = "优秀"
        elif health_score >= self._report_config.health_score_thresholds["good"]:
            health_level = "良好"
        elif health_score >= self._report_config.health_score_thresholds["normal"]:
            health_level = "一般"
        elif health_score >= self._report_config.health_score_thresholds["poor"]:
            health_level = "较差"
        else:
            health_level = "差"

        # 判断风险等级 - 通过配置类集中管理
        if health_score >= self._report_config.risk_level_thresholds["low"]:
            risk_level = "低"
        elif health_score >= self._report_config.risk_level_thresholds["mild"]:
            risk_level = "轻"
        elif health_score >= self._report_config.risk_level_thresholds["moderate"]:
            risk_level = "中"
        else:
            risk_level = "高"

        context.health_score = health_score
        context.health_level = health_level
        context.risk_level = risk_level
        context.health_assessment = {
            "health_score": health_score,
            "health_level": health_level,
            "risk_level": risk_level,
            "disease_risks": [],
            "score_breakdown": {
                "method": "rule_engine",
                "anomaly_count": len(context.anomalies),
                "risk_factor_count": len(context.risk_factors),
                "disease_count": len(diseases)
            },
            "reasoning": "使用规则引擎评估（降级模式）",
            "degraded": True,
            "degraded_reason": "健康评估模型不可用"
        }

        logger.info(f"[ReportStrategy] 规则引擎健康评估完成: "
                   f"health_score={health_score}, health_level={health_level}, risk_level={risk_level}")

    # 报告生成最大重试次数（基于设计文档：最多重试2次） - 通过配置类集中管理
    MAX_REPORT_GENERATION_RETRY = 2

    def _validate_report_generation_result(self, report_content: str) -> tuple:
        """
        校验报告生成结果

        校验规则（基于设计文档定义的"校验失败"条件）：
        1. LLM生成结果不为空
        2. LLM生成结果格式正确（包含Markdown标题结构）
        3. LLM生成结果内容完整（包含必要部分）

        Args:
            report_content: 报告内容

        Returns:
            (校验是否通过, 失败原因)
        """
        # 1. 空值检查：LLM生成结果为空
        if not report_content or not report_content.strip():
            return False, "报告内容为空"

        stripped_content = report_content.strip()

        # 2. 格式检查：LLM生成结果格式不正确（无法解析为报告）
        # 报告应以Markdown标题开头
        if not stripped_content.startswith("#"):
            return False, "报告格式不正确：未以Markdown标题开头"

        # 检查是否包含基本的Markdown标题结构
        if "##" not in stripped_content:
            return False, "报告格式不正确：缺少章节标题结构"

        # 3. 完整性检查：LLM生成结果内容不完整（缺少必要部分）
        required_sections = ["健康综合评分", "免责声明"]
        missing_sections = []
        for section in required_sections:
            if section not in stripped_content:
                missing_sections.append(section)
        if missing_sections:
            return False, f"报告内容不完整：缺少{','.join(missing_sections)}"

        logger.info(f"[ReportStrategy] 报告校验通过: report_len={len(report_content)}")
        return True, ""

    def _handle_report_generation_failure(
        self, context: ReportContextBody, fail_reason: str
    ) -> str:
        """
        处理REPORT_GENERATION校验失败

        基于设计文档：当REPORT_GENERATION校验失败时，状态转换回REPORT_GENERATION自身重新生成，
        最多重试2次。重试耗尽后降级为模板报告。

        Args:
            context: 上下文数据
            fail_reason: 失败原因

        Returns:
            下一个状态："REPORT_GENERATION"(重试) 或 "ASSEMBLY"(降级为模板报告)
        """
        if context.report_generation_retry_count < self.MAX_REPORT_GENERATION_RETRY:
            context.report_generation_retry_count += 1
            logger.warning(
                f"[ReportStrategy] 报告校验失败，第{context.report_generation_retry_count}次重试: "
                f"fail_reason={fail_reason}"
            )
            logger.warning(f"[REPORT_RETRY] retry_count={context.report_generation_retry_count}/{self.MAX_REPORT_GENERATION_RETRY}, reason={fail_reason}")
            return "REPORT_GENERATION"
        else:
            # 重试耗尽，降级为模板报告
            logger.warning(
                f"[ReportStrategy] 报告校验失败，重试耗尽(retry_count={context.report_generation_retry_count})，"
                f"降级为模板报告: fail_reason={fail_reason}"
            )
            logger.error(f"[REPORT_RETRY_EXHAUSTED] max_retries={self.MAX_REPORT_GENERATION_RETRY}, degrading to template report")
            context.report_content = self._generate_template_report(context)
            context.degraded = context.degraded or True
            context.degraded_reason = (
                (context.degraded_reason + "; " if context.degraded_reason else "") +
                f"报告校验失败(重试{context.report_generation_retry_count}次): {fail_reason}"
            )
            logger.info(f"[DEGRADE_MARK] mark=degraded, propagated_to=context, reason=报告校验失败(重试{context.report_generation_retry_count}次): {fail_reason}")
            return "STREAMING"

    def _generate_template_report(self, context: ReportContextBody) -> str:
        """
        生成模板报告（LLM故障降级）

        使用新的UserProfile字段：
        - user_id, gender, birth_date, height, weight
        - past_medical_history, family_history, allergy_history, surgical_history, medical_compliance

        Args:
            context: 上下文数据

        Returns:
            模板报告内容
        """
        logger.info("[ReportStrategy] 生成模板报告")

        # 用户信息
        user_info = ""
        if context.user_profile:
            # 计算年龄
            age = "未知"
            birth_date = context.user_profile.get('birth_date')
            if birth_date:
                try:
                    from datetime import datetime
                    birth = datetime.strptime(birth_date, "%Y-%m-%d")
                    today = datetime.now()
                    age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
                except (ValueError, TypeError):
                    profile_age = context.user_profile.get('age')
                    age = str(int(profile_age)) if isinstance(profile_age, (int, float)) and profile_age > 0 else "未知"
            else:
                profile_age = context.user_profile.get('age')
                age = str(int(profile_age)) if isinstance(profile_age, (int, float)) and profile_age > 0 else "未知"

            user_info = f"""
**用户信息**
- 年龄：{age}
- 性别：{context.user_profile.get('gender', '未知')}
- 既往病史：{context.user_profile.get('past_medical_history', '无')}
- 家族病史：{context.user_profile.get('family_history', '无')}
- 过敏史：{context.user_profile.get('allergy_history', '无')}
- 手术史：{context.user_profile.get('surgical_history', '无')}
"""

        # 异常指标
        anomaly_text = "暂无异常指标"
        if context.anomalies:
            anomaly_items = []
            for a in context.anomalies:
                indicator = a.get("indicator_name", "")
                anomaly_type = a.get("anomaly_type", "")
                value = a.get("anomaly_value", a.get("value", ""))
                parts = [p for p in [indicator, anomaly_type, value] if p]
                anomaly_items.append(f"- {'：'.join(parts)}" if len(parts) > 1 else f"- {parts[0]}" if parts else "")
            anomaly_items = [item for item in anomaly_items if item]
            anomaly_text = "\n".join(anomaly_items) if anomaly_items else "暂无异常指标"

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

        # 维度摘要
        dimension_text = ""
        if context.dimension_summaries:
            dimension_items = []
            dimension_names = {
                "disease_risk": "疾病风险",
                "medication": "用药建议",
                "treatment": "治疗方案",
                "dietary": "饮食建议",
                "checkup": "检查建议",
                "complication": "并发症预警",
                "prevention": "预防措施",
                "susceptible": "易感人群"
            }
            for dim_key, dim_data in context.dimension_summaries.items():
                dim_name = dimension_names.get(dim_key, dim_key)
                summary = dim_data.get("summary", "")
                if summary:
                    dimension_items.append(f"### {dim_name}\n{summary[:self._report_config.template_summary_truncate_len]}")
            if dimension_items:
                dimension_text = "\n\n".join(dimension_items)

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

## 四、8维度健康分析
{dimension_text if dimension_text else "暂无详细分析"}

## 五、健康建议

根据您的健康评分和风险评估结果，建议：
1. 定期进行健康体检，关注异常指标变化
2. 保持良好的生活习惯，合理饮食、适量运动
3. 如有不适症状，及时就医咨询
4. 遵医嘱用药，不自行调整药物剂量

## 六、免责声明
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
            dimension_summaries=context.dimension_summaries,
            health_assessment=context.health_assessment,
            error_code=context.error_code,
            error_message=context.error_message,
            degraded=context.degraded,
            degraded_reason=context.degraded_reason
        )

        # 计算报告字数
        if context.report_content:
            result_data.word_count = len(context.report_content)

        return result_data


def _get_report_transition_trigger(from_state: str, to_state: str) -> str:
    """Derive a short snake_case trigger identifier for a report state transition."""
    triggers = {
        ("INITIAL", "DATA_PREPARE"): "initial_complete",
        ("DATA_PREPARE", "DATA_PARSE"): "data_prepared",
        ("DATA_PARSE", "COMPREHENSIVE_HEALTH_ANALYSIS"): "data_parsed",
        ("COMPREHENSIVE_HEALTH_ANALYSIS", "REPORT_GENERATION"): "analysis_complete",
        ("REPORT_GENERATION", "STREAMING"): "llm_ready",
        ("REPORT_GENERATION", "REPORT_GENERATION"): "retry_generation",
        ("STREAMING", "ASSEMBLY"): "stream_complete",
        ("ASSEMBLY", "FINISHED"): "assembly_complete",
    }
    return triggers.get((from_state, to_state), "state_handler")


def _get_report_transition_reason(from_state: str, to_state: str, context: ReportContextBody) -> str:
    """Derive a brief human-readable reason for a report state transition."""
    reasons = {
        ("INITIAL", "DATA_PREPARE"): "request_received",
        ("DATA_PREPARE", "DATA_PARSE"): f"degradation_level={context.degradation_level}",
        ("DATA_PARSE", "COMPREHENSIVE_HEALTH_ANALYSIS"): f"anomalies={len(context.anomalies)},entities={sum(len(v) for v in context.medical_entities.values() if isinstance(v, list)) if isinstance(context.medical_entities, dict) else 0}",
        ("COMPREHENSIVE_HEALTH_ANALYSIS", "REPORT_GENERATION"): f"health_score={context.health_score},degraded={context.degraded}",
        ("REPORT_GENERATION", "STREAMING"): "stream_start",
        ("REPORT_GENERATION", "REPORT_GENERATION"): f"retry_count={context.report_generation_retry_count}",
        ("STREAMING", "ASSEMBLY"): "stream_end",
        ("ASSEMBLY", "FINISHED"): f"report_length={len(context.report_content) if context.report_content else 0}",
    }
    return reasons.get((from_state, to_state), "state_handler_return")
