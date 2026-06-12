# -*- coding: utf-8 -*-
"""
综合健康分析Agent

该模块实现ComprehensiveHealthAnalysisStrategy类，用于健康报告生成业务中的综合健康分析环节。
基于设计文档《项目业务详细设计v5.16》第3.3节的设计实现。

主要功能：
1. v8扁平状态机（10个主状态，无嵌套子状态机）：BuildQueries → PlanRetrieval → InitRetrievalContext → ParallelDimensionRetrieve → InterDimensionSync → HybridRelevance → EvaluateSufficiency → RefineKnowledge → HealthAssess → Output
2. 检索chain循环：InterDimensionSync → BuildQueries（有跨维度重复且chain_loop_count < max）
3. Agent循环：EvaluateSufficiency → BuildQueries（不充分且agent_retrieval_loop_count < max）
4. v8新增：黑名单过滤、维度标记跳过、知识表（dimension_table/knowledge_cross_refs）
5. 降级策略

v5.16核心变更：
- 向量检索定位从"知识的直接来源"改为"知识图谱检索的增强模式"
- 图谱查询输出规范化：source_entity(必需)+relation_type(必需)+target_entity(必需)+content(可选)
- 混合相关性评分公式简化：主流程 0.60*user_relevance + 0.40*dimension_relevance；降级流程 0.50*user_relevance + 0.30*dimension_relevance + 0.20*vector_score
- 低质知识过滤：过滤content为空/过短/仅名称的知识
- 图谱检索故障降级：向量增强检索结果直接作为降级知识，添加_degraded标记
"""

import copy
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import fields
from typing import Any, Dict, List, Optional, Tuple

from src.orchestration.exceptions import EngineUnavailableError
from src.orchestration.agent.agent_strategy import AgentStrategy
from src.orchestration.agent.data_classes import AgentContext, AgentResult
from src.orchestration.agent.agent_resource import AgentResource
from src.orchestration.state_machine.state_machine import StateMachine
from src.orchestration.chain.data_classes import ChainContext
from src.orchestration.agent.comprehensive_health_analysis_strategy.comprehensive_health_analysis_context import (
    DimensionKnowledge,
    SharedMemory,
    RetrievalStats,
    HealthAssessment,
    ComprehensiveHealthAnalysisContextBody,
)
from src.orchestration.agent.comprehensive_health_analysis_strategy.comprehensive_health_analysis_result import (
    ComprehensiveHealthAnalysisResultData,
)
from src.config.business.report_service_config import get_runtime_config
from src.errors import ErrorCode, MilvusUnavailableError, Neo4jConnectionError, LLMServiceError, HealthAssessmentError
from src.utils.logger import log_arch_event

logger = logging.getLogger(__name__)


# ============================================================================
# 常量定义 - 通过配置类集中管理
# ============================================================================

class _LazyConfig:
    """延迟加载配置代理，每次属性访问时从ConfigManager获取真实配置"""

    def __getattr__(self, name):
        real = get_runtime_config()
        return getattr(real, name)

_config = _LazyConfig()

# 模块级常量：先使用默认值初始化，在execute()入口通过_refresh_config()刷新
# 原因：模块导入时ConfigManager尚未初始化，get_runtime_config()会返回默认值
# 在请求处理时刷新可获取application.yaml中的正确配置

MAX_STEPS = _config.max_steps
MAX_PROMPT_CHARS = _config.max_prompt_chars
DIMENSION_MAX_KNOWLEDGE_ITEMS = _config.dimension_max_knowledge_items
SUFFICIENCY_THRESHOLD = _config.sufficiency_threshold
RELEVANCE_ALPHA = _config.relevance_alpha
RELEVANCE_BETA = _config.relevance_beta
RELEVANCE_GAMMA = _config.relevance_gamma
DEGRADED_ALPHA = _config.degraded_alpha
DEGRADED_BETA = _config.degraded_beta
DEGRADED_GAMMA = _config.degraded_gamma
RELEVANCE_THRESHOLD = _config.relevance_threshold
MAX_RETRIEVE_ATTEMPTS = _config.max_retrieve_attempts
VECTOR_CANDIDATE_TOP_K = _config.vector_candidate_top_k
VECTOR_CANDIDATE_THRESHOLD = _config.vector_candidate_threshold
VECTOR_DEFAULT_SCORE = _config.vector_default_score
LOW_QUALITY_MIN_CONTENT_LEN = _config.low_quality_min_content_len
DIMENSION_WEIGHTS = _config.dimension_weights
DIMENSION_NAMES = list(DIMENSION_WEIGHTS.keys()) if isinstance(DIMENSION_WEIGHTS, dict) else DIMENSION_WEIGHTS
STATE_TIMEOUTS = _config.analysis_state_timeouts

# 维度显示名称映射（代码key → 文档编号+中文名）
DIMENSION_DISPLAY_NAMES = {
    "disease_risk": "D1:疾病风险",
    "medication": "D2:用药建议",
    "treatment": "D3:治疗方案",
    "dietary": "D4:饮食建议",
    "checkup": "D5:检查建议",
    "complication": "D6:并发症预警",
    "prevention": "D7:预防措施",
    "susceptible": "D8:易感人群",
}


def _refresh_config():
    """刷新模块级常量，从ConfigManager获取application.yaml中的正确配置值。

    必须在execute()入口处调用，因为此时ConfigManager已完成初始化。
    模块导入时ConfigManager尚未加载YAML配置，导致默认值不正确。
    """
    global MAX_STEPS, MAX_PROMPT_CHARS, DIMENSION_MAX_KNOWLEDGE_ITEMS
    global SUFFICIENCY_THRESHOLD, RELEVANCE_ALPHA, RELEVANCE_BETA, RELEVANCE_GAMMA
    global DEGRADED_ALPHA, DEGRADED_BETA, DEGRADED_GAMMA
    global RELEVANCE_THRESHOLD, MAX_RETRIEVE_ATTEMPTS
    global VECTOR_CANDIDATE_TOP_K, VECTOR_CANDIDATE_THRESHOLD, VECTOR_DEFAULT_SCORE
    global LOW_QUALITY_MIN_CONTENT_LEN, DIMENSION_WEIGHTS, DIMENSION_NAMES, STATE_TIMEOUTS

    MAX_STEPS = _config.max_steps
    MAX_PROMPT_CHARS = _config.max_prompt_chars
    DIMENSION_MAX_KNOWLEDGE_ITEMS = _config.dimension_max_knowledge_items
    SUFFICIENCY_THRESHOLD = _config.sufficiency_threshold
    RELEVANCE_ALPHA = _config.relevance_alpha
    RELEVANCE_BETA = _config.relevance_beta
    RELEVANCE_GAMMA = _config.relevance_gamma
    DEGRADED_ALPHA = _config.degraded_alpha
    DEGRADED_BETA = _config.degraded_beta
    DEGRADED_GAMMA = _config.degraded_gamma
    RELEVANCE_THRESHOLD = _config.relevance_threshold
    MAX_RETRIEVE_ATTEMPTS = _config.max_retrieve_attempts
    VECTOR_CANDIDATE_TOP_K = _config.vector_candidate_top_k
    VECTOR_CANDIDATE_THRESHOLD = _config.vector_candidate_threshold
    VECTOR_DEFAULT_SCORE = _config.vector_default_score
    LOW_QUALITY_MIN_CONTENT_LEN = _config.low_quality_min_content_len
    DIMENSION_WEIGHTS = _config.dimension_weights
    DIMENSION_NAMES = list(DIMENSION_WEIGHTS.keys()) if isinstance(DIMENSION_WEIGHTS, dict) else DIMENSION_WEIGHTS
    STATE_TIMEOUTS = _config.analysis_state_timeouts


# ============================================================================
# ComprehensiveHealthAnalysisStrategy类
# ============================================================================

class ComprehensiveHealthAnalysisStrategy(
    AgentStrategy[ComprehensiveHealthAnalysisContextBody, ComprehensiveHealthAnalysisResultData]
):
    """
    综合健康分析Agent
    
    基于设计文档《项目业务详细设计v5》第3.3节设计实现。
    
    核心特点：
    - 统一管理：单个Agent管理8维度检索+健康评估，避免维度间重复检索
    - LLM作为决策者：大语言模型根据当前上下文动态决定检索策略和健康评估
    - 动态策略调整：根据检索结果实时调整检索策略
    - 健康评估集成：在知识精炼后直接执行健康评估
    - 降级保障：Agent失败时自动回退到固定流程
    
    v8扁平状态机（10个主状态，无嵌套子状态机）：
    BuildQueries → PlanRetrieval → InitRetrievalContext → ParallelDimensionRetrieve →
    InterDimensionSync → HybridRelevance → EvaluateSufficiency → RefineKnowledge → HealthAssess → Output

    检索chain：BuildQueries → PlanRetrieval → InitRetrievalContext → ParallelDimensionRetrieve → InterDimensionSync
    循环路径：InterDimensionSync → BuildQueries（有跨维度重复且chain_loop_count < max）
    循环路径：EvaluateSufficiency → BuildQueries（不充分且agent_retrieval_loop_count < max）
    """
    
    def execute(
        self,
        context: AgentContext[ComprehensiveHealthAnalysisContextBody],
        resource: AgentResource
    ) -> AgentResult[ComprehensiveHealthAnalysisResultData]:
        """
        执行综合健康分析Agent策略
        
        Args:
            context: Agent输入数据容器
            resource: Agent资源类
            
        Returns:
            AgentResult: Agent输出数据容器
        """
        _refresh_config()
        start_time = time.time()
        logger.info(f"[ComprehensiveHealthAnalysisStrategy] 开始执行: session_id={context.session_id}")
        log_arch_event(
            logger,
            component="ComprehensiveHealthAnalysisStrategy",
            stage="ORCHESTRATION",
            event="strategy_execute",
            status="start",
            design_id="BIZ-3.7",
        )

        body = context.body
        if body is None:
            logger.warning("[ComprehensiveHealthAnalysisStrategy] 输入数据为空")
            return AgentResult(
                session_id=context.session_id,
                data=ComprehensiveHealthAnalysisResultData(
                    error_code=ErrorCode.UNKNOWN,
                    error_message="输入数据为空"
                )
            )
        
        # 初始化状态机
        state_machine = StateMachine(context.session_id)
        self._register_state_transitions(state_machine)
        
        # 注册状态处理器
        self._state_handlers = {
            "BuildQueries": self._handle_build_queries,
            "PlanRetrieval": self._handle_plan_retrieval,
            "InitRetrievalContext": self._handle_init_retrieval_context,
            "ParallelDimensionRetrieve": self._handle_parallel_dimension_retrieve,
            "InterDimensionSync": self._handle_inter_dimension_sync,
            "HybridRelevance": self._handle_hybrid_relevance,
            "EvaluateSufficiency": self._handle_evaluate_sufficiency,
            "RefineKnowledge": self._handle_refine_knowledge,
            "HealthAssess": self._handle_health_assess,
            "Output": self._handle_output,
        }
        
        # 执行状态机
        current_state = body.current_state
        max_iterations = _config.analysis_max_iterations
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"[ComprehensiveHealthAnalysisStrategy] 状态转换: "
                       f"current_state={current_state}, iteration={iteration}")
            
            handler = self._state_handlers.get(current_state)
            if handler is None:
                logger.error(f"[ComprehensiveHealthAnalysisStrategy] 未知状态: {current_state}")
                body.current_state = "ERROR"
                break
            
            timeout = STATE_TIMEOUTS.get(current_state)
            try:
                if timeout:
                    next_state = self._execute_with_timeout(handler, body, resource, timeout)
                else:
                    next_state = self._execute_handler(handler, body, resource)
                logger.info(f"[STATE_TRANSITION_REASON] current_state={current_state}, next_state={next_state}, reason=状态处理器正常返回")
            except TimeoutError as te:
                logger.error(f"[ComprehensiveHealthAnalysisStrategy] 状态超时: "
                           f"state={current_state}, timeout={timeout}s")
                next_state = self._handle_timeout(body, current_state, te)
                logger.info(f"[STATE_TRANSITION_REASON] current_state={current_state}, next_state={next_state}, reason=状态超时")
            except EngineUnavailableError as eue:
                logger.error(f"[ComprehensiveHealthAnalysisStrategy] SGLang引擎已死，跳过后续LLM调用: "
                           f"state={current_state}, error_type={type(eue).__name__}")
                body.degraded = True
                body.degraded_reason = f"SGLang引擎崩溃({current_state}状态)，跳过后续LLM调用"
                next_state = self._handle_engine_dead(body, current_state, eue)
                logger.info(f"[STATE_TRANSITION_REASON] current_state={current_state}, next_state={next_state}, reason=SGLang引擎崩溃")
            except Exception as e:
                logger.error(f"[ComprehensiveHealthAnalysisStrategy] 状态处理异常: "
                           f"state={current_state}, error={type(e).__name__}")
                next_state = self._handle_error(body, e)
                logger.info(f"[STATE_TRANSITION_REASON] current_state={current_state}, next_state={next_state}, reason=状态处理异常:{type(e).__name__}")
            
            state_machine.transition(current_state, next_state,
                trigger=_get_analysis_trigger(current_state, next_state),
                reason=_get_analysis_reason(current_state, next_state, body))
            current_state = next_state
            body.current_state = current_state

            if current_state in ("Output", "ERROR"):
                break
        
        # 构建结果
        result_data = self._build_result(body)
        
        elapsed = time.time() - start_time
        logger.info(f"[ComprehensiveHealthAnalysisStrategy] 执行完成: "
                   f"session_id={context.session_id}, elapsed={elapsed:.2f}s, "
                   f"degraded={result_data.degraded}")
        
        return AgentResult(session_id=context.session_id, data=result_data)
    
    # ========================================================================
    # 状态机注册
    # ========================================================================
    
    def _register_state_transitions(self, state_machine: StateMachine) -> None:
        """注册状态转换规则"""
        # Agent检索超时配置
        state_timeouts = _config.analysis_state_timeouts

        # v8扁平状态机转换规则
        state_machine.add_state_transition("BuildQueries", ["PlanRetrieval", "ERROR"])
        state_machine.add_state_transition("PlanRetrieval", ["InitRetrievalContext", "ERROR"])
        state_machine.add_state_transition("InitRetrievalContext", ["ParallelDimensionRetrieve", "ERROR"])
        state_machine.add_state_transition("ParallelDimensionRetrieve", ["InterDimensionSync", "ERROR"])
        state_machine.add_state_transition("InterDimensionSync", ["BuildQueries", "HybridRelevance", "ERROR"])
        state_machine.add_state_transition("HybridRelevance", ["EvaluateSufficiency", "ERROR"])
        state_machine.add_state_transition("EvaluateSufficiency", ["BuildQueries", "RefineKnowledge", "ERROR"])
        state_machine.add_state_transition("RefineKnowledge", ["HealthAssess", "ERROR"])
        state_machine.add_state_transition("HealthAssess", ["Output", "ERROR"])
        state_machine.add_state_transition("Output", [])
        state_machine.add_state_transition("ERROR", ["Output"])
    
    def _merge_timeout_context(
        self,
        target: ComprehensiveHealthAnalysisContextBody,
        source: ComprehensiveHealthAnalysisContextBody
    ) -> None:
        for context_field in fields(ComprehensiveHealthAnalysisContextBody):
            setattr(target, context_field.name, copy.deepcopy(getattr(source, context_field.name)))

    def _execute_handler(
        self,
        handler,
        context: ComprehensiveHealthAnalysisContextBody,
        resource: AgentResource
    ) -> str:
        """执行状态处理器（无超时）
        
        模型服务层已将适配层的EngineDeadException转换为编排层的EngineUnavailableError，
        此处无需再进行异常转换。
        """
        try:
            return handler(context, resource)
        except EngineUnavailableError:
            raise
        except Exception as e:
            # 兜底：检查异常名称中是否包含EngineDead，防止遗漏的适配层异常
            error_name = type(e).__name__
            error_msg = type(e).__name__
            if "EngineDead" in error_name or "EngineDead" in error_msg:
                raise EngineUnavailableError(f"SGLang引擎已不可用: {error_name}: {error_msg}") from e
            raise

    def _execute_with_timeout(
        self,
        handler,
        context: ComprehensiveHealthAnalysisContextBody,
        resource: AgentResource,
        timeout_seconds: int
    ) -> str:
        """带超时执行状态处理器
        
        模型服务层已将适配层的EngineDeadException转换为编排层的EngineUnavailableError，
        此处无需再进行异常转换。
        """
        execution_context = copy.deepcopy(context)
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(handler, execution_context, resource)
        try:
            next_state = future.result(timeout=timeout_seconds)
            self._merge_timeout_context(context, execution_context)
            return next_state
        except FuturesTimeoutError:
            future.cancel()
            raise TimeoutError(f"State execution timed out after {timeout_seconds} seconds")
        except EngineUnavailableError:
            raise
        except Exception as e:
            # 兜底：检查异常名称中是否包含EngineDead，防止遗漏的适配层异常
            error_name = type(e).__name__
            error_msg = type(e).__name__
            if "EngineDead" in error_name or "EngineDead" in error_msg:
                raise EngineUnavailableError(f"SGLang引擎已不可用: {error_name}: {error_msg}") from e
            raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
    
    # ========================================================================
    # 主状态处理器
    # ========================================================================
    
    def _handle_build_queries(
        self,
        context: ComprehensiveHealthAnalysisContextBody,
        resource: AgentResource
    ) -> str:
        """
        BuildQueries状态：构建8维度查询文本
        
        基于维度类型和用户数据构建8个维度的查询文本：
        - D1:疾病风险: 异常指标 + 风险因子
        - D2:用药建议: 既往病史 + 当前用药
        - D3:治疗方案: 疾病诊断
        - D4:饮食建议: 疾病 + BMI
        - D5:检查建议: 异常指标 + 疾病
        - D6:并发症预警: 疾病 + 病史
        - D7:预防措施: 风险因子 + 疾病
        - D8:易感人群: 年龄 + 性别 + 病史
        """
        logger.info("[ComprehensiveHealthAnalysisStrategy] BuildQueries: 构建维度查询")
        stage_start_time = time.time()
        logger.info("[STAGE_ENTER] BuildQueries")

        # v8: 只为未标记充分的维度构建查询
        sufficient_dims = {dim for dim, dk in context.dimension_knowledge.items() if dk.is_sufficient}
        if sufficient_dims:
            logger.info(f"[BuildQueries] v8维度标记: 跳过已充分维度={sufficient_dims}")

        # 提取疾病实体
        diseases = context.medical_entities.get("diseases", [])
        disease_names = [d.get("entity_name", d.get("name", "")) for d in diseases if d]
        disease_names = [n for n in disease_names if n]
        
        # 提取药物实体
        medications = context.medical_entities.get("medications", [])
        medication_names = [m.get("entity_name", m.get("name", "")) for m in medications if m]
        medication_names = [n for n in medication_names if n]
        
        # 提取异常指标名称
        anomaly_names = [a.get("indicator_name", a.get("name", "")) for a in context.anomalies if a]
        anomaly_names = [n for n in anomaly_names if n]
        
        # 提取风险因子名称
        risk_factor_names = [r.get("factor_name", r.get("name", "")) for r in context.risk_factors if r]
        risk_factor_names = [n for n in risk_factor_names if n]
        
        # 用户档案信息
        age = context.user_profile.get("age", -1)
        gender = context.user_profile.get("gender", "")
        medical_history = context.user_profile.get("past_medical_history", "")
        
        # 构建查询文本
        dimension_queries = {}
        
        user_context_parts = []
        if isinstance(age, int) and age > 0:
            user_context_parts.append(f"{age}岁")
        if gender:
            user_context_parts.append(f"{gender}性")
        if medical_history:
            user_context_parts.append(medical_history)
        user_context = " ".join(user_context_parts) if user_context_parts else ""
        
        # D1: 疾病风险
        if disease_names:
            base_query = f"{' '.join(disease_names[:_config.query_disease_limit])} 风险因素 并发症"
        elif anomaly_names:
            base_query = f"{' '.join(anomaly_names[:_config.query_anomaly_limit])} 风险因素"
        else:
            base_query = "常见疾病风险因素"
        if user_context:
            dimension_queries["disease_risk"] = f"{base_query} {user_context}"
        else:
            dimension_queries["disease_risk"] = base_query
        
        # D2: 用药建议
        if disease_names and medication_names:
            base_query = f"{' '.join(disease_names[:_config.query_entity_limit])} {' '.join(medication_names[:_config.query_entity_limit])} 用药指导 禁忌"
        elif disease_names:
            base_query = f"{' '.join(disease_names[:_config.query_entity_limit])} 常用药物 用药指导"
        else:
            base_query = "常见疾病用药建议"
        if user_context:
            dimension_queries["medication"] = f"{base_query} {user_context}"
        else:
            dimension_queries["medication"] = base_query
        
        # D3: 治疗方案
        if disease_names:
            base_query = f"{' '.join(disease_names[:_config.query_entity_limit])} 治疗方案 指南"
        else:
            base_query = "常见疾病治疗方案"
        if user_context:
            dimension_queries["treatment"] = f"{base_query} {user_context}"
        else:
            dimension_queries["treatment"] = base_query
        
        # D4: 饮食建议
        if disease_names:
            base_query = f"{' '.join(disease_names[:_config.query_entity_limit])} 饮食禁忌 营养建议"
        else:
            base_query = "健康饮食建议"
        if user_context:
            dimension_queries["dietary"] = f"{base_query} {user_context}"
        else:
            dimension_queries["dietary"] = base_query
        
        # D5: 检查建议
        if disease_names:
            base_query = f"{' '.join(disease_names[:_config.query_entity_limit])} 定期检查 筛查项目"
        elif anomaly_names:
            base_query = f"{' '.join(anomaly_names[:_config.query_entity_limit])} 相关检查项目"
        else:
            base_query = "常规体检项目建议"
        if user_context:
            dimension_queries["checkup"] = f"{base_query} {user_context}"
        else:
            dimension_queries["checkup"] = base_query
        
        # D6: 并发症预警
        if disease_names:
            base_query = f"{' '.join(disease_names[:_config.query_entity_limit])} 并发症 预防"
        else:
            base_query = "常见疾病并发症预警"
        if user_context:
            dimension_queries["complication"] = f"{base_query} {user_context}"
        else:
            dimension_queries["complication"] = base_query
        
        # D7: 预防措施
        if disease_names or risk_factor_names:
            entities = disease_names[:_config.query_entity_limit] if disease_names else risk_factor_names[:_config.query_entity_limit]
            base_query = f"{' '.join(entities)} 预防措施 生活方式"
        else:
            base_query = "疾病预防措施"
        if user_context:
            dimension_queries["prevention"] = f"{base_query} {user_context}"
        else:
            dimension_queries["prevention"] = base_query
        
        # D8: 易感人群
        if disease_names:
            base_query = f"{' '.join(disease_names[:_config.query_entity_limit])} 易感人群 风险因素"
        else:
            base_query = "常见疾病易感人群"
        if user_context:
            dimension_queries["susceptible"] = f"{base_query} {user_context}"
        else:
            dimension_queries["susceptible"] = base_query
        
        context.dimension_queries = dimension_queries

        # v8: 删除已标记充分维度的查询
        for dim_name in list(dimension_queries.keys()):
            if dim_name in sufficient_dims:
                del dimension_queries[dim_name]
                logger.info(f"[BuildQueries] v8: 跳过已充分维度{dim_name}")

        # 填充NER实体分类（供Agent检索使用）
        context.ner_entities = {
            "disease_names": disease_names,
            "medication_names": medication_names,
            "symptom_names": [a.get("indicator_name", a.get("name", "")) for a in context.anomalies if a.get("indicator_name", a.get("name", ""))],
        }

        query_lengths = {dim: len(q) for dim, q in dimension_queries.items()}
        logger.info(f"[BUILD_QUERIES] dimension_count={len(dimension_queries)}, query_lengths={query_lengths}")

        # 初始化维度知识结构
        for dim_name in DIMENSION_NAMES:
            if dim_name in sufficient_dims:
                continue
            context.dimension_knowledge[dim_name] = DimensionKnowledge(
                dimension_name=dim_name,
                query=dimension_queries.get(dim_name, "")
            )

        logger.info(f"[ComprehensiveHealthAnalysisStrategy] BuildQueries: "
                   f"构建了{len(dimension_queries)}个维度查询")
        logger.info(f"[STAGE_EXIT] BuildQueries, duration={time.time() - stage_start_time:.2f}s")

        return "PlanRetrieval"

    def _handle_plan_retrieval(
        self,
        context: ComprehensiveHealthAnalysisContextBody,
        resource: AgentResource
    ) -> str:
        """
        PlanRetrieval状态：Qwen3结构化决策 #1

        为8维度选择检索路径和关注实体。
        同时执行向量预扫描，为后续图查询提供neo4j_node_id映射。

        降级：Qwen3决策失败 → 使用DIMENSION_RECOMMENDATIONS推荐路径
        """
        logger.info("[ComprehensiveHealthAnalysisStrategy] PlanRetrieval: Qwen3规划检索路径")
        stage_start_time = time.time()
        logger.info("[STAGE_ENTER] PlanRetrieval")

        # v8: 只为未标记充分的维度规划检索策略
        sufficient_dims = {dim for dim, dk in context.dimension_knowledge.items() if dk.is_sufficient}

        # 向量预扫描：NER实体 → neo4j_node_id
        vector_tool = resource.get_tool_handler("vector_retrieval_tool")
        from src.orchestration.agent.comprehensive_health_analysis_strategy.vector_prescan import VectorPrescan
        prescan = VectorPrescan()

        all_entity_names = []
        for names in context.ner_entities.values():
            all_entity_names.extend(names)

        try:
            context.prescan_results = prescan.prescan_entities(all_entity_names, vector_tool)
        except Exception as e:
            logger.warning(f"[PlanRetrieval] 向量预扫描失败: {type(e).__name__}, 继续使用NER实体")
            context.prescan_results = {}

        # Qwen3规划检索路径
        from src.orchestration.agent.comprehensive_health_analysis_strategy.retrieval_planner import RetrievalPlanner
        planner = RetrievalPlanner()

        try:
            context.retrieval_plan = planner.plan(
                dimension_queries=context.dimension_queries,
                ner_entities=context.ner_entities,
                model_service=resource.model_service,
            )
        except Exception as e:
            logger.warning(f"[PlanRetrieval] Qwen3规划失败({type(e).__name__})，使用推荐路径")
            from src.orchestration.agent.comprehensive_health_analysis_strategy.path_registry import get_recommended_paths_for_dimension
            max_paths = _config.agent_max_paths_per_dimension
            from src.orchestration.agent.comprehensive_health_analysis_strategy.retrieval_planner import RetrievalPlan
            context.retrieval_plan = {
                dim: RetrievalPlan(paths=get_recommended_paths_for_dimension(dim)[:max_paths], entities=[])
                for dim in context.dimension_queries
            }

        # 记录每维度已选路径
        context.dimension_used_paths = {
            dim: plan.paths for dim, plan in context.retrieval_plan.items()
        }

        plan_summary = {dim: plan.paths for dim, plan in context.retrieval_plan.items()}
        logger.info(f"[PlanRetrieval] 规划完成: {plan_summary}")
        logger.info(f"[STAGE_EXIT] PlanRetrieval, duration={time.time() - stage_start_time:.2f}s")

        return "InitRetrievalContext"

    def _handle_evaluate_sufficiency(
        self,
        context: ComprehensiveHealthAnalysisContextBody,
        resource: AgentResource
    ) -> str:
        """
        EvaluateSufficiency状态：Qwen3结构化决策 #2

        8维度并发评估每维度知识充分性。
        不充分时输出replace_indices（要替换的知识索引），而非supplement路径。
        删除replace项→加入黑名单→保存保留知识→进入PatchPathPlan。
        """
        logger.info("[ComprehensiveHealthAnalysisStrategy] EvaluateSufficiency: 评估知识充分性")
        stage_start_time = time.time()
        logger.info("[STAGE_ENTER] EvaluateSufficiency")

        # v8: 只对未标记充分的维度评估
        sufficient_dims = {dim for dim, dk in context.dimension_knowledge.items() if dk.is_sufficient}

        context.supplement_round += 1
        max_rounds = _config.max_agent_retrieval_loops

        model_service = resource.model_service
        from src.orchestration.agent.comprehensive_health_analysis_strategy.sufficiency_evaluator import SufficiencyEvaluator
        evaluator = SufficiencyEvaluator()

        dims_to_evaluate = {dim: dk for dim, dk in context.dimension_knowledge.items() if dim not in sufficient_dims}

        try:
            context.sufficiency_results = evaluator.evaluate(
                dimension_knowledge=dims_to_evaluate,
                dimension_used_paths=context.dimension_used_paths,
                supplement_round=context.iteration_round + 1,
                model_service=model_service,
            )
        except Exception as e:
            logger.warning(f"[EvaluateSufficiency] 评估失败({type(e).__name__})，使用规则引擎评分")
            from src.orchestration.agent.comprehensive_health_analysis_strategy.sufficiency_evaluator import SufficiencyEvaluator
            fallback_evaluator = SufficiencyEvaluator()
            context.sufficiency_results = {
                dim: fallback_evaluator.evaluate_rule_based(context.dimension_knowledge[dim])
                for dim in dims_to_evaluate
            }

        # 处理不充分维度：删除replace项、加入黑名单、保存保留知识
        insufficient_dims = []
        vacancy_dimensions = []

        for dim_name, suf in context.sufficiency_results.items():
            if suf.sufficient:
                # DIFF-01 fix: 充分时标记维度，避免重复评估
                dim_know = context.dimension_knowledge.get(dim_name)
                if dim_know:
                    dim_know.is_sufficient = True
                logger.info(f"[EvaluateSufficiency] 维度{dim_name}: 充分，标记is_sufficient=True")
                continue

            if suf.replace_indices:
                insufficient_dims.append(dim_name)
                vacancy_dimensions.append(dim_name)

                dim_know = context.dimension_knowledge.get(dim_name)
                if not dim_know:
                    continue

                items = dim_know.refined_knowledge
                remove_set = set(suf.replace_indices)
                removed_entities = []
                for idx in suf.replace_indices:
                    if 0 <= idx < len(items):
                        neo4j_id = items[idx].get("neo4j_id", items[idx].get("neo4j_node_id", ""))
                        if neo4j_id:
                            context.knowledge_blacklist.add(neo4j_id)
                            removed_entities.append(neo4j_id)

                kept = [item for i, item in enumerate(items) if i not in remove_set]
                dim_know.refined_knowledge = kept

                # 保存保留知识
                context.retained_knowledge[dim_name] = [item.copy() for item in kept]

                logger.info(
                    f"[EvaluateSufficiency] 维度{dim_name}: "
                    f"删除{len(removed_entities)}项(索引={suf.replace_indices}), "
                    f"保留{len(kept)}项, 黑名单新增={removed_entities}"
                )
            else:
                # 不充分但无replace_indices，仍标记为不充分维度
                insufficient_dims.append(dim_name)
                vacancy_dimensions.append(dim_name)

        logger.info(
            f"[EvaluateSufficiency] 第{context.iteration_round + 1}/{max_rounds}轮: "
            f"不充分维度={len(insufficient_dims)}/{len(context.sufficiency_results)}, "
            f"维度列表={insufficient_dims}"
        )

        # 判断是否需要迭代补充（v8: 返回BuildQueries而非PatchPathPlan）
        if insufficient_dims and context.agent_retrieval_loop_count < max_rounds:
            context.agent_retrieval_loop_count += 1
            context.chain_loop_count = 0  # v8: 重置chain_loop_count
            context.vacancy_dimensions = vacancy_dimensions
            context.is_partial_retrieve = True
            logger.info(
                f"[STAGE_EXIT] EvaluateSufficiency, v8: 转入BuildQueries(检索chain入口), "
                f"agent_retrieval_loop_count={context.agent_retrieval_loop_count}/{max_rounds}, "
                f"duration={time.time() - stage_start_time:.2f}s"
            )
            return "BuildQueries"
        else:
            if insufficient_dims:
                logger.warning(
                    f"[EvaluateSufficiency] 已达最大迭代轮次{max_rounds}，"
                    f"仍有{len(insufficient_dims)}个维度不充分，强制进入RefineKnowledge"
                )
            logger.info(f"[STAGE_EXIT] EvaluateSufficiency, 转入RefineKnowledge, duration={time.time() - stage_start_time:.2f}s")
            return "RefineKnowledge"

    def _handle_hybrid_relevance(
        self,
        context: ComprehensiveHealthAnalysisContextBody,
        resource: AgentResource
    ) -> str:
        """
        HybridRelevance状态：Qwen3混合相关性评估

        对每个维度的每项知识打分，程序计算混合分数。
        结果写入DimensionKnowledge.hybrid_scores，供EvaluateSufficiency参考。
        """
        logger.info("[ComprehensiveHealthAnalysisStrategy] HybridRelevance: 混合相关性评估")
        stage_start_time = time.time()
        logger.info("[STAGE_ENTER] HybridRelevance")

        # v8: 只为未标记充分的维度评分
        sufficient_dims = {dim for dim, dk in context.dimension_knowledge.items() if dk.is_sufficient}
        dims_to_evaluate = {dim: dk for dim, dk in context.dimension_knowledge.items() if dim not in sufficient_dims}

        model_service = resource.model_service
        if not model_service:
            logger.warning("[HybridRelevance] 无模型服务，跳过相关性评估")
            logger.info(f"[STAGE_EXIT] HybridRelevance, duration={time.time() - stage_start_time:.2f}s")
            return "EvaluateSufficiency"

        if not dims_to_evaluate:
            logger.info("[HybridRelevance] 所有维度已充分，跳过评估")
            logger.info(f"[STAGE_EXIT] HybridRelevance, duration={time.time() - stage_start_time:.2f}s")
            return "EvaluateSufficiency"

        user_info = self._build_user_info_summary(context)

        from src.orchestration.agent.comprehensive_health_analysis_strategy.knowledge_relevance_evaluator import KnowledgeRelevanceEvaluator
        evaluator = KnowledgeRelevanceEvaluator()

        try:
            evaluator.evaluate(
                dimension_knowledge=dims_to_evaluate,
                user_info=user_info,
                model_service=model_service,
            )
        except Exception as e:
            logger.warning(
                f"[HybridRelevance] 评估异常({type(e).__name__})，跳过"
            )

        logger.info(f"[STAGE_EXIT] HybridRelevance, duration={time.time() - stage_start_time:.2f}s")
        return "EvaluateSufficiency"

    def _build_user_info_summary(
        self,
        context: ComprehensiveHealthAnalysisContextBody,
    ) -> str:
        """构建用户信息摘要，供KnowledgeSelection/HybridRelevance/PatchPathPlan使用"""
        profile = context.user_profile
        age = profile.get("age", -1)
        age_str = f"{age}岁" if isinstance(age, int) and age > 0 else "未知"
        gender = profile.get("gender", "未知")
        history = profile.get("past_medical_history", "无")
        family = profile.get("family_history", "无")

        anomaly_names = [
            a.get("indicator_name", a.get("name", ""))
            for a in context.anomalies if a
        ]
        anomaly_names = [n for n in anomaly_names if n]
        anomalies_str = "、".join(anomaly_names[:5]) if anomaly_names else "无"

        disease_names = context.ner_entities.get("disease_names", [])
        diseases_str = "、".join(disease_names[:5]) if disease_names else "无"

        return (
            f"年龄={age_str}, 性别={gender}, "
            f"病史={history}, 家族史={family}, "
            f"异常指标={anomalies_str}, 疾病={diseases_str}"
        )

    def _handle_init_retrieval_context(
        self,
        context: ComprehensiveHealthAnalysisContextBody,
        resource: AgentResource
    ) -> str:
        """InitRetrievalContext状态：初始化检索上下文"""
        logger.info("[ComprehensiveHealthAnalysisStrategy] InitRetrievalContext: 初始化检索上下文")
        stage_start_time = time.time()
        logger.info("[STAGE_ENTER] InitRetrievalContext")

        if context.shared_memory.common_knowledge:
            logger.info("[InitRetrievalContext] 已初始化，快速跳过")
            logger.info(f"[STAGE_EXIT] InitRetrievalContext, duration={time.time() - stage_start_time:.2f}s")
            return "ParallelDimensionRetrieve"

        context.shared_memory = SharedMemory()
        context.retrieval_stats = RetrievalStats()
        context.chain_loop_count = 0

        logger.info("[InitRetrievalContext] 首次初始化完成")
        logger.info(f"[STAGE_EXIT] InitRetrievalContext, duration={time.time() - stage_start_time:.2f}s")
        return "ParallelDimensionRetrieve"

    def _handle_parallel_dimension_retrieve(
        self,
        context: ComprehensiveHealthAnalysisContextBody,
        resource: AgentResource
    ) -> str:
        """ParallelDimensionRetrieve状态：8维度并行检索"""
        logger.info("[ComprehensiveHealthAnalysisStrategy] ParallelDimensionRetrieve: 执行维度检索")
        stage_start_time = time.time()
        logger.info("[STAGE_ENTER] ParallelDimensionRetrieve")

        neo4j_tool = resource.get_tool_handler("neo4j_medical_tool")
        from src.orchestration.agent.comprehensive_health_analysis_strategy.retrieval_executor import RetrievalExecutor
        executor = RetrievalExecutor()

        # v8: 只为未标记充分的维度执行检索
        sufficient_dims = {dim for dim, dk in context.dimension_knowledge.items() if dk.is_sufficient}
        active_plan = {
            dim: plan for dim, plan in context.retrieval_plan.items()
            if dim not in sufficient_dims
        }

        if not active_plan:
            logger.warning("[ParallelDimensionRetrieve] 无需检索的维度（全部已充分），跳过")
            logger.info(f"[STAGE_EXIT] ParallelDimensionRetrieve, duration={time.time() - stage_start_time:.2f}s")
            return "InterDimensionSync"

        logger.info(f"[ParallelDimensionRetrieve] v8: 检索维度={list(active_plan.keys())}, 跳过充分维度={sufficient_dims}")

        try:
            retrieval_results = executor.execute_plan(
                plan=active_plan,
                prescan_results=context.prescan_results,
                neo4j_tool=neo4j_tool,
                context_entities=context.ner_entities,
            )

            # v8: 黑名单过滤检索结果
            blacklist = context.knowledge_blacklist
            filtered_count = 0
            for dim_name, knowledge_items in retrieval_results.items():
                if blacklist:
                    original_count = len(knowledge_items)
                    knowledge_items = [
                        item for item in knowledge_items
                        if item.get("neo4j_id", item.get("neo4j_node_id", "")) not in blacklist
                    ]
                    filtered_count += original_count - len(knowledge_items)
                    retrieval_results[dim_name] = knowledge_items
                if blacklist:
                    logger.info(f"[ParallelDimensionRetrieve] v8黑名单过滤: 维度{dim_name}, 黑名单数={len(blacklist)}, 过滤={filtered_count}条")

            # 将结果写入维度知识
            for dim_name, knowledge_items in retrieval_results.items():
                if dim_name in context.dimension_knowledge:
                    dim_know = context.dimension_knowledge[dim_name]
                    dim_know.candidate_knowledge = knowledge_items
                    dim_know.raw_knowledge = knowledge_items

            total_items = sum(len(v) for v in retrieval_results.values())
            context.retrieval_stats.total_retrieval_count = total_items
            logger.info(f"[ParallelDimensionRetrieve] 检索完成: 总条目={total_items}")

        except Exception as e:
            logger.error(f"[ParallelDimensionRetrieve] 检索失败: {type(e).__name__}: {str(e)}")

        logger.info(f"[STAGE_EXIT] ParallelDimensionRetrieve, duration={time.time() - stage_start_time:.2f}s")
        return "InterDimensionSync"

    def _handle_inter_dimension_sync(
        self,
        context: ComprehensiveHealthAnalysisContextBody,
        resource: AgentResource
    ) -> str:
        """InterDimensionSync状态：跨维度去重+黑名单更新+知识表维护"""
        logger.info("[ComprehensiveHealthAnalysisStrategy] InterDimensionSync: 跨维度同步")
        stage_start_time = time.time()
        logger.info("[STAGE_ENTER] InterDimensionSync")

        # v8: 只对未标记充分维度的知识进行去重
        sufficient_dims = {dim for dim, dk in context.dimension_knowledge.items() if dk.is_sufficient}

        # Step 1: 跨维度重复检测——按neo4j ID分组
        neo4j_id_map: Dict[str, List[Tuple[str, Dict]]] = {}
        for dim_name, dim_knowledge in context.dimension_knowledge.items():
            if dim_name in sufficient_dims:
                continue
            for item in dim_knowledge.candidate_knowledge:
                neo4j_id = item.get("neo4j_id", item.get("neo4j_node_id", ""))
                if neo4j_id:
                    if neo4j_id not in neo4j_id_map:
                        neo4j_id_map[neo4j_id] = []
                    neo4j_id_map[neo4j_id].append((dim_name, item))

        # Step 2: 跨维度去重——按检索分数保留
        has_cross_dim_duplicates = False
        for neo4j_id, items in neo4j_id_map.items():
            if len(items) > 1:
                has_cross_dim_duplicates = True
                # 按检索分数保留在最高维度，从其他维度移除
                best_dim, best_item = max(items, key=lambda x: x[1].get("retrieval_score", x[1].get("score", 1.0)))
                for dim_name, item in items:
                    if dim_name != best_dim:
                        dim_knowledge = context.dimension_knowledge[dim_name]
                        dim_knowledge.candidate_knowledge = [
                            ki for ki in dim_knowledge.candidate_knowledge
                            if ki.get("neo4j_id", ki.get("neo4j_node_id", "")) != neo4j_id
                        ]
                        # Step 3: 更新知识表——记录跨维度引用
                        if neo4j_id not in context.knowledge_cross_refs:
                            context.knowledge_cross_refs[neo4j_id] = [best_dim]
                        if dim_name not in context.knowledge_cross_refs[neo4j_id]:
                            context.knowledge_cross_refs[neo4j_id].append(dim_name)

        # 更新维度表
        for dim_name, dim_knowledge in context.dimension_knowledge.items():
            if dim_name in sufficient_dims:
                continue
            context.dimension_table[dim_name] = [
                item.get("neo4j_id", item.get("neo4j_node_id", ""))
                for item in dim_knowledge.candidate_knowledge
                if item.get("neo4j_id", item.get("neo4j_node_id", ""))
            ]

        # Step 4: 更新黑名单——所有当前保留的知识项neo4j ID加入黑名单
        for dim_name, dim_knowledge in context.dimension_knowledge.items():
            for item in dim_knowledge.candidate_knowledge:
                neo4j_id = item.get("neo4j_id", item.get("neo4j_node_id", ""))
                if neo4j_id:
                    context.knowledge_blacklist.add(neo4j_id)

        # 将candidate_knowledge设为refined_knowledge供后续状态使用
        for dim_name, dim_knowledge in context.dimension_knowledge.items():
            if dim_name in sufficient_dims:
                continue
            dim_knowledge.refined_knowledge = dim_knowledge.candidate_knowledge

        # Step 5: 判断下一状态
        max_chain_loops = _config.max_chain_loops
        if has_cross_dim_duplicates and context.chain_loop_count < max_chain_loops:
            context.chain_loop_count += 1
            logger.info(f"[InterDimensionSync] v8: 有跨维度重复, 回到检索chain, "
                        f"chain_loop_count={context.chain_loop_count}/{max_chain_loops}")
            logger.info(f"[STAGE_EXIT] InterDimensionSync, duration={time.time() - stage_start_time:.2f}s")
            return "BuildQueries"
        else:
            reason = "无跨维度重复" if not has_cross_dim_duplicates else f"chain_loop_count({context.chain_loop_count})>=max({max_chain_loops})"
            logger.info(f"[InterDimensionSync] v8: {reason}, 进入HybridRelevance")
            logger.info(f"[STAGE_EXIT] InterDimensionSync, duration={time.time() - stage_start_time:.2f}s")
            return "HybridRelevance"
    
    def _handle_refine_knowledge(
        self,
        context: ComprehensiveHealthAnalysisContextBody,
        resource: AgentResource
    ) -> str:
        """
        RefineKnowledge状态：Qwen3-4B知识精炼（批量推理模式）

        处理逻辑：
        1. 收集所有需要精炼的维度
        2. 批量构建知识精炼Prompt
        3. 调用model_service.call_model_batch()批量精炼
        4. 解析每个维度的精炼结果
        5. 降级策略：批量→串行→规则摘要
        """
        logger.info("[ComprehensiveHealthAnalysisStrategy] RefineKnowledge: 开始知识精炼（批量推理模式）")
        stage_start_time = time.time()
        logger.info("[STAGE_ENTER] RefineKnowledge")

        # v5设计验证日志：知识精炼
        dims_with_knowledge = sum(1 for dk in context.dimension_knowledge.values() if dk.refined_knowledge)
        batch_mode = hasattr(resource.model_service, 'call_model_batch') if resource.model_service else False
        logger.info(f"[REFINE_KNOWLEDGE] dimensions={len(context.dimension_knowledge)}, dims_with_knowledge={dims_with_knowledge}, batch_mode={batch_mode}")

        # 获取LLM服务
        model_service = resource.model_service

        # === 收集需要精炼的维度 ===
        dims_to_refine = []  # 需要LLM精炼的维度
        for dim_name, dim_knowledge in context.dimension_knowledge.items():
            if not dim_knowledge.refined_knowledge:
                # 无知识的维度，直接构建空摘要
                context.dimension_summaries[dim_name] = {
                    "summary": "",
                    "key_entities": [],
                    "knowledge_items": []
                }
                continue
            dims_to_refine.append((dim_name, dim_knowledge))

        # === 批量推理：一次性精炼所有维度 ===
        batch_refine_results = {}  # dim_name -> refined_summary
        if dims_to_refine and model_service and hasattr(model_service, 'call_model_batch'):
            batch_start_time = time.time()
            logger.info(f"[ComprehensiveHealthAnalysisStrategy] RefineKnowledge: "
                       f"开始批量知识精炼，维度数={len(dims_to_refine)}")
            logger.info("[DEGRADE_LEVEL] level=0, 正常批量推理")

            try:
                # 构建所有维度的精炼prompt
                batch_prompts = []
                batch_dim_names = []
                batch_prompt_summaries = []  # 记录每个维度的prompt摘要
                for dim_name, dim_knowledge in dims_to_refine:
                    system_msg = "你是一位医学知识精炼专家。你的任务是对检索到的医学知识进行精炼整合，只做去冗余、去重复、去不相关内容的处理工作。请严格按照JSON格式输出，不要添加任何解释或回答用户问题。"
                    user_msg = self._build_refine_prompt(context, dim_knowledge)

                    if len(user_msg) > MAX_PROMPT_CHARS:
                        user_msg = user_msg[:MAX_PROMPT_CHARS]
                        logger.warning(f"[ComprehensiveHealthAnalysisStrategy] RefineKnowledge: "
                                      f"维度{dim_name}的Prompt被截断")

                    prompt = f"{system_msg}\n\n{user_msg}"
                    batch_prompts.append(prompt)
                    batch_dim_names.append(dim_name)
                    batch_prompt_summaries.append(user_msg)

                refine_prompt_lengths = {dim_name: len(prompt) for dim_name, prompt in zip(batch_dim_names, batch_prompt_summaries)}
                logger.info(f"[BATCH_REFINE_START] dimension_count={len(batch_dim_names)}, prompt_lengths={refine_prompt_lengths}")
                logger.info(f"[LLM_INPUT] 批量知识精炼, 维度数={len(batch_dim_names)}, prompt_lengths={refine_prompt_lengths}")

                # 批量调用
                batch_llm_results = model_service.call_model_batch(
                    prompts=batch_prompts,
                    max_tokens=_config.batch_refine_max_tokens,
                    timeout=_config.batch_refine_timeout
                )

                batch_elapsed = time.time() - batch_start_time
                logger.info(f"[LLM_DURATION] 批量知识精炼, 维度数={len(batch_dim_names)}, "
                           f"耗时={batch_elapsed:.2f}s")

                for idx, dim_name in enumerate(batch_dim_names):
                    llm_result = batch_llm_results[idx] if idx < len(batch_llm_results) else None
                    result_len = len(llm_result) if isinstance(llm_result, str) else 0
                    logger.info(f"[LLM_OUTPUT] 批量知识精炼[{idx+1}/{len(batch_dim_names)}], 维度={dim_name}, result_len={result_len}")

                logger.info(f"[ComprehensiveHealthAnalysisStrategy] RefineKnowledge: "
                           f"批量知识精炼完成，维度数={len(batch_dim_names)}, elapsed={batch_elapsed:.2f}s")

                # 解析每个维度的精炼结果
                for i, dim_name in enumerate(batch_dim_names):
                    llm_result = batch_llm_results[i] if i < len(batch_llm_results) else None
                    if llm_result and isinstance(llm_result, str) and len(llm_result.strip()) > 0:
                        batch_refine_results[dim_name] = llm_result.strip()
                        logger.info(f"[ComprehensiveHealthAnalysisStrategy] RefineKnowledge: "
                                   f"维度{dim_name}批量精炼完成, 摘要长度={len(llm_result.strip())}")
                        # 防重复效果日志
                        _finish_reason = "stop" if llm_result.strip().endswith("}") else "length"
                        logger.info(f"[REPETITION_CHECK] dimension={dim_name}, finish_reason_present=True, duration={batch_elapsed:.2f}s")
                    else:
                        logger.warning(f"[ComprehensiveHealthAnalysisStrategy] RefineKnowledge: "
                                      f"维度{dim_name}批量精炼返回空结果，将使用规则摘要")
                        # 防重复效果日志
                        logger.info(f"[REPETITION_CHECK] dimension={dim_name}, finish_reason_present=True, duration={batch_elapsed:.2f}s")

                # 批量精炼后：记录精炼完成数量
                refined_count = len(batch_refine_results)
                logger.info(f"[BATCH_REFINE_COMPLETE] refined_count={refined_count}")

            except Exception as e:
                batch_elapsed = time.time() - batch_start_time
                logger.error(f"[ComprehensiveHealthAnalysisStrategy] RefineKnowledge: "
                            f"批量知识精炼失败，降级为串行推理: elapsed={batch_elapsed:.2f}s, error={type(e).__name__}")
                logger.warning(f"[DEGRADE_LEVEL] level=1, 降级为串行推理, reason={type(e).__name__}")

                # === 二级降级：逐维度串行推理 ===
                for dim_name, dim_knowledge in dims_to_refine:
                    if dim_name in batch_refine_results:
                        continue  # 已有结果的跳过
                    try:
                        serial_start_time = time.time()
                        prompt = self._build_refine_prompt(context, dim_knowledge)
                        if len(prompt) > MAX_PROMPT_CHARS:
                            prompt = prompt[:MAX_PROMPT_CHARS]

                        messages = [
                            {"role": "system", "content": "你是一位医学知识精炼专家。你的任务是对检索到的医学知识进行精炼整合，只做去冗余、去重复、去不相关内容的处理工作。请严格按照JSON格式输出，不要添加任何解释或回答用户问题。"},
                            {"role": "user", "content": prompt}
                        ]
                        logger.info(f"[LLM_INPUT] 串行降级知识精炼, 维度={dim_name}, prompt_len={len(prompt)}")
                        logger.info(f"[ComprehensiveHealthAnalysisStrategy.RefineKnowledge] 串行降级：调用LLM进行知识精炼，维度={dim_name}")
                        refined_result = model_service.call_model(messages)
                        serial_elapsed = time.time() - serial_start_time
                        logger.info(f"[LLM_DURATION] 串行降级知识精炼, 维度={dim_name}, 耗时={serial_elapsed:.2f}s")
                        if refined_result and isinstance(refined_result, str) and len(refined_result.strip()) > 0:
                            batch_refine_results[dim_name] = refined_result.strip()
                            logger.info(f"[LLM_OUTPUT] 串行降级知识精炼, 维度={dim_name}, result_len={len(refined_result.strip())}")
                            logger.info(f"[ComprehensiveHealthAnalysisStrategy] RefineKnowledge: "
                                       f"维度{dim_name}串行降级精炼完成, 摘要长度={len(refined_result.strip())}")
                            # 防重复效果日志
                            logger.info(f"[REPETITION_CHECK] dimension={dim_name}, finish_reason_present=True, duration={serial_elapsed:.2f}s")
                        else:
                            logger.info(f"[LLM_OUTPUT] 串行降级知识精炼, 维度={dim_name}, result_len=0")
                            logger.warning(f"[ComprehensiveHealthAnalysisStrategy] RefineKnowledge: "
                                          f"维度{dim_name}串行降级精炼返回空结果，使用规则摘要")
                            # 防重复效果日志
                            logger.info(f"[REPETITION_CHECK] dimension={dim_name}, finish_reason_present=True, duration={serial_elapsed:.2f}s")
                    except Exception as inner_e:
                        serial_elapsed = time.time() - serial_start_time
                        logger.info(f"[LLM_DURATION] 串行降级知识精炼, 维度={dim_name}, 耗时={serial_elapsed:.2f}s(异常)")
                        logger.warning(f"[ComprehensiveHealthAnalysisStrategy] RefineKnowledge: "
                                      f"维度{dim_name}串行降级精炼失败: error_type={type(inner_e).__name__}，使用规则摘要")
                        # 防重复效果日志
                        logger.info(f"[REPETITION_CHECK] dimension={dim_name}, finish_reason_present=True, duration={serial_elapsed:.2f}s")

        elif dims_to_refine:
            # model_service不支持call_model_batch，直接使用串行推理
            logger.warning("[ComprehensiveHealthAnalysisStrategy] RefineKnowledge: "
                          "model_service不支持call_model_batch，使用串行推理")
            logger.warning("[DEGRADE_LEVEL] level=1, 降级为串行推理, reason=model_service不支持call_model_batch")
            for dim_name, dim_knowledge in dims_to_refine:
                try:
                    serial_start_time = time.time()
                    prompt = self._build_refine_prompt(context, dim_knowledge)
                    if len(prompt) > MAX_PROMPT_CHARS:
                        prompt = prompt[:MAX_PROMPT_CHARS]

                    messages = [
                        {"role": "system", "content": "你是一位医学知识精炼专家。你的任务是对检索到的医学知识进行精炼整合，只做去冗余、去重复、去不相关内容的处理工作。请严格按照JSON格式输出，不要添加任何解释或回答用户问题。"},
                        {"role": "user", "content": prompt}
                    ]
                    logger.info(f"[LLM_INPUT] 串行知识精炼, 维度={dim_name}, prompt_len={len(prompt)}")
                    refined_result = model_service.call_model(messages)
                    serial_elapsed = time.time() - serial_start_time
                    logger.info(f"[LLM_DURATION] 串行知识精炼, 维度={dim_name}, 耗时={serial_elapsed:.2f}s")
                    if refined_result and isinstance(refined_result, str) and len(refined_result.strip()) > 0:
                        batch_refine_results[dim_name] = refined_result.strip()
                        logger.info(f"[LLM_OUTPUT] 串行知识精炼, 维度={dim_name}, result_len={len(refined_result.strip())}")
                        # 防重复效果日志
                        logger.info(f"[REPETITION_CHECK] dimension={dim_name}, finish_reason_present=True, duration={serial_elapsed:.2f}s")
                    else:
                        logger.info(f"[LLM_OUTPUT] 串行知识精炼, 维度={dim_name}, result_len=0")
                        logger.warning(f"[ComprehensiveHealthAnalysisStrategy] RefineKnowledge: "
                                      f"维度{dim_name}串行精炼返回空结果，使用规则摘要")
                        # 防重复效果日志
                        logger.info(f"[REPETITION_CHECK] dimension={dim_name}, finish_reason_present=True, duration={serial_elapsed:.2f}s")
                except Exception as e:
                    serial_elapsed = time.time() - serial_start_time
                    logger.info(f"[LLM_DURATION] 串行知识精炼, 维度={dim_name}, 耗时={serial_elapsed:.2f}s(异常)")
                    logger.warning(f"[ComprehensiveHealthAnalysisStrategy] RefineKnowledge: "
                                  f"维度{dim_name}串行精炼失败: {type(e).__name__}，使用规则摘要")
                    # 防重复效果日志
                    logger.info(f"[REPETITION_CHECK] dimension={dim_name}, finish_reason_present=True, duration={serial_elapsed:.2f}s")

        # === 处理每个维度的精炼结果 ===
        for dim_name, dim_knowledge in dims_to_refine:
            try:
                refined_summary = batch_refine_results.get(dim_name)

                if refined_summary:
                    dim_knowledge.summary = refined_summary
                    logger.info(f"[ComprehensiveHealthAnalysisStrategy] RefineKnowledge: "
                               f"维度{dim_name}LLM精炼完成, 摘要长度={len(refined_summary)}")
                else:
                    # 三级降级：规则摘要
                    summary_parts = []
                    for item in dim_knowledge.refined_knowledge[:_config.rule_knowledge_item_limit]:
                        source_entity = item.get("source_entity", item.get("entity_name", item.get("name", "")))
                        relation_type = item.get("relation_type", "unknown")
                        target_entity = item.get("target_entity", item.get("entity_name", item.get("name", "")))
                        content = item.get("content", item.get("description", ""))
                        if target_entity:
                            summary_parts.append(f"[{source_entity}] -{relation_type}-> [{target_entity}]: \"{self._truncate_by_sentence(content, _config.rule_content_truncate_len)}\"")

                    dim_knowledge.summary = "\n".join(summary_parts)
                    logger.info(f"[ComprehensiveHealthAnalysisStrategy] RefineKnowledge: "
                               f"维度{dim_name}使用规则摘要, 摘要长度={len(dim_knowledge.summary)}")
                    logger.warning(f"[DEGRADE_LEVEL] level=2, 降级为规则引擎, reason=维度{dim_name}LLM精炼结果为空")

                context.dimension_summaries[dim_name] = {
                    "summary": dim_knowledge.summary,
                    "key_entities": [
                        item.get("entity_name", item.get("name", ""))
                        for item in dim_knowledge.refined_knowledge[:_config.rule_entity_limit]
                    ],
                    "knowledge_items": dim_knowledge.refined_knowledge[:_config.knowledge_item_display_limit]
                }

            except Exception as e:
                logger.error(f"[ComprehensiveHealthAnalysisStrategy] RefineKnowledge: "
                           f"维度{dim_name}精炼失败: {type(e).__name__}")
                # 使用原始知识作为摘要
                context.dimension_summaries[dim_name] = {
                    "summary": "知识精炼失败，使用原始知识",
                    "key_entities": [],
                    "knowledge_items": dim_knowledge.refined_knowledge[:_config.knowledge_item_display_limit]
                }

        logger.info(f"[ComprehensiveHealthAnalysisStrategy] RefineKnowledge: "
                   f"精炼了{len(context.dimension_summaries)}个维度")
        logger.info(f"[STAGE_EXIT] RefineKnowledge, duration={time.time() - stage_start_time:.2f}s")

        return "HealthAssess"
    
    def _handle_health_assess(
        self,
        context: ComprehensiveHealthAnalysisContextBody,
        resource: AgentResource
    ) -> str:
        """
        HealthAssess状态：调用HealthAssessmentChain执行健康评估
        
        设计要点：
        1. 不能跳过，必须执行
        2. 检索结果为0时，仅基于用户信息评估
        3. 检索不充分时，使用可用知识+用户信息继续
        4. 失败时，标记降级标志，LLM在报告生成期间评估
        """
        logger.info("[ComprehensiveHealthAnalysisStrategy] HealthAssess: 开始健康评估")
        stage_start_time = time.time()
        logger.info("[STAGE_ENTER] HealthAssess")

        # 检查检索结果
        has_knowledge = any(
            len(summary.get("knowledge_items", [])) > 0
            for summary in context.dimension_summaries.values()
        )
        
        if not has_knowledge:
            logger.warning("[ComprehensiveHealthAnalysisStrategy] HealthAssess: "
                          "检索结果为0，仅基于用户信息进行健康评估")

        # v5设计验证日志：HealthAssess状态进入
        health_assessment_available = False
        try:
            health_assessment_chain = resource.get_chain("health_assessment_chain")
            if health_assessment_chain and hasattr(health_assessment_chain, '_resource'):
                health_assessment_available = getattr(health_assessment_chain._resource, 'health_assessment_model', None) is not None
        except Exception as e:
            logger.warning("[ComprehensiveHealthAnalysisStrategy] HealthAssess: 健康评估链可用性探测失败: "
                           f"error_type={type(e).__name__}")
        logger.info(f"[HEALTH_ASSESS_STATE] has_knowledge={has_knowledge}, health_assessment_available={health_assessment_available}")
        
        try:
            # 尝试获取HealthAssessmentChain
            health_assessment_chain = resource.get_chain("health_assessment_chain")
            
            if health_assessment_chain:
                # 构建HealthAssessmentContextBody
                from src.orchestration.chain.health_assessment_chain.health_assessment_chain import (
                    HealthAssessmentContextBody
                )
                ha_body = HealthAssessmentContextBody(
                    dimension_summaries=context.dimension_summaries,
                    anomalies=context.anomalies,
                    risk_factors=context.risk_factors,
                    medical_entities=context.medical_entities,
                    user_profile=context.user_profile
                )
                chain_context = ChainContext(
                    session_id=context.user_profile.get("session_id", "unknown"),
                    body=ha_body
                )
                chain_result = health_assessment_chain.execute(chain_context, external_degraded=context.degraded)
                
                if chain_result.data:
                    data = chain_result.data
                    if hasattr(data, 'to_dict'):
                        data_dict = data.to_dict()
                    elif isinstance(data, dict):
                        data_dict = data
                    else:
                        data_dict = vars(data) if hasattr(data, '__dict__') else {}
                    
                    context.health_assessment = HealthAssessment(
                        health_score=data_dict.get("health_score"),
                        health_level=data_dict.get("health_level"),
                        risk_level=data_dict.get("risk_level"),
                        disease_risks=data_dict.get("disease_risks", []),
                        score_breakdown=data_dict.get("score_breakdown", {}),
                        reasoning=data_dict.get("reasoning", ""),
                        degraded=data_dict.get("degraded", False),
                        degraded_reason=data_dict.get("degraded_reason", "")
                    )
                    # 将Chain返回的降级状态同步到Agent的context
                    if context.health_assessment.degraded and not context.degraded:
                        context.degraded = True
                        if not context.degraded_reason:
                            context.degraded_reason = context.health_assessment.degraded_reason
                else:
                    raise ValueError("HealthAssessmentChain返回空结果")
            else:
                # 降级：使用规则引擎评估
                logger.warning("[ComprehensiveHealthAnalysisStrategy] HealthAssess: "
                              "HealthAssessmentChain未注册，使用规则引擎评估")
                context.health_assessment = self._fallback_rule_assessment(context)
                context.health_assessment.degraded = True
                context.health_assessment.degraded_reason = "HealthAssessmentChain未注册"
                context.degraded = True
                if not context.degraded_reason:
                    context.degraded_reason = context.health_assessment.degraded_reason

        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"[ComprehensiveHealthAnalysisStrategy] HealthAssess: "
                        f"健康评估失败: error_type={error_type}")

            # 标记降级标志，LLM在报告生成期间评估
            safe_reason = f"HealthAssessmentChain执行失败({error_type})"
            context.health_assessment = HealthAssessment(
                health_score=None,
                health_level=None,
                risk_level=None,
                disease_risks=[],
                score_breakdown={},
                reasoning="",
                degraded=True,
                degraded_reason=safe_reason
            )
            context.degraded = True
            if not context.degraded_reason:
                context.degraded_reason = context.health_assessment.degraded_reason
        
        logger.info(f"[STAGE_EXIT] HealthAssess, duration={time.time() - stage_start_time:.2f}s")
        
        return "Output"
    
    def _handle_output(
        self,
        context: ComprehensiveHealthAnalysisContextBody,
        resource: AgentResource
    ) -> str:
        """
        Output状态：输出最终结果
        
        处理逻辑：
        1. 汇总8维度摘要
        2. 汇总健康评估结果
        3. 整理提交给LLM撰写报告的完整信息
        4. 输出最终结果
        """
        logger.info("[ComprehensiveHealthAnalysisStrategy] Output: 输出最终结果")
        stage_start_time = time.time()
        logger.info("[STAGE_ENTER] Output")
        
        # 计算检索统计信息
        total_knowledge_count = sum(
            len(dim_knowledge.refined_knowledge)
            for dim_knowledge in context.dimension_knowledge.values()
        )
        context.retrieval_stats.total_retrieval_count = total_knowledge_count
        
        logger.info(f"[ComprehensiveHealthAnalysisStrategy] Output: "
                   f"dimension_summaries={len(context.dimension_summaries)}, "
                   f"total_knowledge={total_knowledge_count}, "
                   f"health_assessment={'有' if context.health_assessment else '无'}")
        # v5设计验证日志：Agent输出
        logger.info(f"[AGENT_OUTPUT] dimension_summaries_count={len(context.dimension_summaries)}, health_assessment={context.health_assessment is not None}")
        logger.info(f"[STAGE_EXIT] Output, duration={time.time() - stage_start_time:.2f}s")
        
        return "Output"
    
    # ========================================================================
    # ParallelRetrieve子状态处理器
    # ========================================================================
    
    def _init_retrieval_context(
        self,
        context: ComprehensiveHealthAnalysisContextBody,
        resource: AgentResource
    ) -> None:
        """
        子状态1: InitRetrievalContext - 初始化检索上下文
        
        处理逻辑：
        1. CreateDimensionTasks: 为每个维度创建检索任务
        2. InitSharedMemory: 初始化维度间共享内存
        """
        logger.info("[ComprehensiveHealthAnalysisStrategy] InitRetrievalContext: 初始化检索上下文")
        
        # 初始化共享内存
        context.shared_memory = SharedMemory(
            common_knowledge={},
            cross_references={},
            shared_entities={}
        )
        
        # 重置检索统计
        context.retrieval_stats = RetrievalStats()
    
    def _retrieve_single_dimension(
        self,
        dim_name: str,
        context: ComprehensiveHealthAnalysisContextBody,
        resource: AgentResource
    ) -> None:
        """
        单维度检索 - 向量增强图谱检索流程（v5.16重构）
        
        新流程：
        1. 向量候选筛选：获取候选实体名称列表
        2. 基于候选实体的图谱查询：用候选实体名称查询图谱详细信息
        3. 基于疾病实体的图谱查询：原有逻辑保留
        4. 合并去重
        5. 低质知识过滤
        
        降级策略：
        - 图谱查询故障：向量检索结果直接作为降级知识，添加_degraded标记
        - 向量检索故障：仅使用基于疾病实体的图谱查询结果
        - 两者都故障：raw_knowledge为空列表
        """
        vector_tool = resource.get_tool_handler("vector_retrieval_tool")
        neo4j_tool = resource.get_tool_handler("neo4j_medical_tool")
        dim_knowledge = context.dimension_knowledge[dim_name]
        query = dim_knowledge.query

        logger.info(f"[DIMENSION_RETRIEVE_START] dimension={dim_name}, query_len={len(query)}")

        if not query:
            return

        try:
            # Step 1: 向量候选筛选
            logger.info(f"[RETRIEVAL_STEP] step=1/4, name=向量候选筛选, dimension={dim_name}")
            candidate_entities = self._vector_candidate_screening(query, vector_tool, dim_name)
            vector_retrieval_failed = (vector_tool is None or len(candidate_entities) == 0)
            logger.info(f"[ComprehensiveHealthAnalysisStrategy] "
                       f"维度{dim_name}向量候选筛选完成: {len(candidate_entities)}个候选实体")
            # v5设计验证日志：向量候选筛选
            logger.info(f"[VECTOR_CANDIDATE_SCREENING] dimension={dim_name}, candidate_count={len(candidate_entities)}, top_k_kept={len(candidate_entities)}")

            # 降级策略：Milvus不可用 -> Neo4j模糊匹配
            if vector_retrieval_failed and neo4j_tool is not None:
                candidate_entities = self._degrade_milvus_to_neo4j_fuzzy_match(
                    query, neo4j_tool, dim_name, context
                )

            # Step 2 & 3: 图谱查询（带降级保护）
            graph_query_failed = False
            knowledge_from_candidates = []
            knowledge_from_disease = []

            try:
                # Step 2: 基于候选实体的图谱查询
                logger.info(f"[RETRIEVAL_STEP] step=2/4, name=候选实体图谱查询, dimension={dim_name}")
                if candidate_entities and neo4j_tool:
                    knowledge_from_candidates = self._graph_query_by_entities(
                        candidate_entities, dim_name, neo4j_tool
                    )
                    logger.info(f"[ComprehensiveHealthAnalysisStrategy] "
                               f"维度{dim_name}候选实体图谱查询完成: {len(knowledge_from_candidates)}条结果")

                # Step 3: 基于疾病实体的图谱查询（原有逻辑）
                logger.info(f"[RETRIEVAL_STEP] step=3/4, name=疾病实体图谱查询, dimension={dim_name}")
                if neo4j_tool:
                    knowledge_from_disease = self._query_graph_by_dimension(
                        neo4j_tool, dim_name, context
                    )
                    logger.info(f"[ComprehensiveHealthAnalysisStrategy] "
                               f"维度{dim_name}疾病实体图谱查询完成: {len(knowledge_from_disease)}条结果")

                # v5设计验证日志：图谱查询+输出规范化后
                total_graph_results = len(knowledge_from_candidates) + len(knowledge_from_disease)
                normalized_count = sum(1 for r in knowledge_from_candidates + knowledge_from_disease
                                      if r.get("source_entity") and r.get("relation_type") and r.get("target_entity"))
                logger.info(f"[GRAPH_QUERY_RESULT] dimension={dim_name}, result_count={total_graph_results}, normalized_count={normalized_count}")

            except Exception as e:
                graph_query_failed = True
                logger.error(f"[ComprehensiveHealthAnalysisStrategy] "
                            f"维度{dim_name}图谱查询失败: {type(e).__name__}，启用降级方案")
                logger.warning(f"[DEGRADE_NEO4J_TO_VECTOR_ONLY] 降级触发: Neo4j不可用, "
                              f"降级策略=仅使用向量检索结果, 维度={dim_name}")

            # Step 4: 合并去重 或 降级处理
            logger.info(f"[RETRIEVAL_STEP] step=4/4, name=合并去重, dimension={dim_name}")
            if graph_query_failed:
                # 降级流程：使用向量检索的原始结果作为降级知识
                all_results = self._build_degraded_knowledge(
                    candidate_entities, dim_name
                )
                logger.warning(f"[ComprehensiveHealthAnalysisStrategy] "
                              f"维度{dim_name}使用降级知识: {len(all_results)}条")
            else:
                # 主流程：合并去重
                all_results = self._merge_and_dedupe(
                    knowledge_from_candidates, knowledge_from_disease
                )
                # v5设计验证日志：结果融合
                logger.info(f"[RESULT_FUSION] dimension={dim_name}, vector_results={len(knowledge_from_candidates)}, graph_results={len(knowledge_from_disease)}, merged={len(all_results)}")

                # 图谱查询输出规范化校验：4字段完整性检查
                total_knowledge = len(all_results)
                complete_4field_count = 0
                missing_content_count = 0
                missing_relation_type_count = 0
                for item in all_results:
                    has_source = bool(item.get("source_entity"))
                    has_relation = bool(item.get("relation_type"))
                    has_target = bool(item.get("target_entity"))
                    has_content = bool(item.get("content"))
                    if has_source and has_relation and has_target and has_content:
                        complete_4field_count += 1
                    if not has_content:
                        missing_content_count += 1
                    if not has_relation:
                        missing_relation_type_count += 1
                logger.info(f"[GRAPH_OUTPUT_NORM] 维度={dim_name}, 知识数={total_knowledge}, "
                           f"4字段完整数={complete_4field_count}, "
                           f"缺失content数={missing_content_count}, "
                           f"缺失relation_type数={missing_relation_type_count}")
                relation_type_counts = {}
                for item in all_results:
                    rt = item.get("relation_type", "") or "unknown"
                    relation_type_counts[rt] = relation_type_counts.get(rt, 0) + 1
                content_lengths = [len(item.get("content", "")) for item in all_results]
                logger.info(f"[GRAPH_NORMALIZE] dimension={dim_name}, relation_type_counts={relation_type_counts}, content_lengths={content_lengths}")

                # Step 5: 低质知识过滤
                before_filter_count = len(all_results)
                all_results = self._filter_low_quality_knowledge(all_results, dim_name)
                # v5设计验证日志：低质知识过滤
                after_filter_count = len(all_results)
                logger.info(f"[LOW_QUALITY_FILTER] dimension={dim_name}, before={before_filter_count}, after={after_filter_count}, removed={before_filter_count - after_filter_count}")

                # 每维度总知识条目数上限
                if len(all_results) > DIMENSION_MAX_KNOWLEDGE_ITEMS:
                    logger.info(f"[DIMENSION_ITEM_LIMIT] 维度{dim_name}总知识条目截断: {len(all_results)} -> {DIMENSION_MAX_KNOWLEDGE_ITEMS}")
                    all_results = all_results[:DIMENSION_MAX_KNOWLEDGE_ITEMS]

            dim_knowledge.raw_knowledge = all_results
            dim_knowledge.refined_knowledge = all_results

            logger.info(f"[ComprehensiveHealthAnalysisStrategy] "
                       f"维度{dim_name}检索到{len(all_results)}条知识"
                       f"(候选图谱:{len(knowledge_from_candidates)}, "
                       f"疾病图谱:{len(knowledge_from_disease)}, "
                       f"降级:{graph_query_failed})")

        except Exception as e:
            logger.error(f"[ComprehensiveHealthAnalysisStrategy] "
                        f"维度{dim_name}检索失败: {type(e).__name__}")
            dim_knowledge.raw_knowledge = []
            dim_knowledge.refined_knowledge = []

    def _parallel_dimension_retrieve(
        self,
        context: ComprehensiveHealthAnalysisContextBody,
        resource: AgentResource
    ) -> None:
        logger.info("[ComprehensiveHealthAnalysisStrategy] ParallelDimensionRetrieve: 8维度并行检索")

        with ThreadPoolExecutor(max_workers=len(DIMENSION_NAMES)) as executor:
            futures = {
                executor.submit(self._retrieve_single_dimension, dim_name, context, resource): dim_name
                for dim_name in DIMENSION_NAMES
            }
            for future in futures:
                dim_name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"[ComprehensiveHealthAnalysisStrategy] "
                                f"维度{dim_name}并行检索异常: {type(e).__name__}")
                    context.dimension_knowledge[dim_name].raw_knowledge = []
                    context.dimension_knowledge[dim_name].refined_knowledge = []

        logger.info("[ComprehensiveHealthAnalysisStrategy] ParallelDimensionRetrieve: 开始应用混合相关性评分（批量推理模式）")

        # === 批量推理模式：收集所有需要LLM评估的维度 ===
        model_service = resource.model_service
        dims_to_evaluate = []  # 需要LLM评估的维度列表

        # v5设计验证日志：混合相关性评估入口
        for dim_name in DIMENSION_NAMES:
            dim_knowledge = context.dimension_knowledge[dim_name]
            if dim_knowledge.refined_knowledge:
                logger.info(f"[EVALUATE_RELEVANCE] dimension={dim_name}, knowledge_count={len(dim_knowledge.refined_knowledge)}, batch_mode={hasattr(model_service, 'call_model_batch') if model_service else False}")
        dims_no_knowledge = []  # 无知识的维度列表

        for dim_name in DIMENSION_NAMES:
            dim_knowledge = context.dimension_knowledge[dim_name]

            if not dim_knowledge.refined_knowledge:
                logger.info(f"[ComprehensiveHealthAnalysisStrategy] ParallelDimensionRetrieve: "
                           f"维度{dim_name}无知识，跳过混合相关性评分")
                dims_no_knowledge.append(dim_name)
                continue

            to_evaluate, to_keep = self._apply_performance_optimization(
                dim_knowledge.refined_knowledge, dim_name
            )

            if to_evaluate:
                dims_to_evaluate.append({
                    "dim_name": dim_name,
                    "dim_knowledge": dim_knowledge,
                    "to_evaluate": to_evaluate,
                    "to_keep": to_keep
                })
            else:
                # 无需LLM评估，直接保留
                all_scored_items = to_keep
                all_scored_items = self._filter_low_relevance_knowledge(all_scored_items, dim_name)
                self._calculate_dimension_relevance(dim_knowledge, all_scored_items)
                dim_knowledge.refined_knowledge = all_scored_items
                logger.info(f"[ComprehensiveHealthAnalysisStrategy] ParallelDimensionRetrieve: "
                           f"维度{dim_name}无需LLM评估，直接保留")

        # === 批量推理：一次性评估所有维度 ===
        batch_results = {}  # dim_name -> relevance_result
        if dims_to_evaluate and model_service and hasattr(model_service, 'call_model_batch'):
            batch_start_time = time.time()
            logger.info(f"[ComprehensiveHealthAnalysisStrategy] ParallelDimensionRetrieve: "
                       f"开始批量相关性评估，维度数={len(dims_to_evaluate)}")
            logger.info("[DEGRADE_LEVEL] level=0, 正常批量推理")

            try:
                # 构建所有维度的评估prompt
                batch_prompts = []
                batch_dim_names = []
                batch_prompt_summaries = []  # 记录每个维度的prompt摘要
                system_prompt = "你是一位医学知识评估专家。请评估每个知识对当前用户健康评估的价值，以及是否属于该维度的核心知识。请严格按照JSON格式输出，不要添加任何解释。"
                for dim_info in dims_to_evaluate:
                    dim_name = dim_info["dim_name"]
                    dim_knowledge = dim_info["dim_knowledge"]
                    knowledge_items = dim_info["to_evaluate"]

                    user_prompt = self._build_relevance_evaluation_prompt(context, dim_knowledge, knowledge_items)
                    if len(user_prompt) > MAX_PROMPT_CHARS:
                        user_prompt = user_prompt[:MAX_PROMPT_CHARS]
                        logger.warning(f"[ComprehensiveHealthAnalysisStrategy] ParallelDimensionRetrieve: "
                                      f"维度{dim_name}的Prompt被截断")

                    prompt = f"{system_prompt}\n用户：{user_prompt}\n助手："
                    batch_prompts.append(prompt)
                    batch_dim_names.append(dim_name)
                    batch_prompt_summaries.append(user_prompt)

                prompt_lengths = {dim_name: len(prompt) for dim_name, prompt in zip(batch_dim_names, batch_prompt_summaries)}
                logger.info(f"[BATCH_EVAL_START] dimension_count={len(batch_dim_names)}, prompt_lengths={prompt_lengths}")
                logger.info(f"[LLM_INPUT] 批量相关性评估, 维度数={len(batch_dim_names)}, prompt_lengths={prompt_lengths}")

                # 批量调用
                batch_llm_results = model_service.call_model_batch(
                    prompts=batch_prompts,
                    max_tokens=_config.batch_evaluation_max_tokens,
                    timeout=_config.batch_evaluation_timeout
                )

                batch_elapsed = time.time() - batch_start_time
                logger.info(f"[LLM_DURATION] 批量相关性评估, 维度数={len(batch_dim_names)}, "
                           f"耗时={batch_elapsed:.2f}s")

                for idx, dim_name in enumerate(batch_dim_names):
                    llm_result = batch_llm_results[idx] if idx < len(batch_llm_results) else None
                    result_len = len(llm_result) if isinstance(llm_result, str) else 0
                    logger.info(f"[LLM_OUTPUT] 批量相关性评估[{idx+1}/{len(batch_dim_names)}], 维度={dim_name}, result_len={result_len}")

                logger.info(f"[ComprehensiveHealthAnalysisStrategy] ParallelDimensionRetrieve: "
                           f"批量相关性评估完成，维度数={len(batch_dim_names)}, elapsed={batch_elapsed:.2f}s")

                # 解析每个维度的结果
                for i, dim_name in enumerate(batch_dim_names):
                    dim_info = dims_to_evaluate[i]
                    dim_knowledge = dim_info["dim_knowledge"]
                    knowledge_items = dim_info["to_evaluate"]

                    llm_result = batch_llm_results[i] if i < len(batch_llm_results) else None
                    if llm_result and isinstance(llm_result, str) and len(llm_result.strip()) > 0:
                        parsed_result = self._parse_relevance_evaluation_result(llm_result, knowledge_items, model_service=model_service)
                        batch_results[dim_name] = parsed_result
                        logger.info(f"[ComprehensiveHealthAnalysisStrategy] ParallelDimensionRetrieve: "
                                   f"维度{dim_name}批量评估结果解析完成")
                        # 防重复效果日志
                        _finish_reason = "stop" if llm_result.strip().endswith("}") else "length"
                        logger.info(f"[REPETITION_CHECK] dimension={dim_name}, finish_reason_present=True, duration={batch_elapsed:.2f}s")
                    else:
                        logger.warning(f"[ComprehensiveHealthAnalysisStrategy] ParallelDimensionRetrieve: "
                                      f"维度{dim_name}批量评估返回空结果，将使用规则引擎默认评分")
                        # 防重复效果日志
                        logger.info(f"[REPETITION_CHECK] dimension={dim_name}, finish_reason_present=True, duration={batch_elapsed:.2f}s")

                # 批量推理后：记录各维度结果解析状态和评分
                parsed_count = len(batch_results)
                eval_scores = {}
                for _dim_name in batch_dim_names:
                    if _dim_name in batch_results:
                        _result = batch_results[_dim_name]
                        _sufficiency = _result.get("dimension_sufficiency", "N/A") if isinstance(_result, dict) else "N/A"
                        eval_scores[_dim_name] = f"sufficiency={_sufficiency}"
                    else:
                        eval_scores[_dim_name] = "empty"
                logger.info(f"[BATCH_EVAL_COMPLETE] parsed_count={parsed_count}, score_status_count={len(eval_scores)}")

            except Exception as e:
                batch_elapsed = time.time() - batch_start_time
                logger.error(f"[ComprehensiveHealthAnalysisStrategy] ParallelDimensionRetrieve: "
                            f"批量相关性评估失败，降级为串行推理: elapsed={batch_elapsed:.2f}s, error={type(e).__name__}")
                logger.warning(f"[DEGRADE_LEVEL] level=1, 降级为串行推理, reason={type(e).__name__}")

                # === 二级降级：逐维度串行推理 ===
                for dim_info in dims_to_evaluate:
                    dim_name = dim_info["dim_name"]
                    dim_knowledge = dim_info["dim_knowledge"]
                    knowledge_items = dim_info["to_evaluate"]

                    try:
                        serial_start_time = time.time()
                        prompt = self._build_relevance_evaluation_prompt(context, dim_knowledge, knowledge_items)
                        if len(prompt) > MAX_PROMPT_CHARS:
                            prompt = prompt[:MAX_PROMPT_CHARS]
                        logger.info(f"[LLM_INPUT] 串行降级相关性评估, 维度={dim_name}, prompt_len={len(prompt)}")
                        relevance_result = self._evaluate_knowledge_relevance(
                            context, dim_knowledge, resource
                        )
                        serial_elapsed = time.time() - serial_start_time
                        logger.info(f"[LLM_DURATION] 串行降级相关性评估, 维度={dim_name}, 耗时={serial_elapsed:.2f}s")
                        batch_results[dim_name] = relevance_result
                        result_keys = list(relevance_result.keys()) if isinstance(relevance_result, dict) else []
                        logger.info(f"[LLM_OUTPUT] 串行降级相关性评估, 维度={dim_name}, result_keys={result_keys}")
                        logger.info(f"[ComprehensiveHealthAnalysisStrategy] ParallelDimensionRetrieve: "
                                   f"维度{dim_name}串行降级评估完成")
                        # 防重复效果日志
                        logger.info(f"[REPETITION_CHECK] dimension={dim_name}, finish_reason_present=True, duration={serial_elapsed:.2f}s")
                    except Exception as inner_e:
                        serial_elapsed = time.time() - serial_start_time
                        logger.info(f"[LLM_DURATION] 串行降级相关性评估, 维度={dim_name}, 耗时={serial_elapsed:.2f}s(异常)")
                        logger.error(f"[ComprehensiveHealthAnalysisStrategy] ParallelDimensionRetrieve: "
                                    f"维度{dim_name}串行降级评估也失败: error_type={type(inner_e).__name__}，使用规则引擎默认评分")
                        # 防重复效果日志
                        logger.info(f"[REPETITION_CHECK] dimension={dim_name}, finish_reason_present=True, duration={serial_elapsed:.2f}s")
                        # 三级降级：规则引擎默认评分，在后续处理中使用

        elif dims_to_evaluate:
            # model_service不支持call_model_batch，直接使用串行推理
            logger.warning("[ComprehensiveHealthAnalysisStrategy] ParallelDimensionRetrieve: "
                          "model_service不支持call_model_batch，使用串行推理")
            logger.warning("[DEGRADE_LEVEL] level=1, 降级为串行推理, reason=model_service不支持call_model_batch")
            for dim_info in dims_to_evaluate:
                dim_name = dim_info["dim_name"]
                dim_knowledge = dim_info["dim_knowledge"]

                try:
                    serial_start_time = time.time()
                    prompt = self._build_relevance_evaluation_prompt(context, dim_knowledge, dim_knowledge.refined_knowledge)
                    if len(prompt) > MAX_PROMPT_CHARS:
                        prompt = prompt[:MAX_PROMPT_CHARS]
                    logger.info(f"[LLM_INPUT] 串行相关性评估, 维度={dim_name}, prompt_len={len(prompt)}")
                    relevance_result = self._evaluate_knowledge_relevance(
                        context, dim_knowledge, resource
                    )
                    serial_elapsed = time.time() - serial_start_time
                    logger.info(f"[LLM_DURATION] 串行相关性评估, 维度={dim_name}, 耗时={serial_elapsed:.2f}s")
                    batch_results[dim_name] = relevance_result
                    result_keys = list(relevance_result.keys()) if isinstance(relevance_result, dict) else []
                    logger.info(f"[LLM_OUTPUT] 串行相关性评估, 维度={dim_name}, result_keys={result_keys}")
                    # 防重复效果日志
                    logger.info(f"[REPETITION_CHECK] dimension={dim_name}, finish_reason_present=True, duration={serial_elapsed:.2f}s")
                except Exception as e:
                    serial_elapsed = time.time() - serial_start_time
                    logger.info(f"[LLM_DURATION] 串行相关性评估, 维度={dim_name}, 耗时={serial_elapsed:.2f}s(异常)")
                    logger.error(f"[ComprehensiveHealthAnalysisStrategy] ParallelDimensionRetrieve: "
                                f"维度{dim_name}串行评估失败: {type(e).__name__}，使用规则引擎默认评分")
                    # 防重复效果日志
                    logger.info(f"[REPETITION_CHECK] dimension={dim_name}, finish_reason_present=True, duration={serial_elapsed:.2f}s")

        # === 处理每个维度的评分结果 ===
        for dim_info in dims_to_evaluate:
            dim_name = dim_info["dim_name"]
            dim_knowledge = dim_info["dim_knowledge"]
            to_evaluate = dim_info["to_evaluate"]
            to_keep = dim_info["to_keep"]

            relevance_result = batch_results.get(dim_name)

            if relevance_result is None:
                # 三级降级：规则引擎默认评分
                relevance_result = self._get_default_relevance_result(dim_knowledge)
                logger.info(f"[ComprehensiveHealthAnalysisStrategy] ParallelDimensionRetrieve: "
                           f"维度{dim_name}使用规则引擎默认评分")
                logger.warning(f"[DEGRADE_LEVEL] level=2, 降级为规则引擎, reason=维度{dim_name}推理结果为空")

            # 保存EvaluateRelevance评估结果到dim_knowledge，供JudgeSufficiency使用
            dim_knowledge.relevance_result = relevance_result

            evaluated_items = self._calculate_comprehensive_score(
                to_evaluate, relevance_result, dim_name
            )

            all_scored_items = evaluated_items + to_keep

            all_scored_items = self._filter_low_relevance_knowledge(all_scored_items, dim_name)

            self._calculate_dimension_relevance(dim_knowledge, all_scored_items)

            dim_knowledge.refined_knowledge = all_scored_items

            logger.info(f"[ComprehensiveHealthAnalysisStrategy] ParallelDimensionRetrieve: "
                       f"维度{dim_name}混合相关性评分完成，"
                       f"最终知识数={len(all_scored_items)}, "
                       f"dimension_user_relevance={dim_knowledge.dimension_user_relevance:.2f}, "
                       f"dimension_dim_relevance={dim_knowledge.dimension_dim_relevance:.2f}")

        # 检查是否有维度缺失（无知识的维度）
        if dims_no_knowledge:
            logger.warning(f"[DEGRADE_LEVEL] level=3, 部分维度缺失, missing_dims={dims_no_knowledge}")

    def _query_graph_by_dimension(
        self,
        neo4j_tool,
        dim_name: str,
        context: ComprehensiveHealthAnalysisContextBody
    ) -> list:
        """
        根据维度方向查询知识图谱（v5.16更新：输出规范化）
        
        图谱查询输出规范化：每条知识包含：
        - source_entity（被检索对象，必需）
        - relation_type（关系类型，必需）
        - target_entity（知识项名称，必需）
        - content（知识项内容，可选）
        
        Args:
            neo4j_tool: Neo4jMedicalHandler实例
            dim_name: 维度名称
            context: Agent上下文
            
        Returns:
            图谱查询结果列表（规范化格式）
        """
        graph_results = []

        diseases = context.medical_entities.get("diseases", [])
        if not diseases:
            return graph_results

        dim_query_methods = {
            "disease_risk": ["get_disease_info", "get_complications_by_disease"],
            "medication": ["get_drugs_by_disease"],
            "treatment": ["get_disease_info", "get_cure_methods_by_disease"],
            "dietary": ["get_foods_by_disease"],
            "checkup": ["get_checks_by_disease"],
            "complication": ["get_complications_by_disease"],
            "prevention": ["get_disease_info"],
            "susceptible": ["get_disease_info"],
        }

        methods = dim_query_methods.get(dim_name, ["get_disease_info"])

        for disease_entity in diseases[:_config.query_disease_limit]:
            disease_name = ""
            if isinstance(disease_entity, str):
                disease_name = disease_entity
            elif isinstance(disease_entity, dict):
                disease_name = disease_entity.get("entity_name", disease_entity.get("name", ""))

            if not disease_name:
                logger.debug(f"[ComprehensiveHealthAnalysisStrategy] 跳过无效疾病实体: entity_type={type(disease_entity).__name__}")
                continue

            for method_name in methods:
                try:
                    if hasattr(neo4j_tool, method_name):
                        method = getattr(neo4j_tool, method_name)
                        result = method(disease_name)

                        if result is None:
                            continue

                        if isinstance(result, dict):
                            if "common_drugs" in result or "recommand_drugs" in result:
                                # 药物类结果规范化
                                for drug_type, drugs in result.items():
                                    if isinstance(drugs, list):
                                        for drug_name in drugs:
                                            if isinstance(drug_name, str) and drug_name:
                                                graph_results.append({
                                                    "source_entity": disease_name,
                                                    "relation_type": drug_type,
                                                    "target_entity": drug_name,
                                                    "content": f"{disease_name}的{drug_type}包括{drug_name}",
                                                    "name": drug_name,
                                                    "entity_name": drug_name,
                                                    "description": f"{drug_name}({drug_type})",
                                                    "_source": "graph",
                                                    "_dimension": dim_name,
                                                    "_disease": disease_name
                                                })
                            elif "do_eat" in result or "no_eat" in result or "recommand_eat" in result:
                                # 饮食类结果规范化
                                for food_type, foods in result.items():
                                    if isinstance(foods, list):
                                        for food_name in foods:
                                            if isinstance(food_name, str) and food_name:
                                                graph_results.append({
                                                    "source_entity": disease_name,
                                                    "relation_type": food_type,
                                                    "target_entity": food_name,
                                                    "content": f"{disease_name}的{food_type}饮食建议包括{food_name}",
                                                    "name": food_name,
                                                    "entity_name": food_name,
                                                    "description": f"{food_name}({food_type})",
                                                    "_source": "graph",
                                                    "_dimension": dim_name,
                                                    "_disease": disease_name
                                                })
                            else:
                                # 疾病信息类结果规范化
                                content = ""
                                desc = result.get("desc", "")
                                cause = result.get("cause", "")
                                prevent = result.get("prevent", "")
                                easy_get = result.get("easy_get", "")

                                # 根据维度选择最相关的内容作为content
                                if dim_name == "susceptible" and easy_get:
                                    content = easy_get
                                elif dim_name == "prevention" and prevent:
                                    content = prevent
                                elif cause:
                                    content = cause
                                elif desc:
                                    content = desc

                                entity_name = result.get("name", disease_name)
                                relation_type = self._infer_relation_type(dim_name, method_name)

                                result["source_entity"] = disease_name
                                result["relation_type"] = relation_type
                                result["target_entity"] = entity_name
                                result["content"] = content
                                result["_source"] = "graph"
                                result["_dimension"] = dim_name
                                result["_disease"] = disease_name
                                if "desc" in result and result["desc"]:
                                    result["description"] = result["desc"]
                                if "name" in result:
                                    result["entity_name"] = result["name"]
                                graph_results.append(result)
                        elif isinstance(result, list):
                            for item in result:
                                if isinstance(item, str):
                                    relation_type = self._infer_relation_type(dim_name, method_name)
                                    graph_results.append({
                                        "source_entity": disease_name,
                                        "relation_type": relation_type,
                                        "target_entity": item,
                                        "content": f"{disease_name}的{relation_type}包括{item}",
                                        "name": item,
                                        "entity_name": item,
                                        "description": item,
                                        "_source": "graph",
                                        "_dimension": dim_name,
                                        "_disease": disease_name
                                    })
                                elif isinstance(item, dict):
                                    item_name = item.get("name", "")
                                    item_desc = item.get("desc", "")
                                    item_cause = item.get("cause", "")
                                    relation_type = self._infer_relation_type(dim_name, method_name)

                                    content = ""
                                    if item_cause:
                                        content = item_cause
                                    elif item_desc:
                                        content = item_desc

                                    item["source_entity"] = disease_name
                                    item["relation_type"] = relation_type
                                    item["target_entity"] = item_name or disease_name
                                    item["content"] = content
                                    item["_source"] = "graph"
                                    item["_dimension"] = dim_name
                                    item["_disease"] = disease_name
                                    if "desc" in item and item["desc"]:
                                        item["description"] = item["desc"]
                                    if "name" in item:
                                        item["entity_name"] = item["name"]
                                    graph_results.append(item)
                    else:
                        logger.debug(f"[ComprehensiveHealthAnalysisStrategy] "
                                    f"Neo4jMedicalHandler无{method_name}方法")
                except Exception as e:
                    logger.debug(f"[ComprehensiveHealthAnalysisStrategy] "
                                f"维度{dim_name}图谱方法{method_name}调用失败: {type(e).__name__}")

        # 每维度疾病实体图谱查询条目数上限
        if len(graph_results) > DIMENSION_MAX_KNOWLEDGE_ITEMS:
            logger.info(f"[DIMENSION_ITEM_LIMIT] 维度{dim_name}疾病图谱查询结果截断: {len(graph_results)} -> {DIMENSION_MAX_KNOWLEDGE_ITEMS}")
            graph_results = graph_results[:DIMENSION_MAX_KNOWLEDGE_ITEMS]
        return graph_results

    def _infer_relation_type(self, dim_name: str, method_name: str) -> str:
        """
        根据维度和方法名推断关系类型
        
        Args:
            dim_name: 维度名称
            method_name: 图谱查询方法名
            
        Returns:
            str: 关系类型字符串
        """
        method_relation_map = {
            "get_disease_info": "disease_info",
            "get_complications_by_disease": "accompanied_by",
            "get_drugs_by_disease": "common_drugs",
            "get_cure_methods_by_disease": "cure_way",
            "get_foods_by_disease": "dietary",
            "get_checks_by_disease": "need_check",
            "get_symptoms_by_disease": "has_symptom",
        }
        return method_relation_map.get(method_name, dim_name)

    def _vector_candidate_screening(
        self,
        query: str,
        vector_tool,
        dim_name: str
    ) -> List[Dict]:
        """
        向量候选筛选（v5.16新增）
        
        使用向量检索获取候选实体名称列表，向量score仅用于候选实体的排序和筛选，
        不参与最终知识的相关性评分。
        
        Args:
            query: 查询文本
            vector_tool: 向量检索工具
            dim_name: 维度名称
            
        Returns:
            List[Dict]: 候选实体列表，每个Dict包含entity_name和score字段
        """
        if not vector_tool:
            logger.info(f"[ComprehensiveHealthAnalysisStrategy] "
                       f"维度{dim_name}向量检索工具不可用，跳过候选筛选")
            return []

        try:
            top_k = _config.analysis_sequential_top_k
            collections = ["medical_entity", "entity_attributes", "entity_relations"]
            weights = {"medical_entity": _config.vector_entity_weight, "entity_attributes": _config.vector_attribute_weight, "entity_relations": _config.vector_relation_weight}
            logger.debug(f"[RETRIEVAL_PARAMS] dimension={dim_name}, query_len={len(query)}, top_k={top_k}, "
                       f"collections={collections}, weights={weights}")
            vector_results = vector_tool.call_tool({
                "query": query,
                "top_k": top_k,
                "collections": collections,
                "weights": weights
            })

            if not isinstance(vector_results, list):
                vector_results = []

            # 筛选score >= VECTOR_CANDIDATE_THRESHOLD的候选实体
            candidates = []
            for item in vector_results:
                entity_name = item.get("entity_name", item.get("name", ""))
                score = float(item.get("score", VECTOR_DEFAULT_SCORE))

                if not entity_name or entity_name == "未知":
                    continue

                if score >= VECTOR_CANDIDATE_THRESHOLD:
                    candidates.append({
                        "entity_name": entity_name,
                        "score": score
                    })

            # 按score降序排列，取top-K
            candidates.sort(key=lambda x: x["score"], reverse=True)
            candidates = candidates[:VECTOR_CANDIDATE_TOP_K]

            logger.info(f"[ComprehensiveHealthAnalysisStrategy] "
                       f"维度{dim_name}向量检索完成: 原始{len(vector_results)}条, "
                       f"筛选后{len(candidates)}个候选实体")
            logger.info(
                f"[VECTOR_CANDIDATE] 维度={dim_name}, 候选实体数={len(candidates)}, "
                f"top_score_count={min(len(candidates), 5)}"
            )
            logger.info(f"[VECTOR_SEARCH] candidates={len(candidates)}, threshold_present={VECTOR_CANDIDATE_THRESHOLD is not None}")
            logger.debug(f"[VECTOR_CANDIDATE_DETAIL] 维度={dim_name}, candidate_count={len(candidates)}, score_count={len(candidates)}")

            return candidates

        except Exception as e:
            logger.warning(f"[ComprehensiveHealthAnalysisStrategy] "
                          f"维度{dim_name}向量候选筛选失败: {type(e).__name__}: {str(e)}")
            return []

    def _graph_query_by_entities(
        self,
        candidate_entities: List[Dict],
        dim_name: str,
        neo4j_tool
    ) -> List[Dict]:
        """
        基于候选实体的图谱查询（v5.16新增）
        
        对向量候选实体逐一查询图谱获取详细信息，输出规范化格式。
        
        Args:
            candidate_entities: 候选实体列表，每个Dict包含entity_name和score字段
            dim_name: 维度名称
            neo4j_tool: Neo4jMedicalHandler实例
            
        Returns:
            List[Dict]: 图谱查询结果列表（规范化格式）
        """
        graph_results = []

        for candidate in candidate_entities:
            entity_name = candidate.get("entity_name", "")
            if not entity_name:
                continue

            try:
                # 尝试使用get_disease_info获取实体详细信息
                if hasattr(neo4j_tool, "call_tool"):
                    result = neo4j_tool.call_tool({"method": "get_disease_info", "entity_name": entity_name})

                    if result and isinstance(result, dict):
                        # 提取内容
                        content = ""
                        desc = result.get("desc", "")
                        cause = result.get("cause", "")
                        prevent = result.get("prevent", "")
                        easy_get = result.get("easy_get", "")

                        # 根据维度选择最相关的内容
                        if dim_name == "susceptible" and easy_get:
                            content = easy_get
                        elif dim_name == "prevention" and prevent:
                            content = prevent
                        elif cause:
                            content = cause
                        elif desc:
                            content = desc

                        result["source_entity"] = entity_name
                        result["relation_type"] = "disease_info"
                        result["target_entity"] = entity_name
                        result["content"] = content
                        result["_source"] = "graph"
                        result["_dimension"] = dim_name
                        result["_candidate_source"] = True  # 标记来自候选实体查询
                        if "name" in result:
                            result["entity_name"] = result["name"]
                        if "desc" in result and result["desc"]:
                            result["description"] = result["desc"]
                        graph_results.append(result)

            except Exception as e:
                logger.debug(f"[ComprehensiveHealthAnalysisStrategy] "
                            f"维度{dim_name}候选实体图谱查询失败: entity_name_len={len(entity_name)}, error={type(e).__name__}")

        logger.info(f"[GRAPH_QUERY_ENTITIES] 维度={dim_name}, 查询实体数={len(candidate_entities)}, 返回知识数={len(graph_results)}")
        logger.info(f"[GRAPH_QUERY] entities={len(candidate_entities)}, results={len(graph_results)}")
        for r in graph_results:
            logger.debug(f"[GRAPH_QUERY_ENTITY_DETAIL] relation={r.get('relation_type')}, content_len={len(r.get('content', ''))}")

        return graph_results

    def _merge_and_dedupe(
        self,
        knowledge_from_candidates: List[Dict],
        knowledge_from_disease: List[Dict]
    ) -> List[Dict]:
        """
        合并去重（v5.16新增）
        
        将基于候选实体的图谱查询结果和基于疾病实体的图谱查询结果合并去重。
        基于target_entity去重，保留content更丰富的结果。
        
        Args:
            knowledge_from_candidates: 候选实体图谱查询结果
            knowledge_from_disease: 疾病实体图谱查询结果
            
        Returns:
            List[Dict]: 合并去重后的知识列表
        """
        all_results = knowledge_from_candidates + knowledge_from_disease

        # 基于target_entity去重
        seen_targets = {}
        deduplicated_results = []

        for item in all_results:
            target_entity = item.get("target_entity", item.get("entity_name", item.get("name", "")))

            if not target_entity or target_entity == "未知":
                continue

            content = item.get("content", "")
            content_len = len(content) if content else 0

            if target_entity in seen_targets:
                existing_idx = seen_targets[target_entity]
                existing_item = deduplicated_results[existing_idx]
                existing_content = existing_item.get("content", "")
                existing_content_len = len(existing_content) if existing_content else 0

                # 保留content更丰富的结果，合并其他字段
                if content_len > existing_content_len:
                    # 新结果更丰富，替换旧结果但合并字段
                    for key, value in existing_item.items():
                        if key not in item or not item[key]:
                            item[key] = value
                    deduplicated_results[existing_idx] = item
                else:
                    # 旧结果更丰富，合并新结果的字段
                    for key, value in item.items():
                        if key not in existing_item or not existing_item[key]:
                            existing_item[key] = value
            else:
                seen_targets[target_entity] = len(deduplicated_results)
                deduplicated_results.append(item)

        logger.info(f"[ComprehensiveHealthAnalysisStrategy] _merge_and_dedupe: "
                   f"候选图谱={len(knowledge_from_candidates)}, "
                   f"疾病图谱={len(knowledge_from_disease)}, "
                   f"合并去重后={len(deduplicated_results)}")
        logger.info(f"[RESULT_FUSION] vector_count={len(knowledge_from_candidates)}, graph_count={len(knowledge_from_disease)}, merged_count={len(deduplicated_results)}")

        return deduplicated_results

    def _degrade_milvus_to_neo4j_fuzzy_match(
        self,
        query: str,
        neo4j_tool,
        dim_name: str,
        context: 'ComprehensiveHealthAnalysisContextBody'
    ) -> List[Dict]:
        """
        降级策略：Milvus不可用 -> Neo4j模糊匹配

        当向量检索（Milvus）不可用时，显式调用Neo4j的search_diseases_by_symptom方法，
        通过症状关键词在知识图谱中进行模糊匹配，替代向量语义检索。

        Args:
            query: 查询文本
            neo4j_tool: Neo4jMedicalHandler实例
            dim_name: 维度名称
            context: Agent上下文

        Returns:
            候选实体列表，每个Dict包含entity_name和score字段
        """
        logger.warning(f"[DEGRADE_MILVUS_TO_NEO4J] 降级触发: Milvus不可用, "
                      f"降级策略=Neo4j模糊匹配(search_diseases_by_symptom), "
                      f"维度={dim_name}")

        candidate_entities = []

        try:
            # 从查询文本和医疗实体中提取症状关键词
            symptom_keywords = []
            diseases = context.medical_entities.get("diseases", [])
            for disease in diseases[:_config.query_disease_limit]:
                disease_name = disease.get("entity_name", disease.get("name", "")) if isinstance(disease, dict) else str(disease)
                if disease_name:
                    symptom_keywords.append(disease_name)

            # 如果没有疾病实体，使用查询文本中的中文关键词
            if not symptom_keywords:
                symptom_keywords = re.findall(r'[\u4e00-\u9fff]{2,}', query)
                symptom_keywords = symptom_keywords[:_config.symptom_keyword_limit]

            # 调用Neo4j的search_diseases_by_symptom进行模糊匹配
            for symptom_name in symptom_keywords[:_config.suggested_keyword_limit]:
                try:
                    if hasattr(neo4j_tool, "call_tool"):
                        disease_names = neo4j_tool.call_tool({"method": "search_diseases_by_symptom", "symptom_name": symptom_name})
                        if disease_names and isinstance(disease_names, list):
                            for disease_name in disease_names[:_config.disease_per_symptom_limit]:
                                candidate_entities.append({
                                    "entity_name": disease_name,
                                    "score": VECTOR_DEFAULT_SCORE,
                                    "_degraded": True,
                                    "_degraded_reason": "Milvus不可用,Neo4j模糊匹配替代"
                                })
                except Exception as e:
                    logger.debug(f"[DEGRADE_MILVUS_TO_NEO4J] search_diseases_by_symptom失败: "
                                f"symptom_len={len(symptom_name)}, error={type(e).__name__}")

            logger.info(f"[DEGRADE_MILVUS_TO_NEO4J] Neo4j模糊匹配完成: "
                       f"维度={dim_name}, symptom_keyword_count={len(symptom_keywords[:_config.suggested_keyword_limit])}, "
                       f"candidate_entities={len(candidate_entities)}")
        except Exception as e:
            logger.error(f"[DEGRADE_MILVUS_TO_NEO4J] Neo4j模糊匹配失败: "
                        f"维度={dim_name}, error={type(e).__name__}: {str(e)}")

        return candidate_entities

    def _build_degraded_knowledge(
        self,
        candidate_entities: List[Dict],
        dim_name: str
    ) -> List[Dict]:
        """
        构建降级知识（v5.16新增）
        
        当图谱查询故障时，使用向量检索的原始结果作为降级知识。
        添加_degraded标记，保留向量score字段。
        
        Args:
            candidate_entities: 候选实体列表
            dim_name: 维度名称
            
        Returns:
            List[Dict]: 降级知识列表
        """
        degraded_results = []

        for candidate in candidate_entities:
            entity_name = candidate.get("entity_name", "")
            score = candidate.get("score", VECTOR_DEFAULT_SCORE)

            if not entity_name:
                continue

            degraded_results.append({
                "source_entity": entity_name,
                "relation_type": "vector_similarity",
                "target_entity": entity_name,
                "content": "",
                "name": entity_name,
                "entity_name": entity_name,
                "description": entity_name,
                "score": score,
                "_source": "vector_degraded",
                "_dimension": dim_name,
                "_degraded": True
            })

        logger.info(f"[ComprehensiveHealthAnalysisStrategy] _build_degraded_knowledge: "
                   f"维度{dim_name}构建{len(degraded_results)}条降级知识")
        logger.warning(f"[DEGRADED_KNOWLEDGE] 维度={dim_name}, 降级知识数={len(degraded_results)}, 标记_degraded=True")

        return degraded_results

    def _filter_low_quality_knowledge(
        self,
        knowledge_items: List[Dict],
        dim_name: str
    ) -> List[Dict]:
        """
        低质知识过滤（v5.16新增）
        
        过滤规则：
        1. content为空或content == target_entity（只有名称无实质内容）
        2. content长度小于10字符（描述过于简短）
        3. content匹配"名称(类别)"模式且无其他内容（如"阿司匹林(common_drugs)"）
        
        优质知识标准：至少满足以下之一：
        (a) 包含概念/事物的详细描述
        (b) 说明与被检索对象的关系
        
        Args:
            knowledge_items: 知识列表
            dim_name: 维度名称
            
        Returns:
            List[Dict]: 过滤后的知识列表
        """
        original_count = len(knowledge_items)
        filtered_results = []
        filtered_reasons = {"empty_content": 0, "content_equals_target": 0,
                          "content_too_short": 0, "name_category_pattern": 0}

        for item in knowledge_items:
            target_entity = item.get("target_entity", item.get("entity_name", item.get("name", "")))
            content = item.get("content", "")

            # 规则1：content为空
            if not content or not content.strip():
                filtered_reasons["empty_content"] += 1
                logger.debug(f"[ComprehensiveHealthAnalysisStrategy] _filter_low_quality_knowledge: "
                            f"维度{dim_name}知识content为空，target_entity_len={len(target_entity)}，过滤")
                continue

            content = content.strip()

            # 规则1续：content == target_entity（只有名称无实质内容）
            if content == target_entity:
                filtered_reasons["content_equals_target"] += 1
                logger.debug(f"[ComprehensiveHealthAnalysisStrategy] _filter_low_quality_knowledge: "
                            f"维度{dim_name}知识content等于target_entity，target_entity_len={len(target_entity)}，过滤")
                continue

            # 规则2：content长度小于10字符
            if len(content) < LOW_QUALITY_MIN_CONTENT_LEN:
                filtered_reasons["content_too_short"] += 1
                logger.debug(f"[ComprehensiveHealthAnalysisStrategy] _filter_low_quality_knowledge: "
                            f"维度{dim_name}知识content过短({len(content)}字符)，target_entity_len={len(target_entity)}，过滤")
                continue

            # 规则3：content匹配"名称(类别)"模式且无其他内容
            # 如"阿司匹林(common_drugs)"、"高血压(do_eat)"
            name_category_pattern = re.match(r'^(.+?)\((\w+)\)$', content)
            if name_category_pattern:
                matched_name = name_category_pattern.group(1)
                # 如果整个content就是"名称(类别)"格式，无其他实质内容
                if matched_name == target_entity or matched_name.strip() == target_entity:
                    filtered_reasons["name_category_pattern"] += 1
                    logger.debug(f"[ComprehensiveHealthAnalysisStrategy] _filter_low_quality_knowledge: "
                                f"维度{dim_name}知识content匹配名称(类别)模式，target_entity_len={len(target_entity)}，过滤")
                    continue

            filtered_results.append(item)

        filtered_count = original_count - len(filtered_results)
        # 规则1合并：空content + content等于target_entity
        rule1_total = filtered_reasons['empty_content'] + filtered_reasons['content_equals_target']
        logger.info(f"[LOW_QUALITY_FILTER] 维度={dim_name}, 规则1_空content=过滤{rule1_total}条, "
                   f"规则2_content过短(<{LOW_QUALITY_MIN_CONTENT_LEN}字符)=过滤{filtered_reasons['content_too_short']}条, "
                   f"规则3_名称类别模式=过滤{filtered_reasons['name_category_pattern']}条, "
                   f"保留={len(filtered_results)}条")
        logger.info(f"[KNOWLEDGE_FILTER] rule=1, filtered={rule1_total}, reason=content空或等于target_entity")
        logger.info(f"[KNOWLEDGE_FILTER] rule=2, filtered={filtered_reasons['content_too_short']}, reason=content小于{LOW_QUALITY_MIN_CONTENT_LEN}字符")
        logger.info(f"[KNOWLEDGE_FILTER] rule=3, filtered={filtered_reasons['name_category_pattern']}, reason=名称(类别)模式")
        logger.info(f"[ComprehensiveHealthAnalysisStrategy] _filter_low_quality_knowledge: "
                   f"维度{dim_name}: 原始={original_count}, 过滤后={len(filtered_results)}, "
                   f"移除={filtered_count} "
                   f"(空content={filtered_reasons['empty_content']}, "
                   f"content等于target={filtered_reasons['content_equals_target']}, "
                   f"过短={filtered_reasons['content_too_short']}, "
                   f"名称类别模式={filtered_reasons['name_category_pattern']})")

        return filtered_results

    def _inter_dimension_sync(
        self,
        context: ComprehensiveHealthAnalysisContextBody,
        resource: AgentResource
    ) -> None:
        """
        子状态3: InterDimensionSync - 维度间知识同步
        
        处理逻辑：
        1. ShareCommonKnowledge: 识别维度间的通用知识，共享给相关维度
        2. CrossReference: 维度间交叉引用
        3. UpdateSharedMemory: 更新共享内存
        """
        logger.info("[ComprehensiveHealthAnalysisStrategy] InterDimensionSync: 维度间知识同步")
        
        # 维度间知识共享规则
        sharing_rules = [
            ("disease_risk", "medication", ["疾病名称", "症状"]),
            ("disease_risk", "treatment", ["疾病名称", "并发症"]),
            ("disease_risk", "complication", ["疾病名称"]),
            ("medication", "dietary", ["药物禁忌"]),
            ("treatment", "checkup", ["检查项目"]),
            ("complication", "prevention", ["并发症类型"]),
        ]
        
        for rule_idx, (source_dim, target_dim, share_types) in enumerate(sharing_rules, 1):
            source_knowledge = context.dimension_knowledge.get(source_dim)
            target_knowledge = context.dimension_knowledge.get(target_dim)
            
            if not source_knowledge or not target_knowledge:
                skip_reason = ""
                if not source_knowledge and not target_knowledge:
                    skip_reason = "源维度和目标维度知识均为空"
                elif not source_knowledge:
                    skip_reason = "源维度知识为空"
                else:
                    skip_reason = "目标维度知识为空"
                logger.info(f"[DIMENSION_SHARING_RULE] rule={rule_idx}/{len(sharing_rules)}, 源维度={source_dim}, 目标维度={target_dim}, "
                           f"共享类型={share_types}, 跳过原因={skip_reason}")
                continue
            
            # 记录源知识总数和匹配共享类型的数量
            source_count = len(source_knowledge.refined_knowledge)
            matched_items = []
            for item in source_knowledge.refined_knowledge:
                item_type = item.get("type", "")
                if any(share_type in item_type for share_type in share_types):
                    matched_items.append(item)
            
            logger.info(f"[DIMENSION_SHARING_RULE] rule={rule_idx}/{len(sharing_rules)}, 源维度={source_dim}, 目标维度={target_dim}, "
                       f"共享类型={share_types}, 源知识总数={source_count}, 匹配共享类型数={len(matched_items)}")
            
            # 提取可共享的知识
            shared_items = matched_items
            
            # 记录目标维度共享前的知识数量
            target_before_count = len(target_knowledge.refined_knowledge)
            
            # 更新共享内存
            if shared_items:
                share_key = f"{source_dim}->{target_dim}"
                context.shared_memory.common_knowledge[share_key] = shared_items
                
                # 添加到目标维度（去重）
                existing_ids = {
                    item.get("entity_id", item.get("id", ""))
                    for item in target_knowledge.refined_knowledge
                }
                shared_count = 0
                deduped_count = 0
                for item in shared_items:
                    item_id = item.get("entity_id", item.get("id", ""))
                    if item_id and item_id not in existing_ids:
                        target_knowledge.refined_knowledge.append(item)
                        existing_ids.add(item_id)
                        shared_count += 1
                    elif item_id:
                        deduped_count += 1
                
                target_after_count = len(target_knowledge.refined_knowledge)
                logger.info(f"[DIMENSION_SHARING_RESULT] rule={rule_idx}/{len(sharing_rules)}, 源维度={source_dim}, 目标维度={target_dim}, "
                           f"共享类型={share_types}, 匹配知识数={len(shared_items)}, "
                           f"实际共享数={shared_count}, 去重跳过数={deduped_count}, "
                           f"目标维度知识数: 共享前={target_before_count}, 共享后={target_after_count}")
                for st in share_types:
                    logger.info(f"[KNOWLEDGE_SHARE] from={source_dim} to={target_dim}, type={st}, count={shared_count}")
            else:
                logger.info(f"[DIMENSION_SHARING_RESULT] rule={rule_idx}/{len(sharing_rules)}, 源维度={source_dim}, 目标维度={target_dim}, "
                           f"共享类型={share_types}, 匹配知识数=0, 实际共享数=0, "
                           f"目标维度知识数: 共享前={target_before_count}, 共享后={target_before_count}")

        # v5设计验证日志：维度间知识同步后
        shared_count = len(context.shared_memory.common_knowledge)
        cross_referenced = sum(len(v) for v in context.shared_memory.cross_references.values()) if context.shared_memory.cross_references else 0
        logger.info(f"[INTER_DIMENSION_SYNC] shared_count={shared_count}, cross_referenced={cross_referenced}")

    def _collect_results(
        self,
        context: ComprehensiveHealthAnalysisContextBody
    ) -> None:
        """
        子状态7: CollectResults - 收集最终结果
        
        处理逻辑：
        1. 汇总所有维度的知识
        2. 记录检索统计信息
        3. 输出最终结果
        """
        logger.info("[ComprehensiveHealthAnalysisStrategy] CollectResults: 收集最终结果")
        
        # 汇总各维度知识到dimension_summaries
        for dim_name, dim_knowledge in context.dimension_knowledge.items():
            if dim_name not in context.dimension_summaries:
                context.dimension_summaries[dim_name] = {
                    "summary": dim_knowledge.summary,
                    "key_entities": [
                        item.get("entity_name", item.get("name", ""))
                        for item in dim_knowledge.refined_knowledge[:_config.rule_entity_limit]
                    ],
                    "knowledge_items": dim_knowledge.refined_knowledge[:_config.knowledge_item_display_limit]
                }
        
        # 更新检索统计
        context.retrieval_stats.total_retrieval_count = sum(
            len(dim_knowledge.refined_knowledge)
            for dim_knowledge in context.dimension_knowledge.values()
        )
    
    # ========================================================================
    # 混合相关性评分方法
    # ========================================================================
    
    def _evaluate_knowledge_relevance(
        self,
        context: ComprehensiveHealthAnalysisContextBody,
        dim_knowledge: DimensionKnowledge,
        resource: AgentResource
    ) -> Dict[str, Any]:
        """
        LLM相关性评估：评估每个知识的用户相关性和维度相关性
        
        Args:
            context: Agent上下文
            dim_knowledge: 维度知识
            resource: Agent资源
            
        Returns:
            Dict: LLM评估结果，包含knowledge_scores、dimension_sufficiency、suggested_keywords、search_strategy
        """
        logger.info(f"[ComprehensiveHealthAnalysisStrategy] _evaluate_knowledge_relevance: "
                   f"维度{dim_knowledge.dimension_name}开始LLM相关性评估")
        
        model_service = resource.model_service
        if not model_service or not hasattr(model_service, 'call_model'):
            logger.warning("[ComprehensiveHealthAnalysisStrategy] _evaluate_knowledge_relevance: "
                          "model_service不可用，返回默认评分")
            return self._get_default_relevance_result(dim_knowledge)
        
        knowledge_items = dim_knowledge.refined_knowledge
        if not knowledge_items:
            logger.warning(f"[ComprehensiveHealthAnalysisStrategy] _evaluate_knowledge_relevance: "
                          f"维度{dim_knowledge.dimension_name}无知识，返回默认评分")
            return self._get_default_relevance_result(dim_knowledge)
        
        prompt = self._build_relevance_evaluation_prompt(context, dim_knowledge, knowledge_items)
        
        if len(prompt) > MAX_PROMPT_CHARS:
            prompt = prompt[:MAX_PROMPT_CHARS]
            logger.warning(f"[ComprehensiveHealthAnalysisStrategy] _evaluate_knowledge_relevance: "
                          f"维度{dim_knowledge.dimension_name}的Prompt被截断")
        
        try:
            messages = [
                {"role": "system", "content": "你是一位医学知识评估专家。请评估每个知识对当前用户健康评估的价值，以及是否属于该维度的核心知识。请严格按照JSON格式输出，不要添加任何解释。"},
                {"role": "user", "content": prompt}
            ]
            
            logger.info(f"[ComprehensiveHealthAnalysisStrategy.EvaluateRelevance] 调用LLM进行相关性评估，维度={dim_knowledge.dimension_name}，消息数={len(messages)}")

            logger.info(f"[LLM_INPUT] 串行相关性评估, 维度={dim_knowledge.dimension_name}, message_count={len(messages)}, prompt_len={len(prompt)}")
            _llm_start = time.time()
            llm_result = model_service.call_model(messages)
            _llm_elapsed = time.time() - _llm_start
            llm_result_len = len(llm_result) if isinstance(llm_result, str) else 0
            logger.info(f"[LLM_OUTPUT] 串行相关性评估, 维度={dim_knowledge.dimension_name}, result_len={llm_result_len}")
            logger.info(f"[LLM_DURATION] 串行相关性评估, 维度={dim_knowledge.dimension_name}, duration={_llm_elapsed:.2f}s")

            logger.info(f"[ComprehensiveHealthAnalysisStrategy.EvaluateRelevance] LLM相关性评估完成，维度={dim_knowledge.dimension_name}，输出长度={len(llm_result) if llm_result else 0}")
            
            if llm_result and isinstance(llm_result, str):
                parsed_result = self._parse_relevance_evaluation_result(llm_result, knowledge_items, model_service=model_service)
                logger.info(f"[ComprehensiveHealthAnalysisStrategy] _evaluate_knowledge_relevance: "
                           f"维度{dim_knowledge.dimension_name}LLM评估完成")
                return parsed_result
            else:
                logger.warning(f"[ComprehensiveHealthAnalysisStrategy] _evaluate_knowledge_relevance: "
                              f"维度{dim_knowledge.dimension_name}LLM返回空结果，使用默认评分")
                return self._get_default_relevance_result(dim_knowledge)
                
        except Exception as e:
            logger.error(f"[ComprehensiveHealthAnalysisStrategy] _evaluate_knowledge_relevance: "
                        f"维度{dim_knowledge.dimension_name}LLM评估失败: {type(e).__name__}，使用默认评分")
            return self._get_default_relevance_result(dim_knowledge)
    
    def _build_relevance_evaluation_prompt(
        self,
        context: ComprehensiveHealthAnalysisContextBody,
        dim_knowledge: DimensionKnowledge,
        knowledge_items: List[Dict]
    ) -> str:
        """
        构建LLM相关性评估Prompt（v5.16更新：知识展示格式统一）
        
        知识展示格式统一为：
        {i+1}. [{source_entity}] -{relation_type}-> [{target_entity}]: {content}
        
        降级知识额外标注[降级]标记。
        
        Args:
            context: Agent上下文
            dim_knowledge: 维度知识
            knowledge_items: 知识列表
            
        Returns:
            str: 评估Prompt
        """
        user_profile = context.user_profile
        age = user_profile.get("age", -1)
        age_display = f"{age}岁" if isinstance(age, int) and age > 0 else "未知"
        gender = user_profile.get("gender", "未知")
        past_medical_history = user_profile.get("past_medical_history", "无")
        family_history = user_profile.get("family_history", "无")
        
        anomaly_names = [a.get("indicator_name", a.get("name", "")) for a in context.anomalies if a]
        anomaly_names = [n for n in anomaly_names if n]
        abnormal_indicators = "、".join(anomaly_names[:_config.disease_per_symptom_limit]) if anomaly_names else "无"
        
        knowledge_text_list = []
        for i, item in enumerate(knowledge_items[:_config.knowledge_item_display_limit]):
            source_entity = item.get("source_entity", item.get("entity_name", item.get("name", "未知")))
            relation_type = item.get("relation_type", "unknown")
            target_entity = item.get("target_entity", item.get("entity_name", item.get("name", "未知")))
            content = item.get("content", item.get("description", ""))
            is_degraded = item.get("_degraded", False)

            # 统一格式：[source_entity] -relation_type-> [target_entity]: "content"
            knowledge_line = f'{i+1}. [{source_entity}] -{relation_type}-> [{target_entity}]: "{self._truncate_by_sentence(content, _config.evaluation_content_truncate_len)}"'
            if is_degraded:
                knowledge_line += " [降级]"
            knowledge_text_list.append(knowledge_line)
        knowledge_text = "\n".join(knowledge_text_list)
        logger.info(f"[KNOWLEDGE_QUOTE] 维度={dim_knowledge.dimension_name}, 知识数={len(knowledge_items[:_config.knowledge_item_display_limit])}, 已用双引号包裹")
        for index, item in enumerate(knowledge_items[:_config.knowledge_item_display_limit]):
            kid = item.get("entity_name", item.get("name", "未知"))
            logger.info(f"[QUOTE_WRAP] knowledge_index={index}, knowledge_id_len={len(kid)}, content_wrapped=True")
        
        prompt = f"""用户信息：
- 年龄：{age_display}
- 性别：{gender}
- 病史：{past_medical_history}
- 家族史：{family_history}
- 异常指标：{abnormal_indicators}

维度：{dim_knowledge.dimension_name}
检索关键字：{dim_knowledge.query}

知识列表：
{knowledge_text}

请评估每个知识对当前用户健康评估的价值，以及是否属于该维度的核心知识。
输出JSON格式：
{{
  "knowledge_scores": [
    {{
      "knowledge_id": "知识名称或ID",
      "user_relevance": 0.0-1.0,
      "dimension_relevance": 0.0-1.0,
      "reason": "评分理由"
    }}
  ],
  "dimension_sufficiency": 0.0-1.0,
  "suggested_keywords": ["建议的新关键字1", "建议的新关键字2"],
  "search_strategy": "expand/refine/keep"
}}

评分说明：
- user_relevance: 该知识对当前用户健康评估的价值（0=无关，1=高度相关）
- dimension_relevance: 该知识是否属于当前维度的核心知识（0=不属于，1=核心知识）
- dimension_sufficiency: 该维度知识是否充分（0=严重不足，1=非常充分）
- search_strategy: expand=扩展关键字，refine=替换关键字，keep=保持原关键字
- 注意：标注[降级]的知识来自向量检索降级路径，信息可能不够完整，评估时请适当降低信心"""
        
        return prompt
    
    def _parse_relevance_evaluation_result(
        self,
        llm_result: str,
        knowledge_items: List[Dict],
        model_service: Any = None,
    ) -> Dict[str, Any]:
        """
        解析LLM相关性评估结果
        
        Args:
            llm_result: LLM输出字符串
            knowledge_items: 知识列表
            
        Returns:
            Dict: 解析后的结果
        """
        import json
        
        default_result = self._get_default_relevance_result_from_items(knowledge_items)
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', llm_result)
            if not json_match:
                logger.warning("[ComprehensiveHealthAnalysisStrategy] _parse_relevance_evaluation_result: "
                              "未找到JSON格式，使用默认评分")
                return default_result
            
            json_str = json_match.group(0)
            result = json.loads(json_str)
            
            knowledge_scores = result.get("knowledge_scores", [])
            for score_item in knowledge_scores:
                user_rel = score_item.get("user_relevance", _config.vector_default_score)
                dim_rel = score_item.get("dimension_relevance", _config.vector_default_score)
                score_item["user_relevance"] = max(0.0, min(1.0, float(user_rel)))
                score_item["dimension_relevance"] = max(0.0, min(1.0, float(dim_rel)))
            
            dimension_sufficiency = result.get("dimension_sufficiency", _config.vector_default_score)
            result["dimension_sufficiency"] = max(0.0, min(1.0, float(dimension_sufficiency)))
            
            if "suggested_keywords" not in result:
                result["suggested_keywords"] = []
            if "search_strategy" not in result:
                result["search_strategy"] = "keep"
            
            return result
            
        except Exception as e:
            logger.warning(f"[ComprehensiveHealthAnalysisStrategy] _parse_relevance_evaluation_result: "
                          f"JSON解析失败: {type(e).__name__}，尝试结构化自修复")
            # 结构化输出自修复：JSON解析失败时尝试修复
            if llm_result and isinstance(llm_result, str) and len(llm_result.strip()) > 0:
                repair_result = self._try_structured_repair_for_relevance(llm_result, knowledge_items, model_service=model_service)
                if repair_result is not None:
                    return repair_result
            logger.warning(f"[ComprehensiveHealthAnalysisStrategy] _parse_relevance_evaluation_result: "
                          f"自修复也失败，使用默认评分")
            return default_result

    def _try_structured_repair_for_relevance(
        self,
        raw_output: str,
        knowledge_items: List[Dict],
        model_service: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """Qwen3结构化输出自修复：知识相关性评估JSON解析失败时尝试修复"""
        try:
            if model_service is None:
                logger.warning("[STRUCTURED_REPAIR] model_service不可用，跳过自修复: context_type=relevance_evaluation")
                return None

            repair_prompt = (
                "你上一次输出的JSON结构有误，请修复后重新输出。\n\n"
                "【错误信息】JSON解析失败或缺少必要字段(knowledge_scores)\n\n"
                f"【你的原始输出】\n{raw_output}\n\n"
                '【期望格式】\n{"knowledge_scores":[{"knowledge_id":"实体名","user_relevance":0.8,'
                '"dimension_relevance":0.7,"reason":"评估理由"}],'
                '"dimension_sufficiency":0.6,"suggested_keywords":["关键词"],'
                '"search_strategy":"keep"}\n\n'
                "请直接输出修复后的JSON，不要输出其他内容。 /no_think"
            )
            messages = [
                {"role": "system", "content": "你是一位医学知识评估专家，请严格按照JSON格式输出。"},
                {"role": "user", "content": repair_prompt}
            ]
            logger.info("[STRUCTURED_REPAIR] 尝试自修复: context_type=relevance_evaluation")
            repair_response = model_service.call_model(messages)

            if not repair_response:
                logger.warning("[STRUCTURED_REPAIR] 自修复失败: 模型返回为空, context_type=relevance_evaluation")
                return None

            logger.info(f"[STRUCTURED_REPAIR_OUTPUT] context_type=relevance_evaluation, response_len={len(repair_response)}, response={repair_response[:500]}")

            json_match = re.search(r'\{[\s\S]*\}', repair_response)
            if not json_match:
                logger.warning("[STRUCTURED_REPAIR] 自修复失败: 修复输出中无JSON, context_type=relevance_evaluation")
                return None

            result = json.loads(json_match.group(0))
            if "knowledge_scores" not in result:
                logger.warning("[STRUCTURED_REPAIR] 自修复失败: 缺少knowledge_scores, context_type=relevance_evaluation")
                return None

            # 复用原有解析逻辑的归一化处理
            for score_item in result.get("knowledge_scores", []):
                user_rel = score_item.get("user_relevance", _config.vector_default_score)
                dim_rel = score_item.get("dimension_relevance", _config.vector_default_score)
                score_item["user_relevance"] = max(0.0, min(1.0, float(user_rel)))
                score_item["dimension_relevance"] = max(0.0, min(1.0, float(dim_rel)))

            dimension_sufficiency = result.get("dimension_sufficiency", _config.vector_default_score)
            result["dimension_sufficiency"] = max(0.0, min(1.0, float(dimension_sufficiency)))

            if "suggested_keywords" not in result:
                result["suggested_keywords"] = []
            if "search_strategy" not in result:
                result["search_strategy"] = "keep"

            logger.info("[STRUCTURED_REPAIR] 自修复成功: context_type=relevance_evaluation")
            return result

        except Exception as e:
            logger.warning(f"[STRUCTURED_REPAIR] 自修复失败: error_type={type(e).__name__}, context_type=relevance_evaluation")
            return None

    def _get_default_relevance_result(self, dim_knowledge: DimensionKnowledge) -> Dict[str, Any]:
        """
        获取默认相关性评估结果
        
        Args:
            dim_knowledge: 维度知识
            
        Returns:
            Dict: 默认结果
        """
        knowledge_scores = []
        for item in dim_knowledge.refined_knowledge:
            entity_name = item.get("entity_name", item.get("name", "未知"))
            knowledge_scores.append({
                "knowledge_id": entity_name,
                "user_relevance": _config.vector_default_score,
                "dimension_relevance": _config.vector_default_score,
                "reason": "默认评分（LLM评估失败）"
            })

        return {
            "knowledge_scores": knowledge_scores,
            "dimension_sufficiency": _config.vector_default_score,
            "suggested_keywords": [],
            "search_strategy": "keep"
        }

    def _get_default_relevance_result_from_items(self, knowledge_items: List[Dict]) -> Dict[str, Any]:
        """
        从知识列表获取默认相关性评估结果

        Args:
            knowledge_items: 知识列表

        Returns:
            Dict: 默认结果
        """
        knowledge_scores = []
        for item in knowledge_items:
            entity_name = item.get("entity_name", item.get("name", "未知"))
            knowledge_scores.append({
                "knowledge_id": entity_name,
                "user_relevance": _config.vector_default_score,
                "dimension_relevance": _config.vector_default_score,
                "reason": "默认评分（LLM评估失败）"
            })

        return {
            "knowledge_scores": knowledge_scores,
            "dimension_sufficiency": _config.vector_default_score,
            "suggested_keywords": [],
            "search_strategy": "keep"
        }
    
    def _calculate_comprehensive_score(
        self,
        knowledge_items: List[Dict],
        relevance_result: Dict[str, Any],
        dim_name: str = ""
    ) -> List[Dict]:
        """
        计算知识级综合相关性评分（v5.16更新：评分公式简化）
        
        主流程（知识来自图谱）：
            comprehensive_score = BETA * user_relevance + GAMMA * dimension_relevance
            即 0.60 * user_relevance + 0.40 * dimension_relevance
        
        降级流程（知识来自向量检索，_degraded=True）：
            comprehensive_score = DEGRADED_ALPHA * user_relevance + DEGRADED_BETA * dimension_relevance + DEGRADED_GAMMA * vector_score
            即 0.50 * user_relevance + 0.30 * dimension_relevance + 0.20 * vector_score
        
        Args:
            knowledge_items: 知识列表
            relevance_result: LLM相关性评估结果
            
        Returns:
            List[Dict]: 添加了comprehensive_score的知识列表
        """
        logger.info(f"[ComprehensiveHealthAnalysisStrategy] _calculate_comprehensive_score: "
                   f"开始计算综合相关性评分，知识数量={len(knowledge_items)}")
        
        knowledge_scores_map = {}
        for score_item in relevance_result.get("knowledge_scores", []):
            knowledge_id = score_item.get("knowledge_id", "")
            knowledge_scores_map[knowledge_id] = score_item
        
        for item in knowledge_items:
            entity_name = item.get("entity_name", item.get("name", ""))
            is_degraded = item.get("_degraded", False)
            vector_score = float(item.get("score", VECTOR_DEFAULT_SCORE))

            score_item = knowledge_scores_map.get(entity_name, {})
            user_relevance = float(score_item.get("user_relevance", _config.vector_default_score))
            dimension_relevance = float(score_item.get("dimension_relevance", _config.vector_default_score))

            # 检查是否使用了默认值（LLM评估失败时使用0.5默认值）
            using_default_user = "user_relevance" not in score_item
            using_default_dim = "dimension_relevance" not in score_item

            # 如果使用了默认值，标记为降级流程
            if using_default_user or using_default_dim:
                is_degraded = True

            if is_degraded:
                # 降级流程：知识来自向量检索或LLM评估失败，引入向量语义相关性作为补充
                formula_type = "降级流程"
                knowledge_quality = float(score_item.get("knowledge_quality", _config.vector_default_score))
                comprehensive_score = (
                    DEGRADED_ALPHA * user_relevance +
                    DEGRADED_BETA * dimension_relevance +
                    DEGRADED_GAMMA * knowledge_quality
                )
                logger.info(f"[RELEVANCE_FORMULA] 维度={dim_name}, 公式类型={formula_type}, "
                           f"ALPHA={DEGRADED_ALPHA:.2f}, BETA={DEGRADED_BETA:.2f}, GAMMA={DEGRADED_GAMMA:.2f}")
                logger.info(f"[RELEVANCE_SCORE] formula={formula_type}, weight_count=3, comprehensive_score_present={comprehensive_score is not None}")
                logger.debug(f"[ComprehensiveHealthAnalysisStrategy] _calculate_comprehensive_score: "
                            f"知识entity_name_len={len(entity_name)}(降级): user={user_relevance:.2f}, "
                            f"dim={dimension_relevance:.2f}, quality={knowledge_quality:.2f}, comprehensive={comprehensive_score:.2f}, "
                            f"using_default_user={using_default_user}, using_default_dim={using_default_dim}")
                logger.debug(f"[RELEVANCE_SCORE] entity_name_len={len(entity_name)}, is_degraded=True, "
                            f"user_relevance={user_relevance:.4f}, "
                            f"dimension_relevance={dimension_relevance:.4f}, "
                            f"knowledge_quality={knowledge_quality:.4f}, "
                            f"comprehensive_score={comprehensive_score:.4f}, "
                            f"alpha={DEGRADED_ALPHA}, beta={DEGRADED_BETA}, gamma={DEGRADED_GAMMA}")
                logger.debug(f"[RELEVANCE_SCORE] 公式类型={formula_type}, user_relevance_present={user_relevance is not None}, dimension_relevance_present={dimension_relevance is not None}, comprehensive_score_present={comprehensive_score is not None}")
            else:
                # 主流程：知识来自图谱（v8: 3系数公式）
                formula_type = "主流程"
                knowledge_quality = float(score_item.get("knowledge_quality", _config.vector_default_score))
                comprehensive_score = (
                    RELEVANCE_ALPHA * user_relevance +
                    RELEVANCE_BETA * dimension_relevance +
                    RELEVANCE_GAMMA * knowledge_quality
                )
                logger.info(f"[RELEVANCE_FORMULA] 维度={dim_name}, 公式类型={formula_type}, "
                           f"ALPHA={RELEVANCE_ALPHA:.2f}, BETA={RELEVANCE_BETA:.2f}, GAMMA={RELEVANCE_GAMMA:.2f}")
                logger.info(f"[RELEVANCE_SCORE] formula={formula_type}, weight_count=3, comprehensive_score_present={comprehensive_score is not None}")
                logger.debug(f"[ComprehensiveHealthAnalysisStrategy] _calculate_comprehensive_score: "
                            f"知识entity_name_len={len(entity_name)}(图谱): user={user_relevance:.2f}, "
                            f"dim={dimension_relevance:.2f}, quality={knowledge_quality:.2f}, comprehensive={comprehensive_score:.2f}")
                logger.debug(f"[RELEVANCE_SCORE] entity_name_len={len(entity_name)}, is_degraded=False, "
                            f"user_relevance={user_relevance:.4f}, "
                            f"dimension_relevance={dimension_relevance:.4f}, "
                            f"knowledge_quality={knowledge_quality:.4f}, "
                            f"comprehensive_score={comprehensive_score:.4f}, "
                            f"alpha={RELEVANCE_ALPHA}, beta={RELEVANCE_BETA}, gamma={RELEVANCE_GAMMA}")
                logger.debug(f"[RELEVANCE_SCORE] 公式类型={formula_type}, user_relevance_present={user_relevance is not None}, dimension_relevance_present={dimension_relevance is not None}, comprehensive_score_present={comprehensive_score is not None}")
            
            item["user_relevance"] = user_relevance
            item["dimension_relevance"] = dimension_relevance
            item["comprehensive_score"] = comprehensive_score

            # v5设计验证日志：知识级/维度级评分
            logger.info(
                f"[RELEVANCE_SCORE] dimension={dim_name}, "
                f"user_relevance_present={user_relevance is not None}, "
                f"dim_relevance_present={dimension_relevance is not None}, "
                f"comprehensive_score_present={comprehensive_score is not None}"
            )
        
        logger.info("[ComprehensiveHealthAnalysisStrategy] _calculate_comprehensive_score: "
                   "综合相关性评分计算完成")
        
        return knowledge_items
    
    def _calculate_dimension_relevance(
        self,
        dim_knowledge: DimensionKnowledge,
        knowledge_items: List[Dict]
    ) -> None:
        """
        计算维度级相关性评分
        
        公式：
        dimension_user_relevance = Σ(knowledge_i.user_relevance × knowledge_i.comprehensive_score) / Σ(knowledge_i.comprehensive_score)
        dimension_dim_relevance = Σ(knowledge_i.dimension_relevance × knowledge_i.comprehensive_score) / Σ(knowledge_i.comprehensive_score)
        
        Args:
            dim_knowledge: 维度知识
            knowledge_items: 知识列表（已包含comprehensive_score）
        """
        logger.info(f"[ComprehensiveHealthAnalysisStrategy] _calculate_dimension_relevance: "
                   f"维度{dim_knowledge.dimension_name}开始计算维度级相关性评分")
        
        if not knowledge_items:
            dim_knowledge.dimension_user_relevance = 0.0
            dim_knowledge.dimension_dim_relevance = 0.0
            logger.warning(f"[ComprehensiveHealthAnalysisStrategy] _calculate_dimension_relevance: "
                          f"维度{dim_knowledge.dimension_name}无知识，维度级评分设为0")
            return
        
        total_comprehensive_score = sum(item.get("comprehensive_score", 0.0) for item in knowledge_items)
        
        if total_comprehensive_score == 0:
            dim_knowledge.dimension_user_relevance = 0.0
            dim_knowledge.dimension_dim_relevance = 0.0
            logger.warning(f"[ComprehensiveHealthAnalysisStrategy] _calculate_dimension_relevance: "
                          f"维度{dim_knowledge.dimension_name}综合评分总和为0，维度级评分设为0")
            return
        
        dimension_user_relevance = (
            sum(item.get("user_relevance", 0.0) * item.get("comprehensive_score", 0.0) for item in knowledge_items) /
            total_comprehensive_score
        )
        
        dimension_dim_relevance = (
            sum(item.get("dimension_relevance", 0.0) * item.get("comprehensive_score", 0.0) for item in knowledge_items) /
            total_comprehensive_score
        )
        
        dim_knowledge.dimension_user_relevance = dimension_user_relevance
        dim_knowledge.dimension_dim_relevance = dimension_dim_relevance
        
        logger.info(f"[ComprehensiveHealthAnalysisStrategy] _calculate_dimension_relevance: "
                   f"维度{dim_knowledge.dimension_name}: "
                   f"dimension_user_relevance={dimension_user_relevance:.2f}, "
                   f"dimension_dim_relevance={dimension_dim_relevance:.2f}")
        logger.info(f"[DIM_LEVEL_SCORE] 维度={dim_knowledge.dimension_name}, 维度级用户相关性={dimension_user_relevance:.3f}, 维度级维度相关性={dimension_dim_relevance:.3f}")
        
        # 维度级相关性评分详细日志
        logger.debug(f"[RELEVANCE_SCORE] dimension={dim_knowledge.dimension_name}, "
                    f"dimension_user_relevance={dimension_user_relevance:.4f}, "
                    f"dimension_dim_relevance={dimension_dim_relevance:.4f}, "
                    f"total_comprehensive_score={total_comprehensive_score:.4f}, "
                    f"knowledge_count={len(knowledge_items)}")
    
    def _filter_low_relevance_knowledge(
        self,
        knowledge_items: List[Dict],
        dim_name: str
    ) -> List[Dict]:
        """
        过滤低相关性知识
        
        过滤阈值：comprehensive_score < RELEVANCE_THRESHOLD (0.4)
        
        Args:
            knowledge_items: 知识列表
            dim_name: 维度名称
            
        Returns:
            List[Dict]: 过滤后的知识列表
        """
        original_count = len(knowledge_items)
        
        filtered_items = [
            item for item in knowledge_items
            if item.get("comprehensive_score", 0.0) >= RELEVANCE_THRESHOLD
        ]
        
        filtered_count = len(filtered_items)
        removed_count = original_count - filtered_count
        
        logger.info(f"[ComprehensiveHealthAnalysisStrategy] _filter_low_relevance_knowledge: "
                   f"维度{dim_name}: 原始知识数={original_count}, 过滤后={filtered_count}, "
                   f"移除={removed_count} (阈值={RELEVANCE_THRESHOLD})")
        
        return filtered_items
    
    def _update_keywords_and_retrieve(
        self,
        context: ComprehensiveHealthAnalysisContextBody,
        dim_knowledge: DimensionKnowledge,
        relevance_result: Dict[str, Any],
        resource: AgentResource
    ) -> bool:
        """
        关键字更新与重新检索
        
        当维度充分性分数 < SUFFICIENCY_THRESHOLD时，根据LLM建议更新检索关键字并重新检索。
        
        Args:
            context: Agent上下文
            dim_knowledge: 维度知识
            relevance_result: LLM相关性评估结果
            resource: Agent资源
            
        Returns:
            bool: 是否进行了重新检索
        """
        dimension_sufficiency = relevance_result.get("dimension_sufficiency", _config.vector_default_score)
        
        if dimension_sufficiency >= SUFFICIENCY_THRESHOLD:
            logger.info(f"[ComprehensiveHealthAnalysisStrategy] _update_keywords_and_retrieve: "
                       f"维度{dim_knowledge.dimension_name}充分性{dimension_sufficiency:.2f}>=阈值，无需重新检索")
            return False
        
        if dim_knowledge.retrieve_attempts >= MAX_RETRIEVE_ATTEMPTS:
            logger.warning(f"[ComprehensiveHealthAnalysisStrategy] _update_keywords_and_retrieve: "
                          f"维度{dim_knowledge.dimension_name}已达到最大重新检索次数{MAX_RETRIEVE_ATTEMPTS}，停止检索")
            return False
        
        suggested_keywords = relevance_result.get("suggested_keywords", [])
        search_strategy = relevance_result.get("search_strategy", "keep")
        
        if not suggested_keywords or search_strategy == "keep":
            logger.info(f"[ComprehensiveHealthAnalysisStrategy] _update_keywords_and_retrieve: "
                       f"维度{dim_knowledge.dimension_name}search_strategy=keep，无需更新关键字")
            return False
        
        original_query = dim_knowledge.query
        
        if search_strategy == "expand":
            new_keywords = " ".join(suggested_keywords[:_config.suggested_keyword_limit])
            new_query = f"{original_query} {new_keywords}"
            logger.info(f"[ComprehensiveHealthAnalysisStrategy] _update_keywords_and_retrieve: "
                       f"维度{dim_knowledge.dimension_name}expand策略: original_query_len={len(original_query)}, "
                       f"new_query_len={len(new_query)}, keyword_count={len(suggested_keywords[:_config.suggested_keyword_limit])}")
        elif search_strategy == "refine":
            new_query = " ".join(suggested_keywords[:_config.suggested_keyword_limit])
            logger.info(f"[ComprehensiveHealthAnalysisStrategy] _update_keywords_and_retrieve: "
                       f"维度{dim_knowledge.dimension_name}refine策略: original_query_len={len(original_query)}, "
                       f"new_query_len={len(new_query)}, keyword_count={len(suggested_keywords[:_config.suggested_keyword_limit])}")
        else:
            logger.info(f"[ComprehensiveHealthAnalysisStrategy] _update_keywords_and_retrieve: "
                       f"维度{dim_knowledge.dimension_name}未知策略{search_strategy}，保持原查询")
            return False
        
        dim_knowledge.query = new_query
        dim_knowledge.retrieve_attempts += 1
        
        logger.info(f"[ComprehensiveHealthAnalysisStrategy] _update_keywords_and_retrieve: "
                   f"维度{dim_knowledge.dimension_name}开始重新检索 (第{dim_knowledge.retrieve_attempts}次)")
        
        self._retrieve_single_dimension(dim_knowledge.dimension_name, context, resource)
        
        logger.info(f"[ComprehensiveHealthAnalysisStrategy] _update_keywords_and_retrieve: "
                   f"维度{dim_knowledge.dimension_name}重新检索完成，"
                   f"知识数量={len(dim_knowledge.refined_knowledge)}")
        
        return True
    
    def _apply_performance_optimization(
        self,
        knowledge_items: List[Dict],
        dim_name: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        性能优化策略（v5.16更新：移除基于向量score的三层过滤）
        
        v5.16之前：基于向量score的三层过滤（score<0.4丢弃，0.4-0.7评估，>0.7保留）
        v5.16之后：向量score不再作为知识评分属性，改为低质知识过滤+LLM评估
        - 所有知识都进行LLM评估（低质知识已在_retrieve_single_dimension中过滤）
        - 直接返回所有知识供LLM评估
        
        Args:
            knowledge_items: 知识列表
            dim_name: 维度名称
            
        Returns:
            Tuple[List[Dict], List[Dict]]: (需要LLM评估的知识, 直接保留的知识)
        """
        # v5.16: 所有知识都进行LLM评估，不再基于向量score分层
        to_evaluate = list(knowledge_items)
        to_keep = []

        logger.info(f"[ComprehensiveHealthAnalysisStrategy] _apply_performance_optimization: "
                   f"维度{dim_name}: 原始={len(knowledge_items)}, "
                   f"需评估={len(to_evaluate)}, 直接保留={len(to_keep)}")

        # 性能优化层级分布日志
        logger.debug(f"[RESULT_DISTRIBUTION] dimension={dim_name}, total={len(knowledge_items)}, "
                    f"to_evaluate={len(to_evaluate)}, to_keep={len(to_keep)}, "
                    f"strategy=v5.16_all_evaluate")

        return to_evaluate, to_keep
    
    # ========================================================================
    # 辅助方法
    # ========================================================================

    def _truncate_by_sentence(self, text: str, max_length: int) -> str:
        """
        知识内容智能截断：按句子边界截断，保持语义完整性

        截断规则：
        1. 如果文本长度 <= max_length，原样返回
        2. 否则，在 [max_length * 0.5, max_length] 范围内查找最后一个句子边界
        3. 找到句子边界则在该处截断并添加 "..."
        4. 未找到则在 max_length 处截断并添加 "..."

        句子边界字符（中英文）：。？！；.?!;

        Args:
            text: 待截断文本
            max_length: 最大长度

        Returns:
            str: 截断后的文本
        """
        if len(text) <= max_length:
            return text

        # 句子边界字符
        sentence_boundaries = "。？！；.?!;"

        # 在 [max_length * 0.5, max_length] 范围内查找最后一个句子边界
        search_start = int(max_length * 0.5)
        search_end = max_length

        trunc_pos = -1
        for pos in range(search_end - 1, search_start - 1, -1):
            if pos < len(text) and text[pos] in sentence_boundaries:
                trunc_pos = pos + 1  # 在句子边界之后截断
                break

        if trunc_pos > 0:
            result = text[:trunc_pos] + "..."
        else:
            trunc_pos = max_length
            result = text[:max_length] + "..."

        logger.debug(f"[智能截断] 原文长度={len(text)}, 截断后长度={len(result)}, 截断位置={trunc_pos}")
        return result

    def _build_refine_prompt(
        self,
        context: ComprehensiveHealthAnalysisContextBody,
        dim_knowledge: DimensionKnowledge
    ) -> str:
        """
        构建知识精炼Prompt
        
        Args:
            context: 上下文
            dim_knowledge: 维度知识
            
        Returns:
            str: 精炼Prompt
        """
        prompt_parts = [
            "请对以下检索到的医学知识进行精炼整合，只做去冗余、去重复、去不相关内容的处理工作。",
            "",
            f"维度: {dim_knowledge.dimension_name}",
            f"查询: {dim_knowledge.query}",
            "",
            "检索到的知识:",
        ]
        
        for i, item in enumerate(dim_knowledge.refined_knowledge[:_config.knowledge_item_display_limit]):
            source_entity = item.get("source_entity", item.get("entity_name", item.get("name", "未知")))
            relation_type = item.get("relation_type", "unknown")
            target_entity = item.get("target_entity", item.get("entity_name", item.get("name", "未知")))
            content = item.get("content", item.get("description", ""))
            prompt_parts.append(f'{i+1}. [{source_entity}] -{relation_type}-> [{target_entity}]: "{self._truncate_by_sentence(content, _config.refine_content_truncate_len)}"')

        logger.info(f"[KNOWLEDGE_QUOTE] 维度={dim_knowledge.dimension_name}, 知识数={len(dim_knowledge.refined_knowledge[:_config.knowledge_item_display_limit])}, 已用双引号包裹")
        for index, item in enumerate(dim_knowledge.refined_knowledge[:_config.knowledge_item_display_limit]):
            kid = item.get("entity_name", item.get("name", "未知"))
            logger.info(f"[QUOTE_WRAP] knowledge_index={index}, knowledge_id_len={len(kid)}, content_wrapped=True")

        prompt_parts.extend([
            "",
            "精炼要求:",
            "1. 去除冗余和重复信息",
            "2. 去除与查询不相关的内容",
            "3. 整合相似的知识条目",
            "4. 保留关键医学信息",
            "",
            "请严格按照以下JSON格式输出，不要添加任何其他内容:",
            "{",
            '  "dimension": "维度名称",',
            '  "refined_knowledge": [',
            '    {"entity": "实体名称", "content": "精炼后的内容"},',
            '    ...',
            "  ],",
            '  "summary": "该维度知识的简要总结"',
            "}",
        ])
        
        return "\n".join(prompt_parts)
    
    def _fallback_rule_assessment(
        self,
        context: ComprehensiveHealthAnalysisContextBody
    ) -> HealthAssessment:
        """
        降级策略：规则引擎健康评估
        
        当HealthAssessmentChain不可用时，使用规则引擎进行健康评估。
        """
        logger.info("[ComprehensiveHealthAnalysisStrategy] 使用规则引擎进行健康评估")
        
        # 基于异常指标数量和严重程度计算健康评分
        base_score = _config.base_health_score
        
        # 异常指标扣分
        for anomaly in context.anomalies:
            severity = anomaly.get("severity", "normal")
            if severity == "severe":
                base_score -= _config.deduction_severe
            elif severity == "moderate":
                base_score -= _config.deduction_moderate
            elif severity == "mild":
                base_score -= _config.deduction_mild

        # 风险因子扣分
        base_score -= len(context.risk_factors) * _config.deduction_risk_factor

        # 疾病实体扣分
        diseases = context.medical_entities.get("diseases", [])
        base_score -= len(diseases) * _config.deduction_disease
        
        # 确保分数在0-100范围内
        health_score = max(0, min(100, base_score))
        
        # 判断健康等级
        if health_score >= _config.health_score_thresholds["excellent"]:
            health_level = "优秀"
        elif health_score >= _config.health_score_thresholds["good"]:
            health_level = "良好"
        elif health_score >= _config.health_score_thresholds["normal"]:
            health_level = "一般"
        elif health_score >= _config.health_score_thresholds["poor"]:
            health_level = "较差"
        else:
            health_level = "差"
        
        # 判断风险等级
        if health_score >= _config.risk_level_thresholds["low"]:
            risk_level = "低"
        elif health_score >= _config.risk_level_thresholds["mild"]:
            risk_level = "轻"
        elif health_score >= _config.risk_level_thresholds["moderate"]:
            risk_level = "中"
        else:
            risk_level = "高"
        
        return HealthAssessment(
            health_score=health_score,
            health_level=health_level,
            risk_level=risk_level,
            disease_risks=[],
            score_breakdown={
                "method": "rule_engine",
                "anomaly_count": len(context.anomalies),
                "risk_factor_count": len(context.risk_factors),
                "disease_count": len(diseases)
            },
            reasoning="使用规则引擎评估（降级模式）"
        )
    
    def _fallback_sequential_retrieval(
        self,
        context: ComprehensiveHealthAnalysisContextBody,
        resource: AgentResource
    ) -> Dict[str, Dict]:
        """
        降级策略：顺序检索模式
        
        当Agent检索失败时，降级为顺序检索模式。
        先向量检索锚定实体，后图查询做结构化推理增强。
        """
        logger.warning("[ComprehensiveHealthAnalysisStrategy] 降级为顺序检索模式")
        
        # 简化实现：使用BuildQueries阶段构建的查询
        dimension_summaries = {}
        
        for dim_name, query in context.dimension_queries.items():
            dimension_summaries[dim_name] = {
                "summary": "顺序检索结果（降级模式）",
                "key_entities": [],
                "knowledge_items": []
            }
        
        return dimension_summaries
    
    # ========================================================================
    # 错误处理
    # ========================================================================
    
    def _handle_engine_dead(
        self,
        context: ComprehensiveHealthAnalysisContextBody,
        state: str,
        error: EngineUnavailableError
    ) -> str:
        """处理SGLang引擎崩溃：跳过后续LLM调用，使用已有数据+规则引擎降级"""
        logger.warning(f"[ComprehensiveHealthAnalysisStrategy] 引擎崩溃降级: state={state}, "
                      f"跳过后续LLM调用，使用规则引擎降级")
        # v5设计验证日志：降级追踪
        logger.info(f"[DEGRADE_TRIGGER] component=SGLang, reason=引擎崩溃({state}状态), strategy=规则引擎降级")
        logger.info("[DEGRADE_MARK] target=LLM调用, from=正常, to=跳过")
        logger.info("[DEGRADE_STRATEGY] component=SGLang, level=CRITICAL, action=跳过后续LLM调用+规则引擎降级")

        context.degraded = True
        context.degraded_reason = f"SGLang引擎崩溃({state}状态)"
        context.error_code = ErrorCode.SGLANG_ENGINE_DEAD
        context.error_message = f"SGLang引擎崩溃，{state}状态及后续LLM调用已跳过"

        if state == "ParallelDimensionRetrieve":
            # 检索阶段引擎崩溃，收集已有结果继续
            self._collect_results(context)
            return "InterDimensionSync"
        elif state == "RefineKnowledge":
            # 精炼阶段引擎崩溃，使用规则摘要继续
            return "HealthAssess"
        elif state == "HealthAssess":
            # 健康评估阶段引擎崩溃，使用规则引擎评估
            context.health_assessment = self._fallback_rule_assessment(context)
            context.health_assessment.degraded = True
            context.health_assessment.degraded_reason = "SGLang引擎崩溃，使用规则引擎评估"
            return "Output"
        else:
            # 其他状态引擎崩溃，直接跳到Output
            if not context.health_assessment:
                context.health_assessment = self._fallback_rule_assessment(context)
                context.health_assessment.degraded = True
                context.health_assessment.degraded_reason = "SGLang引擎崩溃，使用规则引擎评估"
            return "Output"

    def _handle_error(
        self,
        context: ComprehensiveHealthAnalysisContextBody,
        error: Exception
    ) -> str:
        """处理错误状态"""
        logger.error(f"[ComprehensiveHealthAnalysisStrategy] ERROR: "
                    f"error_type={type(error).__name__}")
        # v5设计验证日志：降级追踪
        logger.info(f"[DEGRADE_TRIGGER] component=状态执行, reason=异常({type(error).__name__}), strategy=ERROR状态")
        logger.info("[DEGRADE_MARK] target=执行流程, from=正常, to=ERROR")
        logger.info("[DEGRADE_STRATEGY] component=状态执行, level=CRITICAL, action=进入ERROR状态")

        error_type = type(error).__name__
        context.error_message = f"状态执行异常({error_type})"
        context.degraded = True
        context.degraded_reason = context.error_message

        # 根据异常类型设置错误码
        if isinstance(error, MilvusUnavailableError):
            context.error_code = ErrorCode.MILVUS_UNAVAILABLE
        elif isinstance(error, Neo4jConnectionError):
            context.error_code = ErrorCode.NEO4J_UNAVAILABLE
        elif isinstance(error, LLMServiceError):
            context.error_code = ErrorCode.LLM_FAILURE
        elif isinstance(error, HealthAssessmentError):
            context.error_code = ErrorCode.HEALTH_ASSESS_FAILURE
        else:
            context.error_code = ErrorCode.UNKNOWN

        return "ERROR"
    
    def _handle_timeout(
        self,
        context: ComprehensiveHealthAnalysisContextBody,
        state: str,
        error: TimeoutError
    ) -> str:
        """处理超时状态"""
        logger.warning(f"[ComprehensiveHealthAnalysisStrategy] 超时降级: state={state}")
        # v5设计验证日志：降级追踪
        logger.info(f"[DEGRADE_TRIGGER] component=状态执行, reason=超时({state}状态), strategy=使用已有结果+规则引擎")
        logger.info(f"[DEGRADE_MARK] target={state}, from=正常, to=降级")
        logger.info("[DEGRADE_STRATEGY] component=状态执行, level=HIGH, action=使用已有结果继续")
        
        context.degraded = True
        context.degraded_reason = f"状态{state}执行超时"
        
        if state == "ParallelDimensionRetrieve":
            context.error_code = ErrorCode.ANALYSIS_TIMEOUT
            context.error_message = "并行检索超时，使用已有结果"
            # 使用已有结果继续
            self._collect_results(context)
            return "InterDimensionSync"
        elif state == "HealthAssess":
            context.error_code = ErrorCode.ANALYSIS_RETRIEVE_TIMEOUT
            context.error_message = "健康评估超时，使用规则引擎评估"
            # 使用规则引擎评估
            context.health_assessment = self._fallback_rule_assessment(context)
            context.health_assessment.degraded = True
            context.health_assessment.degraded_reason = "健康评估超时"
            return "Output"
        else:
            context.error_code = ErrorCode.ANALYSIS_PROCESS_TIMEOUT
            context.error_message = f"状态{state}执行超时"
            return "ERROR"
    
    def _build_result(
        self,
        context: ComprehensiveHealthAnalysisContextBody
    ) -> ComprehensiveHealthAnalysisResultData:
        """构建最终结果"""
        return ComprehensiveHealthAnalysisResultData(
            user_profile=context.user_profile,
            anomalies=context.anomalies,
            risk_factors=context.risk_factors,
            medical_entities=context.medical_entities,
            dimension_summaries=context.dimension_summaries,
            health_assessment=context.health_assessment.to_dict() if context.health_assessment else None,
            retrieval_stats=context.retrieval_stats.to_dict(),
            error_code=context.error_code,
            error_message=context.error_message,
            degraded=context.degraded,
            degraded_reason=context.degraded_reason
        )


def _get_analysis_trigger(from_state: str, to_state: str) -> str:
    """Derive a short snake_case trigger for a comprehensive health analysis sub-state transition."""
    triggers = {
        ("BuildQueries", "PlanRetrieval"): "queries_built",
        ("PlanRetrieval", "InitRetrievalContext"): "plan_ready",
        ("InitRetrievalContext", "ParallelDimensionRetrieve"): "context_initialized",
        ("ParallelDimensionRetrieve", "InterDimensionSync"): "retrieval_done",
        ("InterDimensionSync", "BuildQueries"): "cross_dim_duplicates",
        ("InterDimensionSync", "HybridRelevance"): "synced",
        ("HybridRelevance", "EvaluateSufficiency"): "scored",
        ("EvaluateSufficiency", "BuildQueries"): "insufficient",
        ("EvaluateSufficiency", "RefineKnowledge"): "sufficient",
        ("RefineKnowledge", "HealthAssess"): "refined",
        ("HealthAssess", "Output"): "assessed",
    }
    return triggers.get((from_state, to_state), "state_handler")


def _get_analysis_reason(from_state: str, to_state: str, context) -> str:
    """Derive a brief human-readable reason for a comprehensive health analysis sub-state transition."""
    from src.orchestration.agent.comprehensive_health_analysis_strategy.comprehensive_health_analysis_context import ComprehensiveHealthAnalysisContextBody

    if not isinstance(context, ComprehensiveHealthAnalysisContextBody):
        return ""

    dimension_count = len(context.dimension_queries) if context.dimension_queries else 0
    raw_knowledge_count = sum(len(dk.raw_knowledge) for dk in context.dimension_knowledge.values()) if context.dimension_knowledge else 0

    reasons = {
        ("BuildQueries", "PlanRetrieval"): f"dimensions={dimension_count}",
        ("PlanRetrieval", "InitRetrievalContext"): "paths_planned",
        ("InitRetrievalContext", "ParallelDimensionRetrieve"): "context_initialized",
        ("ParallelDimensionRetrieve", "InterDimensionSync"): f"dim_knowledge_items={raw_knowledge_count}",
        ("InterDimensionSync", "BuildQueries"): f"cross_dim_duplicates,chain_loop={context.chain_loop_count}",
        ("InterDimensionSync", "HybridRelevance"): f"unique_items={raw_knowledge_count}",
        ("HybridRelevance", "EvaluateSufficiency"): "relevance_scored",
        ("EvaluateSufficiency", "BuildQueries"): f"insufficient,agent_loop={context.agent_retrieval_loop_count}",
        ("EvaluateSufficiency", "RefineKnowledge"): f"dim_count={dimension_count}",
        ("RefineKnowledge", "HealthAssess"): "knowledge_refined",
        ("HealthAssess", "Output"): f"health_score={context.health_assessment.health_score}" if context.health_assessment and context.health_assessment.health_score is not None else "assessed",
    }
    return reasons.get((from_state, to_state), "")
