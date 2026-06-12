# -*- coding: utf-8 -*-
"""
知识检索Agent

该模块实现KnowledgeRetrievalStrategy类，用于健康咨询业务中的知识检索环节。
基于设计文档《项目业务详细设计v5》第2.3节的设计实现。

主要功能：
1. ReAct模式：Thought → Action → Observation循环
2. knowledge_retrieval_max_steps限制（默认5，由ConsultServiceConfig配置）
3. knowledge_retrieval_max_prompt_chars限制（默认4000，由ConsultServiceConfig配置）
4. 可用Action列表：vector_search、graph_query、expand_search、finish
5. 降级策略：Agent失败时回退到顺序检索模式
"""

import logging
import time
import json
import re
from typing import Any, Dict, List, Optional

from src.config.business.consult_service_config import get_runtime_config
from src.errors import ErrorCode, MilvusUnavailableError, Neo4jConnectionError, LLMServiceError
from src.orchestration.agent.agent_strategy import AgentStrategy
from src.orchestration.agent.data_classes import AgentContext, AgentResult
from src.orchestration.agent.agent_resource import AgentResource
from src.orchestration.state_machine.state_machine import StateMachine
from src.orchestration.chain.knowledge_retrieval_chain.knowledge_retrieval_chain import (
    KnowledgeRetrievalContextBody as ChainContextBody,
)
from src.orchestration.agent.knowledge_retrieval_strategy.knowledge_retrieval_context import (
    RetrievalStep,
    KnowledgeRetrievalContextBody,
)
from src.orchestration.agent.knowledge_retrieval_strategy.knowledge_retrieval_result import (
    KnowledgeRetrievalResultData,
)

logger = logging.getLogger(__name__)

# 模块级常量 - 使用惰性获取模式，避免模块级求值时ConfigManager未初始化
def _get_max_steps() -> int:
    return get_runtime_config().knowledge_retrieval_max_steps

def _get_max_prompt_chars() -> int:
    return get_runtime_config().knowledge_retrieval_max_prompt_chars

MAX_STEPS = _get_max_steps
MAX_PROMPT_CHARS = _get_max_prompt_chars


class KnowledgeRetrievalStrategy(
    AgentStrategy[KnowledgeRetrievalContextBody, KnowledgeRetrievalResultData]
):
    """
    知识检索Agent
    
    基于设计文档《项目业务详细设计v5》第2.3节设计实现。
    
    核心特点：
    - LLM作为决策者：大语言模型根据当前上下文动态决定下一步检索操作
    - 动态策略调整：根据检索结果实时调整检索策略，而非固定流程
    - 自适应检索：根据问题类型、检索效果自动选择最优检索路径
    - 降级保障：Agent失败时自动回退到固定流程的顺序检索模式
    
    ReAct模式状态循环：
    Thought → Action → Observation → Thought → ...
    
    可用Action列表：
    - vector_search: 向量检索锚定实体、关系或属性
    - graph_query: 图查询做结构化推理增强
    - expand_search: 扩展检索(基于已有结果扩展)
    - finish: 完成检索，返回结果
    """
    
    AVAILABLE_ACTIONS = ["vector_search", "graph_query", "expand_search", "finish"]

    class _LazyConsultConfig:
        """延迟加载配置代理，每次属性访问时从ConfigManager获取真实配置"""
        def __getattr__(self, name):
            return getattr(get_runtime_config(), name)

    _consult_config = _LazyConsultConfig()

    STATE_TIMEOUTS = {
        "Thought": 10,
        "Action": 15,
        "Observation": 5,
    }
    
    def execute(
        self,
        context: AgentContext[KnowledgeRetrievalContextBody],
        resource: AgentResource
    ) -> AgentResult[KnowledgeRetrievalResultData]:
        """
        执行知识检索Agent策略
        
        Args:
            context: Agent输入数据容器
            resource: Agent资源类
            
        Returns:
            AgentResult: Agent输出数据容器
        """
        start_time = time.time()
        logger.info(f"[STAGE_ENTER] KnowledgeRetrievalStrategy, session_id={context.session_id}")
        logger.info(f"[KnowledgeRetrievalStrategy] 开始执行: session_id={context.session_id}")
        
        body = context.body
        if body is None:
            logger.warning("[KnowledgeRetrievalStrategy] 输入数据为空")
            return AgentResult(
                session_id=context.session_id,
                data=KnowledgeRetrievalResultData(
                    error_code=1,
                    error_message="输入数据为空"
                )
            )
        
        state_machine = StateMachine(context.session_id)
        self._register_state_transitions(state_machine)
        
        self._state_handlers = {
            "Thought": self._handle_thought,
            "Action": self._handle_action,
            "Observation": self._handle_observation,
            "Finish": self._handle_finish,
        }
        
        current_state = body.current_state
        max_iterations = self._consult_config.knowledge_retrieval_max_steps * 3 + 5
        iteration = 0
        
        try:
            while iteration < max_iterations:
                iteration += 1
                logger.info(f"[KnowledgeRetrievalStrategy] 状态转换: "
                           f"current_state={current_state}, step={body.current_step}, iteration={iteration}")
                
                if body.current_step >= self._consult_config.knowledge_retrieval_max_steps and current_state != "Finish":
                    logger.warning(f"[KnowledgeRetrievalStrategy] 达到最大步数限制: knowledge_retrieval_max_steps={self._consult_config.knowledge_retrieval_max_steps}")
                    current_state = "Finish"
                    body.current_state = current_state
                
                handler = self._state_handlers.get(current_state)
                if handler is None:
                    logger.error(f"[KnowledgeRetrievalStrategy] 未知状态: {current_state}")
                    body.current_state = "Finish"
                    break
                
                try:
                    next_state = handler(body, resource)
                except Exception as e:
                    logger.error(f"[KnowledgeRetrievalStrategy] 状态处理异常: "
                               f"state={current_state}, error={str(e)}")
                    next_state = self._handle_error(body, e)
                
                state_machine.transition(current_state, next_state,
                    trigger=_get_knowledge_trigger(current_state, next_state),
                    reason=_get_knowledge_reason(current_state, next_state, body))
                logger.info(f"[STATE_TRANSITION] {current_state} -> {next_state}, step={body.current_step}")
                current_state = next_state
                body.current_state = current_state
                
                if current_state == "Finish":
                    break
            
            result_data = self._build_result(body)
            
        except Exception as e:
            logger.error(f"[KnowledgeRetrievalStrategy] Agent执行失败，触发降级: {str(e)}")
            result_data = self._fallback_sequential_retrieval(body, resource)
            result_data.degraded = True
            result_data.degraded_reason = f"Agent执行失败: {str(e)}"
        
        elapsed = time.time() - start_time
        logger.info(f"[STAGE_EXIT] KnowledgeRetrievalStrategy, duration={elapsed:.2f}s, session_id={context.session_id}")
        logger.info(f"[KnowledgeRetrievalStrategy] 执行完成: "
                   f"session_id={context.session_id}, elapsed={elapsed:.2f}s, "
                   f"steps={body.current_step}, results={len(result_data.merged_results)}, "
                   f"degraded={result_data.degraded}")
        
        return AgentResult(session_id=context.session_id, data=result_data)
    
    def _register_state_transitions(self, state_machine: StateMachine) -> None:
        """注册状态转换规则"""
        state_machine.add_state_transition("Thought", ["Action", "Finish"])
        state_machine.add_state_transition("Action", ["Observation", "Finish"])
        state_machine.add_state_transition("Observation", ["Thought", "Finish"])
        state_machine.add_state_transition("Finish", [])
    
    def _handle_thought(
        self,
        context: KnowledgeRetrievalContextBody,
        resource: AgentResource
    ) -> str:
        logger.info(f"[KnowledgeRetrievalStrategy._handle_thought] ReAct.Thought: 思考下一步行动, step={context.current_step}, all_results={len(context.all_results)}")
        
        if context.current_step >= self._consult_config.knowledge_retrieval_max_steps:
            logger.info(f"[KnowledgeRetrievalStrategy._handle_thought] ReAct.Thought: 达到最大步数knowledge_retrieval_max_steps={self._consult_config.knowledge_retrieval_max_steps}, 准备结束")
            context.current_step += 1
            return "Finish"
        
        if len(context.all_results) >= self._consult_config.knowledge_sufficiency_min_count:
            sufficiency = self._calculate_sufficiency(context)
            context.sufficiency_score = sufficiency
            if sufficiency >= self._consult_config.knowledge_fusion_threshold:
                logger.info(f"[KnowledgeRetrievalStrategy._handle_thought] ReAct.Thought: 结果充分(sufficiency={sufficiency:.2f}>={self._consult_config.knowledge_fusion_threshold}), 准备结束")
                context.is_sufficient = True
                context.current_step += 1
                return "Finish"
            else:
                logger.info(f"[KnowledgeRetrievalStrategy._handle_thought] ReAct.Thought: 结果不充分(sufficiency={sufficiency:.2f}<{self._consult_config.knowledge_fusion_threshold}), 继续检索")

        decision = self._make_decision(context, resource)
        
        step = RetrievalStep(
            step_num=context.current_step + 1,
            thought=decision.get("thought", ""),
            action=decision.get("action", "vector_search"),
            action_params=decision.get("params", {})
        )
        context.step_history.append(step)
        
        logger.info(f"[KnowledgeRetrievalStrategy._handle_thought] ReAct.Thought: LLM决策结果 thought={step.thought[:200]}, action={step.action}, "
                   f"params={step.action_params}")
        
        context.current_step += 1
        return "Action"
    
    def _handle_action(
        self,
        context: KnowledgeRetrievalContextBody,
        resource: AgentResource
    ) -> str:
        if not context.step_history:
            logger.warning("[KnowledgeRetrievalStrategy._handle_action] ReAct.Action: 无步骤历史，使用默认vector_search")
            action = "vector_search"
            params = {"query": context.query_text, "top_k": 20}
        else:
            last_step = context.step_history[-1]
            action = last_step.action
            params = last_step.action_params
        
        logger.info(f"[KnowledgeRetrievalStrategy._handle_action] ReAct.Action: 执行 action={action}, params={params}")
        
        try:
            if action == "vector_search":
                results = self._execute_vector_search(context, resource, params)
            elif action == "graph_query":
                results = self._execute_graph_query(context, resource, params)
            elif action == "expand_search":
                results = self._execute_expand_search(context, resource, params)
            elif action == "finish":
                logger.info("[KnowledgeRetrievalStrategy._handle_action] ReAct.Action: finish action执行, 结束检索")
                return "Finish"
            else:
                logger.warning(f"[KnowledgeRetrievalStrategy._handle_action] ReAct.Action: 未知action={action}, 降级使用vector_search")
                results = self._execute_vector_search(context, resource, params)
            
            if context.step_history:
                context.step_history[-1].results = results
            
            context.all_results.extend(results)
            
            logger.info(f"[KnowledgeRetrievalStrategy._handle_action] ReAct.Action: 执行完成, action={action}, 获取{len(results)}条结果, 累计all_results={len(context.all_results)}")
            
        except Exception as e:
            logger.error(f"[KnowledgeRetrievalStrategy._handle_action] ReAct.Action: 执行失败 action={action}, error={str(e)}")
            if context.step_history:
                context.step_history[-1].observation = f"执行失败: {str(e)}"
        
        return "Observation"
    
    def _handle_observation(
        self,
        context: KnowledgeRetrievalContextBody,
        resource: AgentResource
    ) -> str:
        logger.info("[KnowledgeRetrievalStrategy._handle_observation] ReAct.Observation: 观察检索结果")
        
        if context.step_history:
            last_step = context.step_history[-1]
            results_count = len(last_step.results)
            observation = f"检索到{results_count}条结果"
            last_step.observation = observation
            logger.info(f"[KnowledgeRetrievalStrategy._handle_observation] ReAct.Observation: {observation}, step={last_step.step_num}")
        
        sufficiency = self._calculate_sufficiency(context)
        context.sufficiency_score = sufficiency
        
        if sufficiency >= self._consult_config.knowledge_fusion_threshold:
            logger.info(f"[KnowledgeRetrievalStrategy._handle_observation] ReAct.Observation: 结果充分(sufficiency={sufficiency:.2f}>={self._consult_config.knowledge_fusion_threshold}), 准备结束")
            context.is_sufficient = True
            return "Finish"
        
        if context.current_step >= self._consult_config.knowledge_retrieval_max_steps:
            logger.info(f"[KnowledgeRetrievalStrategy._handle_observation] ReAct.Observation: 达到最大步数knowledge_retrieval_max_steps={self._consult_config.knowledge_retrieval_max_steps}, 准备结束")
            return "Finish"
        
        logger.info(f"[KnowledgeRetrievalStrategy._handle_observation] ReAct.Observation: 结果不充分(sufficiency={sufficiency:.2f}<{self._consult_config.knowledge_fusion_threshold}), 继续Thought")
        return "Thought"

    def _handle_finish(
        self,
        context: KnowledgeRetrievalContextBody,
        resource: AgentResource
    ) -> str:
        logger.info(f"[KnowledgeRetrievalStrategy._handle_finish] ReAct.Finish: 完成检索, "
                   f"steps={context.current_step}, results={len(context.all_results)}, "
                   f"anchored_entities={len(context.anchored_entities)}, "
                   f"anchored_relations={len(context.anchored_relations)}, "
                   f"sufficiency_score={context.sufficiency_score:.2f}, "
                   f"is_sufficient={context.is_sufficient}")
        return "Finish"
    
    def _make_decision(
        self,
        context: KnowledgeRetrievalContextBody,
        resource: AgentResource
    ) -> Dict[str, Any]:
        """
        构建Agent决策Prompt并调用LLM决策
        
        限制机制：
        - knowledge_retrieval_max_prompt_chars（默认4000，由ConsultServiceConfig配置）: Agent决策Prompt最大字符数
        """
        prompt = self._build_decision_prompt(context)
        
        if len(prompt) > self._consult_config.knowledge_retrieval_max_prompt_chars:
            prompt = prompt[:self._consult_config.knowledge_retrieval_max_prompt_chars]
            logger.warning(f"[KnowledgeRetrievalStrategy._make_decision] Prompt被截断到{self._consult_config.knowledge_retrieval_max_prompt_chars}字符")
        
        decision = self._call_llm_for_decision(prompt, resource)
        
        if decision.get("action") not in self.AVAILABLE_ACTIONS:
            logger.warning(f"[KnowledgeRetrievalStrategy._make_decision] 无效action={decision.get('action')}, 降级使用默认vector_search")
            decision = {
                "thought": "执行向量检索以锚定相关实体",
                "action": "vector_search",
                "params": {"query": context.query_text, "top_k": 20}
            }
        
        return decision
    
    def _build_decision_prompt(self, context: KnowledgeRetrievalContextBody) -> str:
        """构建Agent决策Prompt"""
        prompt_parts = [
            "你是一个医学知识检索专家，请根据当前上下文决定下一步检索操作。",
            "",
            "## 用户查询",
            context.query_text,
            "",
            "## 已提取实体",
            str(context.extracted_entities[:5]) if context.extracted_entities else "无",
            "",
            "## 当前检索状态",
            f"- 当前步数: {context.current_step}/{self._consult_config.knowledge_retrieval_max_steps}",
            f"- 已检索结果数: {len(context.all_results)}",
            f"- 锚定实体数: {len(context.anchored_entities)}",
            "",
        ]
        
        if context.step_history:
            prompt_parts.append("## 检索历史")
            for step in context.step_history[-3:]:
                prompt_parts.append(f"- Step {step.step_num}: {step.action} -> {len(step.results)}条结果")
            prompt_parts.append("")
        
        if context.all_results:
            prompt_parts.append("## 已检索结果摘要")
            for item in context.all_results[:5]:
                entity = item.get("entity", {})
                name = entity.get("name", entity.get("entity_name", "未知"))
                score = item.get("score", 0)
                prompt_parts.append(f"- {name} (相关度: {score:.2f})")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "## 可用Action列表",
            "1. vector_search: 向量检索锚定实体、关系或属性",
            "2. graph_query: 图查询做结构化推理增强",
            "3. expand_search: 扩展检索(基于已有结果扩展)",
            "4. finish: 完成检索，返回结果",
            "",
            "## 输出格式(JSON)",
            '{"thought": "思考内容", "action": "action名称", "params": {参数}}',
            "",
            "## 决策要求",
            "1. 如果还没有执行过vector_search，优先执行vector_search",
            "2. 如果已有锚定实体但还没有执行graph_query，执行graph_query",
            "3. 如果结果不充分且可以扩展，执行expand_search",
            "4. 如果结果已充分或达到最大步数，执行finish",
        ])
        
        return "\n".join(prompt_parts)
    
    def _call_llm_for_decision(
        self,
        prompt: str,
        resource: AgentResource
    ) -> Dict[str, Any]:
        model_service = resource.model_service
        
        if model_service and hasattr(model_service, 'call_model'):
            try:
                messages = [
                    {"role": "system", "content": "你是一个知识检索决策专家。根据当前检索状态，决定下一步采取的行动。请以JSON格式返回决策结果，包含thought（思考）、action（动作名称）和params（参数）三个字段。可用动作：vector_search、graph_query、expand_search、finish。"},
                    {"role": "user", "content": prompt}
                ]
                logger.info(f"[KnowledgeRetrievalStrategy._call_llm_for_decision] 调用LLM进行检索决策，消息数={len(messages)}")
                # 详细记录LLM决策完整输入和输出
                logger.info(f"[LLM_INPUT] system_prompt: {messages[0].get('content', '') if messages else ''}")
                logger.info(f"[LLM_INPUT] user_prompt: {prompt}")
                _llm_start = time.time()
                response = model_service.call_model(messages)
                _llm_elapsed = time.time() - _llm_start
                logger.info(f"[KnowledgeRetrievalStrategy._call_llm_for_decision] LLM检索决策完成，输出长度={len(response) if response else 0}")
                logger.info(f"[LLM_OUTPUT] {response}")
                logger.info(f"[LLM_DURATION] {_llm_elapsed:.3f}s")
                decision = self._parse_decision_response(response, model_service=model_service)
                if decision:
                    return decision
            except Exception as e:
                logger.warning(f"[KnowledgeRetrievalStrategy._call_llm_for_decision] LLM调用失败: {str(e)}")
                logger.debug(f"[KnowledgeRetrievalStrategy._call_llm_for_decision] LLM调用失败详情 - 输入prompt: {prompt[:500]}")
        
        logger.warning("[KnowledgeRetrievalStrategy._call_llm_for_decision] LLM不可用或返回无效, 使用默认vector_search决策")
        return {
            "thought": "执行向量检索以锚定相关实体",
            "action": "vector_search",
            "params": {}
        }
    
    def _extract_balanced_json(self, text: str) -> Optional[str]:
        """从文本中提取第一个平衡的JSON对象（忽略末尾多余的}）"""
        start = text.find('{')
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

    def _parse_decision_response(self, response: str, model_service: Any = None) -> Optional[Dict[str, Any]]:
        """解析LLM决策响应"""
        try:
            # 多层候选提取：markdown代码块 → 平衡括号提取 → 逐个候选json.loads
            candidates = []
            # 1. markdown代码块
            md_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response)
            if md_match:
                candidates.append(md_match.group(1))
            # 2. 平衡括号提取（找到第一个{对应的闭合}，避免Qwen3末尾多余}干扰）
            balanced = self._extract_balanced_json(response)
            if balanced:
                candidates.append(balanced)
            # 3. 逐个候选尝试json.loads
            for candidate in candidates:
                try:
                    decision = json.loads(candidate)
                    if isinstance(decision, dict) and "action" in decision:
                        return decision
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logger.debug(f"[KnowledgeRetrievalStrategy] 解析LLM决策响应失败: {e}")

        # 结构化输出自修复：JSON解析失败时尝试修复
        if response and len(response.strip()) > 0:
            repair_decision = self._try_structured_repair_for_decision(response, model_service=model_service)
            if repair_decision:
                return repair_decision

        for action in self.AVAILABLE_ACTIONS:
            if action in response.lower():
                return {
                    "thought": f"执行{action}",
                    "action": action,
                    "params": {}
                }

        return None

    def _try_structured_repair_for_decision(self, raw_output: str, model_service: Any = None) -> Optional[Dict[str, Any]]:
        """Qwen3结构化输出自修复：检索决策JSON解析失败时尝试修复"""
        try:
            repair_prompt = (
                "你上一次输出的JSON结构有误，请修复后重新输出。\n\n"
                f"【错误信息】JSON解析失败或缺少action字段\n\n"
                f"【你的原始输出】\n{raw_output}\n\n"
                '【期望格式】\n{"thought":"思考过程","action":"vector_search","params":{}}\n\n'
                "请直接输出修复后的JSON，不要输出其他内容。 /no_think"
            )
            messages = [
                {"role": "system", "content": "你是一个知识检索决策专家，请以JSON格式返回决策结果。"},
                {"role": "user", "content": repair_prompt}
            ]
            logger.info("[STRUCTURED_REPAIR] 尝试自修复: context_type=knowledge_decision")
            if model_service is None:
                logger.warning("[STRUCTURED_REPAIR] model_service不可用，跳过自修复")
                return None
            repair_response = model_service.call_model(messages)
            if repair_response:
                logger.info(f"[STRUCTURED_REPAIR_OUTPUT] context_type=knowledge_decision, response_len={len(repair_response)}, response={repair_response[:500]}")
                candidates = []
                md_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', repair_response)
                if md_match:
                    candidates.append(md_match.group(1))
                balanced = self._extract_balanced_json(repair_response)
                if balanced:
                    candidates.append(balanced)
                for candidate in candidates:
                    try:
                        decision = json.loads(candidate)
                        if isinstance(decision, dict) and "action" in decision:
                            logger.info("[STRUCTURED_REPAIR] 自修复成功: context_type=knowledge_decision")
                            return decision
                    except json.JSONDecodeError:
                        continue
            logger.warning("[STRUCTURED_REPAIR] 自修复失败: context_type=knowledge_decision")
            return None
        except Exception as e:
            logger.warning(f"[STRUCTURED_REPAIR] 自修复失败: error_type={type(e).__name__}, context_type=knowledge_decision")
            return None
    
    def _execute_vector_search(
        self,
        context: KnowledgeRetrievalContextBody,
        resource: AgentResource,
        params: Dict[str, Any]
    ) -> List[Dict]:
        """通过KnowledgeRetrievalChain执行向量检索，超时时降级为关键词匹配"""
        chain = resource.get_chain("knowledge_retrieval_chain")
        if chain is None:
            logger.warning("[KnowledgeRetrievalStrategy._execute_vector_search] knowledge_retrieval_chain未注册, 降级为关键词匹配")
            return self._fallback_keyword_search(context, resource, params)

        try:
            search_start_time = time.time()
            chain_input = ChainContextBody(
                query_text=params.get("query", context.query_text),
                extracted_entities=context.anchored_entities if context.anchored_entities else [],
                intent_label=context.intent_label
            )
            vector_results, anchored_entities, anchored_relations = chain._vector_search_step(chain_input)
            search_elapsed = time.time() - search_start_time

            if search_elapsed > self._consult_config.vector_search_timeout:
                logger.warning(f"[KnowledgeRetrievalStrategy._execute_vector_search] 向量检索超时降级: "
                             f"耗时={search_elapsed:.2f}s > 阈值={self._consult_config.vector_search_timeout}s, "
                             f"降级策略=关键词匹配")
                return self._fallback_keyword_search(context, resource, params)

            context.anchored_entities.extend(anchored_entities)
            context.anchored_relations.extend(anchored_relations)

            logger.info(f"[KnowledgeRetrievalStrategy._execute_vector_search] 向量检索(Chain): "
                       f"results={len(vector_results)}, entities={len(context.anchored_entities)}, "
                       f"elapsed={search_elapsed:.2f}s")

            return vector_results

        except Exception as e:
            logger.error(f"[KnowledgeRetrievalStrategy._execute_vector_search] 向量检索失败: {str(e)}, 降级为关键词匹配")
            return self._fallback_keyword_search(context, resource, params)
    
    def _fallback_keyword_search(
        self,
        context: KnowledgeRetrievalContextBody,
        resource: AgentResource,
        params: Dict[str, Any]
    ) -> List[Dict]:
        """
        降级策略：关键词匹配检索
        
        当向量模型推理超时或失败时，使用关键词匹配作为降级方案。
        通过Neo4j图数据库的模糊匹配功能实现关键词检索。
        
        Args:
            context: 知识检索上下文
            resource: Agent资源类
            params: 检索参数
            
        Returns:
            关键词匹配结果列表，带有_degraded标记
        """
        query = params.get("query", context.query_text)
        logger.info(f"[KnowledgeRetrievalStrategy._fallback_keyword_search] 关键词匹配降级检索: query={query[:50]}...")
        
        results = []
        
        # 尝试使用Neo4j进行关键词匹配
        neo4j_handler = resource.get_tool_handler("neo4j_medical_tool")
        if neo4j_handler is not None:
            try:
                # 从查询中提取关键词
                keywords = query.split()
                for keyword in keywords[:self._consult_config.neo4j_keyword_search_limit]:  # 最多使用配置的关键词数量
                    try:
                        # 使用Neo4j的模糊匹配功能
                        disease_info = neo4j_handler.search_diseases_by_keyword(keyword)
                        if disease_info and isinstance(disease_info, list):
                            for item in disease_info[:self._consult_config.neo4j_keyword_search_limit]:
                                result_item = {
                                    "source": "neo4j_keyword_fallback",
                                    "type": "disease",
                                    "entity": item.get("name", keyword),
                                    "data": item,
                                    "score": self._consult_config.neo4j_degraded_search_score,  # 降级检索使用较低的置信度
                                    "_degraded": True,
                                    "_degraded_reason": "向量检索超时，降级为关键词匹配"
                                }
                                results.append(result_item)
                    except Exception as e:
                        logger.warning(f"[KnowledgeRetrievalStrategy._fallback_keyword_search] 关键词'{keyword}'检索失败: {str(e)}")
            except Exception as e:
                logger.error(f"[KnowledgeRetrievalStrategy._fallback_keyword_search] Neo4j关键词检索失败: {str(e)}")
        else:
            logger.warning("[KnowledgeRetrievalStrategy._fallback_keyword_search] neo4j_medical_tool未注册, 关键词匹配降级也无法执行")
        
        # 去重
        seen_names = set()
        deduplicated_results = []
        for item in results:
            entity_name = item.get("entity", "")
            if entity_name and entity_name not in seen_names:
                seen_names.add(entity_name)
                deduplicated_results.append(item)
        
        logger.info(f"[KnowledgeRetrievalStrategy._fallback_keyword_search] 关键词匹配降级检索完成: "
                   f"query={query[:50]}..., results={len(deduplicated_results)}, "
                   f"降级原因=向量检索超时或失败")
        
        return deduplicated_results
    
    def _execute_graph_query(
        self,
        context: KnowledgeRetrievalContextBody,
        resource: AgentResource,
        params: Dict[str, Any]
    ) -> List[Dict]:
        """通过KnowledgeRetrievalChain执行图查询"""
        chain = resource.get_chain("knowledge_retrieval_chain")
        if chain is None:
            logger.warning("[KnowledgeRetrievalStrategy._execute_graph_query] knowledge_retrieval_chain未注册, 降级: 仅使用向量检索结果")
            return []

        if not context.anchored_entities:
            logger.warning("[KnowledgeRetrievalStrategy._execute_graph_query] 图查询: 无锚定实体, 降级: 跳过图查询")
            return []

        try:
            query = params.get("query", context.query_text)
            results = chain._graph_query_step(context.anchored_entities, context.anchored_relations, query)

            logger.info(f"[KnowledgeRetrievalStrategy._execute_graph_query] 图查询(Chain): "
                       f"entities={len(context.anchored_entities)}, results={len(results)}")
            return results

        except Exception as e:
            logger.error(f"[KnowledgeRetrievalStrategy._execute_graph_query] 图查询失败: {str(e)}")
            return []
    
    def _execute_expand_search(
        self,
        context: KnowledgeRetrievalContextBody,
        resource: AgentResource,
        params: Dict[str, Any]
    ) -> List[Dict]:
        """执行扩展检索"""
        seed_entities = params.get("seed_entities", context.anchored_entities[:self._consult_config.neo4j_keyword_search_limit])

        if not seed_entities:
            logger.warning("[KnowledgeRetrievalStrategy._execute_expand_search] 扩展检索: 无种子实体, 降级: 跳过扩展检索")
            return []
        
        expand_queries = []
        for entity in seed_entities:
            entity_data = entity.get("entity", {})
            name = entity_data.get("name", entity_data.get("entity_name", ""))
            if name:
                expand_queries.append(f"{name} 相关疾病 症状")
                expand_queries.append(f"{name} 并发症 预防")
        
        results = []
        for query in expand_queries[:self._consult_config.neo4j_keyword_search_limit]:
            vector_results = self._execute_vector_search(
                context, resource, {"query": query, "top_k": 10}
            )
            results.extend(vector_results)
        
        logger.info(f"[KnowledgeRetrievalStrategy._execute_expand_search] 扩展检索: seeds={len(seed_entities)}, results={len(results)}")
        return results
    
    def _calculate_sufficiency(self, context: KnowledgeRetrievalContextBody) -> float:
        """计算结果充分性分数"""
        if not context.all_results:
            return 0.0
        
        result_count = len(context.all_results)
        entity_count = len(context.anchored_entities)
        
        count_score = min(1.0, result_count / self._consult_config.sufficiency_count_denominator)
        entity_score = min(1.0, entity_count / self._consult_config.sufficiency_entity_denominator)
        
        avg_relevance = 0.0
        if context.all_results:
            scores = [r.get("score", 0) for r in context.all_results if r.get("score", 0) > 0]
            if scores:
                avg_relevance = sum(scores) / len(scores)
        
        sufficiency = (self._consult_config.sufficiency_count_weight * count_score
                       + self._consult_config.sufficiency_entity_weight * entity_score
                       + self._consult_config.sufficiency_relevance_weight * avg_relevance)
        
        # 充分性判断详细日志
        is_sufficient = sufficiency >= self._consult_config.knowledge_fusion_threshold
        gaps = []
        if not is_sufficient:
            gaps.append(f"充分性分数{sufficiency:.2f}<阈值{self._consult_config.knowledge_fusion_threshold}")
        if result_count < self._consult_config.knowledge_sufficiency_min_count:
            gaps.append(f"结果数量不足({result_count}条,需要>={self._consult_config.knowledge_sufficiency_min_count})")
        if entity_count < 1:
            gaps.append("无锚定实体")
        logger.info(f"[SUFFICIENCY] is_sufficient={is_sufficient}, confidence={sufficiency:.4f}, "
                    f"gaps={gaps}, count_score={count_score:.4f}, entity_score={entity_score:.4f}, "
                    f"avg_relevance={avg_relevance:.4f}, result_count={result_count}, "
                    f"entity_count={entity_count}")

        return sufficiency
    
    def _handle_error(
        self,
        context: KnowledgeRetrievalContextBody,
        error: Exception
    ) -> str:
        """处理错误状态"""
        logger.error(f"[KnowledgeRetrievalStrategy] ERROR: "
                    f"error_type={type(error).__name__}, message={str(error)}")

        context.error_message = str(error)

        if isinstance(error, MilvusUnavailableError):
            context.error_code = ErrorCode.MILVUS_UNAVAILABLE
        elif isinstance(error, Neo4jConnectionError):
            context.error_code = ErrorCode.NEO4J_UNAVAILABLE
        elif isinstance(error, LLMServiceError):
            context.error_code = ErrorCode.LLM_FAILURE
        else:
            context.error_code = ErrorCode.UNKNOWN

        return "Finish"
    
    def _fallback_sequential_retrieval(
        self,
        context: KnowledgeRetrievalContextBody,
        resource: AgentResource
    ) -> KnowledgeRetrievalResultData:
        """
        降级策略：顺序检索模式
        
        先向量检索锚定实体，后图查询做结构化推理增强
        """
        logger.warning("[KnowledgeRetrievalStrategy._fallback_sequential_retrieval] 降级触发: Agent检索失败, 降级策略=顺序检索模式(先向量检索锚定实体,后图查询做结构化推理增强)")
        
        all_results = []
        anchored_entities = []
        anchored_relations = []
        
        try:
            vector_results = self._execute_vector_search(
                context, resource, {"query": context.query_text, "top_k": 20}
            )
            all_results.extend(vector_results)
            anchored_entities = context.anchored_entities.copy()
            anchored_relations = context.anchored_relations.copy()
        except Exception as e:
            logger.error(f"[KnowledgeRetrievalStrategy._fallback_sequential_retrieval] 降级向量检索失败: {str(e)}")
        
        try:
            if anchored_entities:
                graph_results = self._execute_graph_query(
                    context, resource, {"entity_ids": []}
                )
                all_results.extend(graph_results)
        except Exception as e:
            logger.error(f"[KnowledgeRetrievalStrategy._fallback_sequential_retrieval] 降级图查询失败: {str(e)}")
        
        merged_results = self._merge_and_deduplicate(all_results)
        
        return KnowledgeRetrievalResultData(
            merged_results=merged_results,
            anchored_entities=anchored_entities,
            anchored_relations=anchored_relations,
            total_steps=1,
            sufficiency_score=self._calculate_sufficiency(context),
            is_sufficient=len(merged_results) >= self._consult_config.knowledge_sufficiency_min_count,
            degraded=True,
            degraded_reason="Agent失败，降级为顺序检索模式"
        )
    
    def _merge_and_deduplicate(self, results: List[Dict]) -> List[Dict]:
        """合并去重结果"""
        before_count = len(results)
        merged = []
        seen_ids = set()
        
        for item in results:
            entity = item.get("entity", {})
            entity_str = ""
            if isinstance(entity, dict):
                entity_str = entity.get("name", entity.get("entity_name", str(entity)))
            elif isinstance(entity, str):
                entity_str = entity
            else:
                entity_str = str(entity)
            
            item_id = (
                entity_str + "_" + item.get("type", "") + "_" +
                str(entity.get("neo4j_node_id", entity.get("id", "")) if isinstance(entity, dict) else "")
            )
            
            if item_id not in seen_ids:
                seen_ids.add(item_id)
                merged.append(item)
        
        merged.sort(key=lambda x: x.get("score", 0), reverse=True)

        after_count = len(merged[:self._consult_config.knowledge_merge_limit])
        logger.info(f"[KnowledgeRetrievalStrategy._merge_and_deduplicate] 去重完成: before={before_count}, after={after_count}")

        return merged[:self._consult_config.knowledge_merge_limit]
    
    def _build_result(
        self,
        context: KnowledgeRetrievalContextBody
    ) -> KnowledgeRetrievalResultData:
        """构建最终结果"""
        merged_results = self._merge_and_deduplicate(context.all_results)
        logger.info(f"[KnowledgeRetrievalStrategy._build_result] 结果构建完成: "
                   f"merged_results={len(merged_results)}, anchored_entities={len(context.anchored_entities)}, "
                   f"anchored_relations={len(context.anchored_relations)}, total_steps={context.current_step}, "
                   f"sufficiency_score={context.sufficiency_score:.2f}, is_sufficient={context.is_sufficient}")

        return KnowledgeRetrievalResultData(
            merged_results=merged_results,
            anchored_entities=context.anchored_entities,
            anchored_relations=context.anchored_relations,
            total_steps=context.current_step,
            sufficiency_score=context.sufficiency_score,
            is_sufficient=context.is_sufficient,
            degraded=context.degraded,
            degraded_reason=context.degraded_reason,
            error_code=context.error_code,
            error_message=context.error_message
        )


def _get_knowledge_trigger(from_state: str, to_state: str) -> str:
    """Derive a short snake_case trigger for a knowledge retrieval sub-state transition."""
    triggers = {
        ("Thought", "Action"): "decision_made",
        ("Thought", "Finish"): "finish_decided",
        ("Action", "Observation"): "action_executed",
        ("Action", "Finish"): "action_finish",
        ("Observation", "Thought"): "observation_complete",
        ("Observation", "Finish"): "observation_finish",
    }
    return triggers.get((from_state, to_state), "state_handler")


def _get_knowledge_reason(from_state: str, to_state: str, context) -> str:
    """Derive a brief human-readable reason for a knowledge retrieval sub-state transition."""
    from src.orchestration.agent.knowledge_retrieval_strategy.knowledge_retrieval_context import KnowledgeRetrievalContextBody

    if not isinstance(context, KnowledgeRetrievalContextBody):
        return ""

    step = context.current_step
    result_count = len(context.all_results) if context.all_results else 0

    reasons = {
        ("Thought", "Action"): f"step={step},action_selected",
        ("Thought", "Finish"): f"step={step},results={result_count}",
        ("Action", "Observation"): f"step={step},action_executed",
        ("Action", "Finish"): f"step={step},early_finish",
        ("Observation", "Thought"): f"step={step},results={result_count}",
        ("Observation", "Finish"): f"step={step},sufficient",
    }
    return reasons.get((from_state, to_state), "")
