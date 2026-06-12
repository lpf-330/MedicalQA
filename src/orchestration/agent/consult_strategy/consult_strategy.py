# -*- coding: utf-8 -*-
"""
健康咨询策略

该模块实现ConsultStrategy类，用于健康咨询业务。
基于设计文档《项目业务详细设计v5》第2节的设计实现。

主要功能：
1. 7环节固定流程：INITIAL → QUERY_PARSE → KNOWLEDGE_RETRIEVAL → KNOWLEDGE_INTEGRATION → ANSWER_GENERATION → STREAMING → FINISHED
2. KNOWLEDGE_RETRIEVAL环节使用KnowledgeRetrievalStrategy执行ReAct模式检索
3. 降级策略：Agent失败时回退到顺序检索模式
"""
import copy
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict, List, Optional

from src.orchestration.agent.agent_strategy import AgentStrategy
from src.orchestration.agent.data_classes import AgentContext, AgentResult
from src.orchestration.agent.agent_resource import AgentResource
from src.orchestration.state_machine.state_machine import StateMachine
from src.orchestration.chain.data_classes import ChainContext
from src.orchestration.chain.answer_generation_chain.answer_generation_chain import AnswerGenerationContextBody
from src.orchestration.agent.knowledge_retrieval_strategy.knowledge_retrieval_strategy import (
    KnowledgeRetrievalStrategy,
    MAX_STEPS,
    MAX_PROMPT_CHARS
)
from src.orchestration.agent.knowledge_retrieval_strategy.knowledge_retrieval_context import (
    KnowledgeRetrievalContextBody,
)
from src.orchestration.agent.consult_strategy.consult_context import ConsultContextBody
from src.orchestration.agent.consult_strategy.consult_result import ConsultResultData
from src.config.business.consult_service_config import get_runtime_config
from src.errors import ErrorCode, MilvusUnavailableError, Neo4jConnectionError, LLMServiceError
from src.utils.logger import log_arch_event

logger = logging.getLogger(__name__)


class ConsultStrategy(AgentStrategy[ConsultContextBody, ConsultResultData]):

    # 通过配置类集中管理状态超时时间 - 使用惰性获取模式
    @property
    def _get_consult_config(self):
        return get_runtime_config()

    @property
    def _state_timeouts(self):
        return get_runtime_config().state_timeouts

    def __init__(self, knowledge_retrieval_strategy: Optional[KnowledgeRetrievalStrategy] = None):
        self._knowledge_retrieval_strategy = knowledge_retrieval_strategy

    def execute(
        self,
        context: AgentContext[ConsultContextBody],
        resource: AgentResource
    ) -> AgentResult[ConsultResultData]:
        start_time = time.time()
        logger.info(f"[ConsultStrategy] 开始执行策略: session_id={context.session_id}")
        log_arch_event(
            logger,
            component="ConsultStrategy",
            stage="ORCHESTRATION",
            event="strategy_execute",
            status="start",
            design_id="BIZ-2.2",
        )

        body = context.body
        if body is None:
            logger.warning(f"[ConsultStrategy] 输入数据为空: session_id={context.session_id}")
            return AgentResult(
                session_id=context.session_id,
                data=ConsultResultData(answer="输入数据为空", confidence=0.0, error_code=ErrorCode.UNKNOWN, error_message="输入数据为空")
            )

        state_machine = StateMachine(context.session_id)
        self._register_state_transitions(state_machine)

        current_state = body.current_state if body.current_state else "INITIAL"
        body.current_state = current_state

        self._state_handlers = {
            "INITIAL": self._handle_initial,
            "QUERY_PARSE": self._handle_query_parse,
            "KNOWLEDGE_RETRIEVAL": self._handle_knowledge_retrieval,
            "KNOWLEDGE_INTEGRATION": self._handle_knowledge_integration,
            "ANSWER_GENERATION": self._handle_answer_generation,
            "STREAMING": self._handle_streaming,
            "FINISHED": self._handle_finished,
            "ERROR": self._handle_error,
        }

        max_iterations = self._get_consult_config.consult_max_iterations
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"[ConsultStrategy] 状态转换: current_state={current_state}, iteration={iteration}")

            handler = self._state_handlers.get(current_state)
            if handler is None:
                logger.error(f"[ConsultStrategy] 未知状态: {current_state}")
                body.current_state = "ERROR"
                break

            timeout = self._state_timeouts.get(current_state)
            try:
                if timeout and current_state not in ("STREAMING", "FINISHED"):
                    handler_body = copy.deepcopy(body)
                    next_state = self._execute_with_timeout(handler, handler_body, resource, timeout)
                    body.__dict__.update(handler_body.__dict__)
                elif timeout:
                    next_state = self._execute_with_timeout(handler, body, resource, timeout)
                else:
                    next_state = handler(body, resource)
                logger.info(f"[STATE_TRANSITION_REASON] current_state={current_state}, next_state={next_state}, reason=状态处理器正常返回")
            except TimeoutError as te:
                logger.error(f"[ConsultStrategy] 状态超时: state={current_state}, timeout={timeout}s")
                next_state = self._handle_timeout(body, current_state, te, resource)
                logger.info(f"[STATE_TRANSITION_REASON] current_state={current_state}, next_state={next_state}, reason=状态超时")
            except Exception as e:
                logger.error(f"[ConsultStrategy] 状态处理异常: state={current_state}, error={str(e)}")
                next_state = self._handle_error(body, e)
                logger.info(f"[STATE_TRANSITION_REASON] current_state={current_state}, next_state={next_state}, reason=状态处理异常:{str(e)}")

            state_machine.transition(current_state, next_state, trigger=_get_transition_trigger(current_state, next_state), reason=_get_transition_reason(current_state, next_state, body))
            current_state = next_state
            body.current_state = current_state

            if current_state in ("FINISHED", "ERROR"):
                break

        if current_state == "ERROR":
            state_machine.transition(current_state, "FINISHED", trigger="error_resolved", reason="error_state_converted_to_finished")
            current_state = "FINISHED"
            body.current_state = current_state

        result_data = self._build_result(body)

        elapsed = time.time() - start_time
        logger.info(f"[ConsultStrategy] 策略执行完成: session_id={context.session_id}, "
                    f"confidence={result_data.confidence}, elapsed={elapsed:.2f}s")

        return AgentResult(session_id=context.session_id, data=result_data)

    def _register_state_transitions(self, state_machine: StateMachine):
        state_machine.add_state_transition("INITIAL", ["QUERY_PARSE", "ERROR"])
        state_machine.add_state_transition("QUERY_PARSE", ["KNOWLEDGE_RETRIEVAL", "ERROR"])
        state_machine.add_state_transition("KNOWLEDGE_RETRIEVAL", ["KNOWLEDGE_INTEGRATION", "ERROR"])
        state_machine.add_state_transition("KNOWLEDGE_INTEGRATION", ["ANSWER_GENERATION", "ERROR"])
        state_machine.add_state_transition("ANSWER_GENERATION", ["STREAMING", "ERROR"])
        state_machine.add_state_transition("STREAMING", ["FINISHED", "ERROR"])
        state_machine.add_state_transition("ERROR", ["FINISHED"])

    def _execute_with_timeout(self, handler, context, resource, timeout_seconds):
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(handler, context, resource)
        timed_out = False
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            timed_out = True
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError(f"State execution timed out after {timeout_seconds} seconds")
        finally:
            if not timed_out:
                executor.shutdown(wait=True)

    def _handle_initial(self, context: ConsultContextBody, resource: AgentResource) -> str:
        stage_start_time = time.time()
        logger.info("[STAGE_ENTER] INITIAL, question=%s", context.question[:200])
        query_text = context.question
        logger.info(f"[ConsultStrategy._handle_initial] INITIAL环节: 接收请求, query_text={query_text[:100]}...")
        logger.info(f"[ConsultStrategy._handle_initial] INITIAL环节: 构建ConsultContext, session_id={context.session_id}")
        logger.info("[ConsultStrategy._handle_initial] INITIAL环节完成, 转入QUERY_PARSE")
        logger.info(f"[STAGE_EXIT] INITIAL, duration={time.time() - stage_start_time:.2f}s, question={context.question[:200]}")
        return "QUERY_PARSE"

    def _handle_query_parse(self, context: ConsultContextBody, resource: AgentResource) -> str:
        stage_start_time = time.time()
        logger.info("[STAGE_ENTER] QUERY_PARSE, question=%s", context.question[:200])
        query_text = context.question
        logger.info(f"[ConsultStrategy._handle_query_parse] QUERY_PARSE环节: 开始问题解析, query_text={query_text[:100]}...")
        
        context.intent_label = "health_consultation"
        context.extracted_entities = []
        context.is_health_consultation = True

        # NLP实体抽取（best-effort，失败不影响主流程）
        self._extract_entities_with_nlp(context, resource)

        if context.conversation_history:
            rewritten = self._resolve_context_reference(query_text, context.conversation_history, resource)
            context.rewritten_query = rewritten
            logger.info(f"[ConsultStrategy._handle_query_parse] QUERY_PARSE环节: 上下文改写, original={query_text[:50]}..., rewritten={context.rewritten_query[:50]}...")
        else:
            context.rewritten_query = query_text
            logger.info(f"[ConsultStrategy._handle_query_parse] QUERY_PARSE环节: 无对话历史, rewritten_query={context.rewritten_query[:50]}...")
        
        logger.info(f"[ConsultStrategy._handle_query_parse] QUERY_PARSE环节完成: intent_label={context.intent_label}, "
                    f"is_health_consultation={context.is_health_consultation}, "
                    f"rewritten_query={context.rewritten_query[:50]}..., "
                    f"extracted_entities_count={len(context.extracted_entities)}")

        if not context.is_health_consultation:
            logger.info(f"[ConsultStrategy._handle_query_parse] 非健康咨询类问题, 降级执行知识检索, intent_label={context.intent_label}")
            context.knowledge_query = context.rewritten_query or context.question
            logger.info(f"[STAGE_EXIT] QUERY_PARSE, duration={time.time() - stage_start_time:.2f}s, intent_label={context.intent_label}, entity_count={len(context.extracted_entities)}, rewritten_query={context.rewritten_query[:200]}")
            return "KNOWLEDGE_RETRIEVAL"

        logger.info("[ConsultStrategy._handle_query_parse] QUERY_PARSE环节完成, 转入KNOWLEDGE_RETRIEVAL")
        logger.info(f"[STAGE_EXIT] QUERY_PARSE, duration={time.time() - stage_start_time:.2f}s, intent_label={context.intent_label}, entity_count={len(context.extracted_entities)}, rewritten_query={context.rewritten_query[:200]}")

        return "KNOWLEDGE_RETRIEVAL"

    def _extract_entities_with_nlp(self, context: ConsultContextBody, resource: AgentResource) -> None:
        """
        通过IntentClassificationHandler进行意图分类，通过NerModelHandler进行实体提取（best-effort，失败不影响主流程）
        """
        nlp_start_time = time.time()
        query_text = context.question

        # 1. 意图分类（ernie-health-zh）
        intent_handler = resource.get_tool_handler("intent_classification_tool")
        if intent_handler is not None:
            try:
                classify_result = intent_handler.call_tool({
                    "method": "classify_intent",
                    "text": query_text
                })
                nlp_intent_label = classify_result.get("intent_label", "")
                nlp_confidence = classify_result.get("confidence", 0.0)
                logger.info(f"[ConsultStrategy._extract_entities_with_nlp] NLP意图分类结果: "
                            f"intent_label={nlp_intent_label}, confidence={nlp_confidence:.4f}, "
                            f"session_id={context.session_id}")

                if nlp_intent_label and nlp_confidence > self._get_consult_config.intent_classification_threshold:
                    context.intent_label = nlp_intent_label
                    context.is_health_consultation = (nlp_intent_label == "health_consultation")
                    logger.info(f"[ConsultStrategy._extract_entities_with_nlp] NLP意图分类已采纳: "
                                f"intent_label={context.intent_label}, "
                                f"is_health_consultation={context.is_health_consultation}, "
                                f"session_id={context.session_id}")
                else:
                    logger.info(f"[ConsultStrategy._extract_entities_with_nlp] NLP意图分类置信度不足(confidence={nlp_confidence:.4f}), "
                                f"保持规则引擎结果, session_id={context.session_id}")
            except Exception as e:
                logger.warning(f"[ConsultStrategy._extract_entities_with_nlp] 意图分类失败: {str(e)}, session_id={context.session_id}")
        else:
            logger.info(f"[ConsultStrategy._extract_entities_with_nlp] intent_classification_tool未注册, 跳过意图分类, session_id={context.session_id}")

        # 2. 实体提取（nlp_raner NER模型）
        ner_handler = resource.get_tool_handler("ner_model_tool")
        if ner_handler is not None:
            try:
                entities_result = ner_handler.call_tool({
                    "method": "extract_entities",
                    "text": query_text
                })
                if entities_result and isinstance(entities_result, list):
                    context.extracted_entities = entities_result
                    logger.info(f"[ConsultStrategy._extract_entities_with_nlp] NER实体提取完成: "
                                f"entity_count={len(entities_result)}, "
                                f"entities={[e.get('entity_name', '') for e in entities_result]}, "
                                f"session_id={context.session_id}")
                else:
                    logger.info(f"[ConsultStrategy._extract_entities_with_nlp] NER实体提取返回空结果, session_id={context.session_id}")
            except Exception as e:
                logger.warning(f"[ConsultStrategy._extract_entities_with_nlp] NER实体提取失败, "
                               f"将使用规则引擎降级, error={str(e)}, session_id={context.session_id}")
        else:
            logger.info(f"[ConsultStrategy._extract_entities_with_nlp] ner_model_tool未注册, 跳过NER实体提取, session_id={context.session_id}")

        nlp_elapsed = time.time() - nlp_start_time
        logger.info(f"[ConsultStrategy._extract_entities_with_nlp] NLP处理完成: "
                    f"intent_label={context.intent_label}, "
                    f"extracted_entities_count={len(context.extracted_entities)}, "
                    f"elapsed={nlp_elapsed:.3f}s, session_id={context.session_id}")

    def _resolve_context_reference(self, query_text: str, conversation_history: List[Dict[str, str]], resource: AgentResource) -> str:
        if not conversation_history:
            return query_text

        has_reference = any(p in query_text for p in ["它", "他", "她", "这个", "那个", "这些", "那些", "其", "该"])
        if not has_reference:
            logger.info("[ConsultStrategy] 查询中无指代词，无需上下文改写")
            return query_text

        referenced_entity = self._extract_referenced_entity(conversation_history)
        if not referenced_entity:
            logger.info("[ConsultStrategy] 未从对话历史中提取到指代实体，使用原始查询")
            return query_text

        rewritten = query_text
        rewritten = rewritten.replace("它的", f"{referenced_entity}的")
        rewritten = rewritten.replace("它", referenced_entity)
        rewritten = rewritten.replace("他的", f"{referenced_entity}的")
        rewritten = rewritten.replace("他", referenced_entity)
        rewritten = rewritten.replace("她的", f"{referenced_entity}的")
        rewritten = rewritten.replace("她", referenced_entity)
        rewritten = rewritten.replace("这个", referenced_entity)
        rewritten = rewritten.replace("那个", referenced_entity)
        rewritten = rewritten.replace("这些", referenced_entity)
        rewritten = rewritten.replace("那些", referenced_entity)
        rewritten = rewritten.replace("其", f"{referenced_entity}的")
        rewritten = rewritten.replace("该", f"{referenced_entity}的")

        if rewritten == query_text:
            logger.info("[ConsultStrategy] 指代替换未生效，使用原始查询")
            return query_text

        logger.info(f"[ConsultStrategy] 上下文改写成功: '{query_text}' -> '{rewritten}'")
        return rewritten

    def _extract_referenced_entity(self, conversation_history: List[Dict[str, str]]) -> Optional[str]:
        import re
        disease_patterns = [
            r'(\S*病)(?:的|了|是|有|和|与|，|。|\s|$)',
            r'(\S*综合征)(?:的|了|是|有|和|与|，|。|\s|$)',
            r'(\S*炎症)(?:的|了|是|有|和|与|，|。|\s|$)',
            r'(\S*症)(?:的|了|是|有|和|与|，|。|\s|$)',
        ]
        entity_candidates = []

        for msg in reversed(conversation_history):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user" and content:
                for pattern in disease_patterns:
                    matches = re.findall(pattern, content)
                    entity_candidates.extend(matches)

        if entity_candidates:
            entity = entity_candidates[0]
            logger.info(f"[ConsultStrategy] 从对话历史提取指代实体: '{entity}'")
            return entity

        return None

    def _fallback_sequential_retrieval(self, context: ConsultContextBody, resource: AgentResource) -> List[Dict]:
        """
        降级策略：顺序检索模式

        先向量检索锚定实体，后图查询做结构化推理增强。
        基于设计文档《项目业务详细设计v5》降级策略设计实现。

        当KnowledgeRetrievalStrategy执行失败或超时时，自动降级为固定流程的顺序检索模式：
        Step1: 向量检索锚定实体（使用VectorRetrievalHandler调用hybrid_search）
        Step2: 基于向量检索锚定的实体进行图查询结构化推理增强（使用Neo4jMedicalHandler）

        Args:
            context: 咨询上下文
            resource: Agent资源类

        Returns:
            降级检索结果列表，格式与KnowledgeRetrievalStrategy.merged_results兼容
        """
        fallback_start_time = time.time()
        logger.warning(f"[ConsultStrategy._fallback_sequential_retrieval] 降级触发: "
                       f"Agent检索失败, 降级策略=顺序检索模式(先向量检索锚定实体,后图查询做结构化推理增强), "
                       f"session_id={context.session_id}")
        logger.warning(f"[DEGRADE_AGENT_TO_SEQUENTIAL] 降级触发: Agent检索失败, "
                       f"降级策略=顺序检索模式, session_id={context.session_id}")
        logger.warning("[DEGRADE_STRATEGY] from=Agent检索 to=顺序检索模式, reason=Agent检索失败")
        logger.warning("[DEGRADE_TRIGGER] reason=Agent检索失败, level=agent_to_sequential, from_state=KNOWLEDGE_RETRIEVAL")

        all_results = []
        anchored_entities = []  # 锚定实体列表，用于Step2图查询
        vector_retrieval_failed = False  # 向量检索是否失败标记

        # Step1: 向量检索锚定实体
        try:
            vector_handler = resource.get_tool_handler("vector_retrieval_tool")
            if vector_handler is not None:
                query = context.rewritten_query or context.question
                search_result = vector_handler.call_tool({
                    "query": query,
                    "top_k": self._get_consult_config.sequential_top_k,
                    "collections": ["medical_entity", "entity_attributes", "entity_relations"],
                    "weights": self._get_consult_config.sequential_collection_weights
                })

                results_list = []
                if search_result:
                    if isinstance(search_result, list):
                        results_list = search_result
                    elif isinstance(search_result, dict):
                        results_list = search_result.get("results", [])
                        if not results_list:
                            for key, value in search_result.items():
                                if isinstance(value, list):
                                    results_list = value
                                    break

                for item in results_list:
                    if isinstance(item, dict):
                        if "source" not in item:
                            item["source"] = "vector"
                        all_results.append(item)
                        # 收集锚定实体（来自medical_entity集合的结果）
                        collection = item.get("collection", item.get("source", ""))
                        if collection == "medical_entity":
                            anchored_entities.append(item)

                logger.info(f"[ConsultStrategy._fallback_sequential_retrieval] Step1向量检索完成: "
                           f"results={len(all_results)}, anchored_entities={len(anchored_entities)}, "
                           f"session_id={context.session_id}")
            else:
                vector_retrieval_failed = True
                logger.warning(f"[ConsultStrategy._fallback_sequential_retrieval] vector_retrieval_tool未注册, "
                              f"跳过向量检索, session_id={context.session_id}")
        except Exception as e:
            vector_retrieval_failed = True
            logger.error(f"[ConsultStrategy._fallback_sequential_retrieval] Step1向量检索失败: "
                        f"error={str(e)}, session_id={context.session_id}")

        # 降级策略：Milvus不可用 -> Neo4j模糊匹配
        if vector_retrieval_failed:
            all_results = self._degrade_milvus_to_neo4j_fuzzy_match(context, resource, all_results)

        # Step2: 图查询做结构化推理增强(基于向量检索锚定的实体)
        graph_query_failed = False
        try:
            neo4j_handler = resource.get_tool_handler("neo4j_medical_tool")
            if neo4j_handler is not None and anchored_entities:
                # 从锚定实体中提取neo4j_node_id
                entity_ids = []
                for entity_item in anchored_entities[:self._get_consult_config.anchored_entity_limit]:
                    entity_data = entity_item.get("entity", entity_item)
                    if isinstance(entity_data, dict):
                        node_id = entity_data.get("neo4j_node_id")
                        if node_id is not None:
                            entity_ids.append(node_id)

                seen_node_ids = set()
                for neo4j_node_id in entity_ids:
                    # elementId()返回字符串类型，直接使用
                    node_id = str(neo4j_node_id)

                    if node_id in seen_node_ids:
                        continue
                    seen_node_ids.add(node_id)

                    try:
                        disease_info = neo4j_handler.get_disease_by_node_id(node_id)
                        if disease_info:
                            disease_name = disease_info.get("name", "")
                            symptoms = neo4j_handler.get_symptoms_by_node_id(node_id)
                            drugs = neo4j_handler.get_drugs_by_node_id(node_id)
                            foods = neo4j_handler.get_foods_by_node_id(node_id)

                            knowledge_item = {
                                "source": "neo4j",
                                "type": "disease",
                                "entity": disease_name,
                                "data": {
                                    "name": disease_name,
                                    "desc": disease_info.get("desc", ""),
                                    "cause": disease_info.get("cause", ""),
                                    "prevent": disease_info.get("prevent", ""),
                                    "symptoms": symptoms,
                                    "drugs": drugs,
                                    "foods": foods
                                },
                                "score": self._get_consult_config.neo4j_default_score
                            }
                            all_results.append(knowledge_item)
                    except Exception as e:
                        logger.warning(f"[ConsultStrategy._fallback_sequential_retrieval] 图查询node_id={node_id}失败: "
                                      f"error={str(e)}, session_id={context.session_id}")

                logger.info(f"[ConsultStrategy._fallback_sequential_retrieval] Step2图查询完成: "
                           f"entity_ids={len(entity_ids)}, total_results={len(all_results)}, "
                           f"session_id={context.session_id}")
            elif neo4j_handler is None:
                graph_query_failed = True
                logger.warning(f"[ConsultStrategy._fallback_sequential_retrieval] neo4j_medical_tool未注册, "
                              f"跳过图查询, session_id={context.session_id}")
            elif not anchored_entities:
                logger.warning(f"[ConsultStrategy._fallback_sequential_retrieval] 无锚定实体, "
                              f"跳过图查询, session_id={context.session_id}")
        except Exception as e:
            graph_query_failed = True
            logger.error(f"[ConsultStrategy._fallback_sequential_retrieval] Step2图查询失败: "
                        f"error={str(e)}, session_id={context.session_id}")

        # 降级策略：Neo4j不可用 -> 仅使用向量检索结果
        if graph_query_failed:
            all_results = self._degrade_neo4j_to_vector_only(context, all_results)

        # 去重和排序
        merged_results = self._merge_and_deduplicate_fallback_results(all_results)

        fallback_elapsed = time.time() - fallback_start_time
        logger.info(f"[ConsultStrategy._fallback_sequential_retrieval] 降级顺序检索完成: "
                   f"results={len(merged_results)}, elapsed={fallback_elapsed:.2f}s, "
                   f"session_id={context.session_id}")

        return merged_results

    def _degrade_milvus_to_neo4j_fuzzy_match(
        self,
        context: ConsultContextBody,
        resource: AgentResource,
        existing_results: List[Dict]
    ) -> List[Dict]:
        """
        降级策略：Milvus不可用 -> Neo4j模糊匹配

        当向量检索（Milvus）不可用时，显式调用Neo4j的search_diseases_by_symptom方法，
        通过症状关键词在知识图谱中进行模糊匹配，替代向量语义检索。

        Args:
            context: 咨询上下文
            resource: Agent资源类
            existing_results: 已有的检索结果列表

        Returns:
            降级后的检索结果列表
        """
        logger.warning(f"[DEGRADE_MILVUS_TO_NEO4J] 降级触发: Milvus不可用, "
                      f"降级策略=Neo4j模糊匹配(search_diseases_by_symptom), "
                      f"session_id={context.session_id}")
        logger.warning("[DEGRADE_STRATEGY] from=Milvus to=Neo4j模糊匹配, reason=Milvus不可用")
        logger.warning("[DEGRADE_TRIGGER] reason=Milvus不可用, level=milvus_to_neo4j, from_state=KNOWLEDGE_RETRIEVAL")

        degraded_results = list(existing_results)

        try:
            neo4j_handler = resource.get_tool_handler("neo4j_medical_tool")
            if neo4j_handler is not None:
                # 从查询文本中提取症状关键词
                query = context.rewritten_query or context.question
                # 从已有实体中提取症状名称
                symptom_keywords = []
                for entity in context.extracted_entities:
                    entity_name = entity.get("entity_name", "")
                    if entity_name:
                        symptom_keywords.append(entity_name)

                # 如果没有提取到实体，使用查询文本中的关键词
                if not symptom_keywords:
                    # 简单分词：提取查询中的中文词组
                    import re
                    symptom_keywords = re.findall(r'[\u4e00-\u9fff]{2,}', query)
                    symptom_keywords = symptom_keywords[:self._get_consult_config.symptom_keyword_limit]  # 限制关键词数量

                # 调用Neo4j的search_diseases_by_symptom进行模糊匹配
                for symptom_name in symptom_keywords[:self._get_consult_config.fuzzy_match_symptom_limit]:
                    try:
                        disease_names = neo4j_handler.search_diseases_by_symptom(symptom_name)
                        if disease_names and isinstance(disease_names, list):
                            for disease_name in disease_names[:self._get_consult_config.disease_per_symptom_limit]:
                                # 获取疾病详细信息
                                try:
                                    disease_info = neo4j_handler.get_disease_info(disease_name)
                                    if disease_info:
                                        knowledge_item = {
                                            "source": "neo4j_degraded",
                                            "type": "disease_info",
                                            "entity": disease_name,
                                            "data": disease_info,
                                            "score": self._get_consult_config.neo4j_degraded_score,
                                            "_degraded": True,
                                            "_degraded_reason": "Milvus不可用,Neo4j模糊匹配替代"
                                        }
                                        degraded_results.append(knowledge_item)
                                except Exception as e:
                                    logger.debug(f"[DEGRADE_MILVUS_TO_NEO4J] 获取疾病详情失败: "
                                                f"disease={disease_name}, error={str(e)}")
                    except Exception as e:
                        logger.debug(f"[DEGRADE_MILVUS_TO_NEO4J] search_diseases_by_symptom失败: "
                                    f"symptom={symptom_name}, error={str(e)}")

                logger.info(f"[DEGRADE_MILVUS_TO_NEO4J] Neo4j模糊匹配完成: "
                           f"symptom_keywords={symptom_keywords}, "
                           f"results={len(degraded_results)}, "
                           f"session_id={context.session_id}")
            else:
                logger.warning(f"[DEGRADE_MILVUS_TO_NEO4J] neo4j_medical_tool也不可用, "
                              f"无法执行Neo4j模糊匹配, session_id={context.session_id}")
        except Exception as e:
            logger.error(f"[DEGRADE_MILVUS_TO_NEO4J] Neo4j模糊匹配失败: "
                        f"error={str(e)}, session_id={context.session_id}")

        return degraded_results

    def _degrade_neo4j_to_vector_only(
        self,
        context: ConsultContextBody,
        existing_results: List[Dict]
    ) -> List[Dict]:
        """
        降级策略：Neo4j不可用 -> 仅使用向量检索结果

        当图查询（Neo4j）不可用时，显式跳过图查询，
        仅使用向量检索结果，并添加降级标记。

        Args:
            context: 咨询上下文
            existing_results: 已有的检索结果列表（仅包含向量检索结果）

        Returns:
            降级后的检索结果列表（添加降级标记）
        """
        logger.warning(f"[DEGRADE_NEO4J_TO_VECTOR_ONLY] 降级触发: Neo4j不可用, "
                      f"降级策略=仅使用向量检索结果, "
                      f"existing_results={len(existing_results)}, "
                      f"session_id={context.session_id}")
        logger.warning("[DEGRADE_STRATEGY] from=Neo4j to=仅向量检索, reason=Neo4j不可用")
        logger.warning("[DEGRADE_TRIGGER] reason=Neo4j不可用, level=neo4j_to_vector_only, from_state=KNOWLEDGE_RETRIEVAL")

        # 为向量检索结果添加降级标记
        for item in existing_results:
            if isinstance(item, dict):
                item["_degraded"] = True
                item["_degraded_reason"] = "Neo4j不可用,仅使用向量检索结果"

        logger.info(f"[DEGRADE_NEO4J_TO_VECTOR_ONLY] 降级处理完成: "
                   f"results={len(existing_results)}, "
                   f"session_id={context.session_id}")

        return existing_results

    def _merge_and_deduplicate_fallback_results(self, results: List[Dict]) -> List[Dict]:
        """合并去重降级检索结果"""
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

        after_count = len(merged[:self._get_consult_config.knowledge_merge_limit])
        logger.info(f"[ConsultStrategy._merge_and_deduplicate_fallback_results] 去重完成: "
                   f"before={before_count}, after={after_count}")

        return merged[:self._get_consult_config.knowledge_merge_limit]

    def _handle_knowledge_retrieval(self, context: ConsultContextBody, resource: AgentResource) -> str:
        """
        KNOWLEDGE_RETRIEVAL状态：使用KnowledgeRetrievalStrategy执行ReAct模式检索

        基于设计文档《项目业务详细设计v5》第2.3节设计实现。

        核心特点：
        - LLM作为决策者：大语言模型根据当前上下文动态决定下一步检索操作
        - 动态策略调整：根据检索结果实时调整检索策略
        - 限制机制：MAX_STEPS=5, MAX_PROMPT_CHARS=4000
        - 降级保障：Agent失败时自动回退到顺序检索模式
        """
        stage_start_time = time.time()
        logger.info("[STAGE_ENTER] KNOWLEDGE_RETRIEVAL, rewritten_query=%s, intent_label=%s", context.rewritten_query[:200] if context.rewritten_query else context.question[:200], context.intent_label)
        logger.info(f"[ConsultStrategy._handle_knowledge_retrieval] KNOWLEDGE_RETRIEVAL环节: MAX_STEPS={MAX_STEPS()}, MAX_PROMPT_CHARS={MAX_PROMPT_CHARS()}, 超时={self._state_timeouts.get('KNOWLEDGE_RETRIEVAL', 20)}s")
        logger.info(f"[ConsultStrategy._handle_knowledge_retrieval] KNOWLEDGE_RETRIEVAL环节: 输入参数 rewritten_query={context.rewritten_query[:50]}..., intent_label={context.intent_label}")

        try:
            knowledge_retrieval_agent = self._knowledge_retrieval_strategy or KnowledgeRetrievalStrategy()

            agent_context = AgentContext(
                session_id=context.session_id,
                current_state="Thought",
                body=KnowledgeRetrievalContextBody(
                    query_text=context.rewritten_query or context.question,
                    extracted_entities=context.extracted_entities,
                    intent_label=context.intent_label
                )
            )

            agent_result = knowledge_retrieval_agent.execute(agent_context, resource)

            if agent_result.data is None:
                logger.error("[ConsultStrategy._handle_knowledge_retrieval] KnowledgeRetrievalStrategy返回空结果, 降级为顺序检索模式")
                context.knowledge_results = self._fallback_sequential_retrieval(context, resource)
                context.degraded = True
                context.degraded_reason = "KnowledgeRetrievalStrategy返回空结果"
                logger.info("[DEGRADE_MARK] mark=degraded, propagated_to=context, reason=KnowledgeRetrievalStrategy返回空结果")
            elif agent_result.data.degraded:
                logger.warning(f"[ConsultStrategy._handle_knowledge_retrieval] KNOWLEDGE_RETRIEVAL环节: Agent降级执行, 降级原因={agent_result.data.degraded_reason}")
                context.knowledge_results = agent_result.data.merged_results
                context.degraded = True
                context.degraded_reason = agent_result.data.degraded_reason
                logger.info(f"[DEGRADE_MARK] mark=degraded, propagated_to=context, reason={agent_result.data.degraded_reason}")
            elif agent_result.data.total_steps >= MAX_STEPS() and not agent_result.data.is_sufficient:
                logger.warning(f"[ConsultStrategy._handle_knowledge_retrieval] KNOWLEDGE_RETRIEVAL环节: "
                              f"Agent达到最大步数且结果不充分, 降级为顺序检索模式, "
                              f"steps={agent_result.data.total_steps}, sufficiency={agent_result.data.sufficiency_score:.2f}")
                context.knowledge_results = self._fallback_sequential_retrieval(context, resource)
                context.degraded = True
                context.degraded_reason = f"Agent达到最大步数(MAX_STEPS={MAX_STEPS()})且结果不充分"
                logger.info(f"[DEGRADE_MARK] mark=degraded, propagated_to=context, reason=Agent达到最大步数(MAX_STEPS={MAX_STEPS()})且结果不充分")
            else:
                context.knowledge_results = agent_result.data.merged_results

            sufficiency_score_str = f"{agent_result.data.sufficiency_score:.2f}" if agent_result.data else "0"
            logger.info(f"[ConsultStrategy._handle_knowledge_retrieval] KNOWLEDGE_RETRIEVAL环节完成: "
                       f"knowledge_results={len(context.knowledge_results)}, "
                       f"steps={agent_result.data.total_steps if agent_result.data else 0}, "
                       f"sufficiency_score={sufficiency_score_str}, "
                       f"is_sufficient={agent_result.data.is_sufficient if agent_result.data else False}, "
                       f"degraded={context.degraded}")

        except Exception as e:
            logger.error(f"[ConsultStrategy._handle_knowledge_retrieval] KnowledgeRetrievalStrategy执行失败, "
                        f"降级为顺序检索模式: error={str(e)}")
            context.knowledge_results = self._fallback_sequential_retrieval(context, resource)
            context.degraded = True
            context.degraded_reason = f"Agent执行失败: {str(e)}"
            logger.info(f"[DEGRADE_MARK] mark=degraded, propagated_to=context, reason=Agent执行失败: {str(e)}")
            logger.info(f"[ConsultStrategy._handle_knowledge_retrieval] 降级顺序检索完成: "
                       f"knowledge_results={len(context.knowledge_results)}, degraded={context.degraded}")

        logger.info("[ConsultStrategy._handle_knowledge_retrieval] KNOWLEDGE_RETRIEVAL环节完成, 转入KNOWLEDGE_INTEGRATION")
        logger.info(f"[STAGE_EXIT] KNOWLEDGE_RETRIEVAL, duration={time.time() - stage_start_time:.2f}s, knowledge_count={len(context.knowledge_results)}, degraded={context.degraded}")

        return "KNOWLEDGE_INTEGRATION"

    def _handle_knowledge_integration(self, context: ConsultContextBody, resource: AgentResource) -> str:
        stage_start_time = time.time()
        logger.info("[STAGE_ENTER] KNOWLEDGE_INTEGRATION, knowledge_results_count=%d", len(context.knowledge_results))
        logger.info(f"[ConsultStrategy._handle_knowledge_integration] KNOWLEDGE_INTEGRATION环节: 开始知识整合, knowledge_results_count={len(context.knowledge_results)}")

        # Filter low-relevance knowledge - 通过配置类集中管理阈值
        _threshold = self._get_consult_config.knowledge_integration_threshold
        original_count = len(context.knowledge_results)
        filtered_results = [r for r in context.knowledge_results if r.get("score", 0.0) >= _threshold]
        filtered_count = len(filtered_results)
        removed_count = original_count - filtered_count
        logger.info(f"[KNOWLEDGE_INTEGRATION_FILTER] 原始知识数={original_count}, 过滤后={filtered_count}, 移除={removed_count} (阈值={_threshold})")

        knowledge_parts = []
        sources_list = []

        for item in filtered_results:
            source = item.get("source", "")
            item_type = item.get("type", "")
            entity = item.get("entity", "")
            data = item.get("data", {})
            score = item.get("score", 0.0)

            source_info = {
                "source": source,
                "entity": entity,
                "type": item_type,
                "confidence": score if score > 0 else self._get_consult_config.source_default_confidence
            }
            sources_list.append(source_info)

            if source == "neo4j":
                if isinstance(data, dict):
                    if item_type == "disease_info":
                        name = data.get("name", entity)
                        desc = data.get("description", "")
                        knowledge_parts.append(f'疾病名称：{name}\n描述："{desc}"')
                    elif item_type == "symptoms":
                        symptoms_list = data if isinstance(data, list) else [data]
                        symptoms_text = "、".join([s.get("name", str(s)) if isinstance(s, dict) else str(s) for s in symptoms_list])
                        knowledge_parts.append(f'疾病：{entity}的症状："{symptoms_text}"')
                    elif item_type == "drugs":
                        drugs_list = data if isinstance(data, list) else [data]
                        drugs_text = "、".join([d.get("name", str(d)) if isinstance(d, dict) else str(d) for d in drugs_list])
                        knowledge_parts.append(f'疾病：{entity}的常用药物："{drugs_text}"')
                    elif item_type == "foods":
                        foods_list = data if isinstance(data, list) else [data]
                        foods_text = "、".join([f.get("name", str(f)) if isinstance(f, dict) else str(f) for f in foods_list])
                        knowledge_parts.append(f'疾病：{entity}的饮食建议："{foods_text}"')
                    elif item_type == "possible_diseases":
                        diseases_list = data if isinstance(data, list) else [data]
                        diseases_text = "、".join([d.get("name", str(d)) if isinstance(d, dict) else str(d) for d in diseases_list])
                        knowledge_parts.append(f'可能相关的疾病："{diseases_text}"')
                    else:
                        knowledge_parts.append(f'知识来源：{entity} - "{data}"')
                else:
                    knowledge_parts.append(f'知识来源：{entity} - "{data}"')
            elif source == "vector":
                inner_data = data if isinstance(data, dict) else {}
                text = inner_data.get("text", inner_data.get("content", ""))
                if not text:
                    text = item.get("description", item.get("content", ""))
                if not text and isinstance(entity, dict):
                    text = entity.get("attribute_value", entity.get("desc", entity.get("relation_description", "")))
                    if not text:
                        text = str(entity)
                if not text:
                    text = str(data) if data else ""
                knowledge_parts.append(f'相关知识（相关度：{score:.2f}）："{text}"')
            else:
                knowledge_parts.append(f'知识来源：{entity or source} - "{data}"')

        context.knowledge_context = "\n\n".join(knowledge_parts)
        context.sources = sources_list

        logger.info(f"[ConsultStrategy._handle_knowledge_integration] KNOWLEDGE_INTEGRATION环节完成: "
                   f"knowledge_context_len={len(context.knowledge_context)}, sources_count={len(sources_list)}, "
                   f"去重排序过滤完成")
        logger.info("[ConsultStrategy._handle_knowledge_integration] KNOWLEDGE_INTEGRATION环节完成, 转入ANSWER_GENERATION")
        logger.info(f"[STAGE_EXIT] KNOWLEDGE_INTEGRATION, duration={time.time() - stage_start_time:.2f}s, knowledge_context_len={len(context.knowledge_context)}, sources_count={len(sources_list)}")

        return "ANSWER_GENERATION"

    def _handle_answer_generation(self, context: ConsultContextBody, resource: AgentResource) -> str:
        stage_start_time = time.time()
        logger.info("[STAGE_ENTER] ANSWER_GENERATION, knowledge_context_len=%d", len(context.knowledge_context))
        logger.info(f"[ConsultStrategy._handle_answer_generation] ANSWER_GENERATION环节: 开始回答生成, knowledge_context_len={len(context.knowledge_context)}")
        answer_chain = resource.get_chain("answer_generation_chain")
        if answer_chain is None:
            logger.error("[ConsultStrategy._handle_answer_generation] 回答生成链未注册")
            raise ValueError("回答生成链未注册")
        
        chain_context = ChainContext(
            session_id=context.session_id,
            body=AnswerGenerationContextBody(
                query_text=context.rewritten_query or context.question,
                knowledge_context=context.knowledge_context,
                intent_label=context.intent_label,
                chat_history=context.conversation_history
            )
        )
        
        context.stream_generator = answer_chain.execute_stream(chain_context)
        context.is_streaming = True
        logger.info("[ConsultStrategy._handle_answer_generation] ANSWER_GENERATION环节完成: 流式生成器已创建, 转入STREAMING")
        logger.info(f"[STAGE_EXIT] ANSWER_GENERATION, duration={time.time() - stage_start_time:.2f}s, is_streaming={context.is_streaming}")

        return "STREAMING"

    def _handle_streaming(self, context: ConsultContextBody, resource: AgentResource) -> str:
        stage_start_time = time.time()
        logger.info("[STAGE_ENTER] STREAMING, is_streaming=%s", context.is_streaming)
        logger.info("[ConsultStrategy._handle_streaming] STREAMING环节: 流式输出状态, 转入FINISHED")
        logger.info(f"[STAGE_EXIT] STREAMING, duration={time.time() - stage_start_time:.2f}s, answer_length={len(context.answer_text) if context.answer_text else 0}")
        return "FINISHED"

    def _handle_finished(self, context: ConsultContextBody, resource: AgentResource) -> str:
        stage_start_time = time.time()
        logger.info("[STAGE_ENTER] FINISHED, session_id=%s", context.session_id)
        logger.info(f"[ConsultStrategy._handle_finished] FINISHED环节: 策略执行结束, session_id={context.session_id}")
        logger.info(f"[STAGE_EXIT] FINISHED, duration={time.time() - stage_start_time:.2f}s, answer_length={len(context.answer_text) if context.answer_text else 0}")
        return "FINISHED"

    def _handle_error(self, context: ConsultContextBody, error: Exception) -> str:
        logger.error(f"[ConsultStrategy._handle_error] ERROR环节: error_type={type(error).__name__}, message={str(error)}")

        error_message = str(error)
        context.error_message = error_message

        if isinstance(error, MilvusUnavailableError):
            logger.warning(f"[ConsultStrategy._handle_error] 降级触发: Milvus向量库不可用, 降级策略=仅使用Neo4j模糊匹配替代, 降级原因={error_message}")
            logger.warning("[DEGRADE_TRIGGER] reason=Milvus向量库不可用, level=milvus_to_neo4j, from_state=ERROR")
            context.error_code = ErrorCode.MILVUS_UNAVAILABLE
        elif isinstance(error, Neo4jConnectionError):
            logger.warning(f"[ConsultStrategy._handle_error] 降级触发: Neo4j数据库不可用, 降级策略=仅使用向量检索结果, 降级原因={error_message}")
            logger.warning("[DEGRADE_TRIGGER] reason=Neo4j数据库不可用, level=neo4j_to_vector_only, from_state=ERROR")
            context.error_code = ErrorCode.NEO4J_UNAVAILABLE
        elif isinstance(error, LLMServiceError):
            logger.warning(f"[ConsultStrategy._handle_error] 降级触发: LLM调用失败, 降级策略=使用预设模板生成简化回答, 降级原因={error_message}")
            logger.warning("[DEGRADE_TRIGGER] reason=LLM调用失败, level=llm_to_template, from_state=ERROR")
            context.error_code = ErrorCode.LLM_FAILURE
            context.degraded = True
            context.degraded_reason = "LLM调用失败，降级为模板回答"
            logger.info(f"[DEGRADE_MARK] mark=degraded, propagated_to=context, reason={context.degraded_reason}")
            context.answer_text = self._generate_template_answer(context)
        else:
            context.error_code = ErrorCode.UNKNOWN
            if not context.answer_text:
                context.answer_text = f"抱歉，处理过程中出现错误，请稍后重试。错误信息：{error_message}"

        return "ERROR"

    def _handle_timeout(self, context: ConsultContextBody, state: str, error: TimeoutError, resource: AgentResource) -> str:
        logger.warning(f"[ConsultStrategy._handle_timeout] 降级触发: 状态{state}执行超时, 降级策略根据状态选择")

        if state == "INITIAL":
            logger.warning("[ConsultStrategy._handle_timeout] 降级触发: INITIAL超时, 降级策略=返回错误码40001")
            logger.warning(f"[DEGRADE_TRIGGER] reason=INITIAL超时, level=timeout_error, from_state={state}")
            context.error_code = ErrorCode.CONSULT_INITIAL_TIMEOUT
            context.error_message = "请求初始化超时"
            context.answer_text = "抱歉，请求初始化超时，请稍后重试。"
            return "ERROR"
        elif state == "QUERY_PARSE":
            logger.warning("[ConsultStrategy._handle_timeout] 降级触发: QUERY_PARSE超时, 降级策略=返回错误码40002")
            logger.warning(f"[DEGRADE_TRIGGER] reason=QUERY_PARSE超时, level=timeout_error, from_state={state}")
            context.error_code = ErrorCode.CONSULT_QUERY_PARSE_TIMEOUT
            context.error_message = "意图解析超时"
            context.answer_text = "抱歉，请求处理超时，请稍后重试。"
            return "ERROR"
        elif state == "KNOWLEDGE_RETRIEVAL":
            logger.warning("[ConsultStrategy._handle_timeout] 降级触发: KNOWLEDGE_RETRIEVAL超时, 降级策略=Agent失败降级为顺序检索模式")
            logger.warning(f"[DEGRADE_TRIGGER] reason=KNOWLEDGE_RETRIEVAL超时, level=agent_to_sequential, from_state={state}")
            context.error_code = ErrorCode.CONSULT_KNOWLEDGE_RETRIEVAL_TIMEOUT
            context.error_message = "知识检索超时，降级为顺序检索模式"
            # 先尝试顺序检索
            try:
                fallback_results = self._fallback_sequential_retrieval(context, resource)
                if fallback_results:
                    context.knowledge_results = fallback_results
                    context.degraded = True
                    context.degraded_reason = "KNOWLEDGE_RETRIEVAL超时，降级为顺序检索模式"
                    logger.info("[DEGRADE_MARK] mark=degraded, propagated_to=context, reason=KNOWLEDGE_RETRIEVAL超时，降级为顺序检索模式")
                    logger.info(f"[ConsultStrategy._handle_timeout] KNOWLEDGE_RETRIEVAL超时降级: "
                               f"顺序检索成功, results={len(fallback_results)}")
                else:
                    logger.warning("[ConsultStrategy._handle_timeout] KNOWLEDGE_RETRIEVAL超时降级: "
                                  "顺序检索返回空结果, 使用已有部分结果继续")
                    if not context.knowledge_results:
                        context.knowledge_context = ""
            except Exception as fallback_error:
                logger.error(f"[ConsultStrategy._handle_timeout] KNOWLEDGE_RETRIEVAL超时降级: "
                            f"顺序检索也失败, error={str(fallback_error)}, 使用已有部分结果继续")
                if not context.knowledge_results:
                    context.knowledge_context = ""
            return "KNOWLEDGE_INTEGRATION"
        elif state == "ANSWER_GENERATION":
            logger.warning("[ConsultStrategy._handle_timeout] 降级触发: ANSWER_GENERATION超时, 降级策略=使用预设模板生成简化回答")
            logger.warning(f"[DEGRADE_TRIGGER] reason=ANSWER_GENERATION超时, level=llm_to_template, from_state={state}")
            context.error_code = ErrorCode.CONSULT_ANSWER_GENERATION_TIMEOUT
            context.error_message = "回答生成超时，使用模板回答"
            context.degraded = True
            context.degraded_reason = "ANSWER_GENERATION超时，降级为模板回答"
            logger.info(f"[DEGRADE_MARK] mark=degraded, propagated_to=context, reason={context.degraded_reason}")
            context.answer_text = self._generate_template_answer(context)
            return "ERROR"
        else:
            context.error_code = ErrorCode.CONSULT_OTHER_TIMEOUT
            context.error_message = f"状态{state}执行超时"
            return "ERROR"

    def _build_result(self, context: ConsultContextBody) -> ConsultResultData:
        result_data = ConsultResultData(
            answer=context.answer_text,
            session_id=context.session_id,
            sources=context.sources,
            is_health_consultation=context.is_health_consultation,
            error_code=context.error_code,
            error_message=context.error_message
        )

        if context.answer_text:
            result_data.word_count = len(context.answer_text)

        if context.knowledge_results:
            result_data.related_knowledge = [
                item.get("entity", "") or item.get("type", "")
                for item in context.knowledge_results
                if item.get("entity") or item.get("type")
            ]

        result_data.suggestions = self._generate_suggestions(context.question, result_data.related_knowledge)
        result_data.follow_up_questions = self._generate_follow_up_questions(context.question)

        # 通过配置类集中管理置信度阈值
        if context.is_health_consultation and context.knowledge_results:
            result_data.confidence = self._get_consult_config.confidence_high
        elif context.is_health_consultation:
            result_data.confidence = self._get_consult_config.confidence_medium
        else:
            result_data.confidence = self._get_consult_config.confidence_low

        return result_data

    def _generate_suggestions(self, question: str, knowledge: List[str]) -> List[str]:
        suggestions = []

        if "头痛" in question or "头晕" in question:
            suggestions.append("建议保持充足睡眠，避免过度劳累")
            suggestions.append("如症状持续，建议及时就医检查")

        if knowledge:
            suggestions.append("建议了解更多相关知识，做好预防措施")

        if not suggestions:
            suggestions.append("建议保持健康的生活方式")
            suggestions.append("如有不适，请及时就医")

        return suggestions

    def _generate_follow_up_questions(self, question: str) -> List[str]:
        follow_up = []

        if "症状" in question or "不舒服" in question:
            follow_up.append("您的症状持续多长时间了？")
            follow_up.append("是否有其他伴随症状？")

        if "药物" in question or "治疗" in question:
            follow_up.append("您是否对某些药物过敏？")
            follow_up.append("您目前是否正在服用其他药物？")

        if not follow_up:
            follow_up.append("您还有其他健康问题想咨询吗？")

        return follow_up[:self._get_consult_config.follow_up_limit]

    def _generate_template_answer(self, context: ConsultContextBody) -> str:
        template = f"关于您咨询的「{context.question}」：\n\n"
        if context.knowledge_results:
            for item in context.knowledge_results[:self._get_consult_config.template_knowledge_limit]:
                entity = item.get("entity", "")
                data = item.get("data", {})
                if isinstance(data, dict):
                    desc = data.get("description", "")
                    if desc:
                        template += f"- {entity}：{desc}\n"
        template += "\n以上信息仅供参考，不构成医疗建议。如有健康问题，请及时就医。"
        return template


def _get_transition_trigger(from_state: str, to_state: str) -> str:
    """Derive a short snake_case trigger identifier for a state transition."""
    triggers = {
        ("INITIAL", "QUERY_PARSE"): "initial_complete",
        ("QUERY_PARSE", "KNOWLEDGE_RETRIEVAL"): "nlp_complete",
        ("KNOWLEDGE_RETRIEVAL", "KNOWLEDGE_INTEGRATION"): "retrieval_done",
        ("KNOWLEDGE_INTEGRATION", "ANSWER_GENERATION"): "integration_done",
        ("ANSWER_GENERATION", "STREAMING"): "llm_ready",
        ("STREAMING", "FINISHED"): "stream_complete",
    }
    return triggers.get((from_state, to_state), "state_handler")


def _get_transition_reason(from_state: str, to_state: str, context: ConsultContextBody) -> str:
    """Derive a brief human-readable reason for a state transition."""
    reasons = {
        ("INITIAL", "QUERY_PARSE"): "request_received",
        ("QUERY_PARSE", "KNOWLEDGE_RETRIEVAL"): f"intent={context.intent_label}",
        ("KNOWLEDGE_RETRIEVAL", "KNOWLEDGE_INTEGRATION"): f"agent_retrieval_success,knowledge_count={len(context.knowledge_results)}",
        ("KNOWLEDGE_INTEGRATION", "ANSWER_GENERATION"): f"knowledge_integrated,sources={len(context.sources)}",
        ("ANSWER_GENERATION", "STREAMING"): "stream_start",
        ("STREAMING", "FINISHED"): "stream_end",
    }
    return reasons.get((from_state, to_state), "state_handler_return")
