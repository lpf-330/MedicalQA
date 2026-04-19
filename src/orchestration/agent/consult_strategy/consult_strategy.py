# -*- coding: utf-8 -*-
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
from src.orchestration.chain.knowledge_retrieval_chain.knowledge_retrieval_chain import KnowledgeRetrievalContextBody
from src.orchestration.chain.answer_generation_chain.answer_generation_chain import AnswerGenerationContextBody

logger = logging.getLogger(__name__)


@dataclass
class ConsultContextBody:
    question: str
    session_id: str = ""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    current_state: str = "INITIAL"
    extracted_entities: List[Dict] = field(default_factory=list)
    intent_label: str = ""
    knowledge_results: List[Dict] = field(default_factory=list)
    answer_text: str = ""
    sources: List[str] = field(default_factory=list)
    knowledge_context: str = ""
    is_health_consultation: bool = True
    rewritten_query: str = ""
    error_code: int = 0
    error_message: str = ""
    stream_generator: Any = None
    is_streaming: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "session_id": self.session_id,
            "conversation_history": self.conversation_history,
            "user_profile": self.user_profile,
            "current_state": self.current_state,
            "extracted_entities": self.extracted_entities,
            "intent_label": self.intent_label,
            "knowledge_results": self.knowledge_results,
            "answer_text": self.answer_text,
            "sources": self.sources,
            "knowledge_context": self.knowledge_context,
            "is_health_consultation": self.is_health_consultation,
            "rewritten_query": self.rewritten_query,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "stream_generator": self.stream_generator,
            "is_streaming": self.is_streaming
        }


@dataclass
class ConsultResultData:
    answer: str
    suggestions: List[str] = field(default_factory=list)
    related_knowledge: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    confidence: float = 0.0
    session_id: str = ""
    sources: List[str] = field(default_factory=list)
    word_count: int = 0
    is_health_consultation: bool = True
    error_code: int = 0
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "suggestions": self.suggestions,
            "related_knowledge": self.related_knowledge,
            "follow_up_questions": self.follow_up_questions,
            "confidence": self.confidence,
            "session_id": self.session_id,
            "sources": self.sources,
            "word_count": self.word_count,
            "is_health_consultation": self.is_health_consultation,
            "error_code": self.error_code,
            "error_message": self.error_message
        }


class ConsultStrategy(AgentStrategy[ConsultContextBody, ConsultResultData]):

    _STATE_TIMEOUTS = {
        "QUERY_PARSE": 10,
        "KNOWLEDGE_RETRIEVAL": 20,
        "ANSWER_GENERATION": 60,
    }

    def execute(
        self,
        context: AgentContext[ConsultContextBody],
        resource: AgentResource
    ) -> AgentResult[ConsultResultData]:
        start_time = time.time()
        logger.info(f"[ConsultStrategy] 开始执行策略: session_id={context.session_id}")

        body = context.body
        if body is None:
            logger.warning(f"[ConsultStrategy] 输入数据为空: session_id={context.session_id}")
            return AgentResult(
                session_id=context.session_id,
                data=ConsultResultData(answer="输入数据为空", confidence=0.0, error_code=1, error_message="输入数据为空")
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
        }

        max_iterations = 20
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"[ConsultStrategy] 状态转换: current_state={current_state}, iteration={iteration}")

            handler = self._state_handlers.get(current_state)
            if handler is None:
                logger.error(f"[ConsultStrategy] 未知状态: {current_state}")
                body.current_state = "ERROR"
                break

            timeout = self._STATE_TIMEOUTS.get(current_state)
            try:
                if timeout:
                    next_state = self._execute_with_timeout(handler, body, resource, timeout)
                else:
                    next_state = handler(body, resource)
            except TimeoutError as te:
                logger.error(f"[ConsultStrategy] 状态超时: state={current_state}, timeout={timeout}s")
                next_state = self._handle_timeout(body, current_state, te)
            except Exception as e:
                logger.error(f"[ConsultStrategy] 状态处理异常: state={current_state}, error={str(e)}")
                next_state = self._handle_error(body, e)

            state_machine.transition(current_state, next_state)
            current_state = next_state
            body.current_state = current_state

            if current_state in ("FINISHED", "ERROR"):
                break

        if current_state == "ERROR":
            current_state = "FINISHED"
            body.current_state = current_state

        result_data = self._build_result(body)

        elapsed = time.time() - start_time
        logger.info(f"[ConsultStrategy] 策略执行完成: session_id={context.session_id}, "
                    f"confidence={result_data.confidence}, elapsed={elapsed:.2f}s")

        return AgentResult(session_id=context.session_id, data=result_data)

    def _register_state_transitions(self, state_machine: StateMachine):
        state_machine.add_state_transition("INITIAL", ["QUERY_PARSE"])
        state_machine.add_state_transition("QUERY_PARSE", ["KNOWLEDGE_RETRIEVAL", "FINISHED", "ERROR"])
        state_machine.add_state_transition("KNOWLEDGE_RETRIEVAL", ["KNOWLEDGE_INTEGRATION", "ERROR"])
        state_machine.add_state_transition("KNOWLEDGE_INTEGRATION", ["ANSWER_GENERATION", "ERROR"])
        state_machine.add_state_transition("ANSWER_GENERATION", ["STREAMING", "ERROR"])
        state_machine.add_state_transition("STREAMING", ["FINISHED", "ERROR"])
        state_machine.add_state_transition("ERROR", ["FINISHED"])

    def _execute_with_timeout(self, handler, context, resource, timeout_seconds):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(handler, context, resource)
            try:
                return future.result(timeout=timeout_seconds)
            except FuturesTimeoutError:
                raise TimeoutError(f"State execution timed out after {timeout_seconds} seconds")

    def _handle_initial(self, context: ConsultContextBody, resource: AgentResource) -> str:
        query_text = context.question
        logger.info(f"[ConsultStrategy] INITIAL: query_text={query_text[:100]}...")
        return "QUERY_PARSE"

    def _handle_query_parse(self, context: ConsultContextBody, resource: AgentResource) -> str:
        query_text = context.question
        logger.info(f"[ConsultStrategy] QUERY_PARSE: query_text={query_text[:100]}...")
        
        context.intent_label = "health_consultation"
        context.extracted_entities = []
        context.is_health_consultation = True

        if context.conversation_history:
            rewritten = self._resolve_context_reference(query_text, context.conversation_history, resource)
            context.rewritten_query = rewritten
            logger.info(f"[ConsultStrategy] QUERY_PARSE 上下文改写: original={query_text[:50]}..., rewritten={context.rewritten_query[:50]}...")
        else:
            context.rewritten_query = query_text
            logger.info(f"[ConsultStrategy] QUERY_PARSE (无对话历史): rewritten_query={context.rewritten_query[:50]}...")
        
        logger.info(f"[ConsultStrategy] QUERY_PARSE: intent_label={context.intent_label}, "
                    f"is_health_consultation={context.is_health_consultation}, "
                    f"rewritten_query={context.rewritten_query[:50]}...")
        
        return "KNOWLEDGE_RETRIEVAL"

    def _resolve_context_reference(self, query_text: str, conversation_history: List[Dict[str, str]], resource: AgentResource) -> str:
        if not conversation_history:
            return query_text

        has_reference = any(p in query_text for p in ["它", "他", "她", "这个", "那个", "这些", "那些", "其", "该"])
        if not has_reference:
            logger.info(f"[ConsultStrategy] 查询中无指代词，无需上下文改写")
            return query_text

        referenced_entity = self._extract_referenced_entity(conversation_history)
        if not referenced_entity:
            logger.info(f"[ConsultStrategy] 未从对话历史中提取到指代实体，使用原始查询")
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
            logger.info(f"[ConsultStrategy] 指代替换未生效，使用原始查询")
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

    def _handle_knowledge_retrieval(self, context: ConsultContextBody, resource: AgentResource) -> str:
        knowledge_chain = resource.get_chain("knowledge_retrieval_chain")
        if knowledge_chain is None:
            logger.error("[ConsultStrategy] 知识检索链未注册")
            raise ValueError("知识检索链未注册")

        chain_context = ChainContext(
            session_id=context.session_id,
            body=KnowledgeRetrievalContextBody(
                query_text=context.rewritten_query or context.question,
                extracted_entities=context.extracted_entities,
                intent_label=context.intent_label
            )
        )

        chain_result = knowledge_chain.execute(chain_context)
        if chain_result.data is None:
            logger.error("[ConsultStrategy] 知识检索链返回空结果")
            raise ValueError("知识检索链返回空结果")

        context.knowledge_results = chain_result.data.merged_results

        logger.info(f"[ConsultStrategy] KNOWLEDGE_RETRIEVAL: knowledge_results={len(context.knowledge_results)}")

        return "KNOWLEDGE_INTEGRATION"

    def _handle_knowledge_integration(self, context: ConsultContextBody, resource: AgentResource) -> str:
        knowledge_parts = []
        sources_list = []

        for item in context.knowledge_results:
            source = item.get("source", "")
            item_type = item.get("type", "")
            entity = item.get("entity", "")
            data = item.get("data", {})
            score = item.get("score", 0.0)

            # 构建sources字段
            source_info = {
                "source": source,
                "entity": entity,
                "type": item_type,
                "confidence": score if score > 0 else 0.5
            }
            sources_list.append(source_info)

            if source == "neo4j":
                if isinstance(data, dict):
                    if item_type == "disease_info":
                        name = data.get("name", entity)
                        desc = data.get("description", "")
                        knowledge_parts.append(f"疾病名称：{name}\n描述：{desc}")
                    elif item_type == "symptoms":
                        symptoms_list = data if isinstance(data, list) else [data]
                        symptoms_text = "、".join([s.get("name", str(s)) if isinstance(s, dict) else str(s) for s in symptoms_list])
                        knowledge_parts.append(f"疾病：{entity}的症状：{symptoms_text}")
                    elif item_type == "drugs":
                        drugs_list = data if isinstance(data, list) else [data]
                        drugs_text = "、".join([d.get("name", str(d)) if isinstance(d, dict) else str(d) for d in drugs_list])
                        knowledge_parts.append(f"疾病：{entity}的常用药物：{drugs_text}")
                    elif item_type == "foods":
                        foods_list = data if isinstance(data, list) else [data]
                        foods_text = "、".join([f.get("name", str(f)) if isinstance(f, dict) else str(f) for f in foods_list])
                        knowledge_parts.append(f"疾病：{entity}的饮食建议：{foods_text}")
                    elif item_type == "possible_diseases":
                        diseases_list = data if isinstance(data, list) else [data]
                        diseases_text = "、".join([d.get("name", str(d)) if isinstance(d, dict) else str(d) for d in diseases_list])
                        knowledge_parts.append(f"可能相关的疾病：{diseases_text}")
                    else:
                        knowledge_parts.append(f"知识来源：{entity} - {data}")
                else:
                    knowledge_parts.append(f"知识来源：{entity} - {data}")
            elif source == "vector":
                inner_data = data if isinstance(data, dict) else {}
                text = inner_data.get("text", inner_data.get("content", str(data)))
                knowledge_parts.append(f"相关知识（相关度：{score:.2f}）：{text}")
            else:
                knowledge_parts.append(f"知识来源：{entity or source} - {data}")

        context.knowledge_context = "\n\n".join(knowledge_parts)
        context.sources = sources_list

        logger.info(f"[ConsultStrategy] KNOWLEDGE_INTEGRATION: knowledge_context_len={len(context.knowledge_context)}, sources_count={len(sources_list)}")

        return "ANSWER_GENERATION"

    def _handle_answer_generation(self, context: ConsultContextBody, resource: AgentResource) -> str:
        answer_chain = resource.get_chain("answer_generation_chain")
        if answer_chain is None:
            raise ValueError("回答生成链未注册")
        
        chain_context = ChainContext(
            session_id=context.session_id,
            body=AnswerGenerationContextBody(
                query_text=context.question,
                knowledge_context=context.knowledge_context,
                intent_label=context.intent_label,
                chat_history=context.conversation_history
            )
        )
        
        context.stream_generator = answer_chain.execute_stream(chain_context)
        context.is_streaming = True
        
        return "STREAMING"

    def _handle_streaming(self, context: ConsultContextBody, resource: AgentResource) -> str:
        logger.info(f"[ConsultStrategy] STREAMING: 流式输出状态")
        return "FINISHED"

    def _handle_finished(self, context: ConsultContextBody, resource: AgentResource) -> str:
        logger.info(f"[ConsultStrategy] FINISHED: 策略执行结束")
        return "FINISHED"

    def _handle_error(self, context: ConsultContextBody, error: Exception) -> str:
        logger.error(f"[ConsultStrategy] ERROR: error_type={type(error).__name__}, message={str(error)}")

        error_message = str(error)
        context.error_message = error_message

        if "Milvus" in error_message or "milvus" in error_message or "vector" in error_message.lower():
            logger.warning("[ConsultStrategy] 降级策略: Milvus不可用，仅使用Neo4j")
            context.error_code = 1001
        elif "Neo4j" in error_message or "neo4j" in error_message or "graph" in error_message.lower():
            logger.warning("[ConsultStrategy] 降级策略: Neo4j不可用，仅使用向量检索")
            context.error_code = 1002
        elif "LLM" in error_message or "llm" in error_message or "model" in error_message.lower():
            logger.warning("[ConsultStrategy] 降级策略: LLM失败，使用模板回答")
            context.error_code = 1003
            context.answer_text = self._generate_template_answer(context)
        else:
            context.error_code = 9999
            if not context.answer_text:
                context.answer_text = f"抱歉，处理过程中出现错误，请稍后重试。错误信息：{error_message}"

        return "ERROR"

    def _handle_timeout(self, context: ConsultContextBody, state: str, error: TimeoutError) -> str:
        logger.warning(f"[ConsultStrategy] 超时降级: state={state}")

        if state == "QUERY_PARSE":
            context.error_code = 40002
            context.error_message = "意图解析超时"
            context.answer_text = "抱歉，请求处理超时，请稍后重试。"
            return "FINISHED"
        elif state == "KNOWLEDGE_RETRIEVAL":
            logger.warning("[ConsultStrategy] 知识检索超时，使用已有部分结果继续")
            context.error_code = 40003
            context.error_message = "知识检索超时，使用部分结果"
            if not context.knowledge_results:
                context.knowledge_context = ""
            return "ANSWER_GENERATION"
        elif state == "ANSWER_GENERATION":
            logger.warning("[ConsultStrategy] 回答生成超时，降级为模板回答")
            context.error_code = 40004
            context.error_message = "回答生成超时，使用模板回答"
            context.answer_text = self._generate_template_answer(context)
            return "FINISHED"
        else:
            context.error_code = 40005
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

        if context.is_health_consultation and context.knowledge_results:
            result_data.confidence = 0.8
        elif context.is_health_consultation:
            result_data.confidence = 0.5
        else:
            result_data.confidence = 0.3

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

        return follow_up[:3]

    def _generate_template_answer(self, context: ConsultContextBody) -> str:
        template = f"关于您咨询的「{context.question}」：\n\n"
        if context.knowledge_results:
            for item in context.knowledge_results[:3]:
                entity = item.get("entity", "")
                data = item.get("data", {})
                if isinstance(data, dict):
                    desc = data.get("description", "")
                    if desc:
                        template += f"- {entity}：{desc}\n"
        template += "\n以上信息仅供参考，不构成医疗建议。如有健康问题，请及时就医。"
        return template
