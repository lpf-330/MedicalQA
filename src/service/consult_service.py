"""
服务层健康咨询服务模块

该模块定义了ConsultService类，是健康咨询业务的服务类。
"""

import logging
import time
from typing import TYPE_CHECKING, TypeVar, TypeAlias, Any, Dict, List, Generator

if TYPE_CHECKING:
    from src.orchestration.agent.agent import Agent
    from src.orchestration.agent.data_classes import AgentContext, AgentResult

from src.orchestration.agent.data_classes import AgentContext
from src.orchestration.agent.consult_strategy.consult_strategy import ConsultContextBody
from src.schemas.consult_request import ConsultRequest

logger = logging.getLogger(__name__)

T = TypeVar('T')

ConsultContext: TypeAlias = 'AgentContext[Any]'
ConsultResult: TypeAlias = 'AgentResult[Any]'


class ConsultService:

    def __init__(self, agent: 'Agent[Any, Any]') -> None:
        if agent is None:
            raise ValueError("agent不能为None")

        self._agent: 'Agent[Any, Any]' = agent
        logger.info("[ConsultService] 服务初始化完成")

    @property
    def agent(self) -> 'Agent[Any, Any]':
        return self._agent

    def process_consult(self, context: ConsultContext) -> ConsultResult:
        start_time = time.time()

        if context is None:
            logger.error("[ConsultService] context为None")
            raise ValueError("context不能为None")

        if not hasattr(context, 'session_id') or not context.session_id:
            logger.error("[ConsultService] session_id为空")
            raise ValueError("context.session_id不能为空")

        logger.info(f"[ConsultService] 开始处理咨询: session_id={context.session_id}")

        try:
            result = self._agent.run(context)
            return result
        finally:
            agent_resource = self._agent.resources
            if agent_resource is not None and agent_resource.model_service is not None:
                try:
                    agent_resource.model_service.release()
                except Exception as e:
                    logger.error(f"[ConsultService] 释放model service失败: {e}")
            elapsed = time.time() - start_time
            logger.info(f"[ConsultService] 咨询处理完成: session_id={context.session_id}, elapsed={elapsed:.2f}s")

    def process_consult_stream(self, context: ConsultContext) -> Generator[str, None, None]:
        import json
        
        if context is None:
            yield f"event: error\ndata: {json.dumps({'error_code': 400, 'error_message': 'context不能为None'})}\n\n"
            return
        
        if not hasattr(context, 'session_id') or not context.session_id:
            yield f"event: error\ndata: {json.dumps({'error_code': 400, 'error_message': 'session_id不能为空'})}\n\n"
            return
        
        logger.info(f"[ConsultService] 开始流式处理咨询: session_id={context.session_id}")
        
        try:
            result = self._agent.run(context)
            
            if result is not None and result.data is not None:
                body = context.body
                if hasattr(body, 'stream_generator') and body.stream_generator is not None:
                    for token in body.stream_generator:
                        payload = json.dumps({"content": token}, ensure_ascii=False)
                        yield f"event: message\ndata: {payload}\n\n"
                    
                    context.answer_text = getattr(body, 'answer_text', '')
                    context.is_streaming = False
                    
                    end_data = {
                        "session_id": result.session_id,
                        "sources": getattr(body, 'sources', []),
                        "is_health_consultation": getattr(body, 'is_health_consultation', True),
                        "error_code": getattr(body, 'error_code', 0),
                    }
                    yield f"event: end\ndata: {json.dumps(end_data, ensure_ascii=False)}\n\n"
                else:
                    answer = result.data.answer if hasattr(result.data, 'answer') else str(result.data)
                    payload = json.dumps({"content": answer}, ensure_ascii=False)
                    yield f"event: message\ndata: {payload}\n\n"
                    end_data = {"session_id": result.session_id}
                    yield f"event: end\ndata: {json.dumps(end_data, ensure_ascii=False)}\n\n"
            else:
                yield f"event: error\ndata: {json.dumps({'error_code': 500, 'error_message': '处理结果为空'})}\n\n"
        
        except Exception as e:
            logger.error(f"[ConsultService] 流式处理异常: {str(e)}")
            yield f"event: error\ndata: {json.dumps({'error_code': 500, 'error_message': str(e)})}\n\n"
        
        finally:
            agent_resource = self._agent.resources
            if agent_resource is not None and agent_resource.model_service is not None:
                try:
                    agent_resource.model_service.release()
                except Exception as e:
                    logger.error(f"[ConsultService] 释放model service失败: {e}")

    def _build_agent_context(self, request: ConsultRequest) -> AgentContext:
        session_id = request.get_session_id() or request.body.task_id

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

        return agent_context

    def __repr__(self) -> str:
        return f"ConsultService(agent={self._agent})"
