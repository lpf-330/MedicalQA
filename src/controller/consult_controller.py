"""
接入层健康咨询控制器模块

该模块定义了ConsultController类，是健康咨询的API接口控制器。
"""

from typing import TYPE_CHECKING, Dict, AsyncGenerator
import uuid
from datetime import datetime
import json
import logging

if TYPE_CHECKING:
    from src.service.consult_service import ConsultService

from fastapi import HTTPException
from starlette.responses import StreamingResponse

from src.config.business.consult_service_config import get_runtime_config
from src.schemas.consult_request import ConsultRequest
from src.utils.logger import get_logger, set_log_context, clear_log_context, log_arch_event
from src.utils.exception_handler import MedicalQAException, ErrorCode

logger = get_logger(__name__)
business_logger = logging.getLogger('Business')


class ConsultController:

    def __init__(self, consult_service: 'ConsultService') -> None:
        if consult_service is None:
            raise ValueError("consult_service不能为None")

        self._consult_service: 'ConsultService' = consult_service
        logger.info("ConsultController初始化完成")

    @property
    def consult_service(self) -> 'ConsultService':
        return self._consult_service

    def consult(self, request: ConsultRequest) -> StreamingResponse:
        try:
            logger.info("=" * 60)
            logger.info("[ConsultController] 收到健康咨询请求")
            logger.info(f"  request_id: {request.request_id}")
            logger.info("=" * 60)

            question = request.get_question() or ""
            logger.info(f"[REQUEST_BODY] request_id={request.request_id}, task_id={request.body.task_id if request.body else ''}, "
                        f"question_length={len(question)}, "
                        f"chat_history_count={len(request.body.chat_history) if request.body and request.body.chat_history else 0}, "
                        f"session_id={request.get_session_id()}")

            business_logger.info(f"[请求] request_id={request.request_id}, question_length={len(question)}")
            
            self._validate_request(request)
            
            session_id = request.get_session_id()
            if not session_id:
                session_id = self._generate_session_id()
            
            agent_context = self._consult_service.build_agent_context(request)
            if not agent_context.session_id:
                agent_context.session_id = session_id
            
            logger.info(f"[ConsultController] 开始流式处理: session_id={session_id}")

            set_log_context(
                session_id=session_id,
                request_id=request.request_id,
                task_id=request.body.task_id if request.body and request.body.task_id else "",
                business_type="consult",
            )
            log_arch_event(
                logger,
                component="ConsultController",
                stage="CONTROLLER",
                event="consult",
                status="start",
                design_id="ARCH-1.1",
            )

            return StreamingResponse(
                self._log_sse_chunks(self._consult_service.process_consult_stream(agent_context)),
                media_type="text/event-stream"
            )
        
        except HTTPException:
            raise
        
        except MedicalQAException as e:
            logger.error(f"[ConsultController] 业务异常: error_code={e.error_code}, message={e.message}")
            logger.info(f"[API_EXCEPTION] session_id={request.get_session_id()}, exception_type=MedicalQAException, error_code={e.error_code}, message={e.message}")
            business_logger.warning(f"[请求失败] request_id={request.request_id}, error={e.message}")
            return StreamingResponse(
                iter([self._format_sse_error(400, str(e))]),
                media_type="text/event-stream"
            )
        
        except Exception as e:
            logger.exception(f"[ConsultController] 未知异常: {str(e)}")
            logger.info(f"[API_EXCEPTION] session_id={request.get_session_id()}, exception_type={type(e).__name__}, error_code={ErrorCode.SYSTEM_ERROR.value}, message={str(e)}")
            business_logger.error(f"[请求异常] request_id={request.request_id}, error={str(e)}")
            return StreamingResponse(
                iter([self._format_sse_error(500, "系统内部错误，请稍后重试")]),
                media_type="text/event-stream"
            )

    def _validate_request(self, request: ConsultRequest) -> None:
        if not request.body:
            logger.warning(f"[VALIDATION_FAIL] field=body, reason=body不能为空, request_id={request.request_id}")
            raise HTTPException(
                status_code=400,
                detail={"error_code": ErrorCode.PARAM_MISSING.value, "error_message": "body不能为空"}
            )

        if not request.body.task_id:
            logger.warning(f"[VALIDATION_FAIL] field=task_id, reason=task_id不能为空, request_id={request.request_id}")
            raise HTTPException(
                status_code=400,
                detail={"error_code": ErrorCode.PARAM_MISSING.value, "error_message": "task_id不能为空"}
            )

        if request.body.chat_history is None:
            logger.warning(f"[VALIDATION_FAIL] field=chat_history, reason=chat_history不能为空, request_id={request.request_id}")
            raise HTTPException(
                status_code=400,
                detail={"error_code": ErrorCode.PARAM_MISSING.value, "error_message": "chat_history不能为空"}
            )

        if not request.get_question() or not request.get_question().strip():
            logger.warning(f"[VALIDATION_FAIL] field=question, reason=question不能为空, request_id={request.request_id}")
            raise HTTPException(
                status_code=400,
                detail={"error_code": ErrorCode.PARAM_MISSING.value, "error_message": "question不能为空"}
            )

        question = request.get_question()
        max_length = get_runtime_config().question_max_length
        if len(question) > max_length:
            logger.warning(f"[VALIDATION_FAIL] field=question, reason=question长度不能超过{max_length}个字符, request_id={request.request_id}")
            raise HTTPException(
                status_code=400,
                detail={"error_code": ErrorCode.PARAM_INVALID.value, "error_message": f"question长度不能超过{max_length}个字符"}
            )

    def _format_sse_message(self, data: str) -> str:
        payload = json.dumps({"content": data}, ensure_ascii=False)
        return f"event: message\ndata: {payload}\n\n"

    def _format_sse_end(self, data: Dict) -> str:
        payload = json.dumps(data, ensure_ascii=False)
        return f"event: end\ndata: {payload}\n\n"

    def _format_sse_error(self, error_code: int, error_message: str) -> str:
        payload = json.dumps({"error_code": error_code, "error_message": error_message}, ensure_ascii=False)
        return f"event: error\ndata: {payload}\n\n"

    async def _log_sse_chunks(self, generator: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        chunk_count = 0
        try:
            async for chunk in generator:
                chunk_count += 1
                logger.debug(f"[SSE_CHUNK] index={chunk_count}, length={len(chunk)}")
                yield chunk
        finally:
            logger.info(f"[SSE_STREAM_END] total_chunks={chunk_count}")
            clear_log_context()

    def _generate_session_id(self) -> str:
        session_id = f"session_{uuid.uuid4().hex[:16]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        logger.info(f"[SESSION_ID_GENERATED] session_id={session_id}")
        return session_id

    def __repr__(self) -> str:
        return f"ConsultController(consult_service={self._consult_service})"


# FastAPI 路由注册
from fastapi import APIRouter, Request
from src.schemas.consult_request import ConsultRequest, ConsultRequestBody

router = APIRouter(prefix="/api/v1", tags=["consult"])


@router.post("/consult")
async def consult(body: ConsultRequestBody, request_id: str = "default", user_id: str = None, request: Request = None):
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
    controller = request.app.state.consult_controller
    return controller.consult(consult_request)
