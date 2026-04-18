"""
接入层健康咨询控制器模块

该模块定义了ConsultController类，是健康咨询的API接口控制器。
"""

from typing import TYPE_CHECKING, Optional, Dict, Generator
import uuid
from datetime import datetime
import time
import json
import logging

if TYPE_CHECKING:
    from src.service.consult_service import ConsultService
    from src.orchestration.agent.data_classes import AgentContext, AgentResult

from fastapi import HTTPException
from starlette.responses import StreamingResponse

from src.schemas.consult_request import ConsultRequest, ConsultRequestBody
from src.schemas.consult_response import ConsultResponse, ConsultResponseData
from src.utils.logger import get_logger
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
        start_time = time.time()
        
        try:
            logger.info("=" * 60)
            logger.info(f"[ConsultController] 收到健康咨询请求")
            logger.info(f"  request_id: {request.request_id}")
            logger.info("=" * 60)
            
            business_logger.info(f"[请求] request_id={request.request_id}, question={request.get_question()[:100] if request.get_question() else ''}")
            
            self._validate_request(request)
            
            session_id = request.get_session_id()
            if not session_id:
                session_id = self._generate_session_id()
            
            agent_context = self._consult_service._build_agent_context(request)
            if not agent_context.session_id:
                agent_context.session_id = session_id
            
            logger.info(f"[ConsultController] 开始流式处理: session_id={session_id}")
            
            return StreamingResponse(
                self._consult_service.process_consult_stream(agent_context),
                media_type="text/event-stream"
            )
        
        except HTTPException:
            raise
        
        except MedicalQAException as e:
            elapsed = time.time() - start_time
            logger.error(f"[ConsultController] 业务异常: error_code={e.error_code}, message={e.message}")
            business_logger.warning(f"[请求失败] request_id={request.request_id}, error={e.message}")
            return StreamingResponse(
                iter([self._format_sse_error(400, str(e))]),
                media_type="text/event-stream"
            )
        
        except Exception as e:
            elapsed = time.time() - start_time
            logger.exception(f"[ConsultController] 未知异常: {str(e)}")
            business_logger.error(f"[请求异常] request_id={request.request_id}, error={str(e)}")
            return StreamingResponse(
                iter([self._format_sse_error(500, f"系统内部错误: {str(e)}")]),
                media_type="text/event-stream"
            )

    def _validate_request(self, request: ConsultRequest) -> None:
        if not request.body:
            raise HTTPException(
                status_code=400,
                detail={"error_code": ErrorCode.PARAM_MISSING.value, "error_message": "body不能为空"}
            )

        if not request.body.task_id:
            raise HTTPException(
                status_code=400,
                detail={"error_code": ErrorCode.PARAM_MISSING.value, "error_message": "task_id不能为空"}
            )

        if request.body.chat_history is None:
            raise HTTPException(
                status_code=400,
                detail={"error_code": ErrorCode.PARAM_MISSING.value, "error_message": "chat_history不能为空"}
            )

        if not request.get_question() or not request.get_question().strip():
            raise HTTPException(
                status_code=400,
                detail={"error_code": ErrorCode.PARAM_MISSING.value, "error_message": "question不能为空"}
            )

        question = request.get_question()
        if len(question) > 1000:
            raise HTTPException(
                status_code=400,
                detail={"error_code": ErrorCode.PARAM_INVALID.value, "error_message": "question长度不能超过1000个字符"}
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

    def _generate_session_id(self) -> str:
        return f"session_{uuid.uuid4().hex[:16]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    def __repr__(self) -> str:
        return f"ConsultController(consult_service={self._consult_service})"
