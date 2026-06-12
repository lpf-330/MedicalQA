"""
接入层健康报告生成控制器模块

该模块定义了ReportController类，是健康报告生成的API接口控制器。
"""

from typing import TYPE_CHECKING, Dict, AsyncGenerator
import uuid
from datetime import datetime
import json
import logging

if TYPE_CHECKING:
    from src.service.report_service import ReportService

from fastapi import HTTPException
from starlette.responses import StreamingResponse

from src.schemas.report_request import ReportRequest
from src.utils.logger import get_logger, set_log_context, clear_log_context, log_arch_event
from src.utils.exception_handler import MedicalQAException, ErrorCode

logger = get_logger(__name__)
business_logger = logging.getLogger('Business')


class ReportController:
    """
    健康报告生成控制器类

    负责处理健康报告生成的API请求，包括请求验证、上下文构建、服务调用等。

    Attributes:
        _report_service (ReportService): 健康报告服务实例
    """

    def __init__(self, report_service: 'ReportService') -> None:
        """
        初始化ReportController实例

        Args:
            report_service: 健康报告服务实例

        Raises:
            ValueError: report_service为None时抛出
        """
        if report_service is None:
            raise ValueError("report_service不能为None")

        self._report_service: 'ReportService' = report_service
        logger.info("ReportController初始化完成")

    @property
    def report_service(self) -> 'ReportService':
        """
        获取报告服务实例（只读属性）

        Returns:
            ReportService: 健康报告服务实例
        """
        return self._report_service

    def generate_report(self, request: ReportRequest) -> StreamingResponse:
        """
        处理健康报告生成请求

        执行请求验证、上下文构建、服务调用，并返回SSE流式响应。

        Args:
            request: 健康报告生成请求数据

        Returns:
            StreamingResponse: SSE流式响应

        Raises:
            HTTPException: 请求参数错误时抛出
        """
        try:
            logger.info("=" * 60)
            logger.info("[ReportController] 收到健康报告生成请求")
            logger.info(f"  request_id: {request.request_id}")
            logger.info("=" * 60)

            logger.info(f"[REQUEST_BODY] request_id={request.request_id}, task_id={request.body.task_id if request.body else ''}, "
                        f"has_monitoring_data={request.body.monitoring_data is not None if request.body else False}, "
                        f"has_user_profile={request.body.user_profile is not None if request.body else False}, "
                        f"session_id={request.get_session_id()}")

            business_logger.info(f"[请求] request_id={request.request_id}, task_id={request.get_task_id()}")

            # 验证请求参数
            self._validate_request(request)
            logger.info("[ReportController] 请求验证通过")

            # 生成session_id（如果没有提供）
            session_id = request.get_session_id()
            if not session_id:
                session_id = self._generate_session_id()

            # 构建AgentContext
            agent_context = self._report_service.build_agent_context(request)
            if not agent_context.session_id:
                agent_context.session_id = session_id

            logger.info(f"[ReportController] 开始流式处理: session_id={session_id}")

            set_log_context(
                session_id=session_id,
                request_id=request.request_id,
                task_id=request.body.task_id if request.body and request.body.task_id else "",
                business_type="report",
            )
            log_arch_event(
                logger,
                component="ReportController",
                stage="CONTROLLER",
                event="generate_report",
                status="start",
                design_id="ARCH-1.1",
            )

            # 调用服务层处理报告生成流
            return StreamingResponse(
                self._log_sse_chunks(self._report_service.process_report_stream(agent_context)),
                media_type="text/event-stream"
            )

        except HTTPException:
            raise

        except MedicalQAException as e:
            logger.error(f"[ReportController] 业务异常: error_code={e.error_code}, message={e.message}")
            business_logger.warning(f"[请求失败] request_id={request.request_id}, error={e.message}")
            return StreamingResponse(
                iter([self._format_sse_error(400, str(e))]),
                media_type="text/event-stream"
            )

        except Exception as e:
            logger.exception(f"[ReportController] 未知异常: {str(e)}")
            business_logger.error(f"[请求异常] request_id={request.request_id}, error={str(e)}")
            return StreamingResponse(
                iter([self._format_sse_error(500, "系统内部错误，请稍后重试")]),
                media_type="text/event-stream"
            )

    def _validate_request(self, request: ReportRequest) -> None:
        """
        验证请求参数

        验证请求体、task_id、monitoring_data、user_profile等参数的有效性。
        验证新的数据结构（6项监测指标，字符串类型的病史字段）。

        Args:
            request: 健康报告生成请求数据

        Raises:
            HTTPException: 参数验证失败时抛出
        """
        # 验证body不为空
        if not request.body:
            logger.warning(f"[VALIDATION_FAIL] field=body, reason=body不能为空, request_id={request.request_id}")
            raise HTTPException(
                status_code=400,
                detail={"error_code": ErrorCode.PARAM_MISSING.value, "error_message": "body不能为空"}
            )

        # 验证task_id不为空
        if not request.body.task_id:
            logger.warning(f"[VALIDATION_FAIL] field=task_id, reason=task_id不能为空, request_id={request.request_id}")
            raise HTTPException(
                status_code=400,
                detail={"error_code": ErrorCode.PARAM_MISSING.value, "error_message": "task_id不能为空"}
            )

        # 验证monitoring_data不为空
        if not request.body.monitoring_data:
            logger.warning(f"[VALIDATION_FAIL] field=monitoring_data, reason=monitoring_data不能为空, request_id={request.request_id}")
            raise HTTPException(
                status_code=400,
                detail={"error_code": ErrorCode.PARAM_MISSING.value, "error_message": "monitoring_data不能为空"}
            )

        # 验证user_profile不为空
        if not request.body.user_profile:
            logger.warning(f"[VALIDATION_FAIL] field=user_profile, reason=user_profile不能为空, request_id={request.request_id}")
            raise HTTPException(
                status_code=400,
                detail={"error_code": ErrorCode.PARAM_MISSING.value, "error_message": "user_profile不能为空"}
            )

        # 验证监测数据字段（至少包含一项监测指标 - 6项指标）
        monitoring_data = request.body.monitoring_data
        has_monitoring_data = (
            monitoring_data.heart_rate is not None or
            monitoring_data.blood_glucose is not None or
            monitoring_data.perfusion_index is not None or
            monitoring_data.blood_oxygen is not None or
            monitoring_data.sleep is not None or
            monitoring_data.blood_pressure is not None
        )

        if not has_monitoring_data:
            logger.warning(f"[VALIDATION_FAIL] field=monitoring_indicators, reason=至少需要包含一项监测指标, request_id={request.request_id}")
            raise HTTPException(
                status_code=400,
                detail={"error_code": ErrorCode.PARAM_INVALID.value, "error_message": "监测数据至少需要包含一项监测指标（心率、血糖、灌注指数、血氧、睡眠、血压）"}
            )

    def _format_sse_message(self, data: str) -> str:
        """
        格式化SSE message事件

        将数据格式化为SSE message事件格式。

        Args:
            data: 要发送的数据内容

        Returns:
            str: SSE格式的message事件字符串
        """
        payload = json.dumps({"content": data}, ensure_ascii=False)
        return f"event: message\ndata: {payload}\n\n"

    def _format_sse_end(self, data: Dict) -> str:
        """
        格式化SSE end事件

        将数据格式化为SSE end事件格式。

        Args:
            data: 要发送的结束数据

        Returns:
            str: SSE格式的end事件字符串
        """
        payload = json.dumps(data, ensure_ascii=False)
        return f"event: end\ndata: {payload}\n\n"

    def _format_sse_error(self, error_code: int, error_message: str) -> str:
        """
        格式化SSE error事件

        将错误信息格式化为SSE error事件格式。

        Args:
            error_code: 错误码
            error_message: 错误消息

        Returns:
            str: SSE格式的error事件字符串
        """
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
        """
        生成唯一会话ID

        使用UUID和时间戳生成唯一的会话标识符。

        Returns:
            str: 唯一的会话ID
        """
        session_id = f"session_{uuid.uuid4().hex[:16]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        logger.info(f"[SESSION_ID_GENERATED] session_id={session_id}")
        return session_id

    def __repr__(self) -> str:
        """
        返回ReportController的字符串表示

        Returns:
            str: ReportController的字符串表示
        """
        return f"ReportController(report_service={self._report_service})"


# FastAPI 路由注册
from fastapi import APIRouter, Request
from src.schemas.report_request import ReportRequest, ReportRequestBody

router = APIRouter(prefix="/api/v1", tags=["report"])


@router.post("/report")
async def generate_report(body: ReportRequestBody, request_id: str = "default", user_id: str = None, request: Request = None):
    """
    健康报告生成API

    Args:
        body: 报告请求体（包含task_id, monitoring_data, user_profile等）
        request_id: 请求ID（可选）
        user_id: 用户ID（可选）

    Returns:
        健康报告生成结果
    """
    report_request = ReportRequest(
        request_id=request_id,
        user_id=user_id,
        body=body
    )
    controller = request.app.state.report_controller
    return controller.generate_report(report_request)
