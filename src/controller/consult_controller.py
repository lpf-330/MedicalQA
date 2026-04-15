"""
接入层健康咨询控制器模块

该模块定义了ConsultController类，是健康咨询的API接口控制器。
"""

from typing import TYPE_CHECKING, Optional
import uuid
from datetime import datetime
import time
import json

if TYPE_CHECKING:
    from src.service.consult_service import ConsultService
    from src.orchestration.agent.data_classes import AgentContext, AgentResult

from src.schemas.consult_request import ConsultRequest, ConsultRequestBody
from src.schemas.consult_response import ConsultResponse, ConsultResponseData
from src.utils.logger import get_logger
from src.utils.exception_handler import MedicalQAException, ErrorCode

logger = get_logger(__name__)


class ConsultController:
    """
    ConsultController类 - 健康咨询控制器

    健康咨询的API接口方法，负责HTTP协议处理、请求参数校验、协议转换。
    对外提供RESTful API接口，接收总项目后端的请求。

    职责：
        - 对外提供RESTful API接口，接收总项目后端的请求
        - 处理SSE流式响应，实现内容的逐块返回（后续扩展）
        - 请求参数合法性校验、异常统一处理
        - 协议转换，将外部请求转换为内部上下文格式

    使用示例：
        >>> # 创建ConsultService实例（通常在应用启动时创建）
        >>> service = ConsultService(agent=agent)
        >>> # 创建ConsultController实例
        >>> controller = ConsultController(consult_service=service)
        >>> # 处理健康咨询请求
        >>> request = ConsultRequest(
        ...     request_id="req-001",
        ...     body=ConsultRequestBody(question="头痛怎么办？")
        ... )
        >>> response = controller.consult(request)
        >>> print(response.data.result)

    Attributes:
        _consult_service: ConsultService实例，用于处理健康咨询业务逻辑
    """

    def __init__(self, consult_service: 'ConsultService') -> None:
        """
        初始化ConsultController实例

        Args:
            consult_service: ConsultService实例，用于处理健康咨询业务逻辑

        Raises:
            ValueError: consult_service为None时抛出
        """
        if consult_service is None:
            raise ValueError("consult_service不能为None")

        self._consult_service: 'ConsultService' = consult_service
        logger.info("ConsultController初始化完成")

    @property
    def consult_service(self) -> 'ConsultService':
        """
        获取ConsultService实例（只读属性）

        Returns:
            ConsultService: ConsultService实例
        """
        return self._consult_service

    def consult(self, request: ConsultRequest) -> ConsultResponse:
        """
        健康咨询的API接口方法

        该方法接收健康咨询请求，进行参数校验和协议转换，
        调用ConsultService处理业务逻辑，返回健康咨询响应。

        Args:
            request: 健康咨询请求，包含request_id、timestamp、body等信息
                     body包含用户问题、对话历史、用户档案等信息

        Returns:
            ConsultResponse: 健康咨询响应，包含status_code、message、data等信息
                            data包含咨询结果、建议、置信度等信息

        Raises:
            MedicalQAException: 业务异常时抛出
            Exception: 其他异常时抛出

        Example:
            >>> request = ConsultRequest(
            ...     request_id="req-001",
            ...     body=ConsultRequestBody(
            ...         question="头痛怎么办？",
            ...         session_id="session-001",
            ...         user_profile={"age": 45, "gender": "male"}
            ...     )
            ... )
            >>> response = controller.consult(request)
            >>> print(response.status_code)
            200
        """
        start_time = time.time()
        
        try:
            logger.info("=" * 60)
            logger.info(f"[ConsultController] 收到健康咨询请求")
            logger.info(f"  request_id: {request.request_id}")
            logger.info(f"  question: {request.get_question()[:100] if request.get_question() else 'None'}...")
            if request.body:
                logger.info(f"  session_id: {request.get_session_id() or 'None'}")
                if hasattr(request.body, 'user_profile') and request.body.user_profile:
                    logger.info(f"  user_profile: {json.dumps(request.body.user_profile, ensure_ascii=False)}")
            logger.info("=" * 60)

            self._validate_request(request)

            session_id = request.get_session_id()
            if not session_id:
                session_id = self._generate_session_id()
                logger.info(f"[ConsultController] 生成新的session_id: {session_id}")

            from src.orchestration.agent.data_classes import AgentContext

            agent_context = AgentContext(
                session_id=session_id,
                current_state="INIT",
                body=request.body
            )

            logger.info(f"[ConsultController] 开始处理业务逻辑: session_id={session_id}")
            agent_result = self._consult_service.process_consult(agent_context)

            response = self._build_response(request, agent_result)

            elapsed = time.time() - start_time
            logger.info("=" * 60)
            logger.info(f"[ConsultController] 健康咨询请求处理完成")
            logger.info(f"  request_id: {request.request_id}")
            logger.info(f"  status_code: {response.status_code}")
            logger.info(f"  elapsed: {elapsed:.2f}s")
            if response.data:
                answer_preview = str(response.data.result)[:200] if response.data.result else "None"
                logger.info(f"  answer_preview: {answer_preview}...")
            logger.info("=" * 60)

            return response

        except MedicalQAException as e:
            elapsed = time.time() - start_time
            logger.error(f"[ConsultController] 业务异常: request_id={request.request_id}, "
                        f"error_code={e.error_code}, message={e.message}, elapsed={elapsed:.2f}s")
            return self._build_error_response(
                request_id=request.request_id,
                status_code=400,
                message=str(e)
            )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.exception(f"[ConsultController] 未知异常: request_id={request.request_id}, elapsed={elapsed:.2f}s")
            return self._build_error_response(
                request_id=request.request_id,
                status_code=500,
                message=f"系统内部错误: {str(e)}"
            )

    def _validate_request(self, request: ConsultRequest) -> None:
        """
        验证请求参数

        Args:
            request: 健康咨询请求

        Raises:
            MedicalQAException: 参数无效时抛出
        """
        # 验证request_id
        if not request.request_id:
            raise MedicalQAException(
                ErrorCode.PARAM_MISSING,
                "request_id不能为空"
            )

        # 验证body
        if not request.body:
            raise MedicalQAException(
                ErrorCode.PARAM_MISSING,
                "body不能为空"
            )

        # 验证question
        if not request.get_question() or not request.get_question().strip():
            raise MedicalQAException(
                ErrorCode.PARAM_MISSING,
                "question不能为空"
            )

        # 验证question长度
        question = request.get_question()
        if len(question) > 1000:
            raise MedicalQAException(
                ErrorCode.PARAM_INVALID,
                "question长度不能超过1000个字符"
            )

    def _generate_session_id(self) -> str:
        """
        生成session_id

        Returns:
            str: 生成的session_id
        """
        return f"session_{uuid.uuid4().hex[:16]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    def _build_response(
        self,
        request: ConsultRequest,
        agent_result: 'AgentResult'
    ) -> ConsultResponse:
        """
        构建成功响应

        Args:
            request: 健康咨询请求
            agent_result: Agent执行结果

        Returns:
            ConsultResponse: 健康咨询响应
        """
        # 从agent_result中提取数据
        # 注：agent_result.data应该是ConsultResultData类型
        # 由于ConsultResultData还未开发，这里需要根据实际情况处理
        result_data = agent_result.data

        # 构建ConsultResponseData
        if result_data is not None:
            # 如果result_data是ConsultResponseData类型，直接使用
            if isinstance(result_data, ConsultResponseData):
                response_data = result_data
            # 如果result_data是字典类型，转换为ConsultResponseData
            elif isinstance(result_data, dict):
                response_data = ConsultResponseData(**result_data)
            # 其他情况，将result_data作为result字段
            else:
                response_data = ConsultResponseData(
                    result=str(result_data),
                    session_id=agent_result.session_id
                )
        else:
            # 如果result_data为None，返回默认响应
            response_data = ConsultResponseData(
                result="抱歉，无法处理您的咨询请求，请稍后重试。",
                session_id=agent_result.session_id
            )

        # 构建ConsultResponse
        response = ConsultResponse(
            status_code=200,
            message="咨询成功",
            data=response_data,
            request_id=request.request_id
        )

        return response

    def _build_error_response(
        self,
        request_id: str,
        status_code: int,
        message: str
    ) -> ConsultResponse:
        """
        构建错误响应

        Args:
            request_id: 请求ID
            status_code: 状态码
            message: 错误消息

        Returns:
            ConsultResponse: 健康咨询响应
        """
        return ConsultResponse(
            status_code=status_code,
            message=message,
            data=None,
            request_id=request_id
        )

    def __repr__(self) -> str:
        """返回ConsultController的字符串表示"""
        return f"ConsultController(consult_service={self._consult_service})"
