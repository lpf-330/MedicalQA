"""
服务层健康报告生成服务模块

该模块定义了ReportService类，是健康报告生成业务的服务类。
"""

import logging
import time
from typing import TYPE_CHECKING, TypeVar, TypeAlias, Any, Generator

if TYPE_CHECKING:
    from src.orchestration.agent.agent import Agent
    from src.orchestration.agent.data_classes import AgentContext, AgentResult

from src.orchestration.agent.data_classes import AgentContext
from src.orchestration.agent.report_strategy.report_strategy import ReportContextBody
from src.schemas.report_request import ReportRequest

logger = logging.getLogger(__name__)

T = TypeVar('T')

ReportContext: TypeAlias = 'AgentContext[Any]'
ReportResult: TypeAlias = 'AgentResult[Any]'


class ReportService:
    """
    报告服务类

    健康报告生成业务的服务类，负责处理报告生成请求。
    依赖Agent组合容器，通过Agent执行报告生成策略。

    Attributes:
        _agent: Agent组合容器实例
    """

    def __init__(self, agent: 'Agent[Any, Any]') -> None:
        """
        初始化ReportService实例

        Args:
            agent: Agent组合容器实例

        Raises:
            ValueError: agent为None时抛出
        """
        if agent is None:
            raise ValueError("agent不能为None")

        self._agent: 'Agent[Any, Any]' = agent
        logger.info("[ReportService] 服务初始化完成")

    @property
    def agent(self) -> 'Agent[Any, Any]':
        """
        获取Agent实例（只读属性）

        Returns:
            Agent[Any, Any]: Agent组合容器实例
        """
        return self._agent

    def process_report(self, context: ReportContext) -> ReportResult:
        """
        处理报告请求

        验证输入参数，调用Agent执行报告生成策略，并释放资源。

        Args:
            context: Agent输入数据容器，包含报告生成所需的上下文数据

        Returns:
            ReportResult: Agent输出数据容器，包含报告生成结果

        Raises:
            ValueError: context为None或session_id为空时抛出

        Example:
            >>> result = report_service.process_report(context)
            >>> print(result.data.report)
        """
        start_time = time.time()

        # 验证context不为None
        if context is None:
            logger.error("[ReportService] context为None")
            raise ValueError("context不能为None")

        # 验证session_id不为空
        if not hasattr(context, 'session_id') or not context.session_id:
            logger.error("[ReportService] session_id为空")
            raise ValueError("context.session_id不能为空")

        logger.info(f"[ReportService] 开始处理报告: session_id={context.session_id}")

        try:
            # 调用Agent执行报告生成策略
            result = self._agent.run(context)
            return result
        finally:
            # 释放资源
            self._release_resources()
            elapsed = time.time() - start_time
            logger.info(f"[ReportService] 报告处理完成: session_id={context.session_id}, elapsed={elapsed:.2f}s")

    def process_report_stream(self, context: ReportContext) -> Generator[str, None, None]:
        """
        流式处理报告请求

        验证输入参数，调用Agent执行报告生成策略，并以SSE格式流式输出报告内容。

        Args:
            context: Agent输入数据容器，包含报告生成所需的上下文数据

        Yields:
            str: SSE格式的流式数据
                - event: message - 报告内容块
                - event: end - 结束信号
                - event: error - 错误信息

        Example:
            >>> for chunk in report_service.process_report_stream(context):
            ...     print(chunk)
        """
        import json

        # 验证context不为None
        if context is None:
            yield f"event: error\ndata: {json.dumps({'error_code': 400, 'error_message': 'context不能为None'})}\n\n"
            return

        # 验证session_id不为空
        if not hasattr(context, 'session_id') or not context.session_id:
            yield f"event: error\ndata: {json.dumps({'error_code': 400, 'error_message': 'session_id不能为空'})}\n\n"
            return

        logger.info(f"[ReportService] 开始流式处理报告: session_id={context.session_id}")

        try:
            # 调用Agent执行报告生成策略
            result = self._agent.run(context)

            # 处理流式输出
            if result is not None and result.data is not None:
                body = context.body

                # 如果有流式生成器，则流式输出
                if hasattr(body, 'stream_generator') and body.stream_generator is not None:
                    # 流式输出报告内容
                    for token in body.stream_generator:
                        payload = json.dumps({"content": token}, ensure_ascii=False)
                        yield f"event: message\ndata: {payload}\n\n"

                    # 更新上下文状态
                    context.is_streaming = False

                    # 发送结束事件
                    end_data = {
                        "session_id": result.session_id,
                        "health_score": getattr(body, 'health_score', 0),
                        "health_level": getattr(body, 'health_level', ''),
                        "risk_level": getattr(body, 'risk_level', ''),
                        "sources": getattr(body, 'sources', []),
                        "error_code": getattr(body, 'error_code', 0),
                    }
                    yield f"event: end\ndata: {json.dumps(end_data, ensure_ascii=False)}\n\n"
                else:
                    # 如果没有流式生成器，直接输出完整报告
                    report = result.data.report if hasattr(result.data, 'report') else str(result.data)
                    payload = json.dumps({"content": report}, ensure_ascii=False)
                    yield f"event: message\ndata: {payload}\n\n"

                    # 发送结束事件
                    end_data = {
                        "session_id": result.session_id,
                        "health_score": result.data.health_score if hasattr(result.data, 'health_score') else 0,
                        "health_level": result.data.health_level if hasattr(result.data, 'health_level') else '',
                        "risk_level": result.data.risk_level if hasattr(result.data, 'risk_level') else '',
                    }
                    yield f"event: end\ndata: {json.dumps(end_data, ensure_ascii=False)}\n\n"
            else:
                # 结果为空，发送错误事件
                yield f"event: error\ndata: {json.dumps({'error_code': 500, 'error_message': '处理结果为空'})}\n\n"

        except Exception as e:
            # 异常处理，发送错误事件
            logger.error(f"[ReportService] 流式处理异常: {str(e)}")
            yield f"event: error\ndata: {json.dumps({'error_code': 500, 'error_message': str(e)})}\n\n"

        finally:
            # 释放资源
            self._release_resources()

    def _build_agent_context(self, request: ReportRequest) -> AgentContext:
        """
        从ReportRequest构建AgentContext

        将API请求数据转换为Agent执行所需的上下文数据。

        Args:
            request: 健康报告生成请求数据

        Returns:
            AgentContext: Agent输入数据容器

        Example:
            >>> context = report_service._build_agent_context(request)
            >>> print(context.session_id)
        """
        # 获取session_id，优先使用body中的session_id，否则使用task_id
        session_id = request.get_session_id() or request.body.task_id

        # 获取监测数据
        monitoring_data = {}
        if request.body.monitoring_data:
            monitoring_data = request.body.monitoring_data.model_dump()

        # 获取用户档案
        user_profile = {}
        if request.body.user_profile:
            user_profile = request.body.user_profile.model_dump()

        # 创建ReportContextBody
        context_body = ReportContextBody(
            task_id=request.body.task_id,
            monitoring_data=monitoring_data,
            user_profile=user_profile,
            session_id=session_id,
            current_state="INITIAL"
        )

        # 创建AgentContext
        agent_context = AgentContext(
            session_id=session_id,
            current_state="INITIAL",
            body=context_body
        )

        return agent_context

    def _release_resources(self) -> None:
        """
        释放Agent资源

        释放tool handlers和model service资源。
        """
        agent_resource = self._agent.resources
        if agent_resource is not None:
            # 释放tool handlers
            for handler in agent_resource.tool_handlers.values():
                try:
                    handler.release()
                except Exception as e:
                    logger.error(f"[ReportService] 释放tool handler失败: {e}")

            # 释放model service
            if agent_resource.model_service is not None:
                try:
                    agent_resource.model_service.release()
                except Exception as e:
                    logger.error(f"[ReportService] 释放model service失败: {e}")

    def __repr__(self) -> str:
        """
        返回ReportService的字符串表示

        Returns:
            str: ReportService的字符串表示
        """
        return f"ReportService(agent={self._agent})"
