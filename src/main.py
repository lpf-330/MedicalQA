# -*- coding: utf-8 -*-
# ruff: noqa: E402
"""
MedicalQA项目启动文件

基于7层架构设计的健康咨询服务启动入口。
按照《项目架构设计v2.1》和《项目架构原则与使用规范v1》的规范流程启动系统。
"""

import os
import sys
import logging
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 防止子进程重复创建日志文件
_LOG_INITIALIZED = os.environ.get('MEDICALQA_LOG_INITIALIZED', '')

log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

if not _LOG_INITIALIZED:
    os.environ['MEDICALQA_LOG_INITIALIZED'] = 'true'

    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"medical_qa_{timestamp}.log")

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter(log_format))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # G-1: 注册 ContextVarFilter，将 session_id/task_id 注入所有日志记录
    from src.utils.logger import ContextVarFilter
    context_var_filter = ContextVarFilter()
    root_logger.addFilter(context_var_filter)
    business_logger_ref = logging.getLogger('Business')
    business_logger_ref.addFilter(context_var_filter)

    logging.getLogger('neo4j').setLevel(logging.WARNING)

from src.utils.logger import get_logger

business_logger = logging.getLogger('Business')
business_console_handler = logging.StreamHandler(sys.stdout)
business_console_handler.setLevel(logging.INFO)
business_console_handler.setFormatter(logging.Formatter(log_format))
business_logger.addHandler(business_console_handler)
business_logger.setLevel(logging.INFO)
business_logger.propagate = False

logger = get_logger(
    name="Main",
    level="INFO",
    log_file=None,
    console_output=False
)

from src.config.config_manager import ConfigManager
from src.resource_manager import GlobalResourceManager
from src.service.consult_service import ConsultService
from src.service.report_service import ReportService
from src.controller.consult_controller import ConsultController
from src.controller.report_controller import ReportController
from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn


def _check_gpu_vram(vram_sufficient_gb: float = 8.0) -> None:
    """
    系统启动前显存检查

    检查GPU可用显存情况，在显存可能不足时输出警告。
    - 可用显存充足：输出INFO日志，正常启动
    - 可用显存不足：输出WARNING日志，提示显存可能不足
    - CUDA不可用或torch未安装：输出INFO日志，跳过检查
    """
    try:
        import torch
        if not torch.cuda.is_available():
            logger.info("[显存检查] CUDA不可用，跳过显存检查")
            return

        gpu_id = 0
        total_vram = torch.cuda.get_device_properties(gpu_id).total_memory
        allocated_vram = torch.cuda.memory_allocated(gpu_id)
        available_vram = total_vram - allocated_vram
        available_gb = available_vram / (1024 ** 3)
        total_gb = total_vram / (1024 ** 3)
        total_mb = total_vram / (1024 ** 2)
        available_mb = available_vram / (1024 ** 2)
        sufficient = available_gb > vram_sufficient_gb
        logger.info(f"[GPU_CHECK] GPU显存检查: total={total_mb:.0f}MB, available={available_mb:.0f}MB, sufficient={sufficient}")

        if sufficient:
            logger.info(f"[显存检查] GPU显存充足: 总计={total_gb:.1f}GB, 可用={available_gb:.1f}GB")
        else:
            logger.warning(f"[显存检查] GPU显存可能不足: 总计={total_gb:.1f}GB, 可用={available_gb:.1f}GB (建议 > {vram_sufficient_gb:.0f}GB)")
    except ImportError:
        logger.info("[显存检查] torch未安装，跳过显存检查")
    except Exception as e:
        logger.warning(f"[显存检查] 显存检查异常: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("=" * 60)
    logger.info("初始化健康咨询服务...")
    logger.info("=" * 60)

    try:
        logger.info("[STARTUP_SEQUENCE] Step 1: 加载配置")
        logger.info("[STARTUP_SEQUENCE] Step 2: 初始化GlobalResourceManager")
        config_manager = GlobalResourceManager.initialize()
        global_config = config_manager.to_global_config()

        # 注册子进程异常退出保护
        from src.resource_manager.process_manager import ProcessManager
        ProcessManager.setup_signal_handlers()
        ProcessManager.register_atexit()

        # 启动前显存检查
        _check_gpu_vram(global_config.vram_sufficient_gb)

        logger.info("[STARTUP_SEQUENCE] Step 3: 创建初始资源实例")

        # 初始化MCPProxyFactory — 设计依据：2.4.2节工厂使用约定 + 6.1节系统启动流程
        # MCPProxyFactory属于MCP代理层，由main.py在GlobalResourceManager之后、Service创建之前初始化
        from src.mcp.factory.mcp_proxy_factory import MCPProxyFactory
        from src.mcp.factory.tool_proxy_config import ToolProxyConfig, ProxyType

        tool_proxy_configs = {
            "neo4j_medical": ToolProxyConfig(
                proxy_type=ProxyType.FAKE,
                connection_info={"tool_name": "neo4j_medical"}
            ),
            "vector_retrieval": ToolProxyConfig(
                proxy_type=ProxyType.FAKE,
                connection_info={"tool_name": "vector_retrieval"}
            ),
        }

        # 可选工具：intent_classification（配置驱动）
        intent_model_resource_config = config_manager.resource_configs.get("intent_model_config")
        if intent_model_resource_config is not None:
            tool_proxy_configs["intent_classification"] = ToolProxyConfig(
                proxy_type=ProxyType.FAKE,
                connection_info={
                    "tool_name": "intent_classification",
                    "model_path": intent_model_resource_config.model_path,
                    "device": intent_model_resource_config.device,
                    "max_length": intent_model_resource_config.max_length,
                }
            )

        # 可选工具：ner_model（配置驱动）
        ner_model_resource_config = config_manager.resource_configs.get("ner_model_config")
        if ner_model_resource_config is not None:
            tool_proxy_configs["ner_model"] = ToolProxyConfig(
                proxy_type=ProxyType.FAKE,
                connection_info={
                    "tool_name": "ner_model",
                    "model_path": ner_model_resource_config.model_path,
                    "device": ner_model_resource_config.device,
                    "max_length": ner_model_resource_config.max_length,
                }
            )

        factory = MCPProxyFactory.get_instance()
        factory.initialize(tool_proxy_configs)
        logger.info("[STARTUP_SEQUENCE] MCPProxyFactory 初始化完成")

        logger.info("[STARTUP_SEQUENCE] Step 4: 初始化业务组件")
        consult_service = ConsultService(config_manager=config_manager)
        report_service = ReportService(config_manager=config_manager)

        consult_controller = ConsultController(consult_service=consult_service)
        logger.info("[COMPONENT_CREATE] 创建组件: ConsultController, type=ConsultController")

        report_controller = ReportController(report_service=report_service)
        logger.info("[COMPONENT_CREATE] 创建组件: ReportController, type=ReportController")

        app.state.consult_controller = consult_controller
        app.state.report_controller = report_controller

        logger.info("[STARTUP_SEQUENCE] Step 5: 启动服务")
        try:
            warmup_messages = [
                {"role": "system", "content": "你是一个健康咨询助手。"},
                {"role": "user", "content": "你好"}
            ]
            warmup_result = consult_service.consult_model_service.call_model(warmup_messages, timeout=global_config.warmup_timeout)
            logger.info(f"模型预热完成: response_length={len(warmup_result)}")
        except Exception as e:
            logger.warning(f"模型预热失败（不影响服务启动）: {str(e)}")

        logger.info("=" * 60)
        logger.info("健康咨询服务初始化完成")
        logger.info("=" * 60)

        try:
            yield
        finally:
            logger.info("=" * 60)
            logger.info("正在关闭服务...")
            logger.info("=" * 60)
            logger.info("[SHUTDOWN] 系统开始关闭")

            try:
                logger.info("[SHUTDOWN] 执行 GlobalResourceManager.shutdown()")
                GlobalResourceManager.INSTANCE.shutdown()
            except Exception as shutdown_error:
                logger.error(f"[SHUTDOWN] GlobalResourceManager.shutdown() 失败: error_type={type(shutdown_error).__name__}")

            logger.info("[SHUTDOWN] 系统关闭完成")

    except Exception as e:
        logger.error(f"服务生命周期异常: error_type={type(e).__name__}")
        try:
            if GlobalResourceManager.INSTANCE is not None:
                GlobalResourceManager.INSTANCE.shutdown()
        except Exception as shutdown_error:
            logger.error(f"[SHUTDOWN] 初始化失败后关闭异常: error_type={type(shutdown_error).__name__}")
        raise


app = FastAPI(
    title="MedicalQA API Server",
    description="基于7层架构设计的健康咨询API服务",
    version="1.0.0",
    lifespan=lifespan
)

# 注册 Controller 层路由
from src.controller.consult_controller import router as consult_router
from src.controller.report_controller import router as report_router

app.include_router(consult_router)
app.include_router(report_router)


@app.get("/")
async def root():
    """根路径"""
    return {"message": "MedicalQA API Server is running", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """健康检查"""
    stats = GlobalResourceManager.INSTANCE.get_stats()
    return {
        "status": "healthy",
        "resource_stats": stats
    }


def main():
    """启动服务"""
    logger.info("启动MedicalQA服务...")

    try:
        from src.config.config_manager import get_config_manager
        config_manager = get_config_manager()
        global_config = config_manager.to_global_config()

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=global_config.server_port,
            log_level="info",
            timeout_keep_alive=global_config.timeout_keep_alive
        )
    except KeyboardInterrupt:
        logger.info("收到中断信号，服务已停止")


if __name__ == "__main__":
    main()
