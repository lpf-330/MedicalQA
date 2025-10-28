# start_MedicalQA.py

import asyncio
import uvicorn
import sys
from rag_fastapi_service import create_app # 导入创建应用的函数

if __name__ == "__main__":
    try:
        # 日志服务已在 log_service.py 中初始化
        from log_service import get_logger
        logger = get_logger(__name__)

        logger.info("正在启动 Medical RAG API 服务...")

        # 创建应用和依赖实例
        app, vllm_service, neo4j_service = asyncio.run(create_app())

        # 定义关闭时的清理函数
        async def shutdown_event():
            logger.info("Shutting down services...")
            neo4j_service.close()
            logger.info("Neo4j connection closed.")

        app.add_event_handler("shutdown", shutdown_event)

        logger.info("服务启动成功!")
        logger.info("RAG流式API地址: http://0.0.0.0:8001/v1/medical_rag_stream")
        logger.info("原有OpenAI API地址: http://0.0.0.0:8001/v1/chat/completions")
        logger.info("文档地址: http://0.0.0.0:8001/docs")

        # 启动 Uvicorn 服务器
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8001, # 修改端口以区分
            log_level="info", # 保持uvicorn日志级别以便调试
            timeout_keep_alive=60
        )

    except Exception as e:
        from log_service import get_logger
        logger = get_logger(__name__)
        logger.error(f"启动服务时出错: {str(e)}", exc_info=True)
        sys.exit(1)
