# start_MedicalQA.py
import uvicorn
import sys
import os
from datetime import datetime

# 假设 log_service 和 rag_fastapi_service 在同一目录
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from log_service import get_logger
from rag_fastapi_service import app # 直接导入已创建的 app 实例

if __name__ == "__main__":
    try:
        logger = get_logger(__name__)
        logger.info("正在启动 Medical RAG API 服务 ...")

        # 直接传递导入的 app 实例
        uvicorn.run(
            app, # 直接传递 app 对象
            host=["0.0.0.0", "::"],
            # host="::",
            port=8001,
            log_level="info",
            timeout_keep_alive=60,
            proxy_headers=True,
            forwarded_allow_ips='*'
        )


    except Exception as e:
        logger.error(f"启动服务时出错: {str(e)}", exc_info=True)
        sys.exit(1)
