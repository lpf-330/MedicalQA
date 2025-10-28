from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
# 在文件顶部，与 ChatCompletionRequest 一起添加
from vllm.entrypoints.openai.protocol import CompletionRequest, TokenizeRequest, DetokenizeRequest
from vllm.entrypoints.openai.serving_models import OpenAIServingModels, BaseModelPath
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_tokenization import OpenAIServingTokenization

import uvicorn
import logging
import sys
import os
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def create_app():
    """创建API应用"""
    # 1. 创建异步引擎参数
    engine_args = AsyncEngineArgs(
        model="/home/project/MedicalQA/base_model",
	enable_lora=False,
        trust_remote_code=True,
        tokenizer_mode="auto",
        dtype="auto",
        quantization="bitsandbytes",
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
	max_num_batched_tokens=4096,
        max_num_seqs=16,
        enforce_eager=False,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        swap_space=8
    )
    
    # 2. 创建异步引擎
    logger.info("正在初始化模型引擎...")
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    async def get_model_config():
        engine_model_config = await engine.get_model_config()
        return engine_model_config
    
    model_config = asyncio.run(get_model_config())
    
    # 3. 创建模型服务对象
    served_model = "Qwen3-4B-Instruct-2507"
    served_model_path = "/home/project/MedicalQA/base_model" # 确保与 AsyncEngineArgs 中的 model 路径一致
    logger.info(f"正在创建API服务，模型: {served_model}")

    # a. 创建 BaseModelPath 对象列表
    base_model_paths = [BaseModelPath(name=served_model, model_path=served_model_path)]

    # b. 创建 OpenAIServingModels 实例
    openai_serving_models = OpenAIServingModels(
        engine_client=engine,        # 传递引擎客户端
        model_config=model_config,   # 传递模型配置
        base_model_paths=base_model_paths # 传递 BaseModelPath 列表
        # lora_modules=None (默认值，因为我们没有使用 LoRA)
    )

    # 注意: vLLM 0.10.1中chat_template参数是必需的，但可以设为None
    openai_serving_chat = OpenAIServingChat(
        engine,
	model_config,
	openai_serving_models,
        response_role="assistant",
        chat_template=None,
	request_logger=None,
	chat_template_content_format="auto"
    )
    openai_serving_completion = OpenAIServingCompletion(
        engine,
	model_config,
	openai_serving_models,
	request_logger=None
    )
    openai_serving_tokenization = OpenAIServingTokenization(
        engine,
	model_config,
	openai_serving_models,
	request_logger=None,
	chat_template=None,
	chat_template_content_format="auto"
    )
    
    # 4. 创建FastAPI应用并注册路由
    app = FastAPI(
        title="vLLM OpenAPI Server",
        description="API server compatible with OpenAI API",
        version="0.1.0"
    )
    
    # 修复：直接添加路由（vLLM 0.10.1中没有router属性）
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        # 检查请求对象中的 stream 属性
        if request.stream:
            generator = await openai_serving_chat.create_chat_completion(request)
            return StreamingResponse(generator, media_type="text/event-stream")
        else:
            # 如果是非流式请求，等待 create_chat_completion 完成并返回完整响应
            return await openai_serving_chat.create_chat_completion(request)
    
    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        return await openai_serving_completion.create_completion(request)
    
    @app.post("/v1/tokenize")
    async def tokenize(request: TokenizeRequest):
        return await openai_serving_tokenization.create_tokenize(request)
    
    @app.post("/v1/detokenize")
    async def detokenize(request: DetokenizeRequest):
        return await openai_serving_tokenization.create_detokenize(request)
    
    @app.get("/v1/models")
    async def show_available_models():
        models = await openai_serving_chat.show_available_models()
        return models
    
    @app.get("/health")
    async def health():
        """健康检查端点"""
        return {"status": "ok"}
    
    return app

if __name__ == "__main__":
    try:
        logger.info("正在启动vLLM API服务...")
        app = create_app()
        
        # 5. 运行服务器
        logger.info("服务启动成功! 请在新窗口运行对应的 vllm_client.py 与模型交互")
        logger.info("API服务地址: http://0.0.0.0:8000")
        logger.info("文档地址: http://0.0.0.0:8000/docs")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="warning",  # 减少uvicorn的日志输出
            timeout_keep_alive=60
        )
    except Exception as e:
        logger.error(f"启动服务时出错: {str(e)}", exc_info=True)
        sys.exit(1)
