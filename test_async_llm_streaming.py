#!/usr/bin/env python3
"""
测试vLLM AsyncLLM真正的实时流式输出
"""
import asyncio
import time
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM


async def test_streaming():
    print("初始化AsyncLLM...")
    
    engine_args = AsyncEngineArgs(
        model="/home/project/MedicalQA/base_models/Qwen3-4B-Instruct-2507",
        enforce_eager=True,
        max_model_len=8192,
        gpu_memory_utilization=0.8,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    
    prompt = "糖尿病的症状有哪些？请简要回答。"
    
    print(f"\n提示词: {prompt}")
    print("回答: ", end="", flush=True)
    
    sampling_params = SamplingParams(
        max_tokens=200,
        temperature=0.7,
        top_p=0.9,
        output_kind=RequestOutputKind.DELTA,
    )
    
    start_time = time.time()
    first_token_time = None
    token_count = 0
    
    try:
        async for output in engine.generate(
            request_id="test-streaming-001",
            prompt=prompt,
            sampling_params=sampling_params
        ):
            for completion in output.outputs:
                new_text = completion.text
                if new_text:
                    if first_token_time is None:
                        first_token_time = time.time()
                        print(f"\n[首个token延迟: {first_token_time - start_time:.3f}s]")
                        print("回答: ", end="", flush=True)
                    print(new_text, end="", flush=True)
                    token_count += 1
            
            if output.finished:
                break
    
    except Exception as e:
        print(f"\n错误: {e}")
        raise
    finally:
        engine.shutdown()
    
    elapsed = time.time() - start_time
    print(f"\n\n总耗时: {elapsed:.3f}s")
    print(f"Token数量: {token_count}")
    print(f"平均速度: {token_count/elapsed:.1f} tokens/s")


if __name__ == "__main__":
    asyncio.run(test_streaming())
