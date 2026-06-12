#!/usr/bin/env python3
"""MedicalQA 项目依赖验证脚本 — 验证 SGLang 环境迁移后所有重要依赖可用"""

import sys
import time

results = []


def test(name, func):
    """执行单个测试并记录结果"""
    try:
        ok, detail = func()
        results.append((name, "PASS" if ok else "FAIL", detail))
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {detail}")
    except Exception as e:
        results.append((name, "ERROR", str(e)))
        print(f"  [ERROR] {name}: {e}")


# ============================================================
# 1. SGLang 核心
# ============================================================
print("\n=== 1. SGLang 核心 ===")


def t_sglang_version():
    import sglang
    v = sglang.__version__
    return v == "0.5.12.post1", f"v{v} {'LOCKED' if v == '0.5.12.post1' else 'VERSION MISMATCH!'}"


def t_sgl_kernel():
    import sgl_kernel
    return True, f"v{sgl_kernel.__version__}"


def test_sgl_kernel_sm75():
    import sgl_kernel
    import torch
    cc = torch.cuda.get_device_capability(0)
    if cc[0] * 10 + cc[1] != 75:
        return False, f"不是 sm_75, cc={cc}"
    try:
        ops = sgl_kernel.load_utils._load_architecture_specific_ops()
        return True, "sm_75 内核加载成功"
    except Exception as e:
        return False, f"sm_75 内核加载失败: {e}"


def t_flash_attn():
    import flash_attn
    return True, f"v{getattr(flash_attn, '__version__', 'N/A')}"


def t_flashinfer():
    import flashinfer
    return True, f"v{flashinfer.__version__}"


test("sglang 版本锁定", t_sglang_version)
test("sgl-kernel", t_sgl_kernel)
test("sgl-kernel sm_75 内核", test_sgl_kernel_sm75)
test("flash-attn", t_flash_attn)
test("flashinfer", t_flashinfer)

# ============================================================
# 2. PyTorch + CUDA
# ============================================================
print("\n=== 2. PyTorch + CUDA ===")


def t_torch():
    import torch
    return True, f"v{torch.__version__}, CUDA {torch.version.cuda}"


def t_cuda_available():
    import torch
    return torch.cuda.is_available(), f"可用: {torch.cuda.is_available()}"


def t_cuda_gpu():
    import torch
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    return cc == (7, 5), f"{name}, sm_{cc[0]}{cc[1]}"


def t_cuda_op():
    import torch
    x = torch.randn(500, 500, device="cuda")
    y = x @ x.T
    return y.shape == (500, 500), f"矩阵乘法 OK, shape={y.shape}"


def t_cuda_vram():
    import torch
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    used = torch.cuda.memory_allocated(0) / (1024**3)
    free = total - used
    return free > 1.0, f"总 {total:.1f}GB, 已用 {used:.2f}GB, 剩余 {free:.1f}GB"


test("torch", t_torch)
test("CUDA 可用", t_cuda_available)
test("GPU 型号+sm", t_cuda_gpu)
test("CUDA 运算", t_cuda_op)
test("CUDA 显存", t_cuda_vram)

# ============================================================
# 3. Transformers + 模型 API
# ============================================================
print("\n=== 3. Transformers + 模型 API ===")


def t_transformers():
    import transformers
    return True, f"v{transformers.__version__}"


def t_transformers_api():
    from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer, pipeline
    return True, "4个API全部可导入"


def t_sentence_transformers():
    from sentence_transformers import SentenceTransformer
    import sentence_transformers
    return True, f"v{sentence_transformers.__version__}"


def t_peft():
    import peft
    return True, f"v{peft.__version__}"


def t_bitsandbytes():
    import bitsandbytes
    return True, f"v{bitsandbytes.__version__}"


test("transformers", t_transformers)
test("transformers API", t_transformers_api)
test("sentence-transformers", t_sentence_transformers)
test("peft", t_peft)
test("bitsandbytes", t_bitsandbytes)

# ============================================================
# 4. 数据库驱动
# ============================================================
print("\n=== 4. 数据库驱动 ===")


def t_neo4j():
    import neo4j
    return True, f"v{neo4j.__version__}"


def t_pymilvus():
    import pymilvus
    return True, f"v{pymilvus.__version__}"


test("neo4j", t_neo4j)
test("pymilvus", t_pymilvus)

# ============================================================
# 5. Web 框架
# ============================================================
print("\n=== 5. Web 框架 ===")


def t_fastapi():
    import fastapi
    return True, f"v{fastapi.__version__}"


def t_uvicorn():
    import uvicorn
    return True, f"v{uvicorn.__version__}"


def t_starlette():
    import starlette
    return True, f"v{starlette.__version__}"


def t_sse():
    from sse_starlette.sse import EventSourceResponse
    return True, "EventSourceResponse 可导入"


def t_streaming():
    from starlette.responses import StreamingResponse
    from fastapi.responses import StreamingResponse as FStreamingResponse
    return True, "StreamingResponse 可导入"


test("fastapi", t_fastapi)
test("uvicorn", t_uvicorn)
test("starlette", t_starlette)
test("sse-starlette", t_sse)
test("StreamingResponse", t_streaming)

# ============================================================
# 6. 编排层
# ============================================================
print("\n=== 6. 编排层 ===")


def t_langchain():
    import langchain
    return True, f"v{langchain.__version__}"


def t_langchain_core():
    import langchain_core
    return True, f"v{langchain_core.__version__}"


def t_langchain_api():
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.prompts import PromptTemplate
    from langchain_core.tools import Tool
    from langchain_core.chat_history import BaseChatMessageHistory
    return True, "4个API全部可导入"


def t_langgraph():
    import langgraph
    try:
        v = langgraph.__version__
    except AttributeError:
        v = "N/A"
    return True, f"v{v}"


def t_langgraph_api():
    from langgraph.prebuilt import create_react_agent
    return True, "create_react_agent 可导入"


def t_langsmith():
    import langsmith
    return True, f"v{langsmith.__version__}"


test("langchain", t_langchain)
test("langchain-core", t_langchain_core)
test("langchain API", t_langchain_api)
test("langgraph", t_langgraph)
test("langgraph API", t_langgraph_api)
test("langsmith", t_langsmith)

# ============================================================
# 7. 其他项目依赖
# ============================================================
print("\n=== 7. 其他项目依赖 ===")


def t_ray():
    import ray
    return True, f"v{ray.__version__}"


def t_gradio():
    import gradio
    return True, f"v{gradio.__version__}"


def t_mcp():
    import mcp
    return True, f"v{getattr(mcp, '__version__', 'N/A')}"


def t_supervisor():
    import supervisor
    return True, "可导入"


def t_prometheus():
    from prometheus_fastapi_instrumentator import Instrumentator
    return True, "Instrumentator 可导入"


def test_pytest():
    import pytest
    return True, f"v{pytest.__version__}"


def t_pydantic_settings():
    from pydantic_settings import BaseSettings
    return True, "BaseSettings 可导入"


test("ray", t_ray)
test("gradio", t_gradio)
test("mcp", t_mcp)
test("supervisor", t_supervisor)
test("prometheus-fastapi-instrumentator", t_prometheus)
test("pytest", test_pytest)
test("pydantic-settings", t_pydantic_settings)

# ============================================================
# 8. vllm 已移除确认
# ============================================================
print("\n=== 8. vllm 移除确认 ===")


def t_no_vllm():
    try:
        import vllm
        return False, f"vllm 仍然存在! v{vllm.__version__}"
    except ImportError:
        return True, "vllm 已移除"


test("vllm 已移除", t_no_vllm)

# ============================================================
# 9. 项目源码关键 import 测试
# ============================================================
print("\n=== 9. 项目源码关键 import ===")
sys.path.insert(0, "/home/project/MedicalQA")


def t_config_import():
    from src.config.config_manager import ConfigManager
    return True, "ConfigManager 可导入"


def test_schemas_import():
    from src.schemas.consult_request import ConsultRequest
    return True, "ConsultRequest 可导入"


def test_base_adapter():
    # 直接导入模块文件，绕过 adapters/__init__.py 的 vllm import
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "base_adapter", "/home/project/MedicalQA/src/adapters/base_adapter.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return True, "BaseAdapter 类可加载"


# vllm 适配层会失败（预期行为，后续重构）
def test_vllm_adapter_will_fail():
    try:
        from src.adapters.vllm.vllm_adapter_impl import VLLMAdapterImpl
        return False, "vllm 适配层不应可导入（vllm 已移除）"
    except (ImportError, ModuleNotFoundError):
        return True, "vllm 适配层 import 失败（预期行为，后续重构）"


test("ConfigManager", t_config_import)
test("ConsultRequest", test_schemas_import)
test("BaseAdapter", test_base_adapter)
test("vllm适配层预期失败", test_vllm_adapter_will_fail)

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
passed = sum(1 for _, s, _ in results if s == "PASS")
failed = sum(1 for _, s, _ in results if s == "FAIL")
errors = sum(1 for _, s, _ in results if s == "ERROR")
total = len(results)

print(f"总计: {total} 项 | PASS: {passed} | FAIL: {failed} | ERROR: {errors}")

if failed > 0 or errors > 0:
    print("\n失败项:")
    for name, status, detail in results:
        if status in ("FAIL", "ERROR"):
            print(f"  [{status}] {name}: {detail}")
    sys.exit(1)
else:
    print("\n全部通过!")
    sys.exit(0)
