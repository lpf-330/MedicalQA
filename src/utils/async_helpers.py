# -*- coding: utf-8 -*-
"""
异步执行辅助工具

提供同步函数在线程池中异步执行的辅助方法，
解决 asyncio.to_thread() 在 Python 3.11 中不传播 ContextVar 的问题。
"""

import contextvars
from typing import Any, Callable


def run_with_context(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """在线程池中执行同步函数，保留当前ContextVar上下文

    Python 3.11 的 asyncio.to_thread() 不会自动传播 ContextVar 到工作线程，
    导致线程池中的日志丢失 session_id/task_id 关联。
    使用 contextvars.copy_context() 手动包裹，确保上下文正确传播。

    用法:
        result = await asyncio.to_thread(run_with_context, self._agent.run, context)
    """
    ctx = contextvars.copy_context()
    return ctx.run(func, *args, **kwargs)
