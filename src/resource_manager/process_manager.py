# -*- coding: utf-8 -*-
"""
子进程集中管理器

管理所有由项目启动的子进程（如SGLang推理服务进程），
提供atexit和signal处理保证异常退出时的进程清理，
防止孤儿进程占用GPU显存等系统资源。
"""

import atexit
import logging
import os
import signal
import subprocess
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ProcessManager:
    """
    子进程集中管理器

    管理所有由项目启动的子进程，提供：
    - register_process: 注册子进程
    - unregister_process: 注销子进程
    - cleanup_all: 终止所有已注册子进程（用于atexit/signal）
    - setup_signal_handlers: 注册SIGTERM/SIGINT处理器
    - register_atexit: 注册atexit清理函数
    """

    _processes: Dict[str, subprocess.Popen] = {}
    _lock = threading.Lock()
    _initialized = False

    @classmethod
    def register_process(cls, resource_type: str, process: subprocess.Popen) -> None:
        """
        注册子进程到管理器

        Args:
            resource_type: 资源类型标识
            process: subprocess.Popen实例
        """
        with cls._lock:
            cls._processes[resource_type] = process
            logger.info(f"[ProcessManager] 已注册子进程: resource_type={resource_type}, pid={process.pid}")

    @classmethod
    def unregister_process(cls, resource_type: str) -> None:
        """
        从管理器中移除子进程记录（进程已正常退出后调用）

        Args:
            resource_type: 资源类型标识
        """
        with cls._lock:
            if resource_type in cls._processes:
                del cls._processes[resource_type]
                logger.info(f"[ProcessManager] 已注销子进程: resource_type={resource_type}")

    @classmethod
    def is_process_running(cls, resource_type: str) -> bool:
        """
        检查指定资源类型的子进程是否仍在运行

        Args:
            resource_type: 资源类型标识

        Returns:
            bool: 进程是否仍在运行
        """
        with cls._lock:
            process = cls._processes.get(resource_type)
            if process is None:
                return False
            return process.poll() is None

    @classmethod
    def cleanup_all(cls) -> None:
        """
        终止所有已注册的子进程

        用于atexit和signal处理。按SIGTERM→等待→SIGKILL的顺序终止进程。
        """
        with cls._lock:
            if not cls._processes:
                return

            logger.info(f"[ProcessManager] cleanup_all: 开始清理{len(cls._processes)}个子进程")
            for resource_type, process in list(cls._processes.items()):
                pid = process.pid
                if process.poll() is not None:
                    logger.info(f"[ProcessManager] 子进程已退出: resource_type={resource_type}, pid={pid}")
                    continue

                logger.info(f"[ProcessManager] 终止子进程: resource_type={resource_type}, pid={pid}")
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except ProcessLookupError:
                    logger.info(f"[ProcessManager] 进程已不存在: resource_type={resource_type}, pid={pid}")
                    continue
                except Exception as e:
                    logger.warning(f"[ProcessManager] SIGTERM发送失败: resource_type={resource_type}, pid={pid}, error={e}")
                    try:
                        process.kill()
                    except Exception as e:
                        logger.debug(f"[ProcessManager] 终止子进程失败: {e}")

            # 等待进程退出
            for resource_type, process in list(cls._processes.items()):
                pid = process.pid
                try:
                    process.wait(timeout=10)
                    logger.info(f"[ProcessManager] 子进程已退出: resource_type={resource_type}, pid={pid}")
                except subprocess.TimeoutExpired:
                    logger.warning(f"[ProcessManager] 子进程未在10s内退出，发送SIGKILL: resource_type={resource_type}, pid={pid}")
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                        process.wait(timeout=5)
                    except Exception as e:
                        logger.error(f"[ProcessManager] SIGKILL终止失败: resource_type={resource_type}, pid={pid}, error={e}")

            cls._processes.clear()
            logger.info("[ProcessManager] cleanup_all: 所有子进程已清理")

    @classmethod
    def setup_signal_handlers(cls) -> None:
        """
        注册atexit清理函数

        信号处理由uvicorn负责（uvicorn收到SIGINT/SIGTERM后触发lifespan shutdown，
        最终执行GlobalResourceManager.shutdown()终止子进程）。
        atexit作为最后保障，确保异常退出时子进程也能被清理。
        """
        if cls._initialized:
            return

        atexit.register(cls.cleanup_all)
        cls._initialized = True
        logger.info("[ProcessManager] atexit清理函数已注册（信号处理由uvicorn负责）")

    @classmethod
    def register_atexit(cls) -> None:
        """
        注册atexit清理函数

        确保Python解释器退出时执行cleanup_all()。
        """
        atexit.register(cls.cleanup_all)
        logger.info("[ProcessManager] atexit清理函数已注册")
