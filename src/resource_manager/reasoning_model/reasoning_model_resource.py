# -*- coding: utf-8 -*-
"""
SGLang模型资源封装

封装SGLang模型推理资源，实现Resource接口，提供资源生命周期管理。
auto_start=True时，activate()自动启动SGLang服务子进程并等待就绪后连接。
auto_start=False时，仅连接外部已运行的SGLang HTTP服务。
destroy()时，若进程由本项目启动则终止子进程，并断开HTTP连接。
"""

import logging
import ctypes
import os
import signal
import socket
import subprocess
import time
from typing import TYPE_CHECKING, Optional

import requests

from src.resource_manager.resource import Resource
from src.adapters.sglang.sglang_adapter import SGLangAdapter
from src.adapters.sglang.sglang_adapter_impl import SGLangAdapterImpl

if TYPE_CHECKING:
    from src.resource_manager.reasoning_model.reasoning_model_config import ReasoningModelConfig

logger = logging.getLogger(__name__)


class ReasoningModelResource(Resource):
    """
    SGLang模型资源类

    封装SGLang模型推理资源，实现Resource接口。

    auto_start=True时，activate()自动启动SGLang服务子进程；
    auto_start=False时，仅连接外部已运行的SGLang HTTP服务。
    destroy()时，若进程由本项目启动则终止子进程，并断开HTTP连接。

    属性：
        _config: SGLang模型配置
        _adapter: SGLang适配器实例
        _last_used_time: 最后使用时间戳
        _is_active: 资源活跃状态
        _process: SGLang服务子进程实例
        _launched_by_us: 是否由本项目启动的进程
    """

    def __init__(self, config: 'ReasoningModelConfig'):
        self._config = config
        self._adapter: Optional[SGLangAdapter] = None
        self._last_used_time = int(time.time() * 1000)
        self._is_active = False
        self._process: Optional[subprocess.Popen] = None
        self._launched_by_us: bool = False

    def get_type(self) -> str:
        """获取资源类型标识"""
        return "reasoning_model"

    def get_last_used_time(self) -> int:
        """获取最后使用时间戳"""
        return self._last_used_time

    def is_activate(self) -> bool:
        """校验资源活跃状态"""
        return self._is_active

    def activate(self) -> None:
        """
        激活资源

        auto_start=True时：
          1. 检查端口是否已被占用（说明SGLang已在运行）
          2. 未占用则启动SGLang子进程
          3. 等待服务就绪
          4. 连接SGLang HTTP服务
        auto_start=False时：
          仅连接外部已运行的SGLang HTTP服务
        """
        logger.info("[STAGE_ENTER] ReasoningModelResource.activate")
        if self._is_active:
            logger.info("[STAGE_EXIT] ReasoningModelResource.activate, already active")
            return

        config_protocol = self._config.config_protocol
        auto_start = config_protocol.get("auto_start", False)

        if auto_start:
            launch_port = config_protocol.get("launch_port", 30000)
            if self._is_port_in_use(launch_port):
                logger.info(f"[ReasoningModelResource] 端口{launch_port}已被占用，跳过启动，直接连接")
                self._launched_by_us = False
            else:
                logger.info(f"[ReasoningModelResource] 端口{launch_port}未被占用，启动SGLang服务子进程")
                self._start_server_process()
                self._launched_by_us = True
                self._wait_for_server_ready()

        self._adapter = SGLangAdapterImpl()
        self._adapter.connect(
            base_url=config_protocol["base_url"],
            model_name=config_protocol.get("model_name", ""),
            default_temperature=config_protocol.get("default_temperature", 0.0),
            default_max_tokens=config_protocol.get("default_max_tokens", 1),
            default_top_p=config_protocol.get("default_top_p", 0.0),
            default_repetition_penalty=config_protocol.get("default_repetition_penalty", 1.15),
            timeout=config_protocol.get("timeout", 120.0)
        )
        self._is_active = True
        self._last_used_time = int(time.time() * 1000)
        logger.info(f"[STAGE_EXIT] ReasoningModelResource.activate, auto_start={auto_start}, launched_by_us={self._launched_by_us}")

    def deactivate(self) -> None:
        """停用资源（释放回池，保持连接）"""
        logger.info("[STAGE_ENTER] ReasoningModelResource.deactivate")
        if not self._is_active:
            logger.info("[STAGE_EXIT] ReasoningModelResource.deactivate, not active")
            return

        self._is_active = False
        logger.info("[STAGE_EXIT] ReasoningModelResource.deactivate")

    def destroy(self) -> None:
        """
        销毁资源（彻底释放）

        若进程由本项目启动（_launched_by_us=True），则终止子进程。
        然后断开HTTP客户端连接。
        """
        logger.info("[STAGE_ENTER] ReasoningModelResource.destroy")
        if self._launched_by_us and self._process is not None:
            self._stop_server_process()
            self._launched_by_us = False
        if self._adapter is not None:
            self._adapter.disconnect()
        self._adapter = None
        self._is_active = False
        logger.info("[STAGE_EXIT] ReasoningModelResource.destroy")

    def get_adapter(self) -> Optional[SGLangAdapter]:
        """获取SGLang适配器实例"""
        return self._adapter

    def get_async_adapter(self) -> Optional[SGLangAdapter]:
        """获取SGLang异步适配器实例（与同步适配器为同一实例）"""
        return self._adapter

    @staticmethod
    def _set_death_signal() -> None:
        """preexec_fn: 创建新进程组 + 父进程死亡时子进程自动收到SIGTERM

        os.setsid() 创建新会话，允许用 killpg 终止整个进程组。
        prctl(PR_SET_PDEATHSIG) 确保父进程被 SIGKILL 或异常退出时，
        子进程也会自动收到 SIGTERM 而不是变成孤儿进程。
        """
        os.setsid()
        try:
            PR_SET_PDEATHSIG = 1
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
        except Exception as e:
            logger.debug(f"[ReasoningModelResource] 设置进程死亡信号失败: {e}")

    def _start_server_process(self) -> None:
        """构建命令行并启动SGLang服务子进程"""
        config_protocol = self._config.config_protocol
        model_path = config_protocol.get("model_path", "")
        launch_host = config_protocol.get("launch_host", "0.0.0.0")
        launch_port = config_protocol.get("launch_port", 30000)
        launch_args = config_protocol.get("launch_args", "")

        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--host", launch_host,
            "--port", str(launch_port),
        ]
        if launch_args:
            cmd.extend(launch_args.split())

        logger.info(f"[ReasoningModelResource] 启动命令: {' '.join(cmd)}")
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=self._set_death_signal
            )
            logger.info(f"[ReasoningModelResource] SGLang子进程已启动, pid={self._process.pid}")

            try:
                from src.resource_manager.process_manager import ProcessManager
                ProcessManager.register_process("reasoning_model", self._process)
                logger.info("[ReasoningModelResource] 子进程已注册到ProcessManager")
            except Exception as e:
                logger.warning(f"[ReasoningModelResource] 注册到ProcessManager失败: {e}")
        except Exception as e:
            logger.error(f"[ReasoningModelResource] 启动SGLang子进程失败: {e}")
            raise

    def _wait_for_server_ready(self) -> None:
        """轮询/v1/models等待SGLang服务就绪"""
        config_protocol = self._config.config_protocol
        base_url = config_protocol["base_url"]
        startup_timeout = config_protocol.get("startup_timeout", 300)
        health_check_interval = config_protocol.get("health_check_interval", 5.0)
        health_url = f"{base_url}/v1/models"

        logger.info(f"[ReasoningModelResource] 等待服务就绪: {health_url}, timeout={startup_timeout}s, interval={health_check_interval}s")
        start_time = time.time()
        while time.time() - start_time < startup_timeout:
            try:
                resp = requests.get(health_url, timeout=5.0)
                if resp.status_code == 200:
                    elapsed = time.time() - start_time
                    logger.info(f"[ReasoningModelResource] 服务已就绪, elapsed={elapsed:.1f}s")
                    return
            except Exception as e:
                logger.debug(f"[ReasoningModelResource] 服务尚未就绪: {e}")
            time.sleep(health_check_interval)

        elapsed = time.time() - start_time
        logger.error(f"[ReasoningModelResource] 服务启动超时, elapsed={elapsed:.1f}s, timeout={startup_timeout}s")
        self._stop_server_process()
        raise RuntimeError(f"SGLang服务启动超时({startup_timeout}s)")

    def _stop_server_process(self) -> None:
        """SIGTERM/SIGKILL终止SGLang子进程"""
        if self._process is None:
            return

        config_protocol = self._config.config_protocol
        shutdown_timeout = config_protocol.get("shutdown_timeout", 30)
        pid = self._process.pid
        logger.info(f"[ReasoningModelResource] 停止SGLang子进程, pid={pid}, shutdown_timeout={shutdown_timeout}s")

        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            logger.info(f"[ReasoningModelResource] 进程已不存在, pid={pid}")
            self._process = None
            return

        try:
            self._process.wait(timeout=shutdown_timeout)
            logger.info(f"[ReasoningModelResource] SGLang子进程已正常退出, pid={pid}")
        except subprocess.TimeoutExpired:
            logger.warning(f"[ReasoningModelResource] SGLang子进程未在{shutdown_timeout}s内退出，发送SIGKILL")
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                self._process.wait(timeout=5)
                logger.info(f"[ReasoningModelResource] SGLang子进程已被SIGKILL终止, pid={pid}")
            except Exception as e:
                logger.error(f"[ReasoningModelResource] SIGKILL终止失败: {e}")

        try:
            from src.resource_manager.process_manager import ProcessManager
            ProcessManager.unregister_process("reasoning_model")
        except Exception as e:
            logger.warning(f"[ReasoningModelResource] 从ProcessManager注销失败: {e}")

        self._process = None

    @staticmethod
    def _is_port_in_use(port: int) -> bool:
        """检查端口是否被占用"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return False
            except OSError:
                return True
