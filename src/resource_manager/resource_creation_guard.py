# -*- coding: utf-8 -*-
"""
资源创建保护器类

防止内存/显存不足导致系统崩溃，在资源创建前进行检查。
"""

import logging
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


class ResourceCreationGuard:
    """
    资源创建保护器类
    
    在资源创建前检查系统资源（内存、显存），防止因资源不足导致系统崩溃。
    
    Attributes:
        min_memory_mb: 创建资源所需最小内存（MB）
        min_vram_mb: 创建资源所需最小显存（MB）
        enabled: 是否启用创建前检查
    """
    
    SAFETY_MARGIN = 0.2  # 安全余量比例（20%）
    
    def __init__(
        self,
        min_memory_mb: int = 512,
        min_vram_mb: int = 0,
        enabled: bool = True
    ):
        """
        初始化资源创建保护器
        
        Args:
            min_memory_mb: 创建资源所需最小内存（MB）
            min_vram_mb: 创建资源所需最小显存（MB），0表示不检查
            enabled: 是否启用创建前检查
        """
        self._min_memory_mb = min_memory_mb
        self._min_vram_mb = min_vram_mb
        self._enabled = enabled
        logger.info(f"[CREATION_GUARD] 初始化: min_memory_mb={min_memory_mb}, "
                   f"min_vram_mb={min_vram_mb}, enabled={enabled}")
    
    @property
    def min_memory_mb(self) -> int:
        """获取最小内存要求"""
        return self._min_memory_mb
    
    @property
    def min_vram_mb(self) -> int:
        """获取最小显存要求"""
        return self._min_vram_mb
    
    @property
    def enabled(self) -> bool:
        """获取是否启用"""
        return self._enabled
    
    def check_before_creation(self, resource_config: Any = None) -> Tuple[bool, str]:
        """
        创建前检查

        Args:
            resource_config: 资源配置（可选，用于获取特定资源的要求）

        Returns:
            Tuple[bool, str]: (是否允许创建, 原因说明)
        """
        if not self._enabled:
            logger.info("[RESOURCE_CREATE_GUARD] enabled=False, memory_check=skipped, vram_check=skipped, allowed=True, reason=创建前检查已禁用")
            return True, "创建前检查已禁用"

        memory_ok = True
        vram_ok = True
        memory_detail = "skipped"
        vram_detail = "skipped"

        if self._min_memory_mb > 0:
            memory_ok, memory_reason = self._check_memory()
            memory_detail = f"ok={memory_ok}, detail={memory_reason}"
            if not memory_ok:
                logger.info(f"[RESOURCE_CREATE_GUARD] memory_check={memory_ok}, vram_check=skipped, allowed=False, memory_detail={memory_detail}, min_memory_mb={self._min_memory_mb}, min_vram_mb={self._min_vram_mb}")
                return False, memory_reason

        if self._min_vram_mb > 0:
            vram_ok, vram_reason = self._check_vram()
            vram_detail = f"ok={vram_ok}, detail={vram_reason}"
            if not vram_ok:
                logger.info(f"[RESOURCE_CREATE_GUARD] memory_check={memory_ok}, vram_check={vram_ok}, allowed=False, memory_detail={memory_detail}, vram_detail={vram_detail}, min_memory_mb={self._min_memory_mb}, min_vram_mb={self._min_vram_mb}")
                return False, vram_reason

        logger.info(f"[CREATION_GUARD] 创建前检查通过: enabled={self._enabled}, memory_ok={memory_ok}, vram_ok={vram_ok}")
        logger.info(f"[RESOURCE_CREATE_GUARD] memory_check={memory_ok}, vram_check={vram_ok}, allowed=True, memory_detail={memory_detail}, vram_detail={vram_detail}, min_memory_mb={self._min_memory_mb}, min_vram_mb={self._min_vram_mb}")
        return True, "资源检查通过"
    
    def _check_memory(self) -> Tuple[bool, str]:
        """
        检查系统可用内存
        
        Returns:
            Tuple[bool, str]: (是否满足要求, 原因说明)
        """
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            available_mb = memory_info.available / (1024 * 1024)
            required_mb = self._min_memory_mb * (1 + self.SAFETY_MARGIN)
            
            if available_mb < required_mb:
                reason = (f"内存不足: 可用{available_mb:.0f}MB, "
                         f"需要{required_mb:.0f}MB (含{self.SAFETY_MARGIN*100:.0f}%安全余量)")
                logger.warning(f"[CREATION_GUARD] {reason}")
                return False, reason
            
            logger.debug(f"[CREATION_GUARD] 内存检查通过: "
                        f"可用{available_mb:.0f}MB, 需要{required_mb:.0f}MB")
            return True, "内存检查通过"
            
        except ImportError:
            logger.warning("[CREATION_GUARD] psutil未安装，跳过内存检查")
            return True, "psutil未安装，跳过内存检查"
        except Exception as e:
            logger.error(f"[CREATION_GUARD] 内存检查异常: {e}")
            return True, f"内存检查异常: {e}"
    
    def _check_vram(self) -> Tuple[bool, str]:
        """
        检查系统可用显存
        
        Returns:
            Tuple[bool, str]: (是否满足要求, 原因说明)
        """
        try:
            import torch
            if not torch.cuda.is_available():
                logger.debug("[CREATION_GUARD] CUDA不可用，跳过显存检查")
                return True, "CUDA不可用，跳过显存检查"
            
            gpu_id = 0
            total_vram = torch.cuda.get_device_properties(gpu_id).total_memory
            allocated_vram = torch.cuda.memory_allocated(gpu_id)
            available_vram = total_vram - allocated_vram
            available_mb = available_vram / (1024 * 1024)
            required_mb = self._min_vram_mb * (1 + self.SAFETY_MARGIN)
            
            if available_mb < required_mb:
                reason = (f"显存不足: 可用{available_mb:.0f}MB, "
                         f"需要{required_mb:.0f}MB (含{self.SAFETY_MARGIN*100:.0f}%安全余量)")
                logger.warning(f"[CREATION_GUARD] {reason}")
                return False, reason
            
            logger.debug(f"[CREATION_GUARD] 显存检查通过: "
                        f"可用{available_mb:.0f}MB, 需要{required_mb:.0f}MB")
            return True, "显存检查通过"
            
        except ImportError:
            logger.warning("[CREATION_GUARD] torch未安装，跳过显存检查")
            return True, "torch未安装，跳过显存检查"
        except Exception as e:
            logger.error(f"[CREATION_GUARD] 显存检查异常: {e}")
            return True, f"显存检查异常: {e}"
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        获取系统资源信息
        
        Returns:
            Dict[str, Any]: 系统资源信息字典
        """
        info = {
            "memory": {},
            "vram": {},
            "guard_config": {
                "min_memory_mb": self._min_memory_mb,
                "min_vram_mb": self._min_vram_mb,
                "enabled": self._enabled,
                "safety_margin": self.SAFETY_MARGIN
            }
        }
        
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            info["memory"] = {
                "total_mb": memory_info.total / (1024 * 1024),
                "available_mb": memory_info.available / (1024 * 1024),
                "used_mb": memory_info.used / (1024 * 1024),
                "percent": memory_info.percent
            }
        except ImportError:
            info["memory"] = {"error": "psutil未安装"}
        except Exception as e:
            info["memory"] = {"error": str(e)}
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_id = 0
                total = torch.cuda.get_device_properties(gpu_id).total_memory
                allocated = torch.cuda.memory_allocated(gpu_id)
                info["vram"] = {
                    "total_mb": total / (1024 * 1024),
                    "available_mb": (total - allocated) / (1024 * 1024),
                    "allocated_mb": allocated / (1024 * 1024),
                    "device_name": torch.cuda.get_device_name(gpu_id)
                }
            else:
                info["vram"] = {"error": "CUDA不可用"}
        except ImportError:
            info["vram"] = {"error": "torch未安装"}
        except Exception as e:
            info["vram"] = {"error": str(e)}
        
        return info
    
    def __repr__(self) -> str:
        """返回保护器对象的字符串表示"""
        return (f"ResourceCreationGuard(min_memory_mb={self._min_memory_mb}, "
                f"min_vram_mb={self._min_vram_mb}, enabled={self._enabled})")
