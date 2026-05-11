#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# AI辅助生成：GLM-5, 2026-04-18
"""
日志管理模块
提供严格的日志打印与管理功能
"""

import logging
import os
import json
from datetime import datetime
from typing import Dict, Any
import traceback


class DeploymentLogger:
    """部署日志管理器"""
    
    def __init__(self, log_dir: str = "logs", deployment_id: str = None):
        """
        初始化日志管理器
        
        Args:
            log_dir: 日志目录
            deployment_id: 部署ID，用于追溯
        """
        self.log_dir = log_dir
        self.deployment_id = deployment_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.main_log_file = os.path.join(log_dir, f"deployment_{timestamp}.log")
        self.error_log_file = os.path.join(log_dir, f"error_{datetime.now().strftime('%Y%m%d')}.log")
        self.metrics_file = os.path.join(log_dir, f"metrics_{datetime.now().strftime('%Y%m%d')}.json")
        
        self.main_logger = self._setup_logger(
            "deployment",
            self.main_log_file,
            logging.INFO
        )
        
        self.error_logger = self._setup_logger(
            "error",
            self.error_log_file,
            logging.ERROR
        )
        
        self.metrics_data = {
            "deployment_id": self.deployment_id,
            "start_time": datetime.now().isoformat(),
            "steps": [],
            "errors": [],
            "performance": {}
        }
        
        self._log_deployment_start()
    
    def _setup_logger(self, name: str, log_file: str, level: int) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        if logger.handlers:
            logger.handlers.clear()
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _log_deployment_start(self):
        """记录部署开始"""
        self.main_logger.info("=" * 80)
        self.main_logger.info("MedicalEntityVector 完整向量数据库部署")
        self.main_logger.info(f"部署ID: {self.deployment_id}")
        self.main_logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.main_logger.info("=" * 80)
    
    def log_step_start(self, step_id: int, step_name: str, description: str):
        """
        记录步骤开始
        
        Args:
            step_id: 步骤ID
            step_name: 步骤名称
            description: 步骤描述
        """
        self.main_logger.info("=" * 80)
        self.main_logger.info(f"步骤 {step_id}: {step_name}")
        self.main_logger.info(f"描述: {description}")
        self.main_logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.main_logger.info("=" * 80)
        
        self.metrics_data["steps"].append({
            "step_id": step_id,
            "step_name": step_name,
            "description": description,
            "start_time": datetime.now().isoformat(),
            "status": "in_progress"
        })
    
    def log_step_progress(self, current: int, total: int, speed: float = None, eta: str = None):
        """
        记录步骤进度
        
        Args:
            current: 当前进度
            total: 总数
            speed: 速度（条/秒）
            eta: 预计剩余时间
        """
        percentage = (current / total * 100) if total > 0 else 0
        self.main_logger.info(f"进度: {current}/{total} ({percentage:.1f}%)")
        if speed:
            self.main_logger.info(f"速度: {speed:.2f} 条/秒")
        if eta:
            self.main_logger.info(f"预计剩余时间: {eta}")
    
    def log_step_complete(self, step_id: int, duration: float, success_count: int, failed_count: int = 0):
        """
        记录步骤完成
        
        Args:
            step_id: 步骤ID
            duration: 耗时（秒）
            success_count: 成功数量
            failed_count: 失败数量
        """
        self.main_logger.info("=" * 80)
        self.main_logger.info(f"步骤 {step_id}: 完成")
        self.main_logger.info(f"耗时: {duration:.2f} 秒")
        self.main_logger.info(f"成功: {success_count}")
        if failed_count > 0:
            self.main_logger.info(f"失败: {failed_count}")
        self.main_logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.main_logger.info("=" * 80)
        
        for step in self.metrics_data["steps"]:
            if step["step_id"] == step_id:
                step["end_time"] = datetime.now().isoformat()
                step["duration"] = duration
                step["success_count"] = success_count
                step["failed_count"] = failed_count
                step["status"] = "completed"
                break
    
    def log_error(self, step_id: int, step_name: str, error: Exception):
        """
        记录错误
        
        Args:
            step_id: 步骤ID
            step_name: 步骤名称
            error: 错误对象
        """
        error_type = type(error).__name__
        error_message = str(error)
        error_traceback = traceback.format_exc()
        
        self.error_logger.error("=" * 80)
        self.error_logger.error(f"步骤 {step_id}: {step_name} - 失败")
        self.error_logger.error(f"错误类型: {error_type}")
        self.error_logger.error(f"错误信息: {error_message}")
        self.error_logger.error(f"错误堆栈: {error_traceback}")
        self.error_logger.error(f"失败时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.error_logger.error("=" * 80)
        
        self.main_logger.error(f"步骤 {step_id} 失败: {error_message}")
        
        self.metrics_data["errors"].append({
            "step_id": step_id,
            "step_name": step_name,
            "error_type": error_type,
            "error_message": error_message,
            "error_traceback": error_traceback,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_data_extraction(self, entity_type: str, query: str, count: int, duration: float):
        """
        记录数据提取
        
        Args:
            entity_type: 实体类型
            query: 查询条件
            count: 提取数量
            duration: 提取耗时
        """
        self.main_logger.info(f"从Neo4j提取数据: {entity_type}")
        self.main_logger.info(f"查询条件: {query}")
        self.main_logger.info(f"提取数量: {count}")
        self.main_logger.info(f"提取耗时: {duration:.2f} 秒")
    
    def log_vector_generation(self, batch_id: int, total_batches: int, batch_size: int, 
                              speed: float, gpu_memory: float = None, normalization_error: float = None):
        """
        记录向量生成
        
        Args:
            batch_id: 批次ID
            total_batches: 总批次数
            batch_size: 批次大小
            speed: 生成速度
            gpu_memory: GPU显存使用
            normalization_error: 归一化误差
        """
        self.main_logger.info(f"向量生成批次: {batch_id}/{total_batches}")
        self.main_logger.info(f"批次大小: {batch_size}")
        self.main_logger.info(f"生成速度: {speed:.2f} 条/秒")
        if gpu_memory:
            self.main_logger.info(f"GPU显存使用: {gpu_memory:.2f} GB")
        if normalization_error:
            self.main_logger.info(f"向量质量: 归一化误差={normalization_error:.2e}")
    
    def log_vector_import(self, batch_id: int, total_batches: int, count: int, 
                          duration: float, speed: float):
        """
        记录向量导入
        
        Args:
            batch_id: 批次ID
            total_batches: 总批次数
            count: 导入数量
            duration: 导入耗时
            speed: 导入速度
        """
        self.main_logger.info(f"导入批次: {batch_id}/{total_batches}")
        self.main_logger.info(f"导入数量: {count}")
        self.main_logger.info(f"导入耗时: {duration:.2f} 秒")
        self.main_logger.info(f"导入速度: {speed:.2f} 条/秒")
    
    def log_retrieval_test(self, test_case_id: int, query_text: str, latency: float,
                           result_count: int, top1_similarity: float, top5_similarity: float):
        """
        记录检索测试
        
        Args:
            test_case_id: 测试用例ID
            query_text: 查询文本
            latency: 检索延迟
            result_count: 返回结果数
            top1_similarity: Top-1相似度
            top5_similarity: Top-5相似度
        """
        self.main_logger.info(f"检索测试: {test_case_id}")
        self.main_logger.info(f"查询文本: {query_text}")
        self.main_logger.info(f"检索延迟: {latency:.2f} ms")
        self.main_logger.info(f"返回结果数: {result_count}")
        self.main_logger.info(f"Top-1相似度: {top1_similarity:.4f}")
        self.main_logger.info(f"Top-5相似度: {top5_similarity:.4f}")
    
    def log_system_resources(self, gpu_memory: float, total_gpu_memory: float,
                             cpu_usage: float, memory_usage: float, total_memory: float):
        """
        记录系统资源
        
        Args:
            gpu_memory: GPU显存使用
            total_gpu_memory: 总GPU显存
            cpu_usage: CPU使用率
            memory_usage: 内存使用
            total_memory: 总内存
        """
        self.main_logger.info(f"GPU显存使用: {gpu_memory:.2f} GB / {total_gpu_memory:.2f} GB")
        self.main_logger.info(f"CPU使用率: {cpu_usage:.1f}%")
        self.main_logger.info(f"内存使用: {memory_usage:.2f} GB / {total_memory:.2f} GB")
    
    def save_metrics(self):
        """保存性能指标到JSON文件"""
        self.metrics_data["end_time"] = datetime.now().isoformat()
        
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_data, f, ensure_ascii=False, indent=2)
        
        self.main_logger.info(f"性能指标已保存到: {self.metrics_file}")
    
    def log_deployment_complete(self, success: bool):
        """
        记录部署完成
        
        Args:
            success: 是否成功
        """
        self.main_logger.info("=" * 80)
        if success:
            self.main_logger.info("部署成功完成！")
        else:
            self.main_logger.error("部署失败！")
        self.main_logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.main_logger.info("=" * 80)
        
        self.save_metrics()


def create_logger(log_dir: str = "logs", deployment_id: str = None) -> DeploymentLogger:
    """
    创建日志管理器
    
    Args:
        log_dir: 日志目录
        deployment_id: 部署ID
    
    Returns:
        DeploymentLogger实例
    """
    return DeploymentLogger(log_dir, deployment_id)


_global_logger = None

class LoggerAdapter:
    """日志适配器，为标准Logger添加兼容方法"""
    
    def __init__(self, logger):
        self.logger = logger
    
    def __getattr__(self, name):
        """代理所有其他属性到标准Logger"""
        return getattr(self.logger, name)
    
    def log_deployment_step(self, step_name: str, description: str = ""):
        """记录部署步骤"""
        self.logger.info(f"[步骤] {step_name}: {description}")
    
    def log_deployment_success(self, step_name: str, details: str = ""):
        """记录部署成功"""
        self.logger.info(f"[成功] {step_name}: {details}")
    
    def log_deployment_failure(self, step_name: str, error: str):
        """记录部署失败"""
        self.logger.error(f"[失败] {step_name}: {error}")

def get_logger():
    """获取全局日志管理器（兼容旧接口）"""
    global _global_logger
    if _global_logger is None:
        _global_logger = create_logger()
    return LoggerAdapter(_global_logger.main_logger)

def log_deployment_step(step_name: str, description: str = ""):
    """记录部署步骤（兼容旧接口）"""
    logger = get_logger()
    logger.log_deployment_step(step_name, description)

def log_deployment_success(step_name: str, details: str = ""):
    """记录部署成功（兼容旧接口）"""
    logger = get_logger()
    logger.log_deployment_success(step_name, details)

def log_deployment_failure(step_name: str, error: str):
    """记录部署失败（兼容旧接口）"""
    logger = get_logger()
    logger.log_deployment_failure(step_name, error)
