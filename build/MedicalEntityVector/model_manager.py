# AI辅助生成：GLM-5, 2026-04-18
"""
模型管理工具
用于管理base_models目录下的模型
"""

import os
import shutil
from pathlib import Path
from logger import get_logger


class ModelManager:
    def __init__(self, base_path="/home/project/MedicalQA/base_models"):
        self.logger = get_logger()
        self.base_path = Path(base_path)
        
    def list_models(self):
        self.logger.info("=" * 60)
        self.logger.info("已安装模型列表")
        self.logger.info("=" * 60)
        
        models = []
        for item in self.base_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                if item.name.startswith('models--'):
                    model_name = item.name.replace('models--', '').replace('--', '/')
                    models.append({
                        'name': model_name,
                        'path': str(item),
                        'type': 'HuggingFace格式'
                    })
                else:
                    models.append({
                        'name': item.name,
                        'path': str(item),
                        'type': '自定义模型'
                    })
        
        for i, model in enumerate(models, 1):
            self.logger.info(f"{i}. {model['name']}")
            self.logger.info(f"   路径: {model['path']}")
            self.logger.info(f"   类型: {model['type']}")
            self.logger.info("")
        
        return models
    
    def clean_locks(self):
        locks_path = self.base_path / '.locks'
        if locks_path.exists():
            self.logger.info(f"清理锁文件: {locks_path}")
            shutil.rmtree(locks_path)
            self.logger.info("✓ 锁文件已清理")
        else:
            self.logger.info("无需清理，锁文件不存在")
    
    def get_model_info(self, model_name):
        model_path = self.base_path / f"models--{model_name.replace('/', '--')}"
        
        if not model_path.exists():
            self.logger.warning(f"模型不存在: {model_name}")
            return None
        
        info = {
            'name': model_name,
            'path': str(model_path),
            'size': self._get_dir_size(model_path)
        }
        
        return info
    
    def _get_dir_size(self, path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if not os.path.islink(filepath):
                    total_size += os.path.getsize(filepath)
        
        return f"{total_size / 1024 / 1024 / 1024:.2f} GB"


if __name__ == "__main__":
    manager = ModelManager()
    
    print("\n1. 列出所有模型")
    print("-" * 60)
    models = manager.list_models()
    
    print("\n2. 查看向量模型信息")
    print("-" * 60)
    info = manager.get_model_info("BAAI/bge-large-zh-v1.5")
    if info:
        print(f"模型名称: {info['name']}")
        print(f"模型路径: {info['path']}")
        print(f"模型大小: {info['size']}")
    
    print("\n3. 清理锁文件（可选）")
    print("-" * 60)
    print("提示: 锁文件可以安全删除，但建议在模型下载完成后执行")
    print("执行清理: manager.clean_locks()")
