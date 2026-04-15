"""
属性向量生成与导入模块
用于从Disease实体属性生成向量并导入到entity_attributes集合
"""

import json
import os
import time
import numpy as np
from typing import List, Dict, Any, Optional
import torch
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, utility
from config import LOCAL_MODEL_CONFIG, ZILLIZ_CONFIG
from logger import get_logger, log_deployment_step, log_deployment_success, log_deployment_failure


class AttributeVectorGenerator:
    """属性向量生成器"""
    
    def __init__(self):
        self.logger = get_logger()
        self.config = LOCAL_MODEL_CONFIG
        self.model = None
        self.device = None
        self.collection_name = "entity_attributes"
        self.collection = None
        self.input_file = "data/disease_attributes.json"
        
        self.total_count = 0
        self.success_count = 0
        self.failed_count = 0
        self.start_time = None
    
    def load_model(self):
        """加载本地向量模型"""
        try:
            log_deployment_step("加载本地向量模型", "开始")
            
            if torch.cuda.is_available():
                self.device = self.config.get('device', 'cuda')
                torch.cuda.set_per_process_memory_fraction(
                    self.config.get('memory_fraction', 0.95),
                    device=0
                )
                self.logger.info(f"使用GPU设备: {self.device}")
                self.logger.info(f"GPU显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            else:
                self.device = "cpu"
                self.logger.warning("GPU不可用，使用CPU")
            
            self.model = SentenceTransformer(
                self.config['model_name'],
                cache_folder=os.path.expanduser(self.config.get('cache_dir', '~/.cache/huggingface/'))
            )
            
            self.model.to(self.device)
            
            if self.device.startswith('cuda'):
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                self.logger.info(f"模型加载后GPU显存占用: {allocated:.2f} GB")
            
            log_deployment_success("加载本地向量模型")
            
        except Exception as e:
            log_deployment_failure("加载本地向量模型", str(e))
            raise
    
    def load_attributes_data(self) -> List[Dict[str, Any]]:
        """加载属性数据，如果不存在则先提取"""
        try:
            log_deployment_step("加载属性数据", "开始")
            
            if not os.path.exists(self.input_file):
                self.logger.warning(f"属性数据文件不存在: {self.input_file}")
                self.logger.info("开始调用 extract_entity_attributes.py 提取数据...")
                
                from extract_entity_attributes import DiseaseAttributeExtractor
                extractor = DiseaseAttributeExtractor()
                attributes_data = extractor.run()
                
                self.logger.info(f"成功提取 {len(attributes_data)} 条属性数据")
            else:
                with open(self.input_file, 'r', encoding='utf-8') as f:
                    attributes_data = json.load(f)
                
                self.logger.info(f"成功加载属性数据: {len(attributes_data)} 条")
            
            self.total_count = len(attributes_data)
            log_deployment_success(f"加载属性数据 (共 {self.total_count} 条)")
            return attributes_data
            
        except Exception as e:
            error_msg = f"加载属性数据失败: {str(e)}"
            log_deployment_failure("加载属性数据", error_msg)
            raise
    
    def generate_vectors_batch(self, texts: List[str]) -> np.ndarray:
        """批量生成向量"""
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.get('batch_size', 512),
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=self.config.get('normalize', True)
            )
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"批量生成向量失败: {e}")
            raise
    
    def generate_all_vectors(self, attributes_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成所有属性的向量"""
        try:
            log_deployment_step("生成属性向量", "开始")
            
            all_vector_data = []
            batch_size = self.config.get('batch_size', 512)
            
            self.start_time = time.time()
            
            texts = [attr['vector_text'] for attr in attributes_data]
            
            self.logger.info(f"开始生成向量，总数据量: {len(texts)} 条")
            self.logger.info(f"批次大小: {batch_size}")
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_attrs = attributes_data[i:i+batch_size]
                
                try:
                    embeddings = self.generate_vectors_batch(batch_texts)
                    
                    for j, attr in enumerate(batch_attrs):
                        vector_item = {
                            'vector': embeddings[j].tolist(),
                            'entity_name': attr['entity_name'],
                            'entity_type': attr['entity_type'],
                            'attribute_name': attr['attribute_name'],
                            'attribute_value': attr['attribute_value'],
                            'neo4j_node_id': attr['neo4j_node_id']
                        }
                        all_vector_data.append(vector_item)
                    
                    processed_count = min(i + batch_size, len(texts))
                    elapsed_time = time.time() - self.start_time
                    speed = processed_count / elapsed_time if elapsed_time > 0 else 0
                    progress = (processed_count / len(texts)) * 100
                    
                    self.logger.info(
                        f"进度: {processed_count}/{len(texts)} ({progress:.1f}%) | "
                        f"速度: {speed:.1f} 条/秒"
                    )
                    
                    if self.device.startswith('cuda'):
                        torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        self.logger.warning("GPU显存不足，降低批次大小并重试")
                        torch.cuda.empty_cache()
                        batch_size = max(batch_size // 2, 32)
                        
                        embeddings = self.generate_vectors_batch(batch_texts)
                        
                        for j, attr in enumerate(batch_attrs):
                            vector_item = {
                                'vector': embeddings[j].tolist(),
                                'entity_name': attr['entity_name'],
                                'entity_type': attr['entity_type'],
                                'attribute_name': attr['attribute_name'],
                                'attribute_value': attr['attribute_value'],
                                'neo4j_node_id': attr['neo4j_node_id']
                            }
                            all_vector_data.append(vector_item)
                    else:
                        raise
            
            elapsed_time = time.time() - self.start_time
            avg_speed = len(texts) / elapsed_time if elapsed_time > 0 else 0
            
            self.logger.info("=" * 60)
            self.logger.info("向量生成统计:")
            self.logger.info("=" * 60)
            self.logger.info(f"  总属性数: {len(texts)}")
            self.logger.info(f"  总耗时: {elapsed_time:.2f} 秒")
            self.logger.info(f"  平均速度: {avg_speed:.1f} 条/秒")
            self.logger.info("=" * 60)
            
            log_deployment_success("生成属性向量")
            return all_vector_data
            
        except Exception as e:
            log_deployment_failure("生成属性向量", str(e))
            raise
    
    def validate_vectors(self, vector_data: List[Dict[str, Any]]) -> bool:
        """验证向量维度和质量"""
        try:
            log_deployment_step("验证向量质量", "开始")
            
            expected_dimension = self.config.get('dimension', 1024)
            
            sample_size = min(100, len(vector_data))
            sample_indices = np.random.choice(len(vector_data), sample_size, replace=False)
            
            dimension_errors = 0
            range_errors = 0
            normalization_errors = 0
            
            for idx in sample_indices:
                item = vector_data[idx]
                vector = np.array(item['vector'])
                
                if len(vector) != expected_dimension:
                    dimension_errors += 1
                    self.logger.error(
                        f"向量维度错误: {item['entity_name']}-{item['attribute_name']}, "
                        f"期望 {expected_dimension}, 实际 {len(vector)}"
                    )
                
                if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
                    range_errors += 1
                    self.logger.error(
                        f"向量包含无效值: {item['entity_name']}-{item['attribute_name']}"
                    )
                
                norm = np.linalg.norm(vector)
                if self.config.get('normalize', True):
                    if abs(norm - 1.0) > 1e-6:
                        normalization_errors += 1
                        self.logger.warning(
                            f"向量未正确归一化: {item['entity_name']}-{item['attribute_name']}, "
                            f"范数: {norm:.6f}"
                        )
            
            self.logger.info(f"向量质量验证结果:")
            self.logger.info(f"  抽样数量: {sample_size}")
            self.logger.info(f"  维度错误: {dimension_errors}")
            self.logger.info(f"  值范围错误: {range_errors}")
            self.logger.info(f"  归一化错误: {normalization_errors}")
            
            if dimension_errors > 0 or range_errors > 0:
                log_deployment_failure("验证向量质量", "发现严重错误")
                return False
            
            log_deployment_success("验证向量质量")
            return True
            
        except Exception as e:
            log_deployment_failure("验证向量质量", str(e))
            return False
    
    def connect_to_zilliz(self):
        """连接到Zilliz Cloud服务"""
        log_deployment_step("连接Zilliz Cloud服务", "开始")
        
        try:
            connections.connect(
                alias="default",
                token=ZILLIZ_CONFIG["token"],
                uri=ZILLIZ_CONFIG["uri"]
            )
            
            self.logger.info(f"成功连接到Zilliz Cloud: {ZILLIZ_CONFIG['uri']}")
            log_deployment_success("连接Zilliz Cloud服务")
            
        except Exception as e:
            error_msg = f"连接Zilliz Cloud失败: {str(e)}"
            log_deployment_failure("连接Zilliz Cloud服务", error_msg)
            raise ConnectionError(error_msg)
    
    def verify_collection(self):
        """验证集合是否存在"""
        log_deployment_step("验证集合是否存在", "开始")
        
        try:
            if not utility.has_collection(self.collection_name):
                error_msg = (
                    f"集合 '{self.collection_name}' 不存在！\n"
                    f"请先运行 create_entity_attributes_collection.py 创建集合"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            self.collection = Collection(self.collection_name)
            self.collection.load()
            
            self.logger.info(f"集合 '{self.collection_name}' 验证成功")
            self.logger.info(f"当前集合实体数量: {self.collection.num_entities}")
            log_deployment_success("验证集合是否存在")
            
        except Exception as e:
            error_msg = f"验证集合失败: {str(e)}"
            log_deployment_failure("验证集合是否存在", error_msg)
            raise
    
    def insert_batch_with_retry(self, batch_data: List[Dict[str, Any]], batch_idx: int, max_retries: int = 3) -> bool:
        """带重试机制的批量插入"""
        for retry_count in range(max_retries):
            try:
                if retry_count > 0:
                    retry_interval = 2 ** retry_count
                    self.logger.warning(
                        f"批次 {batch_idx + 1} 第 {retry_count} 次重试，等待 {retry_interval} 秒..."
                    )
                    time.sleep(retry_interval)
                
                self.collection.insert(batch_data)
                self.collection.flush()
                return True
                
            except Exception as e:
                error_msg = str(e)
                self.logger.error(
                    f"批次 {batch_idx + 1} 插入失败 (尝试 {retry_count + 1}/{max_retries}): {error_msg}"
                )
                
                if retry_count == max_retries - 1:
                    self.logger.error(f"批次 {batch_idx + 1} 已达到最大重试次数，跳过该批次")
                    return False
        
        return False
    
    def import_vectors_batch(self, vector_data: List[Dict[str, Any]], batch_size: int = 1000):
        """批量导入向量数据"""
        log_deployment_step("批量导入向量数据", "开始")
        
        total_batches = (len(vector_data) + batch_size - 1) // batch_size
        
        self.logger.info("=" * 60)
        self.logger.info("开始导入向量数据")
        self.logger.info(f"总数据量: {len(vector_data)} 条")
        self.logger.info(f"批次大小: {batch_size} 条")
        self.logger.info(f"总批次数: {total_batches}")
        self.logger.info("=" * 60)
        
        import_start_time = time.time()
        batch_success = 0
        batch_failed = 0
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(vector_data))
            batch_data = vector_data[start_idx:end_idx]
            
            progress_pct = (batch_idx + 1) / total_batches * 100
            elapsed_time = time.time() - import_start_time
            speed = self.success_count / elapsed_time if elapsed_time > 0 else 0
            
            self.logger.info(
                f"进度: {batch_idx + 1}/{total_batches} ({progress_pct:.1f}%) | "
                f"成功: {self.success_count} | 失败: {self.failed_count} | "
                f"速度: {speed:.1f} 条/秒"
            )
            
            success = self.insert_batch_with_retry(batch_data, batch_idx)
            
            if success:
                batch_success += 1
                self.success_count += len(batch_data)
                self.logger.info(f"批次 {batch_idx + 1} 导入成功 ({len(batch_data)} 条)")
            else:
                batch_failed += 1
                self.failed_count += len(batch_data)
        
        log_deployment_success(
            f"批量导入向量数据 (成功: {batch_success} 批次, 失败: {batch_failed} 批次)"
        )
    
    def generate_statistics(self):
        """生成统计信息"""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        avg_speed = self.total_count / elapsed_time if elapsed_time > 0 else 0
        
        self.logger.info("=" * 60)
        self.logger.info("属性向量处理统计信息:")
        self.logger.info("=" * 60)
        self.logger.info(f"  总属性数: {self.total_count} 条")
        self.logger.info(f"  成功导入: {self.success_count} 条")
        self.logger.info(f"  失败数量: {self.failed_count} 条")
        self.logger.info(
            f"  成功率: {self.success_count / self.total_count * 100:.2f}%"
            if self.total_count > 0 else "  成功率: 0.00%"
        )
        self.logger.info("-" * 60)
        self.logger.info(f"  总耗时: {elapsed_time:.2f} 秒")
        self.logger.info(f"  平均速度: {avg_speed:.2f} 条/秒")
        self.logger.info("=" * 60)
        
        if self.collection:
            final_count = self.collection.num_entities
            self.logger.info(f"  集合最终实体数量: {final_count}")
    
    def disconnect(self):
        """断开连接"""
        try:
            connections.disconnect("default")
            self.logger.info("已断开与Zilliz Cloud的连接")
        except Exception as e:
            self.logger.warning(f"断开连接时出现警告: {str(e)}")
    
    def run(self):
        """执行完整的属性向量生成与导入流程"""
        self.logger.info("=" * 60)
        self.logger.info("开始执行属性向量生成与导入任务")
        self.logger.info("=" * 60)
        
        try:
            self.load_model()
            
            attributes_data = self.load_attributes_data()
            if not attributes_data:
                self.logger.warning("未加载到任何属性数据")
                return False
            
            vector_data = self.generate_all_vectors(attributes_data)
            
            is_valid = self.validate_vectors(vector_data)
            if not is_valid:
                raise ValueError("向量质量验证失败")
            
            self.connect_to_zilliz()
            self.verify_collection()
            
            self.import_vectors_batch(vector_data, batch_size=1000)
            
            self.generate_statistics()
            
            self.logger.info("=" * 60)
            self.logger.info("属性向量生成与导入任务完成！")
            self.logger.info("=" * 60)
            
            return self.failed_count == 0
            
        except Exception as e:
            self.logger.error(f"属性向量生成与导入任务失败: {str(e)}")
            raise
        finally:
            self.disconnect()


def main():
    """主函数"""
    generator = AttributeVectorGenerator()
    
    try:
        success = generator.run()
        
        if success:
            print("\n✓ 属性向量生成与导入成功!")
            print(f"✓ 处理属性数: {generator.total_count} 条")
            print(f"✓ 成功导入: {generator.success_count} 条")
            print(f"✓ 失败数量: {generator.failed_count} 条")
        else:
            print("\n⚠ 属性向量生成与导入完成，但有部分失败")
            print(f"✓ 成功导入: {generator.success_count} 条")
            print(f"✗ 失败数量: {generator.failed_count} 条")
            print("请查看日志了解详情")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"\n✗ 属性向量生成与导入失败: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
