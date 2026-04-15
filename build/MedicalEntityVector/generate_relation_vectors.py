import json
import os
import sys
import time
import subprocess
import numpy as np
from typing import List, Dict, Any, Tuple
import torch
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, utility
from config import LOCAL_MODEL_CONFIG, ZILLIZ_CONFIG
from logger import get_logger, log_deployment_step, log_deployment_success, log_deployment_failure


class RelationVectorGenerator:
    def __init__(self):
        self.logger = get_logger()
        self.config = LOCAL_MODEL_CONFIG
        self.model = None
        self.device = None
        self.collection_name = "entity_relations"
        self.collection = None
        self.relations_file = "data/relations.json"
        self.batch_size = 1024
        self.import_batch_size = 2000
        self.max_retries = 3
        
        self.total_count = 0
        self.success_count = 0
        self.failed_count = 0
        self.start_time = None
    
    def load_model(self):
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
    
    def ensure_relations_data(self) -> List[Dict[str, Any]]:
        try:
            log_deployment_step("读取关系数据", "开始")
            
            if not os.path.exists(self.relations_file):
                self.logger.info(f"关系数据文件不存在: {self.relations_file}")
                self.logger.info("正在调用 extract_relations.py 提取关系数据...")
                
                extract_script = os.path.join(os.path.dirname(__file__), "extract_relations.py")
                if not os.path.exists(extract_script):
                    error_msg = f"extract_relations.py 脚本不存在: {extract_script}"
                    self.logger.error(error_msg)
                    raise FileNotFoundError(error_msg)
                
                result = subprocess.run(
                    [sys.executable, extract_script],
                    capture_output=True,
                    text=True,
                    cwd=os.path.dirname(__file__)
                )
                
                if result.returncode != 0:
                    error_msg = f"提取关系数据失败: {result.stderr}"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                self.logger.info("关系数据提取完成")
            
            with open(self.relations_file, 'r', encoding='utf-8') as f:
                relations = json.load(f)
            
            self.logger.info(f"成功加载 {len(relations)} 条关系数据")
            log_deployment_success(f"读取关系数据 (共 {len(relations)} 条)")
            
            return relations
            
        except Exception as e:
            log_deployment_failure("读取关系数据", str(e))
            raise
    
    def generate_vectors_batch(self, texts: List[str]) -> np.ndarray:
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"批量生成向量失败: {e}")
            raise
    
    def generate_all_vectors(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            log_deployment_step("生成关系向量", "开始")
            
            texts = []
            for relation in relations:
                description = relation.get('relation_description', '')
                if not description:
                    source_name = relation.get('source_entity_name', '')
                    target_name = relation.get('target_entity_name', '')
                    relation_name = relation.get('relation_name', '')
                    description = f"{source_name} {relation_name} {target_name}"
                texts.append(description)
            
            all_vector_relations = []
            total_relations = len(relations)
            self.start_time = time.time()
            
            self.logger.info(f"开始生成向量，总数量: {total_relations}")
            
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i+self.batch_size]
                batch_relations = relations[i:i+self.batch_size]
                
                try:
                    embeddings = self.generate_vectors_batch(batch_texts)
                    
                    for j, relation in enumerate(batch_relations):
                        vector_relation = {
                            'vector': embeddings[j].tolist(),
                            'relation_type': relation.get('relation_type', ''),
                            'source_entity_name': relation.get('source_entity_name', ''),
                            'source_entity_type': relation.get('source_entity_type', ''),
                            'target_entity_name': relation.get('target_entity_name', ''),
                            'target_entity_type': relation.get('target_entity_type', ''),
                            'relation_description': relation.get('relation_description', ''),
                            'neo4j_relation_id': relation.get('neo4j_relation_id', '')
                        }
                        all_vector_relations.append(vector_relation)
                    
                    processed_count = min(i + self.batch_size, total_relations)
                    elapsed_time = time.time() - self.start_time
                    speed = processed_count / elapsed_time if elapsed_time > 0 else 0
                    progress = (processed_count / total_relations) * 100
                    
                    self.logger.info(
                        f"进度: {processed_count}/{total_relations} ({progress:.1f}%) | "
                        f"速度: {speed:.1f} 条/秒"
                    )
                    
                    if self.device.startswith('cuda'):
                        torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        self.logger.warning("GPU显存不足，降低批次大小并重试")
                        torch.cuda.empty_cache()
                        self.batch_size = max(self.batch_size // 2, 32)
                        
                        embeddings = self.generate_vectors_batch(batch_texts)
                        
                        for j, relation in enumerate(batch_relations):
                            vector_relation = {
                                'vector': embeddings[j].tolist(),
                                'relation_type': relation.get('relation_type', ''),
                                'source_entity_name': relation.get('source_entity_name', ''),
                                'source_entity_type': relation.get('source_entity_type', ''),
                                'target_entity_name': relation.get('target_entity_name', ''),
                                'target_entity_type': relation.get('target_entity_type', ''),
                                'relation_description': relation.get('relation_description', ''),
                                'neo4j_relation_id': relation.get('neo4j_relation_id', '')
                            }
                            all_vector_relations.append(vector_relation)
                    else:
                        raise
            
            elapsed_time = time.time() - self.start_time
            avg_speed = total_relations / elapsed_time if elapsed_time > 0 else 0
            
            self.logger.info("=" * 60)
            self.logger.info("向量生成统计:")
            self.logger.info("=" * 60)
            self.logger.info(f"  总关系数: {total_relations}")
            self.logger.info(f"  总耗时: {elapsed_time:.2f} 秒")
            self.logger.info(f"  平均速度: {avg_speed:.1f} 条/秒")
            self.logger.info("=" * 60)
            
            log_deployment_success("生成关系向量")
            
            return all_vector_relations
            
        except Exception as e:
            log_deployment_failure("生成关系向量", str(e))
            raise
    
    def validate_vectors(self, vector_relations: List[Dict[str, Any]]) -> bool:
        try:
            log_deployment_step("验证向量质量", "开始")
            
            expected_dimension = self.config.get('dimension', 1024)
            
            sample_size = min(100, len(vector_relations))
            sample_indices = np.random.choice(len(vector_relations), sample_size, replace=False)
            
            dimension_errors = 0
            range_errors = 0
            normalization_errors = 0
            
            for idx in sample_indices:
                relation = vector_relations[idx]
                vector = np.array(relation['vector'])
                
                if len(vector) != expected_dimension:
                    dimension_errors += 1
                    self.logger.error(f"向量维度错误: {relation.get('relation_type', 'unknown')}, 期望 {expected_dimension}, 实际 {len(vector)}")
                
                if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
                    range_errors += 1
                    self.logger.error(f"向量包含无效值: {relation.get('relation_type', 'unknown')}")
                
                norm = np.linalg.norm(vector)
                if abs(norm - 1.0) > 1e-6:
                    normalization_errors += 1
                    self.logger.warning(f"向量未正确归一化, 范数: {norm:.6f}")
            
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
    
    def connect_to_zilliz(self) -> bool:
        try:
            log_deployment_step("连接Zilliz Cloud服务", "开始")
            
            if ZILLIZ_CONFIG["uri"] == "YOUR_ZILLIZ_CLOUD_URI_PLACEHOLDER":
                error_msg = (
                    "Zilliz Cloud URI未配置！\n"
                    "请按以下步骤获取实际URI：\n"
                    "1. 登录Zilliz Cloud控制台 (https://cloud.zilliz.com)\n"
                    "2. 选择您的集群\n"
                    "3. 在集群详情页面找到 'Public Endpoint' 或 'URI'\n"
                    "4. 将config.py中的ZILLIZ_CONFIG['uri']替换为实际地址\n"
                    "示例格式: https://inxx-xxxx.api.gcp-us-west1.zillizcloud.com"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            connections.connect(
                alias="default",
                user=ZILLIZ_CONFIG["user"],
                password=ZILLIZ_CONFIG["password"],
                uri=ZILLIZ_CONFIG["uri"]
            )
            
            self.logger.info(f"成功连接到Zilliz Cloud: {ZILLIZ_CONFIG['uri']}")
            log_deployment_success("连接Zilliz Cloud服务")
            return True
            
        except Exception as e:
            error_msg = f"连接Zilliz Cloud失败: {str(e)}"
            log_deployment_failure("连接Zilliz Cloud服务", error_msg)
            raise ConnectionError(error_msg)
    
    def verify_collection(self) -> bool:
        try:
            log_deployment_step("验证集合是否存在", "开始")
            
            if not utility.has_collection(self.collection_name):
                error_msg = f"集合 '{self.collection_name}' 不存在，请先运行 create_entity_relations_collection.py 创建集合"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            self.collection = Collection(self.collection_name)
            self.collection.load()
            
            self.logger.info(f"集合 '{self.collection_name}' 验证成功")
            self.logger.info(f"当前集合实体数量: {self.collection.num_entities}")
            log_deployment_success("验证集合是否存在")
            return True
            
        except Exception as e:
            error_msg = f"验证集合失败: {str(e)}"
            log_deployment_failure("验证集合是否存在", error_msg)
            raise Exception(error_msg)
    
    def insert_batch_with_retry(self, batch_data: List[Dict[str, Any]], batch_idx: int) -> bool:
        for retry_count in range(self.max_retries):
            try:
                if retry_count > 0:
                    retry_interval = 2 ** retry_count
                    self.logger.warning(f"批次 {batch_idx + 1} 第 {retry_count} 次重试，等待 {retry_interval} 秒...")
                    time.sleep(retry_interval)
                
                self.collection.insert(batch_data)
                self.collection.flush()
                return True
                
            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"批次 {batch_idx + 1} 插入失败 (尝试 {retry_count + 1}/{self.max_retries}): {error_msg}")
                
                if retry_count == self.max_retries - 1:
                    self.logger.error(f"批次 {batch_idx + 1} 已达到最大重试次数，跳过该批次")
                    return False
        
        return False
    
    def import_vectors_batch(self, vector_relations: List[Dict[str, Any]]) -> Tuple[int, int]:
        try:
            log_deployment_step("批量导入关系向量", "开始")
            
            total_batches = (len(vector_relations) + self.import_batch_size - 1) // self.import_batch_size
            
            self.logger.info("=" * 60)
            self.logger.info("开始导入关系向量数据")
            self.logger.info(f"总数据量: {len(vector_relations)} 条")
            self.logger.info(f"批次大小: {self.import_batch_size} 条")
            self.logger.info(f"总批次数: {total_batches}")
            self.logger.info(f"最大重试次数: {self.max_retries}")
            self.logger.info("=" * 60)
            
            import_start_time = time.time()
            batch_success = 0
            batch_failed = 0
            failed_batches = []
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * self.import_batch_size
                end_idx = min(start_idx + self.import_batch_size, len(vector_relations))
                batch_data = vector_relations[start_idx:end_idx]
                
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
                    failed_batches.append({
                        "batch_idx": batch_idx + 1,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "count": len(batch_data)
                    })
            
            if failed_batches:
                self.logger.warning("=" * 60)
                self.logger.warning("失败批次详情:")
                for fb in failed_batches:
                    self.logger.warning(
                        f"  批次 {fb['batch_idx']}: 数据索引 {fb['start_idx']}-{fb['end_idx']} ({fb['count']} 条)"
                    )
                self.logger.warning("=" * 60)
            
            log_deployment_success(f"批量导入关系向量 (成功: {batch_success} 批次, 失败: {batch_failed} 批次)")
            return batch_success, batch_failed
            
        except Exception as e:
            log_deployment_failure("批量导入关系向量", str(e))
            raise
    
    def generate_statistics(self):
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        avg_speed = self.success_count / elapsed_time if elapsed_time > 0 else 0
        
        self.logger.info("=" * 60)
        self.logger.info("关系向量导入统计信息:")
        self.logger.info("=" * 60)
        self.logger.info(f"  总数据量: {self.total_count} 条")
        self.logger.info(f"  成功导入: {self.success_count} 条")
        self.logger.info(f"  失败数量: {self.failed_count} 条")
        self.logger.info(f"  成功率: {self.success_count / self.total_count * 100:.2f}%" if self.total_count > 0 else "  成功率: 0.00%")
        self.logger.info("-" * 60)
        self.logger.info(f"  总耗时: {elapsed_time:.2f} 秒")
        self.logger.info(f"  平均速度: {avg_speed:.2f} 条/秒")
        self.logger.info("=" * 60)
        
        if self.collection:
            final_count = self.collection.num_entities
            self.logger.info(f"  集合最终实体数量: {final_count}")
    
    def disconnect(self):
        try:
            connections.disconnect("default")
            self.logger.info("已断开与Zilliz Cloud的连接")
        except Exception as e:
            self.logger.warning(f"断开连接时出现警告: {str(e)}")
    
    def run(self):
        try:
            self.logger.info("=" * 60)
            self.logger.info("开始执行关系向量生成和导入任务")
            self.logger.info("=" * 60)
            
            self.load_model()
            
            relations = self.ensure_relations_data()
            self.total_count = len(relations)
            
            vector_relations = self.generate_all_vectors(relations)
            
            is_valid = self.validate_vectors(vector_relations)
            
            if not is_valid:
                raise ValueError("向量质量验证失败")
            
            self.connect_to_zilliz()
            self.verify_collection()
            
            batch_success, batch_failed = self.import_vectors_batch(vector_relations)
            
            self.generate_statistics()
            
            self.logger.info("=" * 60)
            self.logger.info("关系向量生成和导入任务完成！")
            self.logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"关系向量生成和导入任务失败: {e}")
            return False
        finally:
            self.disconnect()


def main():
    generator = RelationVectorGenerator()
    success = generator.run()
    
    if success:
        print("\n✓ 关系向量生成和导入成功!")
        print(f"✓ 处理关系数量: {generator.total_count} 条")
        print(f"✓ 成功导入数量: {generator.success_count} 条")
        print(f"✓ 失败数量: {generator.failed_count} 条")
        return 0
    else:
        print("\n✗ 关系向量生成和导入失败")
        return 1


if __name__ == "__main__":
    exit(main())
