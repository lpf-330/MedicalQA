import json
import os
import time
from typing import List, Dict, Any, Optional
import requests
from config import VOLCANO_ENGINE_CONFIG, VECTOR_CONFIG
from logger import get_logger


class VectorGenerator:
    def __init__(self):
        self.logger = get_logger()
        self.api_key = VOLCANO_ENGINE_CONFIG["api_key"]
        self.model = VOLCANO_ENGINE_CONFIG["model"]
        self.api_endpoint = "https://ark.cn-beijing.volces.com/api/v3/embeddings"
        self.dimension = VECTOR_CONFIG["dimension"]
        self.batch_size = VECTOR_CONFIG["batch_size"]
        self.batch_interval = VECTOR_CONFIG["batch_interval"]
        self.max_retries = 3
        self.input_file = "data/entities.json"
        self.output_file = "data/vectors.json"
        
    def load_entities(self) -> Optional[List[Dict[str, Any]]]:
        try:
            self.logger.log_deployment_step("加载实体数据", "开始")
            
            if not os.path.exists(self.input_file):
                self.logger.log_deployment_failure("加载实体数据", f"文件不存在: {self.input_file}")
                return None
            
            with open(self.input_file, 'r', encoding='utf-8') as f:
                entities = json.load(f)
            
            self.logger.log_deployment_success(f"加载实体数据 (共 {len(entities)} 条)")
            return entities
        except Exception as e:
            self.logger.log_deployment_failure("加载实体数据", str(e))
            return None
    
    def call_embedding_api(self, texts: List[str]) -> Optional[List[List[float]]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "input": texts
        }
        
        for retry_count in range(self.max_retries):
            try:
                if retry_count > 0:
                    retry_interval = 2 ** retry_count
                    self.logger.warning(f"第 {retry_count} 次重试，等待 {retry_interval} 秒...")
                    time.sleep(retry_interval)
                
                response = requests.post(
                    self.api_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embeddings = [item["embedding"] for item in result["data"]]
                    self.logger.debug(f"成功获取 {len(embeddings)} 个向量")
                    return embeddings
                else:
                    error_msg = f"API调用失败 - 状态码: {response.status_code}, 响应: {response.text}"
                    self.logger.error(error_msg)
                    
            except requests.exceptions.Timeout:
                self.logger.error(f"API调用超时 (尝试 {retry_count + 1}/{self.max_retries})")
            except requests.exceptions.RequestException as e:
                self.logger.error(f"API调用异常 (尝试 {retry_count + 1}/{self.max_retries}): {str(e)}")
            except Exception as e:
                self.logger.error(f"未知错误 (尝试 {retry_count + 1}/{self.max_retries}): {str(e)}")
        
        self.logger.error(f"API调用失败，已达到最大重试次数 {self.max_retries}")
        return None
    
    def generate_vectors_batch(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        vector_data = []
        total_entities = len(entities)
        total_batches = (total_entities + self.batch_size - 1) // self.batch_size
        
        self.logger.info("=" * 50)
        self.logger.info(f"开始生成向量数据")
        self.logger.info(f"总实体数: {total_entities}")
        self.logger.info(f"批次大小: {self.batch_size}")
        self.logger.info(f"总批次数: {total_batches}")
        self.logger.info(f"批次间隔: {self.batch_interval} 秒")
        self.logger.info("=" * 50)
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_entities)
            batch_entities = entities[start_idx:end_idx]
            
            self.logger.log_progress(
                batch_idx + 1, 
                total_batches, 
                f"处理批次 {batch_idx + 1}/{total_batches} (实体 {start_idx + 1}-{end_idx})"
            )
            
            texts = [entity["name"] for entity in batch_entities]
            
            embeddings = self.call_embedding_api(texts)
            
            if embeddings is None:
                self.logger.error(f"批次 {batch_idx + 1} 向量生成失败，跳过该批次")
                continue
            
            for entity, embedding in zip(batch_entities, embeddings):
                vector_item = {
                    "id": entity["id"],
                    "name": entity["name"],
                    "entity_type": entity["entity_type"],
                    "vector": embedding
                }
                vector_data.append(vector_item)
            
            self.logger.info(f"批次 {batch_idx + 1} 完成，已生成 {len(embeddings)} 个向量")
            
            if batch_idx < total_batches - 1:
                self.logger.debug(f"等待 {self.batch_interval} 秒后继续...")
                time.sleep(self.batch_interval)
        
        return vector_data
    
    def save_vectors(self, vector_data: List[Dict[str, Any]]) -> bool:
        try:
            self.logger.log_deployment_step("保存向量数据", "开始")
            
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(vector_data, f, ensure_ascii=False, indent=2)
            
            file_size = os.path.getsize(self.output_file) / (1024 * 1024)
            self.logger.log_deployment_success(f"保存向量数据 (文件大小: {file_size:.2f} MB)")
            return True
        except Exception as e:
            self.logger.log_deployment_failure("保存向量数据", str(e))
            return False
    
    def generate_statistics(self, vector_data: List[Dict[str, Any]], total_entities: int):
        stats = {}
        for item in vector_data:
            entity_type = item["entity_type"]
            stats[entity_type] = stats.get(entity_type, 0) + 1
        
        self.logger.info("=" * 50)
        self.logger.info("向量生成统计信息:")
        self.logger.info("=" * 50)
        
        success_count = 0
        entity_types = ["Disease", "Drug", "Symptom", "Food", "Check", "Department"]
        for entity_type in entity_types:
            count = stats.get(entity_type, 0)
            success_count += count
            self.logger.info(f"  {entity_type}: {count} 条")
        
        self.logger.info("-" * 50)
        self.logger.info(f"  成功生成: {success_count} 条")
        self.logger.info(f"  失败数量: {total_entities - success_count} 条")
        self.logger.info(f"  成功率: {success_count / total_entities * 100:.2f}%")
        self.logger.info("=" * 50)
    
    def run(self):
        self.logger.info("开始执行向量生成任务")
        self.logger.info("=" * 50)
        
        entities = self.load_entities()
        if entities is None or len(entities) == 0:
            self.logger.warning("未加载到任何实体数据")
            return False
        
        vector_data = self.generate_vectors_batch(entities)
        
        if not vector_data:
            self.logger.warning("未生成任何向量数据")
            return False
        
        if not self.save_vectors(vector_data):
            return False
        
        self.generate_statistics(vector_data, len(entities))
        
        self.logger.info("=" * 50)
        self.logger.info("向量生成任务执行完成")
        return True


def main():
    generator = VectorGenerator()
    success = generator.run()
    
    if success:
        print("\n✓ 向量生成成功!")
        print(f"✓ 数据已保存至: data/vectors.json")
    else:
        print("\n✗ 向量生成失败，请查看日志了解详情")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
