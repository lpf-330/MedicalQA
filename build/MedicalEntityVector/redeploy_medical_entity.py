# AI辅助生成：GLM-5, 2026-04-18
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速重新部署medical_entity集合
修复neo4j_node_id字段问题
"""

import json
import time
import sys
import os
from typing import List, Dict, Any
import torch
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from config import NEO4J_CONFIG, ZILLIZ_CONFIG, LOCAL_MODEL_CONFIG
from logger import get_logger


class MedicalEntityRedeployer:
    """medical_entity集合重新部署器"""
    
    def __init__(self):
        self.logger = get_logger()
        self.driver = None
        self.model = None
        self.device = None
        self.collection_name = "medical_entity"
        self.entity_types = ["Disease", "Drug", "Symptom", "Food", "Check", "Department", "Producer", "Cure"]
        
    def connect_neo4j(self) -> bool:
        """连接Neo4j"""
        self.logger.info("连接Neo4j数据库...")
        try:
            self.driver = GraphDatabase.driver(
                NEO4J_CONFIG["uri"],
                auth=(NEO4J_CONFIG["user"], NEO4J_CONFIG["password"])
            )
            self.driver.verify_connectivity()
            self.logger.info("✓ Neo4j连接成功")
            return True
        except Exception as e:
            self.logger.error(f"连接Neo4j失败: {e}")
            return False
    
    def connect_milvus(self) -> bool:
        """连接Milvus"""
        self.logger.info("连接Milvus向量数据库...")
        try:
            connections.connect(
                alias="default",
                uri=ZILLIZ_CONFIG["uri"],
                token=ZILLIZ_CONFIG["token"]
            )
            self.logger.info("✓ Milvus连接成功")
            return True
        except Exception as e:
            self.logger.error(f"连接Milvus失败: {e}")
            return False
    
    def load_model(self) -> bool:
        """加载向量模型"""
        self.logger.info("加载向量模型...")
        try:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
            
            self.model = SentenceTransformer(
                LOCAL_MODEL_CONFIG['model_name'],
                cache_folder=os.path.expanduser(LOCAL_MODEL_CONFIG.get('cache_dir', '~/.cache/huggingface/'))
            )
            self.model.to(self.device)
            self.logger.info(f"✓ 模型加载完成，设备: {self.device}")
            return True
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            return False
    
    def extract_entities_from_neo4j(self) -> List[Dict[str, Any]]:
        """从Neo4j提取实体数据"""
        self.logger.info("从Neo4j提取实体数据...")
        
        all_entities = []
        
        for entity_type in self.entity_types:
            try:
                with self.driver.session() as session:
                    query = f"MATCH (n:{entity_type}) RETURN id(n) as neo4j_id, n.name as name"
                    result = session.run(query)
                    
                    for record in result:
                        entity = {
                            "neo4j_id": str(record["neo4j_id"]),
                            "name": record["name"],
                            "entity_type": entity_type
                        }
                        all_entities.append(entity)
                    
                    self.logger.info(f"  {entity_type}: {len([e for e in all_entities if e['entity_type'] == entity_type])} 条")
            except Exception as e:
                self.logger.error(f"提取 {entity_type} 实体失败: {e}")
        
        self.logger.info(f"✓ 实体提取完成，总计: {len(all_entities)} 条")
        return all_entities
    
    def save_entities_to_json(self, entities: List[Dict]) -> bool:
        """保存实体数据到JSON"""
        self.logger.info("保存实体数据到JSON...")
        try:
            with open("data/entities.json", "w", encoding="utf-8") as f:
                json.dump(entities, f, ensure_ascii=False, indent=2)
            self.logger.info(f"✓ 数据已保存到 data/entities.json")
            return True
        except Exception as e:
            self.logger.error(f"保存数据失败: {e}")
            return False
    
    def drop_collection(self) -> bool:
        """删除旧集合"""
        self.logger.info("删除旧的medical_entity集合...")
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                self.logger.info("✓ 旧集合已删除")
            else:
                self.logger.info("集合不存在，跳过删除")
            return True
        except Exception as e:
            self.logger.error(f"删除集合失败: {e}")
            return False
    
    def create_collection(self) -> bool:
        """创建新集合"""
        self.logger.info("创建新的medical_entity集合...")
        try:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="entity_name", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="entity_type", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="neo4j_node_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="医疗实体向量集合（修复后）"
            )
            
            collection = Collection(self.collection_name, schema)
            self.logger.info("✓ 新集合已创建")
            
            self.logger.info("创建索引...")
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index(field_name="vector", index_params=index_params)
            self.logger.info("✓ 索引已创建")
            
            return True
        except Exception as e:
            self.logger.error(f"创建集合失败: {e}")
            return False
    
    def generate_vectors_and_insert(self, entities: List[Dict]) -> bool:
        """生成向量并插入"""
        self.logger.info(f"生成向量并插入数据，共 {len(entities)} 条...")
        
        collection = Collection(self.collection_name)
        
        batch_size = 1024
        total_batches = (len(entities) + batch_size - 1) // batch_size
        start_time = time.time()
        
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i+batch_size]
            
            texts = [e["name"] for e in batch]
            
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False
            )
            
            insert_data = [
                [e["name"] for e in batch],
                [e["entity_type"] for e in batch],
                [e["neo4j_id"] for e in batch],
                embeddings.tolist()
            ]
            
            collection.insert(insert_data)
            
            progress = (i + len(batch)) / len(entities) * 100
            elapsed = time.time() - start_time
            speed = (i + len(batch)) / elapsed if elapsed > 0 else 0
            
            self.logger.info(f"进度: {i + len(batch)}/{len(entities)} ({progress:.1f}%) | 速度: {speed:.1f} 条/秒")
        
        collection.flush()
        
        final_count = collection.num_entities
        self.logger.info(f"✓ 数据插入完成，最终记录数: {final_count}")
        
        return True
    
    def verify_deployment(self) -> bool:
        """验证部署结果"""
        self.logger.info("验证部署结果...")
        
        collection = Collection(self.collection_name)
        collection.load()
        
        results = collection.query(
            expr="",
            output_fields=["entity_name", "entity_type", "neo4j_node_id"],
            limit=10
        )
        
        self.logger.info("部署后样本数据:")
        for i, r in enumerate(results):
            self.logger.info(f"  {i+1}. entity_name={r['entity_name']}, neo4j_node_id={r['neo4j_node_id']}")
        
        return True
    
    def run(self):
        """执行重新部署"""
        self.logger.info("=" * 60)
        self.logger.info("开始重新部署medical_entity集合")
        self.logger.info("=" * 60)
        
        try:
            if not self.connect_neo4j():
                return False
            
            if not self.connect_milvus():
                return False
            
            if not self.load_model():
                return False
            
            entities = self.extract_entities_from_neo4j()
            
            if not entities:
                self.logger.error("未提取到任何实体数据")
                return False
            
            if not self.save_entities_to_json(entities):
                return False
            
            if not self.drop_collection():
                return False
            
            if not self.create_collection():
                return False
            
            if not self.generate_vectors_and_insert(entities):
                return False
            
            self.verify_deployment()
            
            self.logger.info("=" * 60)
            self.logger.info("✓ 重新部署完成！")
            self.logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"重新部署失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if self.driver:
                self.driver.close()
            try:
                connections.disconnect("default")
            except:
                pass


def main():
    redeployer = MedicalEntityRedeployer()
    success = redeployer.run()
    
    if success:
        print("\n✓ medical_entity集合重新部署成功!")
        print("✓ neo4j_node_id已修复")
    else:
        print("\n✗ 重新部署失败，请查看日志了解详情")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
