# AI辅助生成：GLM-5, 2026-04-18
"""
混合检索服务模块
用于从Zilliz Cloud的三个集合中检索医疗实体信息并进行融合
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymilvus import (
    connections,
    Collection,
    utility
)
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

from config import ZILLIZ_CONFIG, LOCAL_MODEL_CONFIG
from logger import get_logger, log_deployment_step, log_deployment_success, log_deployment_failure


class HybridRetrievalService:
    """混合检索服务"""
    
    def __init__(self):
        self.logger = get_logger()
        self.collections = {}
        self.model = None
        self.device = None
        self.connected = False
        
    def connect_to_zilliz(self) -> bool:
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
            self.connected = True
            return True
            
        except Exception as e:
            error_msg = f"连接Zilliz Cloud失败: {str(e)}"
            log_deployment_failure("连接Zilliz Cloud服务", error_msg)
            self.connected = False
            return False
    
    def load_collections(self) -> bool:
        """加载三个集合"""
        log_deployment_step("加载向量集合", "开始")
        
        collection_names = ["medical_entity", "entity_attributes", "entity_relations"]
        success_count = 0
        
        for collection_name in collection_names:
            try:
                if utility.has_collection(collection_name):
                    collection = Collection(collection_name)
                    collection.load()
                    self.collections[collection_name] = collection
                    self.logger.info(f"集合 '{collection_name}' 加载成功，实体数量: {collection.num_entities}")
                    success_count += 1
                else:
                    self.logger.warning(f"集合 '{collection_name}' 不存在")
            except Exception as e:
                self.logger.error(f"加载集合 '{collection_name}' 失败: {str(e)}")
        
        if success_count == 0:
            log_deployment_failure("加载向量集合", "所有集合加载失败")
            return False
        
        log_deployment_success(f"加载向量集合 (成功: {success_count}/3)")
        return True
    
    def load_model(self) -> bool:
        """加载向量生成模型"""
        log_deployment_step("加载向量生成模型", "开始")
        
        try:
            if torch.cuda.is_available():
                self.device = LOCAL_MODEL_CONFIG.get('device', 'cuda')
                self.logger.info(f"使用GPU设备: {self.device}")
            else:
                self.device = "cpu"
                self.logger.warning("GPU不可用，使用CPU")
            
            import os
            self.model = SentenceTransformer(
                LOCAL_MODEL_CONFIG['model_name'],
                cache_folder=os.path.expanduser(LOCAL_MODEL_CONFIG.get('cache_dir', '~/.cache/huggingface/'))
            )
            
            self.model.to(self.device)
            
            self.logger.info("向量生成模型加载成功")
            log_deployment_success("加载向量生成模型")
            return True
            
        except Exception as e:
            error_msg = f"加载向量生成模型失败: {str(e)}"
            log_deployment_failure("加载向量生成模型", error_msg)
            return False
    
    def generate_query_vector(self, query_text: str) -> Optional[List[float]]:
        """生成查询向量"""
        try:
            if not self.model:
                self.logger.error("模型未加载")
                return None
            
            embedding = self.model.encode(
                [query_text],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=LOCAL_MODEL_CONFIG.get('normalize', True)
            )
            
            return embedding[0].tolist()
            
        except Exception as e:
            self.logger.error(f"生成查询向量失败: {str(e)}")
            return None
    
    def search_entity_names(self, query_vector: List[float], top_k: int = 20) -> List[Dict]:
        """在medical_entity集合中检索实体名称"""
        try:
            if "medical_entity" not in self.collections:
                self.logger.warning("medical_entity集合未加载")
                return []
            
            collection = self.collections["medical_entity"]
            
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 128}
            }
            
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["entity_name", "entity_type", "neo4j_node_id"]
            )
            
            entity_results = []
            for hits in results:
                for hit in hits:
                    entity_results.append({
                        "name": hit.entity.get("entity_name"),
                        "type": hit.entity.get("entity_type"),
                        "similarity": float(hit.distance),
                        "neo4j_id": hit.entity.get("neo4j_node_id")
                    })
            
            self.logger.info(f"实体名称检索完成，返回 {len(entity_results)} 条结果")
            return entity_results
            
        except Exception as e:
            self.logger.error(f"实体名称检索失败: {str(e)}")
            return []
    
    def search_entity_attributes(self, query_vector: List[float], top_k: int = 20) -> List[Dict]:
        """在entity_attributes集合中检索实体属性"""
        try:
            if "entity_attributes" not in self.collections:
                self.logger.warning("entity_attributes集合未加载")
                return []
            
            collection = self.collections["entity_attributes"]
            
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 128}
            }
            
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["entity_name", "entity_type", "attribute_name", "attribute_value", "neo4j_node_id"]
            )
            
            attribute_results = []
            for hits in results:
                for hit in hits:
                    attribute_results.append({
                        "entity_name": hit.entity.get("entity_name"),
                        "entity_type": hit.entity.get("entity_type"),
                        "attribute_name": hit.entity.get("attribute_name"),
                        "attribute_value": hit.entity.get("attribute_value"),
                        "similarity": float(hit.distance),
                        "neo4j_id": hit.entity.get("neo4j_node_id")
                    })
            
            self.logger.info(f"实体属性检索完成，返回 {len(attribute_results)} 条结果")
            return attribute_results
            
        except Exception as e:
            self.logger.error(f"实体属性检索失败: {str(e)}")
            return []
    
    def search_relations(self, query_vector: List[float], top_k: int = 20) -> List[Dict]:
        """在entity_relations集合中检索关系"""
        try:
            if "entity_relations" not in self.collections:
                self.logger.warning("entity_relations集合未加载")
                return []
            
            collection = self.collections["entity_relations"]
            
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 128}
            }
            
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["source_entity_name", "source_entity_type", "target_entity_name", 
                              "target_entity_type", "relation_type", "relation_description", "neo4j_relation_id"]
            )
            
            relation_results = []
            for hits in results:
                for hit in hits:
                    relation_results.append({
                        "source": hit.entity.get("source_entity_name"),
                        "source_type": hit.entity.get("source_entity_type"),
                        "target": hit.entity.get("target_entity_name"),
                        "target_type": hit.entity.get("target_entity_type"),
                        "relation": hit.entity.get("relation_type"),
                        "description": hit.entity.get("relation_description"),
                        "similarity": float(hit.distance),
                        "neo4j_id": hit.entity.get("neo4j_relation_id")
                    })
            
            self.logger.info(f"关系检索完成，返回 {len(relation_results)} 条结果")
            return relation_results
            
        except Exception as e:
            self.logger.error(f"关系检索失败: {str(e)}")
            return []
    
    def fuse_results(
        self,
        entity_results: List[Dict],
        attribute_results: List[Dict],
        relation_results: List[Dict],
        weights: Dict[str, float] = None
    ) -> List[Dict]:
        """融合三个集合的检索结果"""
        try:
            if weights is None:
                weights = {"entity": 0.4, "attribute": 0.35, "relation": 0.25}
            
            fused_entities = {}
            
            for entity in entity_results:
                entity_name = entity.get("name", "")
                if entity_name not in fused_entities:
                    fused_entities[entity_name] = {
                        "name": entity_name,
                        "type": entity.get("type"),
                        "neo4j_id": entity.get("neo4j_id"),
                        "entity_score": entity.get("similarity", 0),
                        "attribute_score": 0,
                        "relation_score": 0,
                        "attributes": [],
                        "relations": []
                    }
                else:
                    if fused_entities[entity_name]["entity_score"] < entity.get("similarity", 0):
                        fused_entities[entity_name]["entity_score"] = entity.get("similarity", 0)
            
            for attr in attribute_results:
                entity_name = attr.get("entity_name", "")
                if entity_name not in fused_entities:
                    fused_entities[entity_name] = {
                        "name": entity_name,
                        "type": attr.get("entity_type"),
                        "neo4j_id": attr.get("neo4j_id"),
                        "entity_score": 0,
                        "attribute_score": attr.get("similarity", 0),
                        "relation_score": 0,
                        "attributes": [],
                        "relations": []
                    }
                
                fused_entities[entity_name]["attributes"].append({
                    "name": attr.get("attribute_name"),
                    "value": attr.get("attribute_value"),
                    "similarity": attr.get("similarity")
                })
                
                if fused_entities[entity_name]["attribute_score"] < attr.get("similarity", 0):
                    fused_entities[entity_name]["attribute_score"] = attr.get("similarity", 0)
            
            for rel in relation_results:
                source_name = rel.get("source", "")
                target_name = rel.get("target", "")
                
                for entity_name in [source_name, target_name]:
                    if entity_name not in fused_entities:
                        fused_entities[entity_name] = {
                            "name": entity_name,
                            "type": rel.get("source_type") if entity_name == source_name else rel.get("target_type"),
                            "neo4j_id": None,
                            "entity_score": 0,
                            "attribute_score": 0,
                            "relation_score": rel.get("similarity", 0),
                            "attributes": [],
                            "relations": []
                        }
                    
                    fused_entities[entity_name]["relations"].append({
                        "source": rel.get("source"),
                        "target": rel.get("target"),
                        "relation": rel.get("relation"),
                        "description": rel.get("description"),
                        "similarity": rel.get("similarity")
                    })
                    
                    if fused_entities[entity_name]["relation_score"] < rel.get("similarity", 0):
                        fused_entities[entity_name]["relation_score"] = rel.get("similarity", 0)
            
            fused_list = []
            for entity_name, entity_data in fused_entities.items():
                fused_score = (
                    entity_data["entity_score"] * weights["entity"] +
                    entity_data["attribute_score"] * weights["attribute"] +
                    entity_data["relation_score"] * weights["relation"]
                )
                
                entity_data["fused_score"] = fused_score
                fused_list.append(entity_data)
            
            fused_list.sort(key=lambda x: x["fused_score"], reverse=True)
            
            self.logger.info(f"结果融合完成，融合后实体数量: {len(fused_list)}")
            return fused_list
            
        except Exception as e:
            self.logger.error(f"结果融合失败: {str(e)}")
            return []
    
    def rerank_results(self, results: List[Dict], query_context: str = None) -> List[Dict]:
        """对融合结果进行重排序"""
        try:
            if not results:
                return results
            
            for result in results:
                boost_factor = 1.0
                
                if result.get("entity_score", 0) > 0.9:
                    boost_factor += 0.1
                
                if result.get("attribute_score", 0) > 0.85:
                    boost_factor += 0.05
                
                if result.get("relation_score", 0) > 0.8:
                    boost_factor += 0.05
                
                if len(result.get("attributes", [])) > 0:
                    boost_factor += 0.02
                
                if len(result.get("relations", [])) > 0:
                    boost_factor += 0.02
                
                result["reranked_score"] = result.get("fused_score", 0) * boost_factor
            
            results.sort(key=lambda x: x.get("reranked_score", 0), reverse=True)
            
            self.logger.info(f"结果重排序完成，返回 {len(results)} 条结果")
            return results
            
        except Exception as e:
            self.logger.error(f"结果重排序失败: {str(e)}")
            return results
    
    def hybrid_search(
        self,
        query_text: str,
        top_k: int = 20,
        weights: Dict[str, float] = None,
        threshold: float = 0.75
    ) -> Dict:
        """混合检索主函数"""
        start_time = time.time()
        
        try:
            self.logger.info(f"开始混合检索，查询文本: {query_text}")
            
            query_vector = self.generate_query_vector(query_text)
            if not query_vector:
                self.logger.error("生成查询向量失败")
                return {
                    "entities": [],
                    "attributes": [],
                    "relations": [],
                    "fused": [],
                    "error": "生成查询向量失败"
                }
            
            entity_results = []
            attribute_results = []
            relation_results = []
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_entity = executor.submit(self.search_entity_names, query_vector, top_k)
                future_attribute = executor.submit(self.search_entity_attributes, query_vector, top_k)
                future_relation = executor.submit(self.search_relations, query_vector, top_k)
                
                for future in as_completed([future_entity, future_attribute, future_relation]):
                    try:
                        result = future.result()
                        if future == future_entity:
                            entity_results = result
                        elif future == future_attribute:
                            attribute_results = result
                        elif future == future_relation:
                            relation_results = result
                    except Exception as e:
                        self.logger.error(f"并行检索任务失败: {str(e)}")
            
            fused_results = self.fuse_results(entity_results, attribute_results, relation_results, weights)
            
            reranked_results = self.rerank_results(fused_results, query_text)
            
            filtered_results = [
                result for result in reranked_results
                if result.get("reranked_score", 0) >= threshold
            ]
            
            elapsed_time = time.time() - start_time
            
            result = {
                "entities": entity_results,
                "attributes": attribute_results,
                "relations": relation_results,
                "fused": filtered_results,
                "total_count": len(filtered_results),
                "elapsed_time": elapsed_time
            }
            
            self.logger.info(f"混合检索完成，耗时: {elapsed_time:.2f}秒，返回 {len(filtered_results)} 条结果")
            return result
            
        except Exception as e:
            self.logger.error(f"混合检索失败: {str(e)}")
            return {
                "entities": [],
                "attributes": [],
                "relations": [],
                "fused": [],
                "error": str(e)
            }
    
    def disconnect(self):
        """断开连接"""
        try:
            if self.connected:
                connections.disconnect("default")
                self.logger.info("已断开与Zilliz Cloud的连接")
                self.connected = False
        except Exception as e:
            self.logger.warning(f"断开连接时出现警告: {str(e)}")
    
    def initialize(self) -> bool:
        """初始化服务"""
        self.logger.info("=" * 60)
        self.logger.info("开始初始化混合检索服务")
        self.logger.info("=" * 60)
        
        try:
            if not self.connect_to_zilliz():
                return False
            
            if not self.load_collections():
                self.logger.warning("部分集合加载失败，服务将以降级模式运行")
            
            if not self.load_model():
                self.logger.error("模型加载失败，无法生成查询向量")
                return False
            
            self.logger.info("=" * 60)
            self.logger.info("混合检索服务初始化完成")
            self.logger.info("=" * 60)
            return True
            
        except Exception as e:
            self.logger.error(f"初始化失败: {str(e)}")
            return False


_service_instance: Optional[HybridRetrievalService] = None


def get_service() -> HybridRetrievalService:
    """获取混合检索服务实例"""
    global _service_instance
    if _service_instance is None:
        _service_instance = HybridRetrievalService()
    return _service_instance


def search_entity_names(query_vector: List[float], top_k: int = 20) -> List[Dict]:
    """在medical_entity集合中检索实体名称"""
    service = get_service()
    return service.search_entity_names(query_vector, top_k)


def search_entity_attributes(query_vector: List[float], top_k: int = 20) -> List[Dict]:
    """在entity_attributes集合中检索实体属性"""
    service = get_service()
    return service.search_entity_attributes(query_vector, top_k)


def search_relations(query_vector: List[float], top_k: int = 20) -> List[Dict]:
    """在entity_relations集合中检索关系"""
    service = get_service()
    return service.search_relations(query_vector, top_k)


def fuse_results(
    entity_results: List[Dict],
    attribute_results: List[Dict],
    relation_results: List[Dict],
    weights: Dict[str, float] = None
) -> List[Dict]:
    """融合三个集合的检索结果"""
    service = get_service()
    return service.fuse_results(entity_results, attribute_results, relation_results, weights)


def rerank_results(results: List[Dict], query_context: str = None) -> List[Dict]:
    """对融合结果进行重排序"""
    service = get_service()
    return service.rerank_results(results, query_context)


def hybrid_search(
    query_text: str,
    top_k: int = 20,
    weights: Dict[str, float] = None,
    threshold: float = 0.75
) -> Dict:
    """混合检索主函数"""
    service = get_service()
    return service.hybrid_search(query_text, top_k, weights, threshold)


def main():
    """主函数 - 测试混合检索服务"""
    service = HybridRetrievalService()
    
    try:
        if not service.initialize():
            print("✗ 服务初始化失败")
            return 1
        
        print("\n" + "=" * 60)
        print("测试混合检索服务")
        print("=" * 60)
        
        test_queries = [
            "高血压的症状",
            "糖尿病的治疗方法",
            "感冒的预防措施"
        ]
        
        for query in test_queries:
            print(f"\n查询: {query}")
            print("-" * 60)
            
            result = service.hybrid_search(query, top_k=10, threshold=0.7)
            
            if "error" in result and result["error"]:
                print(f"检索失败: {result['error']}")
                continue
            
            print(f"检索耗时: {result.get('elapsed_time', 0):.2f}秒")
            print(f"实体数量: {len(result.get('entities', []))}")
            print(f"属性数量: {len(result.get('attributes', []))}")
            print(f"关系数量: {len(result.get('relations', []))}")
            print(f"融合结果数量: {len(result.get('fused', []))}")
            
            if result.get('fused'):
                print("\nTop 3 融合结果:")
                for i, item in enumerate(result['fused'][:3], 1):
                    print(f"  {i}. {item.get('name')} (类型: {item.get('type')}, 得分: {item.get('reranked_score', 0):.4f})")
        
        print("\n" + "=" * 60)
        print("✓ 测试完成")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        return 1
    finally:
        service.disconnect()


if __name__ == "__main__":
    exit(main())
