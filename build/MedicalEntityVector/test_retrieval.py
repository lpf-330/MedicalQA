#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量检索功能测试和性能评估脚本
测试单集合检索、混合检索功能，评估检索性能
"""

import sys
import time
import json
import os
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, utility
from config import ZILLIZ_CONFIG, LOCAL_MODEL_CONFIG
from logger import get_logger


class VectorSearchTester:
    """向量检索测试器"""
    
    def __init__(self):
        self.logger = get_logger()
        self.model = None
        self.device = None
        self.collections = {}
        
        self.test_cases = self._generate_test_cases()
        
    def _generate_test_cases(self) -> List[Dict[str, Any]]:
        """生成测试用例"""
        return [
            {
                "query": "感冒有什么症状？",
                "expected_entities": ["感冒", "上呼吸道感染"],
                "expected_types": ["Disease", "Symptom"],
                "category": "疾病症状查询"
            },
            {
                "query": "高血压应该吃什么药？",
                "expected_entities": ["高血压", "降压药"],
                "expected_types": ["Disease", "Drug"],
                "category": "疾病用药查询"
            },
            {
                "query": "糖尿病的病因是什么？",
                "expected_entities": ["糖尿病"],
                "expected_types": ["Disease"],
                "category": "疾病病因查询"
            },
            {
                "query": "发烧需要做什么检查？",
                "expected_entities": ["发烧", "发热"],
                "expected_types": ["Symptom", "Check"],
                "category": "症状检查查询"
            },
            {
                "query": "胃溃疡不能吃什么？",
                "expected_entities": ["胃溃疡"],
                "expected_types": ["Disease", "Food"],
                "category": "饮食禁忌查询"
            },
            {
                "query": "头痛可能是哪些疾病？",
                "expected_entities": ["头痛"],
                "expected_types": ["Symptom", "Disease"],
                "category": "症状诊断查询"
            },
            {
                "query": "阿司匹林的适应症是什么？",
                "expected_entities": ["阿司匹林"],
                "expected_types": ["Drug", "Disease"],
                "category": "药物适应症查询"
            },
            {
                "query": "心脏病如何预防？",
                "expected_entities": ["心脏病", "冠心病"],
                "expected_types": ["Disease"],
                "category": "疾病预防查询"
            },
            {
                "query": "肺炎的治疗方法有哪些？",
                "expected_entities": ["肺炎"],
                "expected_types": ["Disease", "Cure"],
                "category": "疾病治疗查询"
            },
            {
                "query": "乙肝会传染吗？",
                "expected_entities": ["乙肝", "乙型肝炎"],
                "expected_types": ["Disease"],
                "category": "疾病传染性查询"
            },
            {
                "query": "过敏体质的人容易得什么病？",
                "expected_entities": ["过敏"],
                "expected_types": ["Disease", "Symptom"],
                "category": "易感人群查询"
            },
            {
                "query": "阑尾炎手术需要多长时间恢复？",
                "expected_entities": ["阑尾炎"],
                "expected_types": ["Disease"],
                "category": "疾病恢复查询"
            },
            {
                "query": "哪些科室治疗呼吸系统疾病？",
                "expected_entities": ["呼吸内科", "呼吸科"],
                "expected_types": ["Department"],
                "category": "科室查询"
            },
            {
                "query": "维生素C有什么作用？",
                "expected_entities": ["维生素C"],
                "expected_types": ["Food", "Drug"],
                "category": "营养成分查询"
            },
            {
                "query": "失眠怎么调理？",
                "expected_entities": ["失眠"],
                "expected_types": ["Symptom", "Disease"],
                "category": "症状调理查询"
            },
            {
                "query": "高血压患者可以运动吗？",
                "expected_entities": ["高血压"],
                "expected_types": ["Disease"],
                "category": "疾病生活指导查询"
            },
            {
                "query": "儿童发烧怎么处理？",
                "expected_entities": ["发烧", "发热"],
                "expected_types": ["Symptom"],
                "category": "儿童健康查询"
            },
            {
                "query": "胃镜检查痛苦吗？",
                "expected_entities": ["胃镜"],
                "expected_types": ["Check"],
                "category": "检查项目查询"
            },
            {
                "query": "糖尿病患者可以吃水果吗？",
                "expected_entities": ["糖尿病"],
                "expected_types": ["Disease", "Food"],
                "category": "疾病饮食查询"
            },
            {
                "query": "颈椎病怎么治疗？",
                "expected_entities": ["颈椎病"],
                "expected_types": ["Disease", "Cure"],
                "category": "疾病治疗查询"
            }
        ]
    
    def load_model(self):
        """加载向量模型"""
        self.logger.info("加载向量模型...")
        
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
    
    def connect_milvus(self):
        """连接Milvus"""
        self.logger.info("连接Milvus向量数据库...")
        
        connections.connect(
            alias="default",
            uri=ZILLIZ_CONFIG["uri"],
            token=ZILLIZ_CONFIG["token"]
        )
        
        for name in ["medical_entity", "entity_attributes", "entity_relations"]:
            if utility.has_collection(name):
                self.collections[name] = Collection(name)
                self.collections[name].load()
        
        self.logger.info(f"✓ 已连接 {len(self.collections)} 个集合")
    
    def encode_query(self, query: str) -> np.ndarray:
        """将查询文本编码为向量"""
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding[0]
    
    def search_entities(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索实体名称向量"""
        start_time = time.time()
        
        results = self.collections["medical_entity"].search(
            data=[query_vector.tolist()],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            output_fields=["entity_name", "entity_type"]
        )
        
        latency = (time.time() - start_time) * 1000
        
        entities = []
        for hits in results:
            for hit in hits:
                entities.append({
                    "name": hit.entity.get("entity_name"),
                    "type": hit.entity.get("entity_type"),
                    "score": hit.score
                })
        
        return entities, latency
    
    def search_attributes(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索属性向量"""
        start_time = time.time()
        
        results = self.collections["entity_attributes"].search(
            data=[query_vector.tolist()],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            output_fields=["entity_name", "entity_type", "attribute_name", "attribute_value"]
        )
        
        latency = (time.time() - start_time) * 1000
        
        attributes = []
        for hits in results:
            for hit in hits:
                attributes.append({
                    "entity_name": hit.entity.get("entity_name"),
                    "entity_type": hit.entity.get("entity_type"),
                    "attribute_name": hit.entity.get("attribute_name"),
                    "attribute_value": hit.entity.get("attribute_value"),
                    "score": hit.score
                })
        
        return attributes, latency
    
    def search_relations(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索关系向量"""
        start_time = time.time()
        
        results = self.collections["entity_relations"].search(
            data=[query_vector.tolist()],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            output_fields=["source_entity_name", "source_entity_type", "relation_type", "target_entity_name", "target_entity_type"]
        )
        
        latency = (time.time() - start_time) * 1000
        
        relations = []
        for hits in results:
            for hit in hits:
                relations.append({
                    "source_name": hit.entity.get("source_entity_name"),
                    "source_type": hit.entity.get("source_entity_type"),
                    "relation_type": hit.entity.get("relation_type"),
                    "target_name": hit.entity.get("target_entity_name"),
                    "target_type": hit.entity.get("target_entity_type"),
                    "score": hit.score
                })
        
        return relations, latency
    
    def hybrid_search(self, query_vector: np.ndarray, top_k: int = 10, 
                      weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)) -> Dict[str, Any]:
        """混合检索（三集合融合）"""
        start_time = time.time()
        
        entity_results, entity_latency = self.search_entities(query_vector, top_k=top_k*2)
        attr_results, attr_latency = self.search_attributes(query_vector, top_k=top_k*2)
        relation_results, relation_latency = self.search_relations(query_vector, top_k=top_k*2)
        
        w_entity, w_attr, w_relation = weights
        
        all_results = []
        
        for r in entity_results:
            all_results.append({
                "type": "entity",
                "name": r["name"],
                "entity_type": r["type"],
                "score": r["score"] * w_entity,
                "original_score": r["score"]
            })
        
        for r in attr_results:
            all_results.append({
                "type": "attribute",
                "name": r["entity_name"],
                "entity_type": r["entity_type"],
                "attribute_name": r["attribute_name"],
                "attribute_value": r["attribute_value"],
                "score": r["score"] * w_attr,
                "original_score": r["score"]
            })
        
        for r in relation_results:
            all_results.append({
                "type": "relation",
                "source_name": r["source_name"],
                "relation_type": r["relation_type"],
                "target_name": r["target_name"],
                "score": r["score"] * w_relation,
                "original_score": r["score"]
            })
        
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        total_latency = (time.time() - start_time) * 1000
        
        return {
            "results": all_results[:top_k],
            "latency": total_latency,
            "entity_latency": entity_latency,
            "attr_latency": attr_latency,
            "relation_latency": relation_latency
        }
    
    def evaluate_retrieval(self, results: List[Dict], expected_entities: List[str], 
                          expected_types: List[str]) -> Dict[str, float]:
        """评估检索结果"""
        found_entities = set()
        found_types = set()
        
        for r in results:
            if r.get("name"):
                found_entities.add(r["name"])
            if r.get("entity_name"):
                found_entities.add(r["entity_name"])
            if r.get("source_name"):
                found_entities.add(r["source_name"])
            if r.get("target_name"):
                found_entities.add(r["target_name"])
            
            if r.get("entity_type"):
                found_types.add(r["entity_type"])
            if r.get("source_type"):
                found_types.add(r["source_type"])
            if r.get("target_type"):
                found_types.add(r["target_type"])
        
        expected_entities_set = set(expected_entities)
        expected_types_set = set(expected_types)
        
        entity_precision = len(found_entities & expected_entities_set) / len(found_entities) if found_entities else 0
        entity_recall = len(found_entities & expected_entities_set) / len(expected_entities_set) if expected_entities_set else 0
        
        type_precision = len(found_types & expected_types_set) / len(found_types) if found_types else 0
        type_recall = len(found_types & expected_types_set) / len(expected_types_set) if expected_types_set else 0
        
        return {
            "entity_precision": entity_precision,
            "entity_recall": entity_recall,
            "type_precision": type_precision,
            "type_recall": type_recall
        }
    
    def run_tests(self):
        """运行所有测试"""
        self.logger.info("=" * 80)
        self.logger.info("开始向量检索功能测试和性能评估")
        self.logger.info("=" * 80)
        
        self.load_model()
        self.connect_milvus()
        
        all_metrics = {
            "entity_search": {"latencies": [], "precisions": [], "recalls": []},
            "attribute_search": {"latencies": [], "precisions": [], "recalls": []},
            "relation_search": {"latencies": [], "precisions": [], "recalls": []},
            "hybrid_search": {"latencies": [], "precisions": [], "recalls": []}
        }
        
        detailed_results = []
        
        for i, test_case in enumerate(self.test_cases):
            self.logger.info(f"\n测试用例 {i+1}/{len(self.test_cases)}: {test_case['query']}")
            self.logger.info(f"分类: {test_case['category']}")
            
            query_vector = self.encode_query(test_case["query"])
            
            self.logger.info("\n--- 实体名称检索 ---")
            entity_results, entity_latency = self.search_entities(query_vector)
            self.logger.info(f"延迟: {entity_latency:.2f}ms")
            self.logger.info("Top 5 结果:")
            for j, r in enumerate(entity_results[:5]):
                self.logger.info(f"  {j+1}. {r['name']} ({r['type']}) - {r['score']:.4f}")
            
            self.logger.info("\n--- 属性检索 ---")
            attr_results, attr_latency = self.search_attributes(query_vector)
            self.logger.info(f"延迟: {attr_latency:.2f}ms")
            self.logger.info("Top 5 结果:")
            for j, r in enumerate(attr_results[:5]):
                self.logger.info(f"  {j+1}. {r['entity_name']}.{r['attribute_name']} - {r['score']:.4f}")
            
            self.logger.info("\n--- 关系检索 ---")
            relation_results, relation_latency = self.search_relations(query_vector)
            self.logger.info(f"延迟: {relation_latency:.2f}ms")
            self.logger.info("Top 5 结果:")
            for j, r in enumerate(relation_results[:5]):
                self.logger.info(f"  {j+1}. {r['source_name']} -[{r['relation_type']}]-> {r['target_name']} - {r['score']:.4f}")
            
            self.logger.info("\n--- 混合检索 ---")
            hybrid_result = self.hybrid_search(query_vector)
            self.logger.info(f"总延迟: {hybrid_result['latency']:.2f}ms")
            self.logger.info("Top 5 结果:")
            for j, r in enumerate(hybrid_result["results"][:5]):
                if r["type"] == "entity":
                    self.logger.info(f"  {j+1}. [实体] {r['name']} ({r['entity_type']}) - {r['score']:.4f}")
                elif r["type"] == "attribute":
                    self.logger.info(f"  {j+1}. [属性] {r['name']}.{r['attribute_name']} - {r['score']:.4f}")
                else:
                    self.logger.info(f"  {j+1}. [关系] {r['source_name']}-[{r['relation_type']}]->{r['target_name']} - {r['score']:.4f}")
            
            entity_metrics = self.evaluate_retrieval(entity_results, test_case["expected_entities"], test_case["expected_types"])
            attr_metrics = self.evaluate_retrieval(attr_results, test_case["expected_entities"], test_case["expected_types"])
            relation_metrics = self.evaluate_retrieval(relation_results, test_case["expected_entities"], test_case["expected_types"])
            hybrid_metrics = self.evaluate_retrieval(hybrid_result["results"], test_case["expected_entities"], test_case["expected_types"])
            
            all_metrics["entity_search"]["latencies"].append(entity_latency)
            all_metrics["entity_search"]["precisions"].append(entity_metrics["entity_precision"])
            all_metrics["entity_search"]["recalls"].append(entity_metrics["entity_recall"])
            
            all_metrics["attribute_search"]["latencies"].append(attr_latency)
            all_metrics["attribute_search"]["precisions"].append(attr_metrics["entity_precision"])
            all_metrics["attribute_search"]["recalls"].append(attr_metrics["entity_recall"])
            
            all_metrics["relation_search"]["latencies"].append(relation_latency)
            all_metrics["relation_search"]["precisions"].append(relation_metrics["entity_precision"])
            all_metrics["relation_search"]["recalls"].append(relation_metrics["entity_recall"])
            
            all_metrics["hybrid_search"]["latencies"].append(hybrid_result["latency"])
            all_metrics["hybrid_search"]["precisions"].append(hybrid_metrics["entity_precision"])
            all_metrics["hybrid_search"]["recalls"].append(hybrid_metrics["entity_recall"])
            
            detailed_results.append({
                "query": test_case["query"],
                "category": test_case["category"],
                "entity_latency": entity_latency,
                "attr_latency": attr_latency,
                "relation_latency": relation_latency,
                "hybrid_latency": hybrid_result["latency"],
                "entity_metrics": entity_metrics,
                "attr_metrics": attr_metrics,
                "relation_metrics": relation_metrics,
                "hybrid_metrics": hybrid_metrics
            })
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("性能评估总结")
        self.logger.info("=" * 80)
        
        for search_type, metrics in all_metrics.items():
            self.logger.info(f"\n{search_type}:")
            self.logger.info(f"  平均延迟: {np.mean(metrics['latencies']):.2f}ms")
            self.logger.info(f"  延迟标准差: {np.std(metrics['latencies']):.2f}ms")
            self.logger.info(f"  P95延迟: {np.percentile(metrics['latencies'], 95):.2f}ms")
            self.logger.info(f"  平均精确率: {np.mean(metrics['precisions'])*100:.2f}%")
            self.logger.info(f"  平均召回率: {np.mean(metrics['recalls'])*100:.2f}%")
        
        report = {
            "summary": {
                "test_count": len(self.test_cases),
                "entity_search": {
                    "avg_latency_ms": float(np.mean(all_metrics["entity_search"]["latencies"])),
                    "std_latency_ms": float(np.std(all_metrics["entity_search"]["latencies"])),
                    "p95_latency_ms": float(np.percentile(all_metrics["entity_search"]["latencies"], 95)),
                    "avg_precision": float(np.mean(all_metrics["entity_search"]["precisions"])),
                    "avg_recall": float(np.mean(all_metrics["entity_search"]["recalls"]))
                },
                "attribute_search": {
                    "avg_latency_ms": float(np.mean(all_metrics["attribute_search"]["latencies"])),
                    "std_latency_ms": float(np.std(all_metrics["attribute_search"]["latencies"])),
                    "p95_latency_ms": float(np.percentile(all_metrics["attribute_search"]["latencies"], 95)),
                    "avg_precision": float(np.mean(all_metrics["attribute_search"]["precisions"])),
                    "avg_recall": float(np.mean(all_metrics["attribute_search"]["recalls"]))
                },
                "relation_search": {
                    "avg_latency_ms": float(np.mean(all_metrics["relation_search"]["latencies"])),
                    "std_latency_ms": float(np.std(all_metrics["relation_search"]["latencies"])),
                    "p95_latency_ms": float(np.percentile(all_metrics["relation_search"]["latencies"], 95)),
                    "avg_precision": float(np.mean(all_metrics["relation_search"]["precisions"])),
                    "avg_recall": float(np.mean(all_metrics["relation_search"]["recalls"]))
                },
                "hybrid_search": {
                    "avg_latency_ms": float(np.mean(all_metrics["hybrid_search"]["latencies"])),
                    "std_latency_ms": float(np.std(all_metrics["hybrid_search"]["latencies"])),
                    "p95_latency_ms": float(np.percentile(all_metrics["hybrid_search"]["latencies"], 95)),
                    "avg_precision": float(np.mean(all_metrics["hybrid_search"]["precisions"])),
                    "avg_recall": float(np.mean(all_metrics["hybrid_search"]["recalls"]))
                }
            },
            "detailed_results": detailed_results
        }
        
        report_path = "logs/retrieval_performance_report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"\n✓ 性能报告已保存到: {report_path}")
        
        return report
    
    def cleanup(self):
        """清理资源"""
        try:
            connections.disconnect("default")
        except:
            pass


def main():
    tester = VectorSearchTester()
    try:
        report = tester.run_tests()
        print("\n" + "=" * 80)
        print("测试完成！")
        print("=" * 80)
        return 0
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        tester.cleanup()


if __name__ == "__main__":
    sys.exit(main())
