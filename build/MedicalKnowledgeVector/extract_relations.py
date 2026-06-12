# AI辅助生成：GLM-5, 2026-04-18
import json
import os
from typing import List, Dict, Any
from neo4j import GraphDatabase
from config import NEO4J_CONFIG
from logger import get_logger


ENTITY_TYPE_MAP = {
    "Disease": "疾病",
    "Drug": "药物",
    "Symptom": "症状",
    "Food": "食物",
    "Check": "检查",
    "Department": "科室",
    "Producer": "生产商",
    "Cure": "治疗方法"
}

RELATION_NAME_MAP = {
    "recommand_drug": "好评药品",
    "has_symptom": "症状",
    "recommand_eat": "推荐食谱",
    "need_check": "诊断检查",
    "drugs_of": "生产药品",
    "cure_way": "治疗方法",
    "no_eat": "忌吃",
    "do_eat": "宜吃",
    "acompany_with": "并发症",
    "common_drug": "常用药品",
    "belongs_to": "属于"
}

RELATION_TYPES = list(RELATION_NAME_MAP.keys())


class Neo4jRelationExtractor:
    def __init__(self):
        self.logger = get_logger()
        self.driver = None
        self.output_file = "data/relations.json"
        
    def connect(self):
        try:
            self.logger.log_deployment_step("连接Neo4j数据库", "开始")
            self.driver = GraphDatabase.driver(
                NEO4J_CONFIG['uri'],
                auth=(NEO4J_CONFIG['user'], NEO4J_CONFIG['password'])
            )
            
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                result.single()
            
            self.logger.log_deployment_success("连接Neo4j数据库")
            return True
            
        except Exception as e:
            self.logger.log_deployment_failure("连接Neo4j数据库", str(e))
            raise Exception(f"Neo4j连接失败: {e}")
    
    def build_relation_description(
        self,
        source_entity_type: str,
        source_entity_name: str,
        relation_name: str,
        target_entity_type: str,
        target_entity_name: str
    ) -> str:
        source_type_cn = ENTITY_TYPE_MAP.get(source_entity_type, source_entity_type)
        target_type_cn = ENTITY_TYPE_MAP.get(target_entity_type, target_entity_type)
        
        return f"{source_type_cn}：{source_entity_name} {relation_name} {target_type_cn}：{target_entity_name}"
    
    def extract_relations_by_type(self, session, relation_type: str) -> List[Dict[str, Any]]:
        relations = []
        
        query = f"""
        MATCH (source)-[r:{relation_type}]->(target)
        WHERE source.name IS NOT NULL AND target.name IS NOT NULL
        RETURN 
            labels(source) as source_labels,
            source.name as source_name,
            labels(target) as target_labels,
            target.name as target_name,
            elementId(r) as relation_id
        """
        
        result = session.run(query)
        
        for record in result:
            source_labels = record['source_labels']
            source_name = record['source_name']
            target_labels = record['target_labels']
            target_name = record['target_name']
            relation_id = record['relation_id']
            
            if not source_name or not target_name:
                continue
                
            source_entity_type = source_labels[0] if source_labels else 'Unknown'
            target_entity_type = target_labels[0] if target_labels else 'Unknown'
            
            relation_name = RELATION_NAME_MAP.get(relation_type, relation_type)
            
            relation_description = self.build_relation_description(
                source_entity_type,
                source_name,
                relation_name,
                target_entity_type,
                target_name
            )
            
            relations.append({
                "relation_type": relation_type,
                "relation_name": relation_name,
                "source_entity_name": source_name,
                "source_entity_type": source_entity_type,
                "target_entity_name": target_name,
                "target_entity_type": target_entity_type,
                "relation_description": relation_description,
                "neo4j_relation_id": str(relation_id)
            })
        
        return relations
    
    def deduplicate_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_ids = set()
        unique_relations = []
        
        for relation in relations:
            relation_id = relation['neo4j_relation_id']
            
            if relation_id not in seen_ids:
                seen_ids.add(relation_id)
                unique_relations.append(relation)
        
        return unique_relations
    
    def extract_all_relations(self) -> List[Dict[str, Any]]:
        try:
            self.logger.log_deployment_step("提取所有关系", "开始")
            
            all_relations = []
            relation_counts = {}
            
            with self.driver.session() as session:
                for relation_type in RELATION_TYPES:
                    self.logger.info(f"正在提取关系类型: {relation_type}")
                    
                    relations = self.extract_relations_by_type(session, relation_type)
                    all_relations.extend(relations)
                    
                    relation_counts[relation_type] = len(relations)
                    self.logger.info(f"  {relation_type}: {len(relations)} 条")
            
            self.logger.info("=" * 60)
            self.logger.info("关系提取统计（去重前）:")
            self.logger.info("=" * 60)
            for relation_type, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True):
                relation_name = RELATION_NAME_MAP.get(relation_type, relation_type)
                self.logger.info(f"  {relation_type} ({relation_name}): {count} 条")
            self.logger.info(f"  总计: {len(all_relations)} 条")
            self.logger.info("=" * 60)
            
            unique_relations = self.deduplicate_relations(all_relations)
            
            self.logger.info("=" * 60)
            self.logger.info("关系提取统计（去重后）:")
            self.logger.info("=" * 60)
            
            unique_counts = {}
            for relation in unique_relations:
                rt = relation['relation_type']
                unique_counts[rt] = unique_counts.get(rt, 0) + 1
            
            for relation_type, count in sorted(unique_counts.items(), key=lambda x: x[1], reverse=True):
                relation_name = RELATION_NAME_MAP.get(relation_type, relation_type)
                self.logger.info(f"  {relation_type} ({relation_name}): {count} 条")
            self.logger.info(f"  总计: {len(unique_relations)} 条")
            self.logger.info(f"  去重: {len(all_relations) - len(unique_relations)} 条")
            self.logger.info("=" * 60)
            
            self.logger.log_deployment_success("提取所有关系")
            return unique_relations
            
        except Exception as e:
            self.logger.log_deployment_failure("提取所有关系", str(e))
            raise
    
    def save_to_json(self, relations: List[Dict[str, Any]]):
        try:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(relations, f, ensure_ascii=False, indent=2)
            
            file_size = os.path.getsize(self.output_file) / 1024 / 1024
            self.logger.info(f"关系数据已保存到: {self.output_file}")
            self.logger.info(f"文件大小: {file_size:.2f} MB")
            
        except Exception as e:
            self.logger.error(f"保存关系数据失败: {e}")
            raise
    
    def close(self):
        if self.driver:
            self.driver.close()
            self.logger.info("已断开与Neo4j的连接")
    
    def run(self):
        self.logger.info("=" * 60)
        self.logger.info("开始从Neo4j提取所有关系")
        self.logger.info("=" * 60)
        
        try:
            self.connect()
            
            relations = self.extract_all_relations()
            self.save_to_json(relations)
            
            self.logger.info("=" * 60)
            self.logger.info("关系提取完成！")
            self.logger.info("=" * 60)
            
            return relations
            
        except Exception as e:
            self.logger.error(f"关系提取失败: {e}")
            raise
        finally:
            self.close()


if __name__ == "__main__":
    extractor = Neo4jRelationExtractor()
    relations = extractor.run()
    print(f"\n成功提取 {len(relations)} 条关系")
