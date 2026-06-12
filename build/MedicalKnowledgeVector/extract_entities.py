# AI辅助生成：GLM-5, 2026-04-18
import json
import os
from typing import List, Dict, Any
from neo4j import GraphDatabase
from config import NEO4J_CONFIG
from logger import get_logger


class Neo4jEntityExtractor:
    def __init__(self):
        self.logger = get_logger()
        self.driver = None
        self.entity_types = ["Disease", "Drug", "Symptom", "Food", "Check", "Department"]
        self.output_file = "data/entities.json"
        
    def connect(self):
        try:
            self.logger.log_deployment_step("连接Neo4j数据库", "开始")
            self.driver = GraphDatabase.driver(
                NEO4J_CONFIG["uri"],
                auth=(NEO4J_CONFIG["user"], NEO4J_CONFIG["password"])
            )
            self.driver.verify_connectivity()
            self.logger.log_deployment_success("连接Neo4j数据库")
            return True
        except Exception as e:
            self.logger.log_deployment_failure("连接Neo4j数据库", str(e))
            return False
    
    def close(self):
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j数据库连接已关闭")
    
    def extract_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        entities = []
        try:
            with self.driver.session() as session:
                query = f"MATCH (n:{entity_type}) RETURN elementId(n) as neo4j_id, n.name as name"
                result = session.run(query)
                
                for record in result:
                    entity = {
                        "neo4j_id": str(record["neo4j_id"]),
                        "name": record["name"],
                        "entity_type": entity_type
                    }
                    entities.append(entity)
                
                self.logger.info(f"成功提取 {entity_type} 实体: {len(entities)} 条")
        except Exception as e:
            self.logger.error(f"提取 {entity_type} 实体失败: {str(e)}")
        
        return entities
    
    def extract_all_entities(self) -> List[Dict[str, Any]]:
        all_entities = []
        total_types = len(self.entity_types)
        
        self.logger.log_deployment_step("提取所有实体数据", "开始")
        
        for idx, entity_type in enumerate(self.entity_types, 1):
            self.logger.log_progress(idx, total_types, f"正在提取 {entity_type} 实体")
            entities = self.extract_entities_by_type(entity_type)
            all_entities.extend(entities)
        
        self.logger.info(f"实体提取完成，总计: {len(all_entities)} 条")
        return all_entities
    
    def save_to_json(self, entities: List[Dict[str, Any]]) -> bool:
        try:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(entities, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"数据已保存至: {self.output_file}")
            return True
        except Exception as e:
            self.logger.log_deployment_failure("保存数据到JSON文件", str(e))
            return False
    
    def generate_statistics(self, entities: List[Dict[str, Any]]):
        stats = {}
        for entity in entities:
            entity_type = entity["entity_type"]
            stats[entity_type] = stats.get(entity_type, 0) + 1
        
        self.logger.info("=" * 50)
        self.logger.info("实体统计信息:")
        self.logger.info("=" * 50)
        
        total = 0
        for entity_type in self.entity_types:
            count = stats.get(entity_type, 0)
            total += count
            self.logger.info(f"  {entity_type}: {count} 条")
        
        self.logger.info("-" * 50)
        self.logger.info(f"  总计: {total} 条")
        self.logger.info("=" * 50)
        
        return stats
    
    def run(self):
        self.logger.info("开始执行Neo4j实体数据提取任务")
        self.logger.info("=" * 50)
        
        if not self.connect():
            return False
        
        entities = self.extract_all_entities()
        
        if not entities:
            self.logger.warning("未提取到任何实体数据")
            self.close()
            return False
        
        if not self.save_to_json(entities):
            self.close()
            return False
        
        self.generate_statistics(entities)
        
        self.close()
        
        self.logger.info("=" * 50)
        self.logger.info("Neo4j实体数据提取任务执行完成")
        return True


def main():
    extractor = Neo4jEntityExtractor()
    success = extractor.run()
    
    if success:
        print("\n✓ 实体数据提取成功!")
        print(f"✓ 数据已保存至: data/entities.json")
    else:
        print("\n✗ 实体数据提取失败，请查看日志了解详情")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
