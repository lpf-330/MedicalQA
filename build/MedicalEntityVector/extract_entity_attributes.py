import json
import os
import re
from typing import List, Dict, Any
from neo4j import GraphDatabase
from config import NEO4J_CONFIG
from logger import get_logger


class DiseaseAttributeExtractor:
    ATTRIBUTE_MAPPING = {
        'desc': '疾病描述',
        'prevent': '预防措施',
        'cause': '病因',
        'easy_get': '易感人群',
        'cure_lasttime': '治疗时间',
        'cured_prob': '治愈概率'
    }
    
    MAX_ATTRIBUTE_LENGTH = 10000
    
    def __init__(self):
        self.logger = get_logger()
        self.driver = None
        self.output_file = "data/disease_attributes.json"
        
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
            raise
    
    def clean_attribute_value(self, value: str) -> tuple:
        if not value or not isinstance(value, str):
            return None, 0, 0
        
        original_length = len(value)
        
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        # 使用字节数限制而不是字符数限制
        # Milvus VARCHAR长度限制是基于字节数的
        cleaned_bytes = cleaned.encode('utf-8')
        if len(cleaned_bytes) > self.MAX_ATTRIBUTE_LENGTH:
            # 截断到指定字节数
            cleaned_bytes = cleaned_bytes[:self.MAX_ATTRIBUTE_LENGTH]
            # 解码时可能遇到不完整的UTF-8字符，忽略错误
            cleaned = cleaned_bytes.decode('utf-8', errors='ignore')
        
        cleaned_length = len(cleaned)
        
        return cleaned, original_length, cleaned_length
    
    def build_vector_text(self, entity_name: str, attribute_name_cn: str, attribute_value: str) -> str:
        return f"疾病：{entity_name}，{attribute_name_cn}：{attribute_value}"
    
    def extract_disease_attributes(self) -> List[Dict[str, Any]]:
        try:
            self.logger.log_deployment_step("提取Disease实体属性", "开始")
            
            attributes_data = []
            attribute_stats = {}
            length_stats = {'original': 0, 'cleaned': 0}
            
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (d:Disease)
                    RETURN d.name as name, 
                           d.desc as desc,
                           d.prevent as prevent,
                           d.cause as cause,
                           d.easy_get as easy_get,
                           d.cure_lasttime as cure_lasttime,
                           d.cured_prob as cured_prob,
                           id(d) as neo4j_id
                    ORDER BY d.name
                """)
                
                disease_count = 0
                
                for record in result:
                    name = record['name']
                    neo4j_id = record['neo4j_id']
                    
                    if not name:
                        continue
                    
                    disease_count += 1
                    
                    for attr_key in self.ATTRIBUTE_MAPPING.keys():
                        attr_value = record[attr_key]
                        
                        if not attr_value:
                            continue
                        
                        cleaned_value, original_len, cleaned_len = self.clean_attribute_value(attr_value)
                        
                        if not cleaned_value:
                            continue
                        
                        length_stats['original'] += original_len
                        length_stats['cleaned'] += cleaned_len
                        
                        attr_name_cn = self.ATTRIBUTE_MAPPING[attr_key]
                        vector_text = self.build_vector_text(name, attr_name_cn, cleaned_value)
                        
                        attributes_data.append({
                            "entity_name": name,
                            "entity_type": "Disease",
                            "attribute_name": attr_key,
                            "attribute_name_cn": attr_name_cn,
                            "attribute_value": cleaned_value,
                            "vector_text": vector_text,
                            "neo4j_node_id": str(neo4j_id)
                        })
                        
                        attribute_stats[attr_key] = attribute_stats.get(attr_key, 0) + 1
                
                self.logger.info("=" * 60)
                self.logger.info("Disease实体属性提取统计:")
                self.logger.info("=" * 60)
                self.logger.info(f"  处理疾病数量: {disease_count} 条")
                self.logger.info(f"  提取属性总数: {len(attributes_data)} 条")
                self.logger.info("-" * 60)
                self.logger.info("  各属性统计:")
                for attr_key in self.ATTRIBUTE_MAPPING.keys():
                    count = attribute_stats.get(attr_key, 0)
                    attr_name_cn = self.ATTRIBUTE_MAPPING[attr_key]
                    self.logger.info(f"    {attr_name_cn}({attr_key}): {count} 条")
                self.logger.info("-" * 60)
                self.logger.info(f"  原始文本总长度: {length_stats['original']} 字符")
                self.logger.info(f"  清洗后总长度: {length_stats['cleaned']} 字符")
                if length_stats['original'] > 0:
                    ratio = length_stats['cleaned'] / length_stats['original'] * 100
                    self.logger.info(f"  长度保留比例: {ratio:.2f}%")
                self.logger.info("=" * 60)
            
            self.logger.log_deployment_success("提取Disease实体属性")
            return attributes_data
            
        except Exception as e:
            self.logger.log_deployment_failure("提取Disease实体属性", str(e))
            raise
    
    def save_to_json(self, attributes_data: List[Dict[str, Any]]):
        try:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(attributes_data, f, ensure_ascii=False, indent=2)
            
            file_size = os.path.getsize(self.output_file) / 1024 / 1024
            self.logger.info(f"属性数据已保存到: {self.output_file}")
            self.logger.info(f"文件大小: {file_size:.2f} MB")
            
        except Exception as e:
            self.logger.error(f"保存属性数据失败: {e}")
            raise
    
    def close(self):
        if self.driver:
            self.driver.close()
            self.logger.info("已断开与Neo4j的连接")
    
    def run(self) -> List[Dict[str, Any]]:
        self.logger.info("=" * 60)
        self.logger.info("开始从Neo4j提取Disease实体属性")
        self.logger.info("=" * 60)
        
        try:
            self.connect()
            
            attributes_data = self.extract_disease_attributes()
            self.save_to_json(attributes_data)
            
            self.logger.info("=" * 60)
            self.logger.info("Disease实体属性提取完成！")
            self.logger.info("=" * 60)
            
            return attributes_data
            
        except Exception as e:
            self.logger.error(f"Disease实体属性提取失败: {e}")
            raise
        finally:
            self.close()


if __name__ == "__main__":
    extractor = DiseaseAttributeExtractor()
    attributes = extractor.run()
    print(f"\n成功提取 {len(attributes)} 条属性数据")
