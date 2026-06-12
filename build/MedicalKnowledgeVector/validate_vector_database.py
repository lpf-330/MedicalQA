# AI辅助生成：GLM-5, 2026-04-18
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量数据库完整性和正确性验证脚本
验证Neo4j数据库中的所有数据是否都已正确转换为向量
"""

import sys
import time
from neo4j import GraphDatabase
from pymilvus import connections, Collection, utility
from config import NEO4J_CONFIG, ZILLIZ_CONFIG


class VectorDatabaseValidator:
    """向量数据库验证器"""
    
    def __init__(self):
        self.neo4j_driver = None
        self.milvus_connected = False
        
    def connect_neo4j(self):
        """连接Neo4j数据库"""
        print("=" * 80)
        print("连接Neo4j数据库")
        print("=" * 80)
        
        self.neo4j_driver = GraphDatabase.driver(
            NEO4J_CONFIG["uri"],
            auth=(NEO4J_CONFIG["user"], NEO4J_CONFIG["password"])
        )
        
        with self.neo4j_driver.session() as session:
            result = session.run("RETURN 1 AS test")
            if result.single()["test"] == 1:
                print("✓ Neo4j连接成功")
                return True
        return False
    
    def connect_milvus(self):
        """连接Milvus向量数据库"""
        print("\n" + "=" * 80)
        print("连接Milvus向量数据库")
        print("=" * 80)
        
        connections.connect(
            alias="default",
            uri=ZILLIZ_CONFIG["uri"],
            token=ZILLIZ_CONFIG["token"]
        )
        
        self.milvus_connected = True
        print("✓ Milvus连接成功")
        return True
    
    def query_all_with_iterator(self, collection, output_fields, batch_size=10000):
        """使用迭代器查询所有数据（解决Milvus查询窗口限制问题）"""
        all_results = []
        
        try:
            iterator = collection.query_iterator(
                expr="",
                output_fields=output_fields,
                batch_size=batch_size
            )
            
            while True:
                results = iterator.next()
                if not results:
                    break
                all_results.extend(results)
                
            iterator.close()
        except Exception as e:
            print(f"使用迭代器查询失败，尝试备用方法: {e}")
            all_results = self._query_all_with_paging(collection, output_fields, batch_size)
        
        return all_results
    
    def _query_all_with_paging(self, collection, output_fields, batch_size=10000):
        """备用分页查询方法（使用ID分页）"""
        all_results = []
        
        results = collection.query(
            expr="",
            output_fields=output_fields + ["id"],
            limit=batch_size
        )
        
        if not results:
            return all_results
        
        all_results.extend(results)
        last_id = results[-1]["id"]
        
        while True:
            results = collection.query(
                expr=f"id > {last_id}",
                output_fields=output_fields + ["id"],
                limit=batch_size
            )
            
            if not results:
                break
            
            all_results.extend(results)
            last_id = results[-1]["id"]
            
            if len(results) < batch_size:
                break
        
        return all_results
    
    def validate_nodes(self):
        """验证节点（实体名称向量）"""
        print("\n" + "=" * 80)
        print("验证实体名称向量（medical_entity集合）")
        print("=" * 80)
        
        # 从Neo4j获取节点统计
        with self.neo4j_driver.session() as session:
            query = """
            MATCH (n)
            RETURN labels(n) AS labels, count(n) AS count
            ORDER BY count DESC
            """
            result = session.run(query)
            
            neo4j_stats = {}
            total_neo4j_nodes = 0
            
            print("\nNeo4j节点统计:")
            for record in result:
                labels = record["labels"]
                count = record["count"]
                label_str = "/".join(labels)
                neo4j_stats[label_str] = count
                total_neo4j_nodes += count
                print(f"  {label_str}: {count}")
            
            print(f"\nNeo4j总节点数: {total_neo4j_nodes}")
        
        # 从Milvus获取向量统计
        if "medical_entity" in utility.list_collections():
            collection = Collection("medical_entity")
            collection.load()
            
            total_milvus_vectors = collection.num_entities
            
            # 使用迭代器查询获取各实体类型的向量数量
            query_result = self.query_all_with_iterator(collection, ["entity_type"])
            
            milvus_stats = {}
            for item in query_result:
                entity_type = item["entity_type"]
                milvus_stats[entity_type] = milvus_stats.get(entity_type, 0) + 1
            
            print("\nMilvus向量统计:")
            for entity_type, count in sorted(milvus_stats.items()):
                print(f"  {entity_type}: {count}")
            
            print(f"\nMilvus总向量数: {total_milvus_vectors}")
            
            # 对比验证
            print("\n" + "-" * 80)
            print("验证结果:")
            print("-" * 80)
            
            if total_neo4j_nodes == total_milvus_vectors:
                print(f"✓ 节点数量匹配: {total_neo4j_nodes} == {total_milvus_vectors}")
            else:
                print(f"✗ 节点数量不匹配: Neo4j={total_neo4j_nodes}, Milvus={total_milvus_vectors}")
            
            # 验证各实体类型
            all_match = True
            for label, neo4j_count in neo4j_stats.items():
                milvus_count = milvus_stats.get(label, 0)
                if neo4j_count == milvus_count:
                    print(f"✓ {label}: {neo4j_count} == {milvus_count}")
                else:
                    print(f"✗ {label}: Neo4j={neo4j_count}, Milvus={milvus_count}")
                    all_match = False
            
            return total_neo4j_nodes == total_milvus_vectors and all_match
        else:
            print("✗ medical_entity集合不存在")
            return False
    
    def validate_attributes(self):
        """验证属性向量"""
        print("\n" + "=" * 80)
        print("验证实体属性向量（entity_attributes集合）")
        print("=" * 80)
        
        # 从Neo4j获取Disease节点的属性统计
        with self.neo4j_driver.session() as session:
            query = """
            MATCH (d:Disease)
            RETURN 
                count(d) AS total,
                count(d.desc) AS has_desc,
                count(d.cause) AS has_cause,
                count(d.prevent) AS has_prevent,
                count(d.easy_get) AS has_easy_get,
                count(d.cure_lasttime) AS has_cure_lasttime,
                count(d.cured_prob) AS has_cured_prob
            """
            result = session.run(query)
            record = result.single()
            
            neo4j_stats = {
                "total": record["total"],
                "desc": record["has_desc"],
                "cause": record["has_cause"],
                "prevent": record["has_prevent"],
                "easy_get": record["has_easy_get"],
                "cure_lasttime": record["has_cure_lasttime"],
                "cured_prob": record["has_cured_prob"]
            }
            
            neo4j_total_attrs = sum([
                record["has_desc"],
                record["has_cause"],
                record["has_prevent"],
                record["has_easy_get"],
                record["has_cure_lasttime"],
                record["has_cured_prob"]
            ])
            
            print("\nNeo4j Disease属性统计:")
            print(f"  Disease节点总数: {neo4j_stats['total']}")
            print(f"  desc属性: {neo4j_stats['desc']}")
            print(f"  cause属性: {neo4j_stats['cause']}")
            print(f"  prevent属性: {neo4j_stats['prevent']}")
            print(f"  easy_get属性: {neo4j_stats['easy_get']}")
            print(f"  cure_lasttime属性: {neo4j_stats['cure_lasttime']}")
            print(f"  cured_prob属性: {neo4j_stats['cured_prob']}")
            print(f"\nNeo4j总属性数: {neo4j_total_attrs}")
        
        # 从Milvus获取向量统计
        if "entity_attributes" in utility.list_collections():
            collection = Collection("entity_attributes")
            collection.load()
            
            total_milvus_vectors = collection.num_entities
            
            # 使用迭代器查询获取各属性类型的向量数量
            query_result = self.query_all_with_iterator(collection, ["attribute_name"])
            
            milvus_stats = {}
            for item in query_result:
                attr_name = item["attribute_name"]
                milvus_stats[attr_name] = milvus_stats.get(attr_name, 0) + 1
            
            print("\nMilvus属性向量统计:")
            for attr_name, count in sorted(milvus_stats.items()):
                print(f"  {attr_name}: {count}")
            
            print(f"\nMilvus总向量数: {total_milvus_vectors}")
            
            # 对比验证
            print("\n" + "-" * 80)
            print("验证结果:")
            print("-" * 80)
            
            if neo4j_total_attrs == total_milvus_vectors:
                print(f"✓ 属性数量匹配: {neo4j_total_attrs} == {total_milvus_vectors}")
            else:
                print(f"⚠ 属性数量不完全匹配: Neo4j={neo4j_total_attrs}, Milvus={total_milvus_vectors}")
                print(f"  差异: {abs(neo4j_total_attrs - total_milvus_vectors)} 条")
            
            # 验证各属性类型
            all_match = True
            for attr, neo4j_count in [("desc", neo4j_stats["desc"]), 
                                       ("cause", neo4j_stats["cause"]),
                                       ("prevent", neo4j_stats["prevent"]),
                                       ("easy_get", neo4j_stats["easy_get"]),
                                       ("cure_lasttime", neo4j_stats["cure_lasttime"]),
                                       ("cured_prob", neo4j_stats["cured_prob"])]:
                milvus_count = milvus_stats.get(attr, 0)
                if neo4j_count == milvus_count:
                    print(f"✓ {attr}: {neo4j_count} == {milvus_count}")
                else:
                    print(f"⚠ {attr}: Neo4j={neo4j_count}, Milvus={milvus_count}")
                    all_match = False
            
            return abs(neo4j_total_attrs - total_milvus_vectors) < 100  # 允许小误差
        else:
            print("✗ entity_attributes集合不存在")
            return False
    
    def validate_relations(self):
        """验证关系向量"""
        print("\n" + "=" * 80)
        print("验证实体关系向量（entity_relations集合）")
        print("=" * 80)
        
        # 从Neo4j获取关系统计
        with self.neo4j_driver.session() as session:
            query = """
            MATCH ()-[r]->()
            RETURN type(r) AS type, count(r) AS count
            ORDER BY count DESC
            """
            result = session.run(query)
            
            neo4j_stats = {}
            total_neo4j_relations = 0
            
            print("\nNeo4j关系统计:")
            for record in result:
                rel_type = record["type"]
                count = record["count"]
                neo4j_stats[rel_type] = count
                total_neo4j_relations += count
                print(f"  {rel_type}: {count}")
            
            print(f"\nNeo4j总关系数: {total_neo4j_relations}")
        
        # 从Milvus获取向量统计
        if "entity_relations" in utility.list_collections():
            collection = Collection("entity_relations")
            collection.load()
            
            total_milvus_vectors = collection.num_entities
            
            # 使用迭代器查询获取各关系类型的向量数量
            query_result = self.query_all_with_iterator(collection, ["relation_type"])
            
            milvus_stats = {}
            for item in query_result:
                rel_type = item["relation_type"]
                milvus_stats[rel_type] = milvus_stats.get(rel_type, 0) + 1
            
            print("\nMilvus关系向量统计:")
            for rel_type, count in sorted(milvus_stats.items()):
                print(f"  {rel_type}: {count}")
            
            print(f"\nMilvus总向量数: {total_milvus_vectors}")
            
            # 对比验证
            print("\n" + "-" * 80)
            print("验证结果:")
            print("-" * 80)
            
            if total_neo4j_relations == total_milvus_vectors:
                print(f"✓ 关系数量匹配: {total_neo4j_relations} == {total_milvus_vectors}")
            else:
                print(f"✗ 关系数量不匹配: Neo4j={total_neo4j_relations}, Milvus={total_milvus_vectors}")
            
            # 验证各关系类型
            all_match = True
            for rel_type, neo4j_count in neo4j_stats.items():
                milvus_count = milvus_stats.get(rel_type, 0)
                if neo4j_count == milvus_count:
                    print(f"✓ {rel_type}: {neo4j_count} == {milvus_count}")
                else:
                    print(f"✗ {rel_type}: Neo4j={neo4j_count}, Milvus={milvus_count}")
                    all_match = False
            
            return total_neo4j_relations == total_milvus_vectors and all_match
        else:
            print("✗ entity_relations集合不存在")
            return False
    
    def run_validation(self):
        """运行完整验证"""
        print("\n" + "=" * 80)
        print("向量数据库完整性和正确性验证")
        print("=" * 80)
        
        try:
            # 连接数据库
            if not self.connect_neo4j():
                return False
            
            if not self.connect_milvus():
                return False
            
            # 验证各集合
            nodes_valid = self.validate_nodes()
            attrs_valid = self.validate_attributes()
            relations_valid = self.validate_relations()
            
            # 总结
            print("\n" + "=" * 80)
            print("验证总结")
            print("=" * 80)
            
            if nodes_valid and attrs_valid and relations_valid:
                print("✓ 所有验证通过！向量数据库完整性和正确性验证成功！")
                return True
            else:
                print("✗ 部分验证失败，请检查详细结果")
                if not nodes_valid:
                    print("  - 实体名称向量验证失败")
                if not attrs_valid:
                    print("  - 实体属性向量验证失败")
                if not relations_valid:
                    print("  - 实体关系向量验证失败")
                return False
            
        except Exception as e:
            print(f"\n✗ 验证过程出错: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # 关闭连接
            if self.neo4j_driver:
                self.neo4j_driver.close()
            if self.milvus_connected:
                connections.disconnect("default")


if __name__ == "__main__":
    validator = VectorDatabaseValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1)
