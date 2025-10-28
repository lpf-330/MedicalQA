# neo4j_service.py

from neo4j import GraphDatabase
import logging
# 从 log_service 导入日志记录器
from log_service import get_logger, log_neo4j_query # 导入特定的记录器和辅助函数
from typing import List, Dict, Any, Optional
import os

# 使用 log_service 提供的记录器
logger = get_logger(__name__)

# --- 1. 数据库Schema定义 (统一管理) ---
# 定义数据库中存在的节点类型、关系类型和属性键
# 这样便于维护，当数据库Schema发生变化时，只需修改此处
DB_SCHEMA = {
    "node_types": {
        "Check": "检查",
        "Cure": "治疗方法",
        "Department": "部门",
        "Disease": "疾病",
        "Drug": "药物",
        "Food": "食物",
        "Producer": "厂商",
        "Symptom": "症状"
    },
    "relationship_types": [
        "acompany_with", # 并发症
        "belongs_to",    # 属于
        "common_drug",   # 常用药品
        "cure_way",      # 治疗方法
        "do_eat",        # 宜吃
        "drugs_of",      # 药品
        "has_symptom",   # 有症状
        "need_check",    # 需要检查
        "no_eat",        # 忌吃
        "recommand_drug",# 推荐药品
        "recommand_eat"  # 推荐吃
    ],
    "property_keys": [
        "cause",         # 病因
        "cure_lasttime", # 治愈周期
        "cured_prob",    # 治愈概率
        "desc",          # 描述
        "easy_get",      # 易发人群
        # "name",          # name 已被单独返回
        "prevent"        # 预防
    ]
}

# --- 2. Neo4jService 类 ---
class Neo4jService:
    """
    封装与Neo4j数据库交互的查询方法。
    """
    # 在类定义内部或附近定义常量
    MAX_RESULTS_LIMIT = 30 # 限制每个查询方法返回的最大记录数

    def __init__(self, uri: str, user: str, password: str):
        """
        初始化Neo4j驱动连接。

        Args:
            uri (str): Neo4j数据库的URI (e.g., "bolt://localhost:7687").
            user (str): 用户名。
            password (str): 密码。
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """
        关闭Neo4j驱动连接。
        """
        self.driver.close()

    def find_connections_between_entities(
        self,
        entities: List[Dict[str, str]],
        relationships: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        查找给定实体之间的关系或连接。

        Args:
            entities (List[Dict[str, str]]): 实体列表，例如 [{"name": "Hypertension", "type": "Disease"}, ...]。
            relationships (Optional[List[str]]): 关系类型列表，例如 ["HAS_SYMPTOM", "CAUSES"]。如果为None，则查找任何关系。

        Returns:
            List[Dict[str, Any]]: 查询结果列表，每个元素包含连接信息。结果数量被限制为 MAX_RESULTS_LIMIT。
        """
        # 假设最多处理两个实体
        if len(entities) < 2:
            logger.warning("find_connections_between_entities 需要至少两个实体。")
            return []

        entity1 = entities[0]
        entity2 = entities[1]

        # 验证实体类型是否在定义中
        if entity1['type'] not in DB_SCHEMA['node_types'] or entity2['type'] not in DB_SCHEMA['node_types']:
            logger.error(f"find_connections_between_entities: 未知实体类型 {entity1['type']} 或 {entity2['type']}")
            return []

        # 构建Cypher查询
        if relationships:
            # 验证关系类型是否在定义中
            for rel in relationships:
                if rel not in DB_SCHEMA['relationship_types']:
                    logger.warning(f"find_connections_between_entities: 未知关系类型 {rel}")
            rel_clause = f"-[r:{':|'.join(relationships)}]-"
        else:
            rel_clause = "-[]-"

        query = f"""
        MATCH (e1:{entity1['type']} {{name: $entity1_name}}) {rel_clause} (e2:{entity2['type']} {{name: $entity2_name}})
        RETURN e1.name AS entity1, type(r) AS relationship, e2.name AS entity2, properties(r) AS relationship_properties
        """

        result = self.driver.execute_query(
            query,
            entity1_name=entity1['name'],
            entity2_name=entity2['name']
        )

        records = []
        for record in result.records:
            records.append({
                "entity1": record["entity1"],
                "relationship": record["relationship"],
                "entity2": record["entity2"],
                "relationship_properties": record["relationship_properties"]
            })

        logger.info(f"find_connections_between_entities 查询到 {len(records)} 条记录。")
        
        # --- 修改点：限制返回结果数量 ---
        if len(records) > self.MAX_RESULTS_LIMIT:
            logger.info(f"find_connections_between_entities 结果过多 ({len(records)}), 截断至前 {self.MAX_RESULTS_LIMIT} 条。")
            records = records[:self.MAX_RESULTS_LIMIT]
        # --- 修改点结束 ---
        
        return records

    def find_properties_of_entity(
        self,
        entities: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        获取指定实体的属性（如定义、描述等）。

        Args:
            entities (List[Dict[str, str]]): 实体列表。

        Returns:
            List[Dict[str, Any]]: 实体属性列表。结果数量通常较少（每个实体一条），但为一致性也应用限制。
        """
        records = []
        for entity in entities:
            # 验证实体类型是否在定义中
            if entity['type'] not in DB_SCHEMA['node_types']:
                logger.error(f"find_properties_of_entity: 未知实体类型 {entity['type']}")
                continue

            # 查询所有定义的属性
            property_selections = []
            for prop_key in DB_SCHEMA['property_keys']:
                property_selections.append(f"e.{prop_key} AS {prop_key}")
            property_clause = ", ".join(property_selections)

            query = f"""
            MATCH (e:{entity['type']} {{name: $entity_name}})
            RETURN e.name AS name, {property_clause}
            """
            result = self.driver.execute_query(query, entity_name=entity['name'])

            for record in result.records:
                # 只保留非None的属性
                record_dict = {"name": record["name"]}
                for prop_key in DB_SCHEMA['property_keys']:
                    if record.get(prop_key) is not None:
                        record_dict[prop_key] = record[prop_key]
                records.append(record_dict)

        logger.info(f"find_properties_of_entity 查询到 {len(records)} 条记录。")
        
        # --- 修改点：限制返回结果数量 ---
        if len(records) > self.MAX_RESULTS_LIMIT:
            logger.info(f"find_properties_of_entity 结果过多 ({len(records)}), 截断至前 {self.MAX_RESULTS_LIMIT} 条。")
            records = records[:self.MAX_RESULTS_LIMIT]
        # --- 修改点结束 ---
        
        return records

    def find_related_entities_by_relationship(
        self,
        entities: List[Dict[str, str]],
        relationships: List[str]
    ) -> List[Dict[str, Any]]:
        """
        根据给定实体和关系类型，查找相关的其他实体。

        Args:
            entities (List[Dict[str, str]]): 实体列表。
            relationships (List[str]): 关系类型列表。

        Returns:
            List[Dict[str, Any]]: 相关实体列表。结果数量被限制为 MAX_RESULTS_LIMIT。
        """
        records = []
        for entity in entities:
            # 验证实体类型是否在定义中
            if entity['type'] not in DB_SCHEMA['node_types']:
                logger.error(f"find_related_entities_by_relationship: 未知实体类型 {entity['type']}")
                continue

            # 验证关系类型是否在定义中
            for rel in relationships:
                if rel not in DB_SCHEMA['relationship_types']:
                    logger.warning(f"find_related_entities_by_relationship: 未知关系类型 {rel}")

            # 构建匹配关系的模式
            rel_pattern = f"-[r:{':|'.join(relationships)}]-"
            # 假设关系是实体发出的 (OUTGOING)
            query = f"""
            MATCH (e:{entity['type']} {{name: $entity_name}}) {rel_pattern} (related_entity)
            RETURN e.name AS source_entity, type(r) AS relationship, related_entity.name AS related_name, labels(related_entity) AS related_type
            """
            result = self.driver.execute_query(query, entity_name=entity['name'])

            for record in result.records:
                records.append({
                    "source_entity": record["source_entity"],
                    "relationship": record["relationship"],
                    "related_name": record["related_name"],
                    "related_type": record["related_type"][0] # 取第一个标签作为类型
                })

        logger.info(f"find_related_entities_by_relationship 查询到 {len(records)} 条记录。")
        
        # --- 修改点：限制返回结果数量 ---
        if len(records) > self.MAX_RESULTS_LIMIT:
            logger.info(f"find_related_entities_by_relationship 结果过多 ({len(records)}), 截断至前 {self.MAX_RESULTS_LIMIT} 条。")
            records = records[:self.MAX_RESULTS_LIMIT]
        # --- 修改点结束 ---
        
        return records

    def find_common_connections(
        self,
        entities: List[Dict[str, str]],
        relationships: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        查找多个实体共同连接的其他实体或关系。

        Args:
            entities (List[Dict[str, str]]): 实体列表。
            relationships (Optional[List[str]]): 关系类型列表。

        Returns:
            List[Dict[str, Any]]: 共同连接的实体或关系列表。结果数量被限制为 MAX_RESULTS_LIMIT。
        """
        if len(entities) < 2:
            logger.warning("find_common_connections 需要至少两个实体。")
            return []

        # 验证实体类型
        for entity in entities:
            if entity['type'] not in DB_SCHEMA['node_types']:
                logger.error(f"find_common_connections: 未知实体类型 {entity['type']}")
                return []

        # 构建查询，查找与前两个实体都有关联的节点
        # 这是一个简化的示例，实际场景可能需要更复杂的逻辑
        entity1 = entities[0]
        entity2 = entities[1]

        if relationships:
            # 验证关系类型
            for rel in relationships:
                if rel not in DB_SCHEMA['relationship_types']:
                    logger.warning(f"find_common_connections: 未知关系类型 {rel}")
            rel_clause = f"-[:{':|'.join(relationships)}]-"
        else:
            rel_clause = "-[]-"

        # 更准确的查询：查找共同的目标实体
        query = f"""
        MATCH (e1:{entity1['type']} {{name: $entity1_name}})-[r1]->(common_entity)
        MATCH (e2:{entity2['type']} {{name: $entity2_name}})-[r2]->(common_entity)
        WHERE e1 <> e2
        """
        if relationships:
            query += f"AND type(r1) IN $rels AND type(r2) IN $rels "
        query += "RETURN common_entity.name AS common_connection, labels(common_entity) AS common_type, type(r1) AS relationship1, type(r2) AS relationship2"

        params = {"entity1_name": entity1['name'], "entity2_name": entity2['name']}
        if relationships:
            params["rels"] = relationships

        result = self.driver.execute_query(query, **params)

        records = []
        for record in result.records:
            records.append({
                "common_connection": record["common_connection"],
                "common_type": record["common_type"][0],
                "relationship1": record["relationship1"],
                "relationship2": record["relationship2"]
            })

        logger.info(f"find_common_connections 查询到 {len(records)} 条记录。")
        
        # --- 修改点：限制返回结果数量 ---
        if len(records) > self.MAX_RESULTS_LIMIT:
            logger.info(f"find_common_connections 结果过多 ({len(records)}), 截断至前 {self.MAX_RESULTS_LIMIT} 条。")
            records = records[:self.MAX_RESULTS_LIMIT]
        # --- 修改点结束 ---
        
        return records

    def query_entity_relationships(
        self,
        entities: List[Dict[str, str]],
        direction: str = "BOTH" # "INCOMING", "OUTGOING", "BOTH"
    ) -> List[Dict[str, Any]]:
        """
        获取指定实体拥有的关系类型（出边或入边）。

        Args:
            entities (List[Dict[str, str]]): 实体列表。
            direction (str): 关系方向 ("INCOMING", "OUTGOING", "BOTH")。

        Returns:
            List[Dict[str, Any]]: 关系类型列表。结果数量被限制为 MAX_RESULTS_LIMIT。
        """
        records = []
        for entity in entities:
            # 验证实体类型是否在定义中
            if entity['type'] not in DB_SCHEMA['node_types']:
                logger.error(f"query_entity_relationships: 未知实体类型 {entity['type']}")
                continue

            if direction.upper() == "OUTGOING":
                query = f"""
                MATCH (e:{entity['type']} {{name: $entity_name}})-[r]->()
                RETURN DISTINCT type(r) AS relationship_type
                """
            elif direction.upper() == "INCOMING":
                query = f"""
                MATCH ()-[r]->(e:{entity['type']} {{name: $entity_name}})
                RETURN DISTINCT type(r) AS relationship_type
                """
            else: # BOTH
                query = f"""
                MATCH (e:{entity['type']} {{name: $entity_name}})-[r]-()
                RETURN DISTINCT type(r) AS relationship_type
                """

            result = self.driver.execute_query(query, entity_name=entity['name'])

            for record in result.records:
                records.append({
                    "entity": entity['name'],
                    "relationship_type": record["relationship_type"]
                })

        logger.info(f"query_entity_relationships 查询到 {len(records)} 条记录。")
        
        # --- 修改点：限制返回结果数量 ---
        if len(records) > self.MAX_RESULTS_LIMIT:
            logger.info(f"query_entity_relationships 结果过多 ({len(records)}), 截断至前 {self.MAX_RESULTS_LIMIT} 条。")
            records = records[:self.MAX_RESULTS_LIMIT]
        # --- 修改点结束 ---
        
        return records

    def query_entity_connections(
        self,
        entities: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        获取与指定实体相连的其他实体的类型或名称。

        Args:
            entities (List[Dict[str, str]]): 实体列表。

        Returns:
            List[Dict[str, Any]]: 连接的实体信息列表。结果数量被限制为 MAX_RESULTS_LIMIT。
        """
        records = []
        for entity in entities:
            # 验证实体类型是否在定义中
            if entity['type'] not in DB_SCHEMA['node_types']:
                logger.error(f"query_entity_connections: 未知实体类型 {entity['type']}")
                continue

            query = f"""
            MATCH (e:{entity['type']} {{name: $entity_name}})-[r]-(connected_entity)
            RETURN DISTINCT labels(connected_entity) AS connected_entity_types, count(connected_entity) AS count, type(r) AS relationship_type
            """
            # 或者返回具体的实体名称
            # query = f"""
            # MATCH (e:{entity['type']} {{name: $entity_name}})-[r]-(connected_entity)
            # RETURN connected_entity.name AS connected_entity_name, labels(connected_entity) AS connected_entity_type, type(r) AS relationship_type
            # """
            result = self.driver.execute_query(query, entity_name=entity['name'])

            for record in result.records:
                records.append({
                    "entity": entity['name'],
                    "connected_entity_types": record["connected_entity_types"],
                    "count": record["count"],
                    "relationship_type": record["relationship_type"]
                })

        logger.info(f"query_entity_connections 查询到 {len(records)} 条记录。")
        
        # --- 修改点：限制返回结果数量 ---
        if len(records) > self.MAX_RESULTS_LIMIT:
            logger.info(f"query_entity_connections 结果过多 ({len(records)}), 截断至前 {self.MAX_RESULTS_LIMIT} 条。")
            records = records[:self.MAX_RESULTS_LIMIT]
        # --- 修改点结束 ---
        
        return records

    # def find_entities_by_property(
    #     self,
    #     entities: List[Dict[str, str]], # 这里可能包含属性名和值，需要重新设计参数
    #     property_name: str,
    #     property_value: str
    # ) -> List[Dict[str, Any]]:
    #     """
    #     根据实体的属性（如适应症、作用）反向查找实体。

    #     Args:
    #         entities (List[Dict[str, str]]): 可能包含属性信息的实体列表（此参数可能需要调整）。
    #         property_name (str): 要查询的属性名 (e.g., 'indication').
    #         property_value (str): 属性值 (e.g., 'Hypertension').

    #     Returns:
    #         List[Dict[str, Any]]: 符合属性条件的实体列表。结果数量被限制为 MAX_RESULTS_LIMIT。
    #     """
    #     # 假设我们查询Drug类型实体，其indication属性等于property_value
    #     # 这个方法的参数设计可能需要根据实际Neo4j Schema更灵活地调整
    #     query = f"""
    #     MATCH (e) // 可以指定类型，如 MATCH (e:Drug)
    #     WHERE e[$property_name] = $property_value
    #     RETURN e.name AS name, labels(e) AS type
    #     """
    #     # 或者更具体的类型
    #     # query = f"""
    #     # MATCH (e:Drug)
    #     # WHERE e[$property_name] = $property_value
    #     # RETURN e.name AS name
    #     # """
    #     params = {"property_name": property_name, "property_value": property_value}
    #     result = self.driver.execute_query(query, **params)

    #     records = []
    #     for record in result.records:
    #         records.append({
    #             "name": record["name"],
    #             "type": record["type"][0] # 取第一个标签作为类型
    #         })

    #     logger.info(f"find_entities_by_property (property: {property_name}, value: {property_value}) 查询到 {len(records)} 条记录。")
        
    #     # --- 修改点：限制返回结果数量 ---
    #     if len(records) > self.MAX_RESULTS_LIMIT:
    #         logger.info(f"find_entities_by_property 结果过多 ({len(records)}), 截断至前 {self.MAX_RESULTS_LIMIT} 条。")
    #         records = records[:self.MAX_RESULTS_LIMIT]
    #     # --- 修改点结束 ---
        
    #     return records

    def query_relationship_properties(
        self,
        entities: List[Dict[str, str]],
        relationships: List[str]
    ) -> List[Dict[str, Any]]:
        """
        获取关系本身的属性（如强度、来源）。

        Args:
            entities (List[Dict[str, str]]): 实体列表，用于定位关系。
            relationships (List[str]): 关系类型列表。

        Returns:
            List[Dict[str, Any]]: 关系属性列表。结果数量被限制为 MAX_RESULTS_LIMIT。
        """
        # 假设我们查找两个实体之间的特定关系的属性
        if len(entities) < 2:
             logger.warning("query_relationship_properties 需要至少两个实体来定位关系。")
             return []

        # 验证实体类型
        for entity in entities:
            if entity['type'] not in DB_SCHEMA['node_types']:
                logger.error(f"query_relationship_properties: 未知实体类型 {entity['type']}")
                return []

        # 验证关系类型
        for rel in relationships:
            if rel not in DB_SCHEMA['relationship_types']:
                logger.warning(f"query_relationship_properties: 未知关系类型 {rel}")

        entity1 = entities[0]
        entity2 = entities[1]

        query = f"""
        MATCH (e1:{entity1['type']} {{name: $entity1_name}})-[r:{':|'.join(relationships)}]-(e2:{entity2['type']} {{name: $entity2_name}})
        RETURN e1.name AS entity1, e2.name AS entity2, type(r) AS relationship_type, properties(r) AS properties
        """
        result = self.driver.execute_query(
            query,
            entity1_name=entity1['name'],
            entity2_name=entity2['name']
        )

        records = []
        for record in result.records:
            records.append({
                "entity1": record["entity1"],
                "entity2": record["entity2"],
                "relationship_type": record["relationship_type"],
                "properties": record["properties"]
            })

        logger.info(f"query_relationship_properties 查询到 {len(records)} 条记录。")
        
        # --- 修改点：限制返回结果数量 ---
        if len(records) > self.MAX_RESULTS_LIMIT:
            logger.info(f"query_relationship_properties 结果过多 ({len(records)}), 截断至前 {self.MAX_RESULTS_LIMIT} 条。")
            records = records[:self.MAX_RESULTS_LIMIT]
        # --- 修改点结束 ---
        
        return records

# --- 示例用法 (如果直接运行此文件) ---
# if __name__ == "__main__":
#     # 日志已在 log_service.py 中配置
#     logger = get_logger(__name__)

#     # 请替换为实际的Neo4j连接信息
#     uri = os.getenv("NEO4J_URI", "neo4j+s://74f0576d.databases.neo4j.io")
#     user = os.getenv("NEO4J_USER", "neo4j")
#     password = os.getenv("NEO4J_PASSWORD", "Jvabu2hncwBYexP_vWoQpRQd3tIp0pul7QK4sK6xh_s")

#     neo4j_service = Neo4jService(uri, user, password)

#     try:
#         # 示例调用
#         entities_example = [{"name": "Hypertension", "type": "Disease"}]
#         props = neo4j_service.find_properties_of_entity(entities_example)
#         print("Properties:", props)

#         rels = neo4j_service.query_entity_relationships(entities_example, direction="OUTGOING")
#         print("Outgoing Relationships:", rels)

#     finally:
#         neo4j_service.close()
