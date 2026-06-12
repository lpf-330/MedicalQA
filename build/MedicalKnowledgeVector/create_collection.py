# -*- coding: utf-8 -*-
# AI辅助生成：GLM-5, 2026-04-18
"""
Milvus集合创建模块
用于创建MedicalEntityVector向量数据库的medical_entity集合
"""

from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)
from config import ZILLIZ_CONFIG, MILVUS_CONFIG
from logger import get_logger, log_deployment_step, log_deployment_success, log_deployment_failure


class MedicalEntityCollectionCreator:
    """医疗实体向量集合创建器"""
    
    def __init__(self):
        self.logger = get_logger()
        self.collection_name = MILVUS_CONFIG["collection_name"]
        self.dimension = MILVUS_CONFIG["dimension"]
        self.collection = None
        
    def connect_to_zilliz(self):
        """连接到Zilliz Cloud服务"""
        log_deployment_step("连接Zilliz Cloud服务", "开始")
        
        try:
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
    
    def drop_collection_if_exists(self):
        """如果集合已存在，则删除"""
        log_deployment_step("检查并删除已存在的集合", "开始")
        
        try:
            if utility.has_collection(self.collection_name):
                self.logger.warning(f"集合 '{self.collection_name}' 已存在，正在删除...")
                utility.drop_collection(self.collection_name)
                self.logger.info(f"已删除旧集合 '{self.collection_name}'")
            else:
                self.logger.info(f"集合 '{self.collection_name}' 不存在，将创建新集合")
            
            log_deployment_success("检查并删除已存在的集合")
            return True
            
        except Exception as e:
            error_msg = f"删除集合失败: {str(e)}"
            log_deployment_failure("检查并删除已存在的集合", error_msg)
            raise Exception(error_msg)
    
    def create_collection_schema(self):
        """创建集合字段模式"""
        log_deployment_step("创建集合字段模式", "开始")
        
        try:
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=True,
                    description="主键ID，自增"
                ),
                FieldSchema(
                    name="vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.dimension,
                    description=f"{self.dimension}维向量"
                ),
                FieldSchema(
                    name="entity_name",
                    dtype=DataType.VARCHAR,
                    max_length=255,
                    description="实体名称"
                ),
                FieldSchema(
                    name="entity_type",
                    dtype=DataType.VARCHAR,
                    max_length=50,
                    description="实体类型"
                ),
                FieldSchema(
                    name="neo4j_node_id",
                    dtype=DataType.VARCHAR,
                    max_length=50,
                    description="Neo4j节点ID"
                )
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="医疗实体向量集合，用于存储医疗实体的向量表示",
                enable_dynamic_field=False
            )
            
            self.logger.info("集合字段模式创建成功:")
            self.logger.info(f"  - id: INT64 (主键，自增)")
            self.logger.info(f"  - vector: FLOAT_VECTOR({self.dimension})")
            self.logger.info(f"  - entity_name: VARCHAR(255)")
            self.logger.info(f"  - entity_type: VARCHAR(50)")
            self.logger.info(f"  - neo4j_node_id: VARCHAR(128)")
            
            log_deployment_success("创建集合字段模式")
            return schema
            
        except Exception as e:
            error_msg = f"创建集合字段模式失败: {str(e)}"
            log_deployment_failure("创建集合字段模式", error_msg)
            raise Exception(error_msg)
    
    def create_collection(self, schema):
        """创建集合"""
        log_deployment_step("创建集合", "开始")
        
        try:
            self.collection = Collection(
                name=self.collection_name,
                schema=schema,
                using='default',
                shards_num=2
            )
            
            self.logger.info(f"集合 '{self.collection_name}' 创建成功")
            log_deployment_success("创建集合")
            return True
            
        except Exception as e:
            error_msg = f"创建集合失败: {str(e)}"
            log_deployment_failure("创建集合", error_msg)
            raise Exception(error_msg)
    
    def create_index(self):
        """创建向量索引"""
        log_deployment_step("创建向量索引", "开始")
        
        try:
            index_params = {
                "metric_type": MILVUS_CONFIG["metric_type"],
                "index_type": MILVUS_CONFIG["index_type"],
                "params": {"nlist": 1024}
            }
            
            self.logger.info(f"索引配置:")
            self.logger.info(f"  - 索引类型: {MILVUS_CONFIG['index_type']}")
            self.logger.info(f"  - 相似度度量: {MILVUS_CONFIG['metric_type']}")
            self.logger.info(f"  - nlist: 1024")
            
            self.collection.create_index(
                field_name="vector",
                index_params=index_params
            )
            
            self.logger.info(f"向量索引创建成功")
            log_deployment_success("创建向量索引")
            return True
            
        except Exception as e:
            error_msg = f"创建向量索引失败: {str(e)}"
            log_deployment_failure("创建向量索引", error_msg)
            raise Exception(error_msg)
    
    def load_collection(self):
        """加载集合到内存"""
        log_deployment_step("加载集合到内存", "开始")
        
        try:
            self.collection.load()
            
            self.logger.info(f"集合 '{self.collection_name}' 已加载到内存")
            log_deployment_success("加载集合到内存")
            return True
            
        except Exception as e:
            error_msg = f"加载集合到内存失败: {str(e)}"
            log_deployment_failure("加载集合到内存", error_msg)
            raise Exception(error_msg)
    
    def verify_collection(self):
        """验证集合创建结果"""
        log_deployment_step("验证集合创建结果", "开始")
        
        try:
            collection_info = {
                "name": self.collection.name,
                "schema": self.collection.schema,
                "num_entities": self.collection.num_entities,
                "description": self.collection.description
            }
            
            self.logger.info("集合验证信息:")
            self.logger.info(f"  - 集合名称: {collection_info['name']}")
            self.logger.info(f"  - 实体数量: {collection_info['num_entities']}")
            self.logger.info(f"  - 描述: {collection_info['description']}")
            
            indexes = self.collection.indexes
            for index in indexes:
                self.logger.info(f"  - 索引字段: {index.field_name}")
                self.logger.info(f"    索引类型: {index.params.get('index_type')}")
                self.logger.info(f"    度量类型: {index.params.get('metric_type')}")
            
            log_deployment_success("验证集合创建结果")
            return collection_info
            
        except Exception as e:
            error_msg = f"验证集合失败: {str(e)}"
            log_deployment_failure("验证集合创建结果", error_msg)
            raise Exception(error_msg)
    
    def disconnect(self):
        """断开连接"""
        try:
            connections.disconnect("default")
            self.logger.info("已断开与Zilliz Cloud的连接")
        except Exception as e:
            self.logger.warning(f"断开连接时出现警告: {str(e)}")
    
    def run(self):
        """执行完整的集合创建流程"""
        self.logger.info("=" * 60)
        self.logger.info("开始创建 MedicalEntityVector 向量数据库集合")
        self.logger.info("=" * 60)
        
        try:
            self.connect_to_zilliz()
            self.drop_collection_if_exists()
            schema = self.create_collection_schema()
            self.create_collection(schema)
            self.create_index()
            self.load_collection()
            self.verify_collection()
            
            self.logger.info("=" * 60)
            self.logger.info("集合创建完成！")
            self.logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"集合创建流程失败: {str(e)}")
            raise
        finally:
            self.disconnect()


def main():
    """主函数"""
    creator = MedicalEntityCollectionCreator()
    
    try:
        creator.run()
        print("\n✓ 集合创建成功！")
        print(f"✓ 集合名称: {MILVUS_CONFIG['collection_name']}")
        print(f"✓ 向量维度: {VECTOR_CONFIG['dimension']}")
        print(f"✓ 索引类型: {MILVUS_CONFIG['index_type']}")
        print(f"✓ 度量类型: {MILVUS_CONFIG['metric_type']}")
        
    except Exception as e:
        print(f"\n✗ 集合创建失败: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
