"""
实体属性向量集合创建模块
用于创建MedicalEntityVector向量数据库的entity_attributes集合
"""

from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)
from config import ZILLIZ_CONFIG
from logger import get_logger, log_deployment_step, log_deployment_success, log_deployment_failure


class EntityAttributesCollectionCreator:
    """实体属性向量集合创建器"""
    
    def __init__(self):
        self.logger = get_logger()
        self.collection_name = "entity_attributes"
        self.dimension = 1024
        self.collection = None
        
    def connect_to_zilliz(self):
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
                    name="attribute_name",
                    dtype=DataType.VARCHAR,
                    max_length=50,
                    description="属性名称"
                ),
                FieldSchema(
                    name="attribute_value",
                    dtype=DataType.VARCHAR,
                    max_length=10000,
                    description="属性值"
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
                description="实体属性向量集合，用于存储医疗实体属性的向量表示",
                enable_dynamic_field=False
            )
            
            self.logger.info("集合字段模式创建成功:")
            self.logger.info(f"  - id: INT64 (主键，自增)")
            self.logger.info(f"  - vector: FLOAT_VECTOR({self.dimension})")
            self.logger.info(f"  - entity_name: VARCHAR(255)")
            self.logger.info(f"  - entity_type: VARCHAR(50)")
            self.logger.info(f"  - attribute_name: VARCHAR(50)")
            self.logger.info(f"  - attribute_value: VARCHAR(10000)")
            self.logger.info(f"  - neo4j_node_id: VARCHAR(50)")
            
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
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            
            self.logger.info(f"索引配置:")
            self.logger.info(f"  - 索引类型: IVF_FLAT")
            self.logger.info(f"  - 相似度度量: COSINE")
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
        self.logger.info("开始创建 EntityAttributes 向量数据库集合")
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
    creator = EntityAttributesCollectionCreator()
    
    try:
        creator.run()
        print("\n✓ 集合创建成功！")
        print(f"✓ 集合名称: entity_attributes")
        print(f"✓ 向量维度: 1024")
        print(f"✓ 索引类型: IVF_FLAT")
        print(f"✓ 度量类型: COSINE")
        
    except Exception as e:
        print(f"\n✗ 集合创建失败: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
