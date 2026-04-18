# 维度评估Chain使用示例

## 概述

DimensionEvaluationChain是健康报告生成业务的核心组件，负责对8个维度进行评估。

## 维度列表

1. **疾病风险评估** - 基于异常指标和病史，调用向量检索和图谱查询
2. **用药建议** - 基于既往病史和当前用药，查询药物信息
3. **治疗方案** - 基于疾病诊断，查询治疗方案
4. **饮食建议** - 基于疾病和BMI，查询饮食建议
5. **检查建议** - 基于疾病和异常指标，查询检查项目
6. **并发症预警** - 基于疾病和病史，查询并发症
7. **预防措施** - 基于疾病和风险因子，查询预防措施
8. **易感人群** - 基于年龄、性别、病史，评估易感疾病

## 使用示例

```python
from src.orchestration.chain.dimension_evaluation_chain import (
    DimensionEvaluationChain,
    DimensionEvaluationContextBody,
    DimensionEvaluationResource
)
from src.orchestration.chain.data_classes import ChainContext

# 1. 创建资源（需要初始化Handler）
resource = DimensionEvaluationResource(
    vector_handler=vector_handler,  # 向量检索Handler
    neo4j_handler=neo4j_handler,    # Neo4j医疗Handler
    vector_encode_service=None      # 向量编码服务（可选）
)

# 2. 创建Chain实例
chain = DimensionEvaluationChain(resource)

# 3. 准备输入数据
context_body = DimensionEvaluationContextBody(
    anomalies=[
        {"name": "血压偏高", "value": "150/95"},
        {"name": "血糖偏高", "value": "7.2"}
    ],
    risk_factors=[
        {"name": "高血压家族史"},
        {"name": "糖尿病家族史"}
    ],
    medical_entities=[
        {
            "entity": {
                "neo4j_node_id": "123",
                "entity_type": "Disease",
                "name": "高血压"
            }
        }
    ],
    dimension_id="1"  # 疾病风险评估
)

# 4. 执行评估
chain_context = ChainContext(session_id="session_001", body=context_body)
result = chain.execute(chain_context)

# 5. 获取结果
print(f"维度ID: {result.data.dimension_id}")
print(f"维度名称: {result.data.dimension_name}")
print(f"置信度: {result.data.confidence}")
print(f"评估结果: {result.data.evaluation_result}")
```

## 降级策略

当Handler不可用时，Chain会自动启用降级策略：
- Milvus不可用：使用Neo4j模糊匹配替代
- Neo4j不可用：仅使用向量检索结果
- 两者都不可用：返回基础评估结果，置信度为0

## 输出数据结构

```python
{
    "dimension_id": "1",
    "dimension_name": "疾病风险评估",
    "evaluation_result": {
        "risk_level": "中风险",
        "risk_diseases": ["高血压", "糖尿病"],
        "suggestions": ["建议定期监测血压", "建议控制饮食"],
        "basis": ["disease: 高血压", "disease: 糖尿病"]
    },
    "confidence": 0.85
}
```

## 注意事项

1. 使用前需要初始化VectorRetrievalHandler和Neo4jMedicalHandler
2. medical_entities需要包含neo4j_node_id字段才能进行图谱查询
3. 所有注释均为中文，符合项目规范
4. 实现了完整的降级策略，确保系统稳定性
