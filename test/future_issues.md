# 未来考虑解决的问题

本文档记录了第二轮验收评估测试中发现但暂不修复的问题，供未来参考和处理。

---

## 一、架构合规性问题

### 建议1：文档完善建议

**问题描述**：
- 建议为所有资源封装类的Client类添加更详细的业务方法文档
- 建议为状态机的状态转换图添加可视化文档

**影响范围**：文档层面，不影响功能

**严重程度**：建议

**设计依据**：项目架构原则与使用规范v1.md

**未来修复建议**：
1. 为Client类添加详细的方法文档说明
2. 创建状态机状态转换图的可视化文档

**预估工作量**：2-4小时

**状态**：待未来处理

---

## 二、业务设计合规性问题

### 问题2：图查询缺少基于关系的深度推理

**问题描述**：
- 未实现基于anchored_relations的深度推理功能

**影响范围**：KNOWLEDGE_RETRIEVAL状态的图查询步骤

**严重程度**：中等

**设计依据**：项目业务详细设计v3.md 第174-183行

**详细说明**：
根据设计文档，图查询步骤应该实现基于关系的深度推理：
1. 从anchored_relations中提取关系名称
2. 查询关系相关联的实体
3. 递归查询关联实体的详细信息
4. 将推理结果添加到知识结果中

**未来修复建议**：
```python
# 在 _graph_query_step() 方法中添加关系深度推理
def _graph_query_step(self, anchored_entities, anchored_relations, query_text):
    knowledge_results = []
    
    # 原有的实体查询逻辑
    for entity in anchored_entities:
        # ... 现有代码
    
    # 新增：基于关系的深度推理
    for relation in anchored_relations:
        relation_name = relation.get("name", relation.get("relation_name", ""))
        try:
            # 查询关系相关联的实体
            related_entities = self._resource.neo4j_handler.get_entities_by_relation(relation_name)
            for related_entity in related_entities:
                # 递归查询关联实体的详细信息
                entity_info = self._resource.neo4j_handler.get_disease_info(related_entity)
                if entity_info:
                    knowledge_results.append({
                        "source": "neo4j",
                        "type": "relation_inferred",
                        "relation": relation_name,
                        "entity": related_entity,
                        "data": entity_info,
                        "score": relation.get("score", 0.0) * 0.8  # 推理结果降低权重
                    })
        except Exception as e:
            logger.error(f"[KnowledgeRetrievalChain] 关系深度推理失败: {e}")
    
    return knowledge_results
```

**预估工作量**：3-4小时

**状态**：待未来处理

---

## 三、意图识别功能升级

### 问题：当前意图识别功能已废弃

**问题描述**：
- 原有的意图识别功能使用Apollo-0.5B模型，对医疗健康领域的意图识别能力有限
- 典型的健康咨询问题（如"糖尿病有什么症状？"）置信度仅为0.5499，需要依赖关键词增强才能达到合理水平
- 意图识别功能已暂时注释掉，所有查询默认作为健康咨询处理

**影响范围**：QUERY_PARSE状态

**严重程度**：中等

**设计依据**：项目业务详细设计v3.md

**废弃时间**：2026-04-18

**未来修复建议**：

#### 方案：Neo4j关系意图分析模型 + 查询代码构建模型

将意图分类升级为两阶段智能检索：

**阶段1：Neo4j关系意图分析模型**
- 分析用户输入中涉及的Neo4j数据库关系类型
- 识别用户查询意图（如：症状查询、疾病查询、药物查询、治疗方案查询等）
- 输出：意图类型 + 相关实体 + 关系类型

**阶段2：查询代码构建模型**
- 基于意图分析结果，自动构建Neo4j Cypher查询语句
- 支持复杂的多跳查询和关系推理
- 输出：可执行的Cypher查询语句

**实现步骤**：
1. 训练或微调专门的关系意图分析模型
2. 训练或微调Cypher查询构建模型
3. 集成到QUERY_PARSE状态
4. 替换当前的IntentParseChain

**预估工作量**：8-12小时

**状态**：待未来处理

**相关代码位置**：
- src/orchestration/agent/consult_strategy/consult_strategy.py: _handle_query_parse方法
- src/orchestration/chain/intent_parse_chain/intent_parse_chain.py
- src/tools/intent_classification_tool/intent_classification_tool.py

---

## 四、问题优先级排序

| 序号 | 问题 | 严重程度 | 预估工作量 | 建议处理时间 |
|------|------|---------|-----------|-------------|
| 1 | 意图识别功能升级 | 中等 | 8-12小时 | 下次迭代 |
| 2 | 文档完善建议 | 建议 | 2-4小时 | 下次迭代 |
| 3 | 图查询关系深度推理 | 中等 | 3-4小时 | 下次迭代 |

---

## 五、风险评估

| 风险项 | 风险描述 | 影响程度 | 应对措施 |
|--------|---------|---------|---------|
| 意图识别缺失 | 所有查询都作为健康咨询处理 | 中 | 尽快实现Neo4j关系意图分析模型 |
| 关系深度推理 | 可能增加响应时间 | 中 | 添加超时控制，异步处理 |
| 文档完善 | 不影响功能 | 低 | 逐步完善 |

---

**文档创建时间**: 2026-04-17
**文档版本**: v1.0
**来源**: 第二轮验收评估测试 phase6_fix_work_report.md
