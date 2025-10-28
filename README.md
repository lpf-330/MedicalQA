项目需求：
1、在本地部署的Qwen3-4B-Instruct-2507上加入已经部署在云端的neo4j知识图谱作为回答问题的数据支持，从而实现一个医疗知识问答模型
2、将医疗知识问答模型集成到springboot项目中，给web端提供对话接口
3、项目追求低耦合、高内聚的强拓展性
4、后端返回前端用户问题的回答时采用流式输出

硬件条件：2080ti 22g显存+32g内存

主要技术栈：
1、springboot：Spring Web + HTTP Client (WebClient) +Redis+LangChain4j
2、python：Qwen3-4B-Instruct-2507+neo4j+vLLM+Transformers+FastAPI

服务部署与通信：
1、springboot和vLLM服务部署到一个主机上，他们之间通过本地http+json通信。neo4j部署在云端，通过neo4j协议与vLLM服务通信。
2、redis部署也部署在该主机

Prompt模板：
1、Qwen3-4B-Instruct-2507标准模板上进行针对neo4j进行修改
2、确保最重要的信息（Prompt模板、用户问题）优先保留
3、第一次调用模型要给实体中文名列表（从entity.json中按类型提取出）、关系中文名列表（从relationship.json中提取出）、意图中文名列表（从intent_interface_mapping中提取出），让它去对用户问题进行“解析实体和关系+处理映射+分析意图”，第一次调用模型的输出格式示例如下：
```
{
  "entity_relationship_groups": [
    {
      "entities": [
        {"name": "高血压", "type": "Disease"}, 
        {"name": "头痛", "type": "Symptom"}
      ],
      "relationships": ["有症状", "导致"],
      "intent": "...之间的关系"
    },
    {
      "entities": [
        {"name": "阿司匹林", "type": "Drug"}
      ],
      "relationships": ["治疗"], 
      "intent": "治疗什么"
    }
  ],
  "standalone_entities": [
    {"name": "患者", "type": "Person"}
  ],
  "overall_intent": "查找连接和治疗方法"
}
```
4、第二次调用模型给用户问题和程序所查询到的信息，模型根据信息回答并返回。

RAG流程：
1、两次调用 (LLM解析 + LLM生成)，第一次调用模型处理“解析实体和关系+处理映射+分析意图”，然后程序根据模型解析结果获取neo4j信息并返回；第二次模型根据neo4j回答用户问题。
2、得到第一次调用模型的处理结果后，程序把模型输出结果中的中文关系名通过relationship.json转成英文关系类型（可能会作为neo4j接口的参数），并根据模型输出结果和intent_interface_mapping.json调用对应接口（接口方法在neo4j_service.py）。
3、程序根据第一次调用模型的输出内容，给每个实体关系组根据intent_interface_mapping.json找到并调用对应的neo4j接口。（注意"standalone_entities"与"overall_intent"的处理），并将用户问题和所查询到的信息作为第二次调用模型的输入。

模型与neo4j词义不同问题：
模型在实体表文件内容（按照类型分为几类）、关系表文件内容（包括实体之间的所有关系类型）自动选择最合适的名词。模型在实体表中未找到合适名字时，让模型根据自己知识给出专业医学名词。





neo4j查询接口至少要包括：find_connections_between_entities、find_properties_of_entity、find_related_entities_by_relationship、find_common_connections、query_entity_relationships、query_entity_connections、find_entities_by_property、query_relationship_properties。


选择调用接口的问题：使用意图接口映射方案，意图在模型第一次调用的结果中，neo4j查询接口与意图之间设置映射表，一个意图可能对应多个接口。

springboot与vLLM服务的接口设计：（后面讨论）

会话管理：（后面讨论）

部署提示：
1、将Qwen3-4B-Instruct-2507下载至第一级的base_model目录下
2、注意代码中可能使用了绝对路径，记得修改


优化方案：
1、RAG流程

2、模型与neo4j词义不同问题：
```
一、方案A（暂行）

二、方案B——基于向量化的Neo4j RAG（也可以用来解决“用户问题中实体与关系的排列组合”问题）
1、核心目标： 解决模型输入与Neo4j数据库存储实体名称之间的语义不匹配问题，通过语义相似度而非精确匹配来检索信息。

2、技术选型与架构：
（1）向量数据库 (Vector Store):
推荐：ChromaDB 或 Qdrant。
理由： 对于本地开发和中小规模数据，ChromaDB配置简单，易于上手。Qdrant功能强大，性能较好，也支持本地部署。两者都有良好的Python客户端。
部署： 可通过Docker快速部署 (docker run 或 docker-compose up)。
（2）嵌入模型 (Embedding Model):
推荐：sentence-transformers 模型 (如 all-MiniLM-L6-v2 或 all-mpnet-base-v2) 或 Hugging Face Inference API。
理由： sentence-transformers 模型可以直接在Python环境中加载，与您的FastAPI服务集成。Hugging Face API使用简单，无需本地部署，但有网络依赖和成本。
部署：
本地： 使用transformers库加载模型，可能用FastAPI封装成独立服务。
API： 配置API Key和端点即可。
（3）知识库准备 (离线/定时)：
数据提取： 复用您现有的neo4j Python驱动，编写一个脚本连接到Neo4j。
提取内容： 执行Cypher查询，提取节点的name、description等文本属性，以及重要的关系信息（如"实体A -[关系类型]-> 实体B"）。将这些信息格式化为文本块（Documents）。
向量化与存储： 使用选定的嵌入模型将提取的文本块转换为向量，并利用向量数据库的客户端库将向量及其元数据（如原始文本、Neo4j节点ID）存入向量数据库。
（4）查询处理 (在线)：
集成到FastAPI： 修改您现有的FastAPI服务（或创建新的端点）。
问题向量化： 接收用户问题，使用与知识库准备阶段相同的嵌入模型将其转换为向量。
向量检索： 使用向量数据库客户端，在向量库中进行相似度搜索（如余弦相似度），获取Top-K个最相关的文本块及其元数据。
上下文构建： 将检索到的文本块（可能包含Neo4j中的标准实体名和描述）与原始用户问题一起整合成Prompt。
LLM生成： 调用您现有的vLLM服务（通过openai库调用其API）生成最终回答。
```