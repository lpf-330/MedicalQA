# DiseaseKG 医疗知识图谱构建

从原始医疗数据构建 Neo4j 知识图谱，为 MedicalQA 系统提供图谱查询能力。

## 目录结构

```
diseaseKG/
├── data/
│   └── medical.json            # 原始医疗数据（8808 条疾病记录）
├── dict/                       # 实体词典（用于 NER）
│   ├── check.txt               # 检查项目
│   ├── deny.txt                # 否定词
│   ├── department.txt          # 科室
│   ├── disease.txt             # 疾病
│   ├── drug.txt                # 药品
│   ├── food.txt                # 食物
│   ├── producer.txt            # 生产商
│   └── symptom.txt             # 症状
├── prepare_data/               # 数据预处理
│   ├── build_data.py           # 从原始 CSV 构建JSON
│   ├── data_spider.py          # 数据爬取脚本
│   └── max_cut.py              # 最大匹配分词
├── build/
│   ├── deploy_reliable.py      # 生产部署脚本（批量UNWIND）
│   ├── environment.txt         # 部署环境依赖
│   └── README.md               # 部署详细说明
├── build_json.py               # 构建中间 JSON 格式
├── build_medicalgraph.py       # 构建知识图谱（py2neo，开发用）
├── build_medicalgraph_from_json.py  # 从 JSON 构建图谱
└── pic/                        # 图谱截图
```

## 图谱结构

### 节点类型（8 种）

| 节点 | 数量 | 主要属性 |
|------|------|----------|
| Disease | 8,809 | name, desc, prevent, cause, easy_get, cure_lasttime, cured_prob |
| Drug | 3,828 | name |
| Food | 4,870 | name |
| Check | 3,353 | name |
| Department | 54 | name |
| Producer | 17,201 | name |
| Symptom | 5,998 | name |
| Cure | 544 | name |

### 关系类型（11 种）

| 关系 | 起点→终点 | 含义 |
|------|-----------|------|
| recommand_eat | Disease→Food | 推荐食谱 |
| no_eat | Disease→Food | 忌吃 |
| do_eat | Disease→Food | 宜吃 |
| belongs_to | Disease/Department→Department | 属于 |
| common_drug | Disease→Drug | 常用药品 |
| drugs_of | Producer→Drug | 生产药品 |
| recommand_drug | Disease→Drug | 好评药品 |
| need_check | Disease→Check | 诊断检查 |
| has_symptom | Disease→Symptom | 症状 |
| acompany_with | Disease→Disease | 并发症 |
| cure_way | Disease→Cure | 治疗方法 |

## 使用方式

详细部署说明见 [build/README.md](build/README.md)，推荐使用 `build/deploy_reliable.py` 批量部署。

```bash
# 快速部署
cd build/
conda create -n diseasekg_final python=3.9 -y
conda run -n diseasekg_final pip install neo4j
# 修改 deploy_reliable.py 中的数据库连接信息
conda run -n diseasekg_final python deploy_reliable.py
```

## 节点 ID 说明

Neo4j 5.x 中 `id()` 函数已废弃，本项目已迁移至 `elementId()`。重部署后所有节点和关系使用 `elementId()` 标识，Milvus 向量数据库中的 `neo4j_node_id` 字段存储的也是 `elementId()` 返回值。

详细说明和重部署后数据差异原因见 [build/README.md](build/README.md)。

## 数据来源

`data/medical.json` 由 `prepare_data/` 下的脚本从原始医疗数据整理而来，包含 8808 条疾病的完整信息（症状、用药、饮食、检查、科室等）。
