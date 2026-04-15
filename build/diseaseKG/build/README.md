# DiseaseKG 医疗知识图谱部署说明

## 项目概述
DiseaseKG 是一个医疗知识图谱项目，包含疾病、药物、食物、检查、科室、生产商、症状、治疗方法等实体及其关系。

## 部署前准备

### 1. 环境要求
- Python 3.9+
- Conda (推荐)
- Neo4j 数据库 (本地或云端)

### 2. 数据准备
确保 `data/medical.json` 文件存在，包含完整的医疗数据。

## 部署步骤

### 1. 创建 Conda 环境
```bash
conda create -n diseasekg_final python=3.9 -y
```

### 2. 安装依赖
```bash
conda run -n diseasekg_final pip install neo4j
```

### 3. 配置数据库连接
编辑 `deploy_reliable.py` 文件，修改以下配置：
```python
self.uri = "neo4j+s://your-database-uri"
self.user = "your-username"
self.password = "your-password"
```

### 4. 执行部署
```bash
cd /home/project/MedicalQA/build/diseaseKG
/root/.conda/envs/diseasekg_final/bin/python deploy_reliable.py
```

## 部署脚本说明

### deploy_reliable.py 主要功能
1. **清空数据库**: 确保数据库干净，避免数据冲突
2. **读取数据**: 从 `data/medical.json` 读取医疗数据
3. **创建节点**: 批量创建 8 类节点
   - Disease (疾病)
   - Drug (药物)
   - Food (食物)
   - Check (检查)
   - Department (科室)
   - Producer (生产商)
   - Symptom (症状)
   - Cure (治疗方法)
4. **创建关系**: 批量创建 11 类关系
   - recommand_eat (推荐食谱)
   - no_eat (忌吃)
   - do_eat (宜吃)
   - belongs_to (属于)
   - common_drug (常用药品)
   - drugs_of (生产药品)
   - recommand_drug (好评药品)
   - need_check (诊断检查)
   - has_symptom (症状)
   - acompany_with (并发症)
   - cure_way (治疗方法)
5. **验证完整性**: 部署完成后自动验证数据完整性

### 批量操作优化
- 批量大小: 500 条记录/批次
- 使用 UNWIND 语句批量插入
- 显著提升部署速度 (相比逐条插入快 10 倍以上)

## 部署后验证

### 1. 检查节点数量
```cypher
MATCH (n) RETURN count(n)
```

### 2. 检查关系数量
```cypher
MATCH ()-[r]->() RETURN count(r)
```

### 3. 抽样检查节点
```cypher
MATCH (d:Disease) RETURN d.name, d.desc LIMIT 5
```

### 4. 抽样检查关系
```cypher
MATCH (d:Disease)-[r:has_symptom]->(s:Symptom) 
RETURN d.name, s.name LIMIT 5
```

## 常见问题

### 1. 连接失败
- 检查数据库 URI 是否正确
- 检查用户名和密码是否正确
- 检查网络连接是否正常

### 2. 认证失败
- 确认用户名和密码正确
- 检查数据库是否允许当前 IP 访问

### 3. 部署速度慢
- 检查网络延迟
- 考虑增加批量大小 (batch_size)
- 使用更近的数据库服务器

## 注意事项

1. **数据安全**: 部署脚本包含数据库密码，请妥善保管
2. **数据备份**: 部署前建议备份现有数据
3. **环境清理**: 部署完成后及时删除临时环境
4. **性能优化**: 批量大小可根据网络情况调整

## 部署日志示例
```
============================================================
DiseaseKG 医疗知识图谱可靠部署
============================================================
[11:07:04] 清空数据库以保证完整性...
[11:07:07] 数据库已清空
[11:07:07] 开始创建节点
[11:07:07] 数据读取完成，共 8808 条疾病记录
[11:07:39] Disease 节点创建完成: 8808/8808
[11:07:44] Drug 节点创建完成: 3828/3828
...
[11:44:48] 验证部署完整性...
[11:44:49] 节点总数: 44657
[11:44:49] 关系总数: 312226
[11:44:49] 部署完成！
[11:44:49] 总耗时: 2265.10 秒
============================================================
```

## 联系方式
如有问题，请联系项目负责人。
