# 老人健康监测系统-健康咨询子项目 API文档

**文档版本**: v1.0  
**最后更新**: 2026-04-19  
**API版本**: v1

---

## 目录

- [概述](#概述)
- [通用说明](#通用说明)
- [接口一：健康咨询接口](#接口一健康咨询接口)
- [接口二：健康报告生成接口](#接口二健康报告生成接口)
- [错误码说明](#错误码说明)
- [性能指标](#性能指标)
- [最佳实践](#最佳实践)

---

## 概述

本文档详细描述了老人健康监测系统-健康咨询子项目的API接口规范。系统提供两个核心接口：

1. **健康咨询接口**：基于医疗知识图谱的智能问答服务
2. **健康报告生成接口**：基于监测数据生成个性化的综合健康评估报告

### 基础信息

- **基础URL**: `http://your-domain:8001`
- **API版本**: v1
- **协议**: HTTP/HTTPS
- **数据格式**: JSON
- **字符编码**: UTF-8
- **响应格式**: SSE (Server-Sent Events) 流式返回

---

## 通用说明

### 请求头

所有API请求应包含以下请求头：

```
Content-Type: application/json
Accept: text/event-stream
```

### 认证

当前版本暂不需要认证，后续版本将支持基于JWT的认证机制。

### SSE响应格式

系统采用SSE (Server-Sent Events) 协议进行流式返回，响应格式如下：

#### message事件（内容块）

```
event: message
data: {"content": "内容片段"}

```

#### end事件（结束标识）

```
event: end
data: {"type": "end", ...元数据}

```

#### error事件（错误信息）

```
event: error
data: {"error_code": 40001, "error_message": "参数校验失败"}

```

### 通用错误响应

当请求失败时，系统将返回SSE error事件，包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| error_code | int | 错误码 |
| error_message | string | 错误描述 |

---

## 接口一：健康咨询接口

### 接口描述

提供基于医疗知识图谱的智能问答服务，用户可以询问健康相关问题，系统通过检索医学知识库并利用大语言模型生成专业、易懂的回答。

### 基本信息

- **请求地址**: `/api/v1/consult`
- **请求方法**: POST
- **响应格式**: SSE流式返回

### 请求参数

#### 请求体结构

```json
{
  "request_id": "string",
  "timestamp": "string",
  "user_id": "string",
  "client_info": {
    "client_type": "string",
    "version": "string"
  },
  "body": {
    "task_id": "string",
    "chat_history": [
      {
        "role": "string",
        "content": "string"
      }
    ],
    "question": "string",
    "session_id": "string",
    "conversation_history": [
      {
        "role": "string",
        "content": "string"
      }
    ],
    "user_profile": {
      "age": "integer",
      "gender": "string",
      "medical_history": ["string"],
      "allergies": ["string"]
    },
    "context": {}
  }
}
```

#### 参数说明

##### 顶层参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| request_id | string | 否 | 请求唯一标识符，如未提供系统会自动生成 |
| timestamp | string | 否 | 请求时间戳，格式：YYYY-MM-DDTHH:mm:ss |
| user_id | string | 否 | 用户ID |
| client_info | object | 否 | 客户端信息 |
| body | object | 是 | 健康咨询请求体 |

##### body参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| task_id | string | 是 | 任务标识符，用于关联对话上下文 |
| chat_history | array | 是 | 对话记录列表，包含历史问答内容 |
| question | string | 是 | 用户提出的健康咨询问题，长度≤1000字符 |
| session_id | string | 否 | 会话ID，用于多轮对话的会话标识 |
| conversation_history | array | 否 | 对话历史，包含之前的对话记录 |
| user_profile | object | 否 | 用户健康档案信息 |
| context | object | 否 | 额外的上下文信息 |

##### chat_history数组元素

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| role | string | 是 | 消息角色，取值：user、assistant |
| content | string | 是 | 消息内容 |

##### user_profile对象

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| age | integer | 否 | 用户年龄 |
| gender | string | 否 | 用户性别，取值：male、female、other |
| medical_history | array | 否 | 既往病史列表 |
| allergies | array | 否 | 过敏史列表 |

### 请求示例

#### 示例1：基础咨询

```bash
curl -X POST "http://localhost:8001/api/v1/consult" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "request_id": "req-123456",
    "user_id": "user-001",
    "body": {
      "task_id": "task-001",
      "chat_history": [
        {"role": "user", "content": "我最近总是头痛"}
      ],
      "question": "我最近总是头痛，应该怎么办？"
    }
  }'
```

#### 示例2：多轮对话

```bash
curl -X POST "http://localhost:8001/api/v1/consult" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "request_id": "req-123457",
    "user_id": "user-001",
    "body": {
      "task_id": "task-001",
      "chat_history": [
        {"role": "user", "content": "我最近总是头痛"},
        {"role": "assistant", "content": "请问您的头痛持续多长时间了？"},
        {"role": "user", "content": "大概三天了"}
      ],
      "question": "需要去医院检查吗？",
      "session_id": "session-001",
      "user_profile": {
        "age": 45,
        "gender": "male",
        "medical_history": ["高血压"],
        "allergies": ["青霉素"]
      }
    }
  }'
```

### 响应说明

#### 响应格式

系统采用SSE流式返回，包含多个message事件和一个end事件。

#### message事件

每个message事件包含一个回答片段：

```
event: message
data: {"content": "您好！关于头痛的问题，"}

event: message
data: {"content": "我为您整理了以下几点建议：\n\n"}

event: message
data: {"content": "## 一、可能的原因\n\n"}

event: message
data: {"content": "1. **紧张性头痛**：由于精神紧张、压力大引起..."}

```

#### end事件

最后一个事件携带结构化元数据：

```json
{
  "type": "end",
  "task_id": "task-001",
  "sources": [
    "neo4j_node_id_1",
    "neo4j_node_id_2"
  ],
  "word_count": 450
}
```

##### end事件字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| type | string | 固定为"end"，表示回答生成完成 |
| task_id | string | 任务标识符 |
| sources | array | 知识来源引用列表（Neo4j节点ID数组） |
| word_count | int | 回答总字数 |

### 业务规则

#### 意图识别规则

系统会识别用户问题的意图，区分"健康咨询"、"闲聊"、"非医疗问题"三类意图：

- **健康咨询**：正常处理，返回专业回答
- **闲聊**：返回错误码40002，提示用户这是健康咨询助手
- **非医疗问题**：返回错误码40002，友好拒绝并引导用户回到健康话题

#### 知识来源优先级

| 优先级 | 来源 | 说明 |
|--------|------|------|
| 1 | Neo4j精确匹配 | 实体名称完全匹配 |
| 2 | 向量语义匹配 | 基于语义相似度检索 |
| 3 | 通用知识模板 | 无精确匹配时使用通用模板 |

#### 回答长度控制

- 目标长度：200-800字
- 过短（<200字）：补充相关知识
- 过长（>800字）：精简内容，突出重点

#### 引用标注规则

每个关键知识点必须标注知识来源：

```
根据医学知识库，高血压患者需要注意...【来源：高血压疾病节点】
```

#### 安全免责声明

回答末尾必须包含免责声明：

```
⚠️ 本回答仅供参考，不能替代医生诊断。如有不适，请及时就医。
```

### 性能指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 响应时间 | 5-15秒 | 从接收到首字节的时间 |
| 首字节时间 | ≤30秒 | 开始返回第一个内容块的时间 |
| 超时保护 | 最大60秒 | 超时自动降级为模板回答 |

---

## 接口二：健康报告生成接口

### 接口描述

基于用户的监测数据和健康档案，生成一份综合性的、个性化的、专业的健康评估报告。

### 基本信息

- **请求地址**: `/api/v1/report`
- **请求方法**: POST
- **响应格式**: SSE流式返回

### 请求参数

#### 请求体结构

```json
{
  "request_id": "string",
  "timestamp": "string",
  "user_id": "string",
  "client_info": {
    "client_type": "string",
    "version": "string"
  },
  "body": {
    "task_id": "string",
    "monitoring_data": {
      "heart_rate": {
        "latest": [{}],
        "daily_stats": [{}],
        "weekly_stats": [{}],
        "monthly_stats": [{}]
      },
      "blood_glucose": {},
      "perfusion_index": {},
      "blood_oxygen": {},
      "sleep": {},
      "blood_pressure": {}
    },
    "user_profile": {
      "user_id": "integer",
      "gender": "string",
      "birth_date": "string",
      "height": "float",
      "weight": "float",
      "past_medical_history": "string",
      "family_history": "string",
      "allergy_history": "string",
      "surgical_history": "string",
      "medical_compliance": "string"
    },
    "session_id": "string"
  }
}
```

#### 参数说明

##### 顶层参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| request_id | string | 否 | 请求唯一标识符 |
| timestamp | string | 否 | 请求时间戳 |
| user_id | string | 否 | 用户ID |
| client_info | object | 否 | 客户端信息 |
| body | object | 是 | 健康报告生成请求体 |

##### body参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| task_id | string | 是 | 任务标识符 |
| monitoring_data | object | 是 | 监测数据，包含各项健康指标 |
| user_profile | object | 是 | 用户档案，包含基本信息、病史等 |
| session_id | string | 否 | 会话ID |

##### monitoring_data对象

监测数据包含6项监测指标，每项指标包含4个时间维度的数据：

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| heart_rate | object | 否 | 心率数据 |
| blood_glucose | object | 否 | 血糖数据 |
| perfusion_index | object | 否 | 灌注指数数据 |
| blood_oxygen | object | 否 | 血氧数据 |
| sleep | object | 否 | 睡眠数据 |
| blood_pressure | object | 否 | 血压数据 |

**注意**：至少需要包含一项监测指标。

##### 时间维度说明

每个监测指标包含以下4个时间维度：

| 时间维度 | 数据内容 | 业务含义 |
|----------|----------|----------|
| latest | 当日最新3-5次数据 | 实时健康快照，识别当前异常状态 |
| daily_stats | 最近30天日统计数据 | 短期波动模式，分析日内变异 |
| weekly_stats | 最近12周周统计数据 | 中期变化趋势，评估干预效果 |
| monthly_stats | 最近6个月月统计数据 | 长期基线水平，评估慢性病风险 |

##### 心率数据示例

```json
{
  "heart_rate": {
    "latest": [
      {
        "value": 72,
        "unit": "bpm",
        "time": "2024-01-01 08:00:00"
      }
    ],
    "daily_stats": [
      {
        "date": "2024-01-01",
        "avg": 70,
        "max": 85,
        "min": 62
      }
    ],
    "weekly_stats": [
      {
        "week": "2024-W1",
        "avg": 71,
        "trend": "stable"
      }
    ],
    "monthly_stats": [
      {
        "month": "2024-01",
        "avg": 72,
        "trend": "stable"
      }
    ]
  }
}
```

##### 血压数据示例

```json
{
  "blood_pressure": {
    "latest": [
      {
        "systolic": 120,
        "diastolic": 80,
        "unit": "mmHg",
        "time": "2024-01-01 08:00:00"
      }
    ],
    "daily_stats": [
      {
        "date": "2024-01-01",
        "avg_systolic": 118,
        "avg_diastolic": 79
      }
    ],
    "weekly_stats": [
      {
        "week": "2024-W1",
        "avg_systolic": 119,
        "avg_diastolic": 78,
        "trend": "stable"
      }
    ],
    "monthly_stats": [
      {
        "month": "2024-01",
        "avg_systolic": 120,
        "avg_diastolic": 80,
        "trend": "rising"
      }
    ]
  }
}
```

##### user_profile对象

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| user_id | integer | 否 | 用户ID |
| gender | string | 否 | 性别，取值：male、female、other |
| birth_date | string | 否 | 出生日期，格式：YYYY-MM-DD |
| height | float | 否 | 身高(cm) |
| weight | float | 否 | 体重(kg) |
| past_medical_history | string | 否 | 既往病史，字符串文本类型 |
| family_history | string | 否 | 家族遗传病史，字符串文本类型 |
| allergy_history | string | 否 | 过敏史，字符串文本类型 |
| surgical_history | string | 否 | 手术史，字符串文本类型 |
| medical_compliance | string | 否 | 用药医嘱，取值：好、一般、差 |

**注意**：所有病史字段均为字符串文本类型，便于自然语言处理。

### 请求示例

#### 示例1：完整数据报告

```bash
curl -X POST "http://localhost:8001/api/v1/report" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "request_id": "req-123456",
    "user_id": "user-001",
    "body": {
      "task_id": "task-001",
      "monitoring_data": {
        "heart_rate": {
          "latest": [
            {"value": 72, "unit": "bpm", "time": "2024-01-01 08:00:00"}
          ],
          "daily_stats": [
            {"date": "2024-01-01", "avg": 70, "max": 85, "min": 62}
          ],
          "weekly_stats": [
            {"week": "2024-W1", "avg": 71, "trend": "stable"}
          ],
          "monthly_stats": [
            {"month": "2024-01", "avg": 72, "trend": "stable"}
          ]
        },
        "blood_glucose": {
          "latest": [
            {"value": 5.5, "unit": "mmol/L", "type": "fasting", "time": "2024-01-01 08:00:00"}
          ],
          "daily_stats": [
            {"date": "2024-01-01", "avg": 5.8, "max": 7.2, "min": 5.0}
          ]
        },
        "blood_pressure": {
          "latest": [
            {"systolic": 120, "diastolic": 80, "unit": "mmHg", "time": "2024-01-01 08:00:00"}
          ],
          "daily_stats": [
            {"date": "2024-01-01", "avg_systolic": 118, "avg_diastolic": 79}
          ]
        }
      },
      "user_profile": {
        "user_id": 1,
        "gender": "male",
        "birth_date": "1955-03-15",
        "height": 170.0,
        "weight": 75.0,
        "past_medical_history": "冠心病史5年、高血脂3年、2020年脑梗死",
        "family_history": "父亲有高血压、母亲有糖尿病",
        "allergy_history": "青霉素过敏、海鲜过敏",
        "surgical_history": "2020年胆囊切除术",
        "medical_compliance": "好"
      }
    }
  }'
```

#### 示例2：部分数据报告

```bash
curl -X POST "http://localhost:8001/api/v1/report" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "request_id": "req-123457",
    "user_id": "user-002",
    "body": {
      "task_id": "task-002",
      "monitoring_data": {
        "blood_pressure": {
          "latest": [
            {"systolic": 145, "diastolic": 95, "unit": "mmHg"}
          ]
        }
      },
      "user_profile": {
        "gender": "female",
        "birth_date": "1960-08-20",
        "height": 160.0,
        "weight": 65.0,
        "past_medical_history": "高血压病史10年"
      }
    }
  }'
```

### 响应说明

#### 响应格式

系统采用SSE流式返回，包含多个message事件和一个end事件。

#### message事件

每个message事件包含一个Markdown格式的报告片段：

```
event: message
data: {"content": "# 健康评估报告\n\n"}

event: message
data: {"content": "## 一、健康综合评分\n"}

event: message
data: {"content": "评分：78分（良好）\n\n"}

event: message
data: {"content": "根据您的监测数据和健康档案，"}

event: message
data: {"content": "您的整体健康状况处于良好水平...\n\n"}

```

#### end事件

最后一个事件携带结构化元数据：

```json
{
  "type": "end",
  "summary": {
    "health_score": 78,
    "risk_level": "轻度风险"
  },
  "report_id": "RPT-20240419-001",
  "generation_time": 180000,
  "word_count": 3500
}
```

##### end事件字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| type | string | 固定为"end"，表示报告生成完成 |
| summary.health_score | float | 健康综合评分（0-100） |
| summary.risk_level | string | 最终风险等级（低/轻/中/高） |
| report_id | string | 报告唯一编号，用于追溯和存档 |
| generation_time | int | 总耗时（毫秒） |
| word_count | int | 报告总字数 |

### 报告结构

生成的健康报告采用Markdown格式，包含以下六大章节：

#### 一、健康综合评分

- 健康评分（0-100分）
- 健康等级（优秀/良好/一般/较差/差）
- 评分依据说明

#### 二、监测数据分析

- 各项指标的当前数值、变化趋势、波动情况
- 与正常医学参考范围的对比分析
- 指标之间的关联性和相互影响
- 异常指标的医学意义说明

#### 三、风险评估

- 风险等级（低/轻/中/高）
- 风险疾病列表
- 各风险因素的分析
- 潜在健康威胁和发展趋势

#### 四、各维度评估

包含6个评估维度：

1. **整体健康状态评估**
2. **慢性病管理效果评估**
3. **生活方式健康度评估**
4. **疾病发展趋势评估**
5. **健康风险预警评估**
6. **健康改善空间评估**

#### 五、健康建议

- 详细、具体、可操作的健康建议
- 每条建议明确说明具体做法
- 便于老年用户理解和执行

#### 六、免责声明

```
以上信息仅供参考，不构成医疗建议。如有健康问题，请及时就医。
```

### 业务规则

#### 健康评分规则

采用100分制，扣分制计算：

| 维度 | 满分 | 扣分规则 |
|------|------|----------|
| 基础生理指标 | 30分 | 血压、血糖、BMI等异常扣分 |
| 生活方式 | 25分 | 吸烟、饮酒、缺乏运动等扣分 |
| 病史情况 | 25分 | 慢性病、家族史、未定期体检等扣分 |
| 其他综合 | 20分 | 肝肾功能、睡眠质量、精神状态等扣分 |

#### 健康等级划分

| 评分区间 | 等级 | 视觉标识 | 建议 |
|----------|------|----------|------|
| 90-100分 | 优秀 | 绿色 | 继续保持 |
| 80-89分 | 良好 | 黄色 | 关注改善 |
| 70-79分 | 一般 | 橙色 | 制定改善计划 |
| 60-69分 | 较差 | 红色 | 尽快就医 |
| <60分 | 差 | 深红色 | 立即就医 |

#### 空值降级策略

当数据不完整时，系统会自适应调整报告内容：

| 数据完整度 | 报告类型 | 说明 |
|------------|----------|------|
| 全部6项指标 | 完整版报告 | 3000-4000字 |
| 3-5项指标 | 标准版报告 | 2500-3500字 |
| 1-2项指标 | 精简版报告 | 2000-2500字 |
| 无监测数据 | 基础版报告 | 1500-2000字 |

#### 特殊规则

1. **高风险疾病优先规则**：风险分>80的疾病醒目展示，优先推荐就医
2. **多疾病关联规则**：多个相关疾病时，自动检查并发症关系，提高预警级别
3. **用药冲突检测规则**：推荐药物时检查过敏史和当前用药，排除冲突药物
4. **年龄适配规则**：>60岁用户，报告内容适老化调整

### 性能指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 响应时间 | 3-5分钟 | 从接收到完整报告的总时长 |
| 首字节时间 | ≤30秒 | 开始返回第一个内容块的时间 |
| 超时保护 | 最大300秒 | 超时自动降级为简化版报告 |

---

## 错误码说明

### 通用错误码

| 错误码 | 含义 | 说明 |
|--------|------|------|
| 40001 | 参数校验失败 | 请求参数格式错误或缺失必填字段 |
| 50001 | 知识检索失败 | Neo4j或Milvus检索失败 |
| 50002 | LLM调用失败 | 大语言模型调用失败 |
| 50005 | 系统内部错误 | 未预期的系统异常 |

### 健康咨询专属错误码

| 错误码 | 含义 | 说明 |
|--------|------|------|
| 40002 | 问题不属于健康咨询范围 | 用户问题被识别为闲聊或非医疗问题 |

### 健康报告专属错误码

| 错误码 | 含义 | 说明 |
|--------|------|------|
| 40003 | 健康数据无效或缺失关键字段 | 监测数据或用户档案缺失关键字段 |
| 50003 | 内容校验失败 | 生成的报告内容不符合规范 |
| 50004 | 报告生成超时 | 报告生成超过最大时间限制 |

### 错误响应示例

```
event: error
data: {"error_code": 40001, "error_message": "task_id不能为空"}

```

---

## 性能指标

### 响应时间

| 接口 | 目标时长 | 首字节时间 | 超时保护 |
|------|----------|------------|----------|
| 健康咨询 | 5-15秒 | ≤30秒 | 最大60秒 |
| 健康报告 | 3-5分钟 | ≤30秒 | 最大300秒 |

### 并发能力

- 支持≥100 QPS并发
- 系统可用性≥99.9%

### 检索性能

- 单集合检索延迟：<150ms
- 混合检索延迟：<200ms
- 召回率：>67%

---

## 最佳实践

### 1. 使用SSE客户端

推荐使用支持SSE的客户端库：

**JavaScript示例**：

```javascript
const eventSource = new EventSource('/api/v1/consult', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    request_id: 'req-123456',
    body: {
      task_id: 'task-001',
      chat_history: [
        {role: 'user', content: '我最近总是头痛'}
      ],
      question: '我最近总是头痛，应该怎么办？'
    }
  })
});

eventSource.addEventListener('message', (event) => {
  const data = JSON.parse(event.data);
  console.log('内容片段:', data.content);
  // 实时显示内容
});

eventSource.addEventListener('end', (event) => {
  const data = JSON.parse(event.data);
  console.log('回答完成:', data);
  eventSource.close();
});

eventSource.addEventListener('error', (event) => {
  const data = JSON.parse(event.data);
  console.error('错误:', data.error_code, data.error_message);
  eventSource.close();
});
```

**Python示例**：

```python
import requests
import json

def consult(question, task_id):
    url = 'http://localhost:8001/api/v1/consult'
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream'
    }
    data = {
        'request_id': 'req-123456',
        'body': {
            'task_id': task_id,
            'chat_history': [
                {'role': 'user', 'content': question}
            ],
            'question': question
        }
    }
    
    with requests.post(url, headers=headers, json=data, stream=True) as response:
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('event:'):
                    event_type = line.split(':')[1].strip()
                elif line.startswith('data:'):
                    data_str = line.split(':', 1)[1].strip()
                    data = json.loads(data_str)
                    
                    if event_type == 'message':
                        print(data['content'], end='', flush=True)
                    elif event_type == 'end':
                        print('\n\n回答完成:', data)
                    elif event_type == 'error':
                        print('\n错误:', data['error_code'], data['error_message'])

# 使用示例
consult('我最近总是头痛，应该怎么办？', 'task-001')
```

### 2. 多轮对话管理

对于多轮对话场景，建议：

1. **使用session_id**：为每个会话分配唯一的session_id
2. **传递完整chat_history**：每次请求都传递完整的对话历史
3. **控制对话长度**：建议对话历史不超过10轮

### 3. 健康报告数据准备

对于健康报告生成，建议：

1. **提供完整数据**：尽可能提供完整的监测数据和用户档案
2. **数据格式规范**：严格按照API文档的数据格式提供数据
3. **时间维度完整**：尽量提供4个时间维度的数据，以便进行全面分析

### 4. 错误处理

建议实现以下错误处理机制：

1. **超时重试**：对于超时错误，可以重试请求
2. **降级处理**：对于系统错误，可以降级为模板回答
3. **日志记录**：记录所有错误信息，便于问题排查

### 5. 性能优化

为提升用户体验，建议：

1. **流式显示**：实时显示生成的内容，不要等待全部完成
2. **加载提示**：在首字节返回前显示加载提示
3. **进度反馈**：对于健康报告，可以显示生成进度

---

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0 | 2026-04-19 | 初始版本，定义健康咨询和健康报告生成接口 |

---

## 联系方式

如有问题或建议，请通过项目Issue反馈。

---

**文档编写**: AI Assistant  
**最后更新**: 2026-04-19  
**文档版本**: 1.0
