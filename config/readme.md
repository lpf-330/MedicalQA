# 配置文件说明

## 文件列表

| 文件 | 用途 | 修改频率 |
|------|------|----------|
| `application.yaml` | 运行期主配置：资源连接、资源池参数、业务参数覆盖 | 按环境调整 |
| `clinical_standards.yaml` | 临床标准值：医学参考范围、规则引擎评分映射 | 极少修改，修改前请咨询临床专家 |
| `application.example.yaml` | 配置示例模板（入库），真实application.yaml不入库 | 仅新增字段时更新 |

## 加载机制

所有配置由 `ConfigManager`（`src/config/config_manager.py`）统一加载：

1. 加载 `application.yaml` → 运行期配置
2. 加载 `clinical_standards.yaml` → 临床标准值（通过 `ConfigManager.clinical_standards` 访问）
3. 扫描 `src/config/business/` → 业务配置结构
4. 合并运行期覆盖 → 业务配置实例
5. 扫描 `src/config/resources/` → 资源配置结构
6. 合并运行期覆盖 → 资源配置 + 资源池配置
7. 验证 → 导出 GlobalConfig

## 修改注意事项

- **application.yaml**：包含数据库连接地址、模型端口等敏感信息，不纳入版本控制
- **clinical_standards.yaml**：可纳入版本控制，但修改需谨慎，确认符合临床标准
- **业务/资源配置Python文件**：只定义结构（字段名、类型、默认占位值），真实运行期值由yaml覆盖
- **新增配置字段**：需同步更新Python配置类 + application.yaml，否则ConfigManager会报缺失字段警告
