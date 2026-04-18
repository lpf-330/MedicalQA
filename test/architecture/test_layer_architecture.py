"""
健康报告生成业务 - 架构符合性测试
测试目标：验证项目架构是否符合《项目架构设计v2.1》和《项目架构原则与使用规范v1》
"""
import os
import sys
import inspect
from pathlib import Path
from typing import List, Dict, Any, Tuple

PROJECT_ROOT = Path("/home/project/MedicalQA")
SRC_ROOT = PROJECT_ROOT / "src"

class ArchitectureTestResult:
    def __init__(self):
        self.passed: List[str] = []
        self.failed: List[Tuple[str, str]] = []
        self.warnings: List[Tuple[str, str]] = []
    
    def add_pass(self, test_name: str):
        self.passed.append(test_name)
    
    def add_fail(self, test_name: str, reason: str):
        self.failed.append((test_name, reason))
    
    def add_warning(self, test_name: str, reason: str):
        self.warnings.append((test_name, reason))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tests": len(self.passed) + len(self.failed),
            "passed_count": len(self.passed),
            "failed_count": len(self.failed),
            "warning_count": len(self.warnings),
            "passed": self.passed,
            "failed": self.failed,
            "warnings": self.warnings
        }

def test_layer_architecture(result: ArchitectureTestResult) -> None:
    """测试分层架构"""
    print("\n=== 测试分层架构 ===")
    
    required_layers = {
        "controller": "接入层(Controller)",
        "service": "服务层(Service)",
        "orchestration": "编排层(Orchestration)",
        "tools": "工具层(Tools)",
        "resource_manager": "资源管理层(ResourceManager)",
        "config": "配置层(Config)",
        "adapters": "适配层(Adapters)",
        "schemas": "数据模型层(Schemas)",
        "utils": "工具类层(Utils)"
    }
    
    for layer_dir, layer_name in required_layers.items():
        layer_path = SRC_ROOT / layer_dir
        if layer_path.exists() and layer_path.is_dir():
            py_files = list(layer_path.glob("**/*.py"))
            if len(py_files) > 0:
                result.add_pass(f"分层架构验证 - {layer_name}目录存在且包含Python文件")
                print(f"  ✓ {layer_name}: 存在，包含 {len(py_files)} 个Python文件")
            else:
                result.add_warning(f"分层架构验证 - {layer_name}", "目录存在但没有Python文件")
                print(f"  ⚠ {layer_name}: 目录存在但没有Python文件")
        else:
            result.add_fail(f"分层架构验证 - {layer_name}", "目录不存在")
            print(f"  ✗ {layer_name}: 目录不存在")

def test_model_business_service_layer(result: ArchitectureTestResult) -> None:
    """测试模型业务层"""
    print("\n=== 测试模型业务层 ===")
    
    mbs_path = SRC_ROOT / "orchestration" / "model_business_service"
    
    if not mbs_path.exists():
        result.add_fail("模型业务层验证", "model_business_service目录不存在")
        print("  ✗ model_business_service目录不存在")
        return
    
    impl_path = mbs_path / "Impl"
    if impl_path.exists():
        result.add_pass("模型业务层验证 - Impl目录存在")
        print("  ✓ Impl目录存在")
        
        required_services = ["report_model_service.py", "consult_model_service.py"]
        for service in required_services:
            service_path = impl_path / service
            if service_path.exists():
                result.add_pass(f"模型业务层验证 - {service}存在")
                print(f"  ✓ {service} 存在")
            else:
                result.add_fail(f"模型业务层验证 - {service}", "文件不存在")
                print(f"  ✗ {service} 不存在")
    else:
        result.add_fail("模型业务层验证", "Impl目录不存在")
        print("  ✗ Impl目录不存在")

def test_chain_layer(result: ArchitectureTestResult) -> None:
    """测试Chain层"""
    print("\n=== 测试Chain层 ===")
    
    chain_path = SRC_ROOT / "orchestration" / "chain"
    
    if not chain_path.exists():
        result.add_fail("Chain层验证", "chain目录不存在")
        print("  ✗ chain目录不存在")
        return
    
    required_chains = {
        "data_prepare_chain": "数据准备Chain",
        "multi_analysis_chain": "多维度分析Chain",
        "dimension_evaluation_chain": "维度评估Chain",
        "report_knowledge_retrieval_chain": "报告知识检索Chain",
        "integration_chain": "整合Chain",
        "report_generation_chain": "报告生成Chain"
    }
    
    for chain_dir, chain_name in required_chains.items():
        chain_dir_path = chain_path / chain_dir
        if chain_dir_path.exists():
            py_files = list(chain_dir_path.glob("*.py"))
            if any(f.name == f"{chain_dir}.py" for f in py_files):
                result.add_pass(f"Chain层验证 - {chain_name}")
                print(f"  ✓ {chain_name}: 存在")
            else:
                result.add_fail(f"Chain层验证 - {chain_name}", "主文件不存在")
                print(f"  ✗ {chain_name}: 主文件不存在")
        else:
            result.add_fail(f"Chain层验证 - {chain_name}", "目录不存在")
            print(f"  ✗ {chain_name}: 目录不存在")

def test_tool_layer(result: ArchitectureTestResult) -> None:
    """测试工具层"""
    print("\n=== 测试工具层 ===")
    
    tools_path = SRC_ROOT / "tools"
    
    if not tools_path.exists():
        result.add_fail("工具层验证", "tools目录不存在")
        print("  ✗ tools目录不存在")
        return
    
    required_tools = {
        "neo4j_medical_tool": "Neo4j医疗工具",
        "vector_retrieval_tool": "向量检索工具",
        "intent_classification_tool": "意图分类工具"
    }
    
    for tool_dir, tool_name in required_tools.items():
        tool_dir_path = tools_path / tool_dir
        if tool_dir_path.exists():
            py_files = list(tool_dir_path.glob("*.py"))
            if any(f.name == f"{tool_dir}.py" for f in py_files):
                result.add_pass(f"工具层验证 - {tool_name}")
                print(f"  ✓ {tool_name}: 存在")
            else:
                result.add_fail(f"工具层验证 - {tool_name}", "主文件不存在")
                print(f"  ✗ {tool_name}: 主文件不存在")
        else:
            result.add_fail(f"工具层验证 - {tool_name}", "目录不存在")
            print(f"  ✗ {tool_name}: 目录不存在")

def test_config_layer(result: ArchitectureTestResult) -> None:
    """测试配置层"""
    print("\n=== 测试配置层 ===")
    
    config_path = SRC_ROOT / "config"
    
    if not config_path.exists():
        result.add_fail("配置层验证", "config目录不存在")
        print("  ✗ config目录不存在")
        return
    
    required_configs = {
        "global_config.py": "全局配置",
        "pool_config.py": "资源池配置",
        "base_config.py": "基础配置"
    }
    
    for config_file, config_name in required_configs.items():
        config_file_path = config_path / config_file
        if config_file_path.exists():
            result.add_pass(f"配置层验证 - {config_name}")
            print(f"  ✓ {config_name}: 存在")
        else:
            result.add_fail(f"配置层验证 - {config_name}", "文件不存在")
            print(f"  ✗ {config_name}: 不存在")
    
    resources_path = config_path / "resources"
    if resources_path.exists():
        result.add_pass("配置层验证 - resources目录存在")
        print("  ✓ resources目录存在")
        
        required_resources = ["neo4j_config.py", "vllm_config.py", "milvus_config.py"]
        for res in required_resources:
            res_path = resources_path / res
            if res_path.exists():
                result.add_pass(f"配置层验证 - {res}存在")
                print(f"  ✓ {res} 存在")
            else:
                result.add_fail(f"配置层验证 - {res}", "文件不存在")
                print(f"  ✗ {res} 不存在")
    else:
        result.add_fail("配置层验证", "resources目录不存在")
        print("  ✗ resources目录不存在")
    
    business_path = config_path / "business"
    if business_path.exists():
        result.add_pass("配置层验证 - business目录存在")
        print("  ✓ business目录存在")
        
        required_business = ["report_service_config.py", "consult_service_config.py"]
        for bus in required_business:
            bus_path = business_path / bus
            if bus_path.exists():
                result.add_pass(f"配置层验证 - {bus}存在")
                print(f"  ✓ {bus} 存在")
            else:
                result.add_fail(f"配置层验证 - {bus}", "文件不存在")
                print(f"  ✗ {bus} 不存在")
    else:
        result.add_fail("配置层验证", "business目录不存在")
        print("  ✗ business目录不存在")

def test_resource_manager_layer(result: ArchitectureTestResult) -> None:
    """测试资源管理层"""
    print("\n=== 测试资源管理层 ===")
    
    rm_path = SRC_ROOT / "resource_manager"
    
    if not rm_path.exists():
        result.add_fail("资源管理层验证", "resource_manager目录不存在")
        print("  ✗ resource_manager目录不存在")
        return
    
    required_files = {
        "pool_manager.py": "资源池管理器",
        "global_resource_manager.py": "全局资源管理器",
        "resource_pool.py": "资源池",
        "resource.py": "资源基类",
        "resource_factory.py": "资源工厂"
    }
    
    for file_name, file_desc in required_files.items():
        file_path = rm_path / file_name
        if file_path.exists():
            result.add_pass(f"资源管理层验证 - {file_desc}")
            print(f"  ✓ {file_desc}: 存在")
        else:
            result.add_fail(f"资源管理层验证 - {file_desc}", "文件不存在")
            print(f"  ✗ {file_desc}: 不存在")

def test_controller_layer(result: ArchitectureTestResult) -> None:
    """测试接入层"""
    print("\n=== 测试接入层 ===")
    
    controller_path = SRC_ROOT / "controller"
    
    if not controller_path.exists():
        result.add_fail("接入层验证", "controller目录不存在")
        print("  ✗ controller目录不存在")
        return
    
    required_controllers = ["report_controller.py", "consult_controller.py"]
    for ctrl in required_controllers:
        ctrl_path = controller_path / ctrl
        if ctrl_path.exists():
            result.add_pass(f"接入层验证 - {ctrl}")
            print(f"  ✓ {ctrl}: 存在")
        else:
            result.add_fail(f"接入层验证 - {ctrl}", "文件不存在")
            print(f"  ✗ {ctrl}: 不存在")

def test_service_layer(result: ArchitectureTestResult) -> None:
    """测试服务层"""
    print("\n=== 测试服务层 ===")
    
    service_path = SRC_ROOT / "service"
    
    if not service_path.exists():
        result.add_fail("服务层验证", "service目录不存在")
        print("  ✗ service目录不存在")
        return
    
    required_services = ["report_service.py", "consult_service.py"]
    for svc in required_services:
        svc_path = service_path / svc
        if svc_path.exists():
            result.add_pass(f"服务层验证 - {svc}")
            print(f"  ✓ {svc}: 存在")
        else:
            result.add_fail(f"服务层验证 - {svc}", "文件不存在")
            print(f"  ✗ {svc}: 不存在")

def test_schemas_layer(result: ArchitectureTestResult) -> None:
    """测试数据模型层"""
    print("\n=== 测试数据模型层 ===")
    
    schemas_path = SRC_ROOT / "schemas"
    
    if not schemas_path.exists():
        result.add_fail("数据模型层验证", "schemas目录不存在")
        print("  ✗ schemas目录不存在")
        return
    
    required_schemas = {
        "report_request.py": "报告请求模型",
        "report_response.py": "报告响应模型",
        "consult_request.py": "咨询请求模型",
        "consult_response.py": "咨询响应模型"
    }
    
    for schema_file, schema_name in required_schemas.items():
        schema_path = schemas_path / schema_file
        if schema_path.exists():
            result.add_pass(f"数据模型层验证 - {schema_name}")
            print(f"  ✓ {schema_name}: 存在")
        else:
            result.add_fail(f"数据模型层验证 - {schema_name}", "文件不存在")
            print(f"  ✗ {schema_name}: 不存在")

def run_all_tests() -> Dict[str, Any]:
    """运行所有架构测试"""
    print("=" * 60)
    print("健康报告生成业务 - 架构符合性测试")
    print("=" * 60)
    
    result = ArchitectureTestResult()
    
    test_layer_architecture(result)
    test_model_business_service_layer(result)
    test_chain_layer(result)
    test_tool_layer(result)
    test_config_layer(result)
    test_resource_manager_layer(result)
    test_controller_layer(result)
    test_service_layer(result)
    test_schemas_layer(result)
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"总测试数: {len(result.passed) + len(result.failed)}")
    print(f"通过: {len(result.passed)}")
    print(f"失败: {len(result.failed)}")
    print(f"警告: {len(result.warnings)}")
    
    if result.failed:
        print("\n失败的测试:")
        for name, reason in result.failed:
            print(f"  ✗ {name}: {reason}")
    
    if result.warnings:
        print("\n警告:")
        for name, reason in result.warnings:
            print(f"  ⚠ {name}: {reason}")
    
    return result.to_dict()

if __name__ == "__main__":
    results = run_all_tests()
    
    import json
    report_path = PROJECT_ROOT / "test" / "report" / "health_report" / "architecture_test_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n测试报告已保存到: {report_path}")
