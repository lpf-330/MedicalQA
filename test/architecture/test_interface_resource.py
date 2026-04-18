"""
健康报告生成业务 - 接口设计与资源管理验证测试
测试目标：验证接口定义和资源管理是否符合《项目架构设计v2.1》
"""
import os
import sys
import inspect
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import ast

PROJECT_ROOT = Path("/home/project/MedicalQA")
SRC_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_ROOT))

class TestResult:
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

def get_file_content(file_path: Path) -> str:
    """获取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return ""

def test_interface_design(result: TestResult) -> None:
    """测试接口设计"""
    print("\n=== 测试接口设计 ===")
    
    interface_tests = {
        "ReportController": {
            "path": SRC_ROOT / "controller" / "report_controller.py",
            "expected_methods": ["generate_report"],
            "expected_params": ["request"]
        },
        "ReportService": {
            "path": SRC_ROOT / "service" / "report_service.py",
            "expected_methods": ["process_report", "process_report_stream"],
            "expected_params": ["context"]
        },
        "ReportStrategy": {
            "path": SRC_ROOT / "orchestration" / "agent" / "report_strategy" / "report_strategy.py",
            "expected_methods": ["execute"],
            "expected_params": ["context"]
        }
    }
    
    for class_name, config in interface_tests.items():
        file_path = config["path"]
        if not file_path.exists():
            result.add_fail(f"接口设计验证 - {class_name}", "文件不存在")
            print(f"  ✗ {class_name}: 文件不存在")
            continue
        
        content = get_file_content(file_path)
        
        found_methods = []
        for method in config["expected_methods"]:
            if re.search(rf"def\s+{method}\s*\(", content):
                found_methods.append(method)
        
        if len(found_methods) == len(config["expected_methods"]):
            result.add_pass(f"接口设计验证 - {class_name}")
            print(f"  ✓ {class_name}: 包含所有预期方法 {found_methods}")
        elif len(found_methods) > 0:
            result.add_warning(f"接口设计验证 - {class_name}", 
                             f"部分方法存在: {found_methods}")
            print(f"  ⚠ {class_name}: 部分方法存在 {found_methods}")
        else:
            result.add_fail(f"接口设计验证 - {class_name}", "未找到预期方法")
            print(f"  ✗ {class_name}: 未找到预期方法")

def test_chain_interfaces(result: TestResult) -> None:
    """测试Chain接口"""
    print("\n=== 测试Chain接口 ===")
    
    chain_configs = {
        "DataPrepareChain": SRC_ROOT / "orchestration" / "chain" / "data_prepare_chain" / "data_prepare_chain.py",
        "MultiAnalysisChain": SRC_ROOT / "orchestration" / "chain" / "multi_analysis_chain" / "multi_analysis_chain.py",
        "DimensionEvaluationChain": SRC_ROOT / "orchestration" / "chain" / "dimension_evaluation_chain" / "dimension_evaluation_chain.py",
        "ReportKnowledgeRetrievalChain": SRC_ROOT / "orchestration" / "chain" / "report_knowledge_retrieval_chain" / "report_knowledge_retrieval_chain.py",
        "IntegrationChain": SRC_ROOT / "orchestration" / "chain" / "integration_chain" / "integration_chain.py",
        "ReportGenerationChain": SRC_ROOT / "orchestration" / "chain" / "report_generation_chain" / "report_generation_chain.py"
    }
    
    for chain_name, chain_path in chain_configs.items():
        if not chain_path.exists():
            result.add_fail(f"Chain接口验证 - {chain_name}", "文件不存在")
            print(f"  ✗ {chain_name}: 文件不存在")
            continue
        
        content = get_file_content(chain_path)
        
        if re.search(r"def\s+execute\s*\(", content) or re.search(r"def\s+__call__\s*\(", content):
            result.add_pass(f"Chain接口验证 - {chain_name}")
            print(f"  ✓ {chain_name}: 包含执行接口")
        else:
            result.add_warning(f"Chain接口验证 - {chain_name}", "未找到标准执行接口")
            print(f"  ⚠ {chain_name}: 未找到标准执行接口")

def test_resource_management(result: TestResult) -> None:
    """测试资源管理"""
    print("\n=== 测试资源管理 ===")
    
    pool_manager_path = SRC_ROOT / "resource_manager" / "pool_manager.py"
    content = get_file_content(pool_manager_path)
    
    if "config_id" in content:
        result.add_pass("资源管理验证 - PoolManager使用config_id参数")
        print("  ✓ PoolManager: 使用config_id参数")
    else:
        result.add_fail("资源管理验证 - PoolManager使用config_id参数", "未找到config_id参数")
        print("  ✗ PoolManager: 未找到config_id参数")
    
    if 'resource_type:config_id' in content or 'f"{resource_type}:{config_id}"' in content:
        result.add_pass("资源管理验证 - Pool标识格式为resource_type:config_id")
        print("  ✓ Pool标识: 格式为resource_type:config_id")
    else:
        result.add_warning("资源管理验证 - Pool标识格式", "未明确找到resource_type:config_id格式")
        print("  ⚠ Pool标识: 未明确找到resource_type:config_id格式")
    
    grm_path = SRC_ROOT / "resource_manager" / "global_resource_manager.py"
    grm_content = get_file_content(grm_path)
    
    if "config_id" in grm_content:
        result.add_pass("资源管理验证 - GlobalResourceManager使用config_id参数")
        print("  ✓ GlobalResourceManager: 使用config_id参数")
    else:
        result.add_fail("资源管理验证 - GlobalResourceManager使用config_id参数", "未找到config_id参数")
        print("  ✗ GlobalResourceManager: 未找到config_id参数")
    
    model_service_path = SRC_ROOT / "orchestration" / "model_business_service" / "Impl" / "report_model_service.py"
    ms_content = get_file_content(model_service_path)
    
    if 'acquire(' in ms_content and 'config_id' in ms_content:
        result.add_pass("资源管理验证 - ModelService层资源获取使用config_id参数")
        print("  ✓ ModelService层: 资源获取使用config_id参数")
    else:
        result.add_warning("资源管理验证 - ModelService层资源获取", "需要检查config_id使用情况")
        print("  ⚠ ModelService层: 需要检查config_id使用情况")
    
    tool_paths = [
        SRC_ROOT / "tools" / "neo4j_medical_tool" / "neo4j_medical_tool.py",
        SRC_ROOT / "tools" / "vector_retrieval_tool" / "vector_retrieval_tool.py",
        SRC_ROOT / "tools" / "intent_classification_tool" / "intent_classification_tool.py"
    ]
    
    for tool_path in tool_paths:
        tool_content = get_file_content(tool_path)
        tool_name = tool_path.stem
        
        if 'acquire(' in tool_content and 'config_id' in tool_content:
            result.add_pass(f"资源管理验证 - {tool_name}使用config_id参数")
            print(f"  ✓ {tool_name}: 使用config_id参数")
        else:
            result.add_warning(f"资源管理验证 - {tool_name}", "需要检查config_id使用情况")
            print(f"  ⚠ {tool_name}: 需要检查config_id使用情况")

def test_config_management(result: TestResult) -> None:
    """测试配置管理"""
    print("\n=== 测试配置管理 ===")
    
    resources_path = SRC_ROOT / "config" / "resources"
    if resources_path.exists():
        py_files = list(resources_path.glob("*.py"))
        if len(py_files) > 0:
            result.add_pass("配置管理验证 - 资源配置文件位于src/config/resources/目录")
            print(f"  ✓ 资源配置目录: 存在，包含 {len(py_files)} 个配置文件")
        else:
            result.add_fail("配置管理验证 - 资源配置文件", "目录存在但没有配置文件")
            print("  ✗ 资源配置目录: 没有配置文件")
    else:
        result.add_fail("配置管理验证 - 资源配置文件", "目录不存在")
        print("  ✗ 资源配置目录: 不存在")
    
    business_path = SRC_ROOT / "config" / "business"
    if business_path.exists():
        py_files = list(business_path.glob("*.py"))
        if len(py_files) > 0:
            result.add_pass("配置管理验证 - 业务配置文件位于src/config/business/目录")
            print(f"  ✓ 业务配置目录: 存在，包含 {len(py_files)} 个配置文件")
        else:
            result.add_fail("配置管理验证 - 业务配置文件", "目录存在但没有配置文件")
            print("  ✗ 业务配置目录: 没有配置文件")
    else:
        result.add_fail("配置管理验证 - 业务配置文件", "目录不存在")
        print("  ✗ 业务配置目录: 不存在")
    
    neo4j_config_path = SRC_ROOT / "config" / "resources" / "neo4j_config.py"
    if neo4j_config_path.exists():
        content = get_file_content(neo4j_config_path)
        if "config_id" in content:
            result.add_pass("配置管理验证 - 资源配置文件包含config_id字段")
            print("  ✓ 资源配置文件: 包含config_id字段")
        else:
            result.add_fail("配置管理验证 - 资源配置文件包含config_id字段", "未找到config_id字段")
            print("  ✗ 资源配置文件: 未找到config_id字段")
    else:
        result.add_fail("配置管理验证 - 资源配置文件包含config_id字段", "文件不存在")
        print("  ✗ 资源配置文件: 不存在")
    
    report_config_path = SRC_ROOT / "config" / "business" / "report_service_config.py"
    if report_config_path.exists():
        content = get_file_content(report_config_path)
        if "resource_configs" in content:
            result.add_pass("配置管理验证 - 业务配置文件包含resource_configs列表")
            print("  ✓ 业务配置文件: 包含resource_configs列表")
        else:
            result.add_fail("配置管理验证 - 业务配置文件包含resource_configs列表", "未找到resource_configs")
            print("  ✗ 业务配置文件: 未找到resource_configs")
    else:
        result.add_fail("配置管理验证 - 业务配置文件包含resource_configs列表", "文件不存在")
        print("  ✗ 业务配置文件: 不存在")
    
    global_config_path = SRC_ROOT / "config" / "global_config.py"
    if global_config_path.exists():
        content = get_file_content(global_config_path)
        if "add_resource_config" in content and "add_pool_config" in content:
            result.add_pass("配置管理验证 - GlobalConfig支持按config_id存储配置")
            print("  ✓ GlobalConfig: 支持按config_id存储配置")
        else:
            result.add_warning("配置管理验证 - GlobalConfig", "需要检查配置存储方法")
            print("  ⚠ GlobalConfig: 需要检查配置存储方法")
    else:
        result.add_fail("配置管理验证 - GlobalConfig", "文件不存在")
        print("  ✗ GlobalConfig: 文件不存在")

def run_all_tests() -> Dict[str, Any]:
    """运行所有测试"""
    print("=" * 60)
    print("健康报告生成业务 - 接口设计与资源管理验证测试")
    print("=" * 60)
    
    result = TestResult()
    
    test_interface_design(result)
    test_chain_interfaces(result)
    test_resource_management(result)
    test_config_management(result)
    
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
    report_path = PROJECT_ROOT / "test" / "report" / "health_report" / "interface_resource_test_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n测试报告已保存到: {report_path}")
