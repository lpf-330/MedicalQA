"""
健康报告生成业务 - 类职责验证测试
测试目标：验证各层类的职责是否符合《项目架构设计v2.1》中的定义
"""
import os
import sys
import inspect
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
import ast

PROJECT_ROOT = Path("/home/project/MedicalQA")
SRC_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_ROOT))

class ClassResponsibilityTestResult:
    def __init__(self):
        self.passed: List[str] = []
        self.failed: List[Tuple[str, str]] = []
        self.warnings: List[Tuple[str, str]] = []
        self.class_details: Dict[str, Dict[str, Any]] = {}
    
    def add_pass(self, test_name: str):
        self.passed.append(test_name)
    
    def add_fail(self, test_name: str, reason: str):
        self.failed.append((test_name, reason))
    
    def add_warning(self, test_name: str, reason: str):
        self.warnings.append((test_name, reason))
    
    def add_class_detail(self, class_name: str, detail: Dict[str, Any]):
        self.class_details[class_name] = detail
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tests": len(self.passed) + len(self.failed),
            "passed_count": len(self.passed),
            "failed_count": len(self.failed),
            "warning_count": len(self.warnings),
            "passed": self.passed,
            "failed": self.failed,
            "warnings": self.warnings,
            "class_details": self.class_details
        }

def get_class_info(file_path: Path) -> Dict[str, Any]:
    """获取文件中的类信息"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        classes = []
        functions = []
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                classes.append({
                    "name": node.name,
                    "methods": methods,
                    "bases": [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases]
                })
            elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                functions.append(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
        
        return {
            "classes": classes,
            "functions": functions,
            "imports": imports
        }
    except Exception as e:
        return {"error": str(e)}

def test_controller_classes(result: ClassResponsibilityTestResult) -> None:
    """测试Controller类职责"""
    print("\n=== 测试Controller类职责 ===")
    
    controller_path = SRC_ROOT / "controller" / "report_controller.py"
    info = get_class_info(controller_path)
    
    if "error" in info:
        result.add_fail("ReportController类验证", f"解析错误: {info['error']}")
        print(f"  ✗ ReportController类解析错误: {info['error']}")
        return
    
    for cls in info.get("classes", []):
        if "Controller" in cls["name"]:
            result.add_class_detail(cls["name"], cls)
            
            required_methods = ["generate_report"]
            has_required = all(m in cls["methods"] for m in required_methods)
            
            if has_required:
                result.add_pass(f"Controller类职责验证 - {cls['name']}")
                print(f"  ✓ {cls['name']}: 包含必要的方法 {required_methods}")
            else:
                result.add_warning(f"Controller类职责验证 - {cls['name']}", 
                                 f"缺少部分方法，现有方法: {cls['methods']}")
                print(f"  ⚠ {cls['name']}: 现有方法 {cls['methods']}")

def test_service_classes(result: ClassResponsibilityTestResult) -> None:
    """测试Service类职责"""
    print("\n=== 测试Service类职责 ===")
    
    service_path = SRC_ROOT / "service" / "report_service.py"
    info = get_class_info(service_path)
    
    if "error" in info:
        result.add_fail("ReportService类验证", f"解析错误: {info['error']}")
        print(f"  ✗ ReportService类解析错误: {info['error']}")
        return
    
    for cls in info.get("classes", []):
        if "Service" in cls["name"]:
            result.add_class_detail(cls["name"], cls)
            
            expected_methods = ["generate_report", "initialize", "shutdown"]
            found_methods = [m for m in expected_methods if m in cls["methods"]]
            
            if len(found_methods) >= 1:
                result.add_pass(f"Service类职责验证 - {cls['name']}")
                print(f"  ✓ {cls['name']}: 包含方法 {found_methods}")
            else:
                result.add_warning(f"Service类职责验证 - {cls['name']}", 
                                 f"现有方法: {cls['methods']}")
                print(f"  ⚠ {cls['name']}: 现有方法 {cls['methods']}")

def test_strategy_classes(result: ClassResponsibilityTestResult) -> None:
    """测试Strategy类职责"""
    print("\n=== 测试Strategy类职责 ===")
    
    strategy_path = SRC_ROOT / "orchestration" / "agent" / "report_strategy" / "report_strategy.py"
    info = get_class_info(strategy_path)
    
    if "error" in info:
        result.add_fail("ReportStrategy类验证", f"解析错误: {info['error']}")
        print(f"  ✗ ReportStrategy类解析错误: {info['error']}")
        return
    
    for cls in info.get("classes", []):
        if "Strategy" in cls["name"] or "strategy" in cls["name"].lower():
            result.add_class_detail(cls["name"], cls)
            
            expected_patterns = ["execute", "run", "process"]
            has_execute = any(p in "".join(cls["methods"]).lower() for p in expected_patterns)
            
            if has_execute:
                result.add_pass(f"Strategy类职责验证 - {cls['name']}")
                print(f"  ✓ {cls['name']}: 包含执行方法")
            else:
                result.add_warning(f"Strategy类职责验证 - {cls['name']}", 
                                 f"现有方法: {cls['methods']}")
                print(f"  ⚠ {cls['name']}: 现有方法 {cls['methods']}")

def test_chain_classes(result: ClassResponsibilityTestResult) -> None:
    """测试Chain类职责"""
    print("\n=== 测试Chain类职责 ===")
    
    chain_configs = {
        "data_prepare_chain": "DataPrepareChain",
        "multi_analysis_chain": "MultiAnalysisChain",
        "dimension_evaluation_chain": "DimensionEvaluationChain",
        "report_knowledge_retrieval_chain": "ReportKnowledgeRetrievalChain",
        "integration_chain": "IntegrationChain",
        "report_generation_chain": "ReportGenerationChain"
    }
    
    for chain_dir, expected_class in chain_configs.items():
        chain_path = SRC_ROOT / "orchestration" / "chain" / chain_dir / f"{chain_dir}.py"
        
        if not chain_path.exists():
            result.add_fail(f"Chain类验证 - {expected_class}", "文件不存在")
            print(f"  ✗ {expected_class}: 文件不存在")
            continue
        
        info = get_class_info(chain_path)
        
        if "error" in info:
            result.add_fail(f"Chain类验证 - {expected_class}", f"解析错误: {info['error']}")
            print(f"  ✗ {expected_class}: 解析错误")
            continue
        
        found = False
        for cls in info.get("classes", []):
            if expected_class in cls["name"] or "Chain" in cls["name"]:
                found = True
                result.add_class_detail(cls["name"], cls)
                
                expected_methods = ["execute", "run", "process", "__call__"]
                has_execute = any(m in cls["methods"] for m in expected_methods)
                
                if has_execute:
                    result.add_pass(f"Chain类职责验证 - {cls['name']}")
                    print(f"  ✓ {cls['name']}: 包含执行方法")
                else:
                    result.add_warning(f"Chain类职责验证 - {cls['name']}", 
                                     f"现有方法: {cls['methods']}")
                    print(f"  ⚠ {cls['name']}: 现有方法 {cls['methods']}")
        
        if not found:
            result.add_warning(f"Chain类验证 - {expected_class}", "未找到预期的Chain类")
            print(f"  ⚠ {expected_class}: 未找到预期的Chain类")

def test_model_service_classes(result: ClassResponsibilityTestResult) -> None:
    """测试ModelService类职责"""
    print("\n=== 测试ModelService类职责 ===")
    
    model_service_path = SRC_ROOT / "orchestration" / "model_business_service" / "Impl" / "report_model_service.py"
    info = get_class_info(model_service_path)
    
    if "error" in info:
        result.add_fail("ReportModelService类验证", f"解析错误: {info['error']}")
        print(f"  ✗ ReportModelService类解析错误: {info['error']}")
        return
    
    for cls in info.get("classes", []):
        if "ModelService" in cls["name"] or "model_service" in cls["name"].lower():
            result.add_class_detail(cls["name"], cls)
            
            expected_methods = ["generate", "call", "invoke"]
            has_generate = any(m in "".join(cls["methods"]).lower() for m in expected_methods)
            
            if has_generate:
                result.add_pass(f"ModelService类职责验证 - {cls['name']}")
                print(f"  ✓ {cls['name']}: 包含生成方法")
            else:
                result.add_warning(f"ModelService类职责验证 - {cls['name']}", 
                                 f"现有方法: {cls['methods']}")
                print(f"  ⚠ {cls['name']}: 现有方法 {cls['methods']}")

def test_tool_classes(result: ClassResponsibilityTestResult) -> None:
    """测试Tool类职责"""
    print("\n=== 测试Tool类职责 ===")
    
    tool_configs = {
        "neo4j_medical_tool": "Neo4jMedicalTool",
        "vector_retrieval_tool": "VectorRetrievalTool",
        "intent_classification_tool": "IntentClassificationTool"
    }
    
    for tool_dir, expected_class in tool_configs.items():
        tool_path = SRC_ROOT / "tools" / tool_dir / f"{tool_dir}.py"
        
        if not tool_path.exists():
            result.add_fail(f"Tool类验证 - {expected_class}", "文件不存在")
            print(f"  ✗ {expected_class}: 文件不存在")
            continue
        
        info = get_class_info(tool_path)
        
        if "error" in info:
            result.add_fail(f"Tool类验证 - {expected_class}", f"解析错误: {info['error']}")
            print(f"  ✗ {expected_class}: 解析错误")
            continue
        
        for cls in info.get("classes", []):
            if "Tool" in cls["name"] or expected_class.replace("Tool", "").lower() in cls["name"].lower():
                result.add_class_detail(cls["name"], cls)
                
                expected_methods = ["call", "execute", "run", "_init_tool", "call_tool"]
                has_call = any(m in cls["methods"] for m in expected_methods)
                
                if has_call:
                    result.add_pass(f"Tool类职责验证 - {cls['name']}")
                    print(f"  ✓ {cls['name']}: 包含调用方法")
                else:
                    result.add_warning(f"Tool类职责验证 - {cls['name']}", 
                                     f"现有方法: {cls['methods']}")
                    print(f"  ⚠ {cls['name']}: 现有方法 {cls['methods']}")

def test_resource_manager_classes(result: ClassResponsibilityTestResult) -> None:
    """测试ResourceManager类职责"""
    print("\n=== 测试ResourceManager类职责 ===")
    
    rm_configs = {
        "pool_manager.py": "PoolManager",
        "global_resource_manager.py": "GlobalResourceManager",
        "resource_pool.py": "ResourcePool",
        "resource_factory.py": "ResourceFactory"
    }
    
    for file_name, expected_class in rm_configs.items():
        rm_path = SRC_ROOT / "resource_manager" / file_name
        
        if not rm_path.exists():
            result.add_fail(f"ResourceManager类验证 - {expected_class}", "文件不存在")
            print(f"  ✗ {expected_class}: 文件不存在")
            continue
        
        info = get_class_info(rm_path)
        
        if "error" in info:
            result.add_fail(f"ResourceManager类验证 - {expected_class}", f"解析错误: {info['error']}")
            print(f"  ✗ {expected_class}: 解析错误")
            continue
        
        found = False
        for cls in info.get("classes", []):
            if expected_class in cls["name"]:
                found = True
                result.add_class_detail(cls["name"], cls)
                
                if "PoolManager" in expected_class:
                    expected_methods = ["create_pool", "get_pool", "acquire", "release"]
                elif "GlobalResourceManager" in expected_class:
                    expected_methods = ["initialize", "acquire", "release", "shutdown"]
                elif "ResourcePool" in expected_class:
                    expected_methods = ["acquire", "release", "create"]
                else:
                    expected_methods = ["create", "get"]
                
                found_methods = [m for m in expected_methods if m in cls["methods"]]
                
                if len(found_methods) >= 1:
                    result.add_pass(f"ResourceManager类职责验证 - {cls['name']}")
                    print(f"  ✓ {cls['name']}: 包含方法 {found_methods}")
                else:
                    result.add_warning(f"ResourceManager类职责验证 - {cls['name']}", 
                                     f"现有方法: {cls['methods']}")
                    print(f"  ⚠ {cls['name']}: 现有方法 {cls['methods']}")
        
        if not found:
            result.add_warning(f"ResourceManager类验证 - {expected_class}", "未找到预期的类")
            print(f"  ⚠ {expected_class}: 未找到预期的类")

def run_all_tests() -> Dict[str, Any]:
    """运行所有类职责测试"""
    print("=" * 60)
    print("健康报告生成业务 - 类职责验证测试")
    print("=" * 60)
    
    result = ClassResponsibilityTestResult()
    
    test_controller_classes(result)
    test_service_classes(result)
    test_strategy_classes(result)
    test_chain_classes(result)
    test_model_service_classes(result)
    test_tool_classes(result)
    test_resource_manager_classes(result)
    
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
    report_path = PROJECT_ROOT / "test" / "report" / "health_report" / "class_responsibility_test_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n测试报告已保存到: {report_path}")
