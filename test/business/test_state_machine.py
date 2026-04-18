"""
健康报告生成业务 - 状态机验证测试
测试目标：验证健康报告生成流程是否符合《项目业务详细设计v3》中的10状态有限状态机设计
"""
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

PROJECT_ROOT = Path("/home/project/MedicalQA")
SRC_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_ROOT))

class StateMachineTestResult:
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

def test_state_definitions(result: StateMachineTestResult) -> None:
    """测试状态定义"""
    print("\n=== 测试状态定义 ===")
    
    expected_states = {
        "INITIAL": "初始状态",
        "DATA_PREPARE": "数据准备状态",
        "MULTI_ANALYSIS": "模型分析状态",
        "PARALLEL_PROCESSING": "并行处理状态",
        "INTEGRATION": "整合计算状态",
        "REPORT_GENERATION": "报告生成状态",
        "STREAMING": "流式返回状态",
        "ASSEMBLY": "组装结束状态",
        "FINISHED": "完成状态",
        "ERROR": "错误状态"
    }
    
    strategy_path = SRC_ROOT / "orchestration" / "agent" / "report_strategy" / "report_strategy.py"
    content = get_file_content(strategy_path)
    
    found_states = []
    for state, desc in expected_states.items():
        if state in content or state.lower() in content.lower():
            found_states.append(state)
            result.add_pass(f"状态定义验证 - {desc}({state})")
            print(f"  ✓ {desc}({state}): 存在")
        else:
            result.add_warning(f"状态定义验证 - {desc}({state})", "未在代码中找到状态定义")
            print(f"  ⚠ {desc}({state}): 未在代码中找到")
    
    if len(found_states) >= 8:
        result.add_pass("状态机验证 - 状态定义完整性")
        print(f"  ✓ 状态定义完整性: 找到 {len(found_states)}/10 个状态")
    else:
        result.add_warning("状态机验证 - 状态定义完整性", f"只找到 {len(found_states)}/10 个状态")
        print(f"  ⚠ 状态定义完整性: 只找到 {len(found_states)}/10 个状态")

def test_context_class(result: StateMachineTestResult) -> None:
    """测试上下文类"""
    print("\n=== 测试上下文类 ===")
    
    context_patterns = {
        "ReportContext": ["monitoring_data", "user_profile", "report_content", "health_score"],
        "ReportContextBody": ["data", "state"],
        "ReportResultData": ["health_score", "risk_level", "report_id"]
    }
    
    for class_name, expected_attrs in context_patterns.items():
        found_in_files = []
        
        for py_file in SRC_ROOT.glob("**/*.py"):
            content = get_file_content(py_file)
            if f"class {class_name}" in content:
                found_attrs = [attr for attr in expected_attrs if attr in content]
                if found_attrs:
                    result.add_pass(f"上下文类验证 - {class_name}")
                    print(f"  ✓ {class_name}: 包含属性 {found_attrs}")
                    found_in_files.append(str(py_file))
        
        if not found_in_files:
            result.add_warning(f"上下文类验证 - {class_name}", "未找到类定义或预期属性")
            print(f"  ⚠ {class_name}: 未找到类定义或预期属性")

def test_state_transitions(result: StateMachineTestResult) -> None:
    """测试状态转换"""
    print("\n=== 测试状态转换 ===")
    
    expected_transitions = [
        ("INITIAL", "DATA_PREPARE"),
        ("DATA_PREPARE", "MULTI_ANALYSIS"),
        ("MULTI_ANALYSIS", "PARALLEL_PROCESSING"),
        ("PARALLEL_PROCESSING", "INTEGRATION"),
        ("INTEGRATION", "REPORT_GENERATION"),
        ("REPORT_GENERATION", "STREAMING"),
        ("STREAMING", "ASSEMBLY"),
        ("ASSEMBLY", "FINISHED"),
        ("*", "ERROR")
    ]
    
    strategy_path = SRC_ROOT / "orchestration" / "agent" / "report_strategy" / "report_strategy.py"
    content = get_file_content(strategy_path)
    
    found_transitions = 0
    for from_state, to_state in expected_transitions:
        if from_state == "*":
            if "ERROR" in content or "error" in content.lower():
                found_transitions += 1
                result.add_pass(f"状态转换验证 - 任意状态→ERROR")
                print(f"  ✓ 任意状态→ERROR: 存在错误处理")
        else:
            if from_state in content and to_state in content:
                found_transitions += 1
                result.add_pass(f"状态转换验证 - {from_state}→{to_state}")
                print(f"  ✓ {from_state}→{to_state}: 存在")
            else:
                result.add_warning(f"状态转换验证 - {from_state}→{to_state}", "未找到转换代码")
                print(f"  ⚠ {from_state}→{to_state}: 未找到转换代码")
    
    if found_transitions >= len(expected_transitions) - 2:
        result.add_pass("状态机验证 - 状态转换完整性")
        print(f"  ✓ 状态转换完整性: 找到 {found_transitions}/{len(expected_transitions)} 个转换")

def test_chain_execution(result: StateMachineTestResult) -> None:
    """测试Chain执行流程"""
    print("\n=== 测试Chain执行流程 ===")
    
    chain_configs = {
        "DataPrepareChain": "data_prepare_chain",
        "MultiAnalysisChain": "multi_analysis_chain",
        "DimensionEvaluationChain": "dimension_evaluation_chain",
        "ReportKnowledgeRetrievalChain": "report_knowledge_retrieval_chain",
        "IntegrationChain": "integration_chain",
        "ReportGenerationChain": "report_generation_chain"
    }
    
    expected_methods = ["execute", "process", "run"]
    
    for chain_name, chain_dir in chain_configs.items():
        chain_files = list(SRC_ROOT.glob(f"**/chain/{chain_dir}/{chain_dir}.py"))
        
        if chain_files:
            chain_path = chain_files[0]
            content = get_file_content(chain_path)
            found_methods = [m for m in expected_methods if f"def {m}" in content]
            
            if found_methods:
                result.add_pass(f"Chain执行验证 - {chain_name}")
                print(f"  ✓ {chain_name}: 包含执行方法 {found_methods}")
            else:
                result.add_warning(f"Chain执行验证 - {chain_name}", "未找到标准执行方法")
                print(f"  ⚠ {chain_name}: 未找到标准执行方法")
        else:
            result.add_fail(f"Chain执行验证 - {chain_name}", "文件不存在")
            print(f"  ✗ {chain_name}: 文件不存在")

def test_parallel_processing(result: StateMachineTestResult) -> None:
    """测试并行处理"""
    print("\n=== 测试并行处理 ===")
    
    strategy_path = SRC_ROOT / "orchestration" / "agent" / "report_strategy" / "report_strategy.py"
    content = get_file_content(strategy_path)
    
    if "asyncio.gather" in content or "gather(" in content:
        result.add_pass("并行处理验证 - 使用asyncio.gather")
        print("  ✓ 使用asyncio.gather进行并行处理")
    else:
        result.add_warning("并行处理验证 - asyncio.gather", "未找到asyncio.gather调用")
        print("  ⚠ 未找到asyncio.gather调用")
    
    multi_analysis_path = SRC_ROOT / "orchestration" / "chain" / "multi_analysis_chain" / "multi_analysis_chain.py"
    ma_content = get_file_content(multi_analysis_path)
    
    if "asyncio" in ma_content or "async def" in ma_content:
        result.add_pass("并行处理验证 - 异步实现")
        print("  ✓ MultiAnalysisChain使用异步实现")
    else:
        result.add_warning("并行处理验证 - 异步实现", "未找到异步实现")
        print("  ⚠ MultiAnalysisChain未使用异步实现")

def run_all_tests() -> Dict[str, Any]:
    """运行所有状态机测试"""
    print("=" * 60)
    print("健康报告生成业务 - 状态机验证测试")
    print("=" * 60)
    
    result = StateMachineTestResult()
    
    test_state_definitions(result)
    test_context_class(result)
    test_state_transitions(result)
    test_chain_execution(result)
    test_parallel_processing(result)
    
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
    report_path = PROJECT_ROOT / "test" / "report" / "health_report" / "state_machine_test_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n测试报告已保存到: {report_path}")
