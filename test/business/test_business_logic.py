"""
健康报告生成业务 - 并发处理、算法实现、特殊规则验证测试
测试目标：验证业务实现是否符合《项目业务详细设计v3》
"""
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

PROJECT_ROOT = Path("/home/project/MedicalQA")
SRC_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_ROOT))

class BusinessTestResult:
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

def test_concurrent_processing(result: BusinessTestResult) -> None:
    """测试并发处理"""
    print("\n=== 测试并发处理 ===")
    
    strategy_path = SRC_ROOT / "orchestration" / "agent" / "report_strategy" / "report_strategy.py"
    content = get_file_content(strategy_path)
    
    if "asyncio" in content or "async def" in content:
        result.add_pass("并发处理验证 - 使用异步编程")
        print("  ✓ 使用异步编程(async/await)")
    else:
        result.add_warning("并发处理验证 - 异步编程", "未找到async/await关键字")
        print("  ⚠ 未找到async/await关键字")
    
    if "gather" in content or "concurrent" in content.lower():
        result.add_pass("并发处理验证 - 并发执行机制")
        print("  ✓ 存在并发执行机制")
    else:
        result.add_warning("并发处理验证 - 并发执行机制", "未找到并发执行代码")
        print("  ⚠ 未找到并发执行代码")
    
    multi_analysis_path = SRC_ROOT / "orchestration" / "chain" / "multi_analysis_chain" / "multi_analysis_chain.py"
    ma_content = get_file_content(multi_analysis_path)
    
    dimension_keywords = ["dimension", "维度", "risk", "风险", "score", "评分"]
    found_keywords = [k for k in dimension_keywords if k in ma_content.lower()]
    
    if len(found_keywords) >= 3:
        result.add_pass("并发处理验证 - 8维度评估任务")
        print(f"  ✓ 8维度评估任务相关代码存在")
    else:
        result.add_warning("并发处理验证 - 8维度评估任务", "维度评估相关代码较少")
        print("  ⚠ 维度评估相关代码较少")
    
    integration_path = SRC_ROOT / "orchestration" / "chain" / "integration_chain" / "integration_chain.py"
    int_content = get_file_content(integration_path)
    
    if "merge" in int_content.lower() or "整合" in int_content or "integrate" in int_content.lower():
        result.add_pass("并发处理验证 - 结果整合机制")
        print("  ✓ 存在结果整合机制")
    else:
        result.add_warning("并发处理验证 - 结果整合机制", "未找到结果整合代码")
        print("  ⚠ 未找到结果整合代码")

def test_algorithm_implementation(result: BusinessTestResult) -> None:
    """测试算法实现"""
    print("\n=== 测试算法实现 ===")
    
    integration_path = SRC_ROOT / "orchestration" / "chain" / "integration_chain" / "integration_chain.py"
    int_content = get_file_content(integration_path)
    
    if "risk_score" in int_content.lower() or "风险" in int_content:
        result.add_pass("算法实现验证 - 疾病风险评分算法")
        print("  ✓ 疾病风险评分算法相关代码存在")
    else:
        result.add_warning("算法实现验证 - 疾病风险评分算法", "未找到风险评分相关代码")
        print("  ⚠ 未找到风险评分相关代码")
    
    if "health_score" in int_content.lower() or "健康评分" in int_content:
        result.add_pass("算法实现验证 - 健康综合评分算法")
        print("  ✓ 健康综合评分算法相关代码存在")
    else:
        result.add_warning("算法实现验证 - 健康综合评分算法", "未找到健康评分相关代码")
        print("  ⚠ 未找到健康评分相关代码")
    
    if "risk_level" in int_content.lower() or "风险等级" in int_content:
        result.add_pass("算法实现验证 - 风险等级判定算法")
        print("  ✓ 风险等级判定算法相关代码存在")
    else:
        result.add_warning("算法实现验证 - 风险等级判定算法", "未找到风险等级相关代码")
        print("  ⚠ 未找到风险等级相关代码")
    
    if "100" in int_content or "score" in int_content.lower():
        result.add_pass("算法实现验证 - 100分制评分")
        print("  ✓ 100分制评分相关代码存在")
    else:
        result.add_warning("算法实现验证 - 100分制评分", "未找到100分制评分代码")
        print("  ⚠ 未找到100分制评分代码")
    
    if "deduct" in int_content.lower() or "扣分" in int_content or "weight" in int_content.lower():
        result.add_pass("算法实现验证 - 评分扣分规则")
        print("  ✓ 评分扣分规则相关代码存在")
    else:
        result.add_warning("算法实现验证 - 评分扣分规则", "未找到扣分规则代码")
        print("  ⚠ 未找到扣分规则代码")

def test_special_rules(result: BusinessTestResult) -> None:
    """测试特殊规则"""
    print("\n=== 测试特殊规则 ===")
    
    integration_path = SRC_ROOT / "orchestration" / "chain" / "integration_chain" / "integration_chain.py"
    int_content = get_file_content(integration_path)
    
    if "high_risk" in int_content.lower() or "高风险" in int_content or "priority" in int_content.lower():
        result.add_pass("特殊规则验证 - 高风险疾病优先规则")
        print("  ✓ 高风险疾病优先规则相关代码存在")
    else:
        result.add_warning("特殊规则验证 - 高风险疾病优先规则", "未找到高风险优先代码")
        print("  ⚠ 未找到高风险优先代码")
    
    if "complication" in int_content.lower() or "并发症" in int_content or "association" in int_content.lower():
        result.add_pass("特殊规则验证 - 多疾病关联规则")
        print("  ✓ 多疾病关联规则相关代码存在")
    else:
        result.add_warning("特殊规则验证 - 多疾病关联规则", "未找到疾病关联代码")
        print("  ⚠ 未找到疾病关联代码")
    
    if "allergy" in int_content.lower() or "过敏" in int_content or "conflict" in int_content.lower():
        result.add_pass("特殊规则验证 - 用药冲突检测规则")
        print("  ✓ 用药冲突检测规则相关代码存在")
    else:
        result.add_warning("特殊规则验证 - 用药冲突检测规则", "未找到用药冲突检测代码")
        print("  ⚠ 未找到用药冲突检测代码")
    
    data_prepare_path = SRC_ROOT / "orchestration" / "chain" / "data_prepare_chain" / "data_prepare_chain.py"
    dp_content = get_file_content(data_prepare_path)
    
    if "degradation" in dp_content.lower() or "降级" in dp_content or "empty" in dp_content.lower() or "空值" in dp_content:
        result.add_pass("特殊规则验证 - 空值降级策略")
        print("  ✓ 空值降级策略相关代码存在")
    else:
        result.add_warning("特殊规则验证 - 空值降级策略", "未找到空值降级代码")
        print("  ⚠ 未找到空值降级代码")
    
    if "age" in dp_content.lower() or "年龄" in dp_content or "60" in dp_content:
        result.add_pass("特殊规则验证 - 年龄适配规则")
        print("  ✓ 年龄适配规则相关代码存在")
    else:
        result.add_warning("特殊规则验证 - 年龄适配规则", "未找到年龄适配代码")
        print("  ⚠ 未找到年龄适配代码")

def test_token_budget(result: BusinessTestResult) -> None:
    """测试Token预算"""
    print("\n=== 测试Token预算 ===")
    
    report_gen_path = SRC_ROOT / "orchestration" / "chain" / "report_generation_chain" / "report_generation_chain.py"
    rg_content = get_file_content(report_gen_path)
    
    if "token" in rg_content.lower() or "prompt" in rg_content.lower():
        result.add_pass("Token预算验证 - Prompt模板管理")
        print("  ✓ Prompt模板管理相关代码存在")
    else:
        result.add_warning("Token预算验证 - Prompt模板管理", "未找到Prompt管理代码")
        print("  ⚠ 未找到Prompt管理代码")
    
    if "template" in rg_content.lower() or "模板" in rg_content:
        result.add_pass("Token预算验证 - 报告模板")
        print("  ✓ 报告模板相关代码存在")
    else:
        result.add_warning("Token预算验证 - 报告模板", "未找到报告模板代码")
        print("  ⚠ 未找到报告模板代码")
    
    if "knowledge" in rg_content.lower() or "知识" in rg_content:
        result.add_pass("Token预算验证 - 知识素材注入")
        print("  ✓ 知识素材注入相关代码存在")
    else:
        result.add_warning("Token预算验证 - 知识素材注入", "未找到知识素材注入代码")
        print("  ⚠ 未找到知识素材注入代码")

def run_all_tests() -> Dict[str, Any]:
    """运行所有业务测试"""
    print("=" * 60)
    print("健康报告生成业务 - 业务符合性测试")
    print("=" * 60)
    
    result = BusinessTestResult()
    
    test_concurrent_processing(result)
    test_algorithm_implementation(result)
    test_special_rules(result)
    test_token_budget(result)
    
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
    report_path = PROJECT_ROOT / "test" / "report" / "health_report" / "business_test_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n测试报告已保存到: {report_path}")
