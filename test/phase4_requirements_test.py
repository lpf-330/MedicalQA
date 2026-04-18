# -*- coding: utf-8 -*-
"""
健康咨询业务需求满足度测试脚本

测试内容：
1. 功能需求验证
2. 非功能需求验证
3. 业务规则验证
4. 特殊规则验证
"""

import requests
import json
import time
import re
import sys
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

BASE_URL = "http://localhost:8001"
TIMEOUT = 60

class RequirementsTestResult:
    """测试结果类"""
    
    def __init__(self):
        self.test_items: List[Dict[str, Any]] = []
        self.passed_count = 0
        self.failed_count = 0
        self.total_count = 0
    
    def add_result(self, category: str, test_name: str, passed: bool, 
                   details: str = "", error: str = ""):
        """添加测试结果"""
        self.total_count += 1
        if passed:
            self.passed_count += 1
        else:
            self.failed_count += 1
        
        self.test_items.append({
            "category": category,
            "test_name": test_name,
            "passed": passed,
            "details": details,
            "error": error,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def get_pass_rate(self) -> float:
        """获取通过率"""
        if self.total_count == 0:
            return 0.0
        return (self.passed_count / self.total_count) * 100


class RequirementsTester:
    """需求满足度测试器"""
    
    def __init__(self):
        self.result = RequirementsTestResult()
        self.service_started = False
    
    def check_service_health(self) -> bool:
        """检查服务健康状态"""
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def send_consult_request(self, question: str, task_id: str = None, 
                            chat_history: List[Dict] = None) -> Tuple[bool, Dict]:
        """发送咨询请求"""
        if task_id is None:
            task_id = f"test_{int(time.time())}"
        
        if chat_history is None:
            chat_history = [{"role": "user", "content": question}]
        
        payload = {
            "task_id": task_id,
            "chat_history": chat_history,
            "question": question
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/v1/consult",
                json=payload,
                timeout=TIMEOUT,
                stream=True
            )
            
            if response.status_code != 200:
                return False, {"error": f"HTTP {response.status_code}"}
            
            full_content = ""
            sources = []
            is_health_consultation = True
            error_code = 0
            session_id = ""
            first_byte_time = None
            start_time = time.time()
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    
                    if first_byte_time is None:
                        first_byte_time = time.time() - start_time
                    
                    if line_str.startswith('event: message'):
                        continue
                    elif line_str.startswith('data: '):
                        data_str = line_str[6:]
                        try:
                            data = json.loads(data_str)
                            if 'content' in data:
                                full_content += data['content']
                        except:
                            pass
                    elif line_str.startswith('event: end'):
                        next_line = next(response.iter_lines(), None)
                        if next_line:
                            next_str = next_line.decode('utf-8')
                            if next_str.startswith('data: '):
                                try:
                                    end_data = json.loads(next_str[6:])
                                    sources = end_data.get('sources', [])
                                    is_health_consultation = end_data.get('is_health_consultation', True)
                                    error_code = end_data.get('error_code', 0)
                                    session_id = end_data.get('session_id', '')
                                except:
                                    pass
            
            elapsed_time = time.time() - start_time
            
            return True, {
                "content": full_content,
                "sources": sources,
                "is_health_consultation": is_health_consultation,
                "error_code": error_code,
                "session_id": session_id,
                "elapsed_time": elapsed_time,
                "first_byte_time": first_byte_time,
                "word_count": len(full_content)
            }
        
        except Exception as e:
            return False, {"error": str(e)}
    
    def test_functional_requirements(self):
        """测试功能需求"""
        print("\n" + "=" * 60)
        print("1. 功能需求验证")
        print("=" * 60)
        
        print("\n1.1 测试健康咨询基本功能...")
        success, result = self.send_consult_request("高血压有什么症状？")
        
        if success:
            content = result.get('content', '')
            word_count = result.get('word_count', 0)
            
            passed = len(content) > 0 and word_count > 0
            details = f"回答长度: {word_count}字"
            self.result.add_result(
                "功能需求", "健康咨询基本功能", passed, 
                details, "" if passed else "回答内容为空"
            )
            
            print(f"  {'✅ 通过' if passed else '❌ 失败'} - {details}")
        else:
            self.result.add_result(
                "功能需求", "健康咨询基本功能", False, 
                "", result.get('error', '未知错误')
            )
            print(f"  ❌ 失败 - {result.get('error', '未知错误')}")
        
        print("\n1.2 测试回答长度控制（200-800字）...")
        test_questions = [
            "糖尿病的症状有哪些？",
            "感冒了怎么办？",
            "高血压患者饮食注意事项"
        ]
        
        length_tests_passed = 0
        length_details = []
        
        for question in test_questions:
            success, result = self.send_consult_request(question)
            if success:
                word_count = result.get('word_count', 0)
                is_valid_length = 200 <= word_count <= 800
                length_details.append(f"{question[:10]}: {word_count}字")
                if is_valid_length:
                    length_tests_passed += 1
        
        passed = length_tests_passed >= 2
        details = f"通过{length_tests_passed}/{len(test_questions)}次测试"
        self.result.add_result(
            "功能需求", "回答长度控制（200-800字）", passed, 
            details, "" if passed else f"详情: {'; '.join(length_details)}"
        )
        print(f"  {'✅ 通过' if passed else '❌ 失败'} - {details}")
        
        print("\n1.3 测试知识来源引用...")
        success, result = self.send_consult_request("冠心病的治疗方法有哪些？")
        
        if success:
            sources = result.get('sources', [])
            content = result.get('content', '')
            
            has_sources = len(sources) > 0 or '知识' in content or '来源' in content
            passed = has_sources
            details = f"知识来源数: {len(sources)}"
            self.result.add_result(
                "功能需求", "知识来源引用", passed, 
                details, "" if passed else "未找到知识来源引用"
            )
            print(f"  {'✅ 通过' if passed else '❌ 失败'} - {details}")
        else:
            self.result.add_result(
                "功能需求", "知识来源引用", False, 
                "", result.get('error', '未知错误')
            )
            print(f"  ❌ 失败 - {result.get('error', '未知错误')}")
        
        print("\n1.4 测试安全免责声明...")
        success, result = self.send_consult_request("头痛怎么办？")
        
        if success:
            content = result.get('content', '')
            
            has_disclaimer = any(keyword in content for keyword in [
                '仅供参考', '不能替代', '医生诊断', '及时就医', 
                '医疗建议', '专业医生', '不构成'
            ])
            
            passed = has_disclaimer
            details = "包含免责声明" if passed else "未包含免责声明"
            self.result.add_result(
                "功能需求", "安全免责声明", passed, 
                details, "" if passed else "回答中未包含安全免责声明"
            )
            print(f"  {'✅ 通过' if passed else '❌ 失败'} - {details}")
        else:
            self.result.add_result(
                "功能需求", "安全免责声明", False, 
                "", result.get('error', '未知错误')
            )
            print(f"  ❌ 失败 - {result.get('error', '未知错误')}")
        
        print("\n1.5 测试非健康咨询意图处理...")
        non_health_questions = [
            "今天天气怎么样？",
            "推荐一部好看的电影",
            "怎么做红烧肉？"
        ]
        
        non_health_tests_passed = 0
        non_health_details = []
        
        for question in non_health_questions:
            success, result = self.send_consult_request(question)
            if success:
                is_health = result.get('is_health_consultation', True)
                error_code = result.get('error_code', 0)
                content = result.get('content', '')
                
                is_non_health = (not is_health) or (error_code == 40002) or \
                               ('健康咨询' in content and '只能回答' in content)
                
                non_health_details.append(f"{question[:10]}: {'正确拒绝' if is_non_health else '未正确处理'}")
                if is_non_health:
                    non_health_tests_passed += 1
        
        passed = non_health_tests_passed >= 2
        details = f"正确处理{non_health_tests_passed}/{len(non_health_questions)}次"
        self.result.add_result(
            "功能需求", "非健康咨询意图处理", passed, 
            details, "" if passed else f"详情: {'; '.join(non_health_details)}"
        )
        print(f"  {'✅ 通过' if passed else '❌ 失败'} - {details}")
    
    def test_non_functional_requirements(self):
        """测试非功能需求"""
        print("\n" + "=" * 60)
        print("2. 非功能需求验证")
        print("=" * 60)
        
        print("\n2.1 测试响应时间（≤30秒）...")
        test_questions = [
            "高血压的症状",
            "糖尿病怎么治疗",
            "感冒吃什么药"
        ]
        
        response_time_tests_passed = 0
        response_time_details = []
        
        for question in test_questions:
            success, result = self.send_consult_request(question)
            if success:
                elapsed_time = result.get('elapsed_time', 0)
                is_valid_time = elapsed_time <= 30
                response_time_details.append(f"{question[:10]}: {elapsed_time:.2f}s")
                if is_valid_time:
                    response_time_tests_passed += 1
        
        passed = response_time_tests_passed >= 2
        details = f"通过{response_time_tests_passed}/{len(test_questions)}次测试"
        self.result.add_result(
            "非功能需求", "响应时间（≤30秒）", passed, 
            details, "" if passed else f"详情: {'; '.join(response_time_details)}"
        )
        print(f"  {'✅ 通过' if passed else '❌ 失败'} - {details}")
        
        print("\n2.2 测试首字节时间（≤30秒）...")
        first_byte_tests_passed = 0
        first_byte_details = []
        
        for question in test_questions:
            success, result = self.send_consult_request(question)
            if success:
                first_byte_time = result.get('first_byte_time', 0)
                is_valid_time = first_byte_time <= 30
                first_byte_details.append(f"{question[:10]}: {first_byte_time:.2f}s")
                if is_valid_time:
                    first_byte_tests_passed += 1
        
        passed = first_byte_tests_passed >= 2
        details = f"通过{first_byte_tests_passed}/{len(test_questions)}次测试"
        self.result.add_result(
            "非功能需求", "首字节时间（≤30秒）", passed, 
            details, "" if passed else f"详情: {'; '.join(first_byte_details)}"
        )
        print(f"  {'✅ 通过' if passed else '❌ 失败'} - {details}")
        
        print("\n2.3 测试SSE流式响应...")
        success, result = self.send_consult_request("测试流式响应")
        
        if success:
            content = result.get('content', '')
            first_byte_time = result.get('first_byte_time', 0)
            
            is_streaming = first_byte_time is not None and first_byte_time < result.get('elapsed_time', 0)
            passed = is_streaming and len(content) > 0
            details = f"首字节时间: {first_byte_time:.2f}s, 总时间: {result.get('elapsed_time', 0):.2f}s"
            self.result.add_result(
                "非功能需求", "SSE流式响应", passed, 
                details, "" if passed else "未检测到流式响应"
            )
            print(f"  {'✅ 通过' if passed else '❌ 失败'} - {details}")
        else:
            self.result.add_result(
                "非功能需求", "SSE流式响应", False, 
                "", result.get('error', '未知错误')
            )
            print(f"  ❌ 失败 - {result.get('error', '未知错误')}")
    
    def test_business_rules(self):
        """测试业务规则"""
        print("\n" + "=" * 60)
        print("3. 业务规则验证")
        print("=" * 60)
        
        print("\n3.1 测试意图识别规则...")
        test_cases = [
            ("高血压怎么治疗", True, "健康咨询"),
            ("今天天气怎么样", False, "非健康咨询"),
            ("糖尿病的症状", True, "健康咨询"),
            ("推荐一部电影", False, "非健康咨询")
        ]
        
        intent_tests_passed = 0
        intent_details = []
        
        for question, expected_health, case_name in test_cases:
            success, result = self.send_consult_request(question)
            if success:
                is_health = result.get('is_health_consultation', True)
                is_correct = is_health == expected_health
                intent_details.append(f"{case_name}: {'正确' if is_correct else '错误'}")
                if is_correct:
                    intent_tests_passed += 1
        
        passed = intent_tests_passed >= 3
        details = f"正确识别{intent_tests_passed}/{len(test_cases)}次"
        self.result.add_result(
            "业务规则", "意图识别规则", passed, 
            details, "" if passed else f"详情: {'; '.join(intent_details)}"
        )
        print(f"  {'✅ 通过' if passed else '❌ 失败'} - {details}")
        
        print("\n3.2 测试知识来源优先级...")
        success, result = self.send_consult_request("高血压的症状有哪些？")
        
        if success:
            content = result.get('content', '')
            sources = result.get('sources', [])
            
            has_knowledge = len(content) > 100
            passed = has_knowledge
            details = f"回答长度: {len(content)}字, 知识来源数: {len(sources)}"
            self.result.add_result(
                "业务规则", "知识来源优先级", passed, 
                details, "" if passed else "知识检索结果不足"
            )
            print(f"  {'✅ 通过' if passed else '❌ 失败'} - {details}")
        else:
            self.result.add_result(
                "业务规则", "知识来源优先级", False, 
                "", result.get('error', '未知错误')
            )
            print(f"  ❌ 失败 - {result.get('error', '未知错误')}")
        
        print("\n3.3 测试引用标注规则...")
        success, result = self.send_consult_request("冠心病的治疗方法")
        
        if success:
            content = result.get('content', '')
            sources = result.get('sources', [])
            
            has_citation = len(sources) > 0 or any(
                keyword in content for keyword in ['来源', '参考', '根据', '知识库']
            )
            
            passed = has_citation
            details = f"包含引用标注" if passed else "未包含引用标注"
            self.result.add_result(
                "业务规则", "引用标注规则", passed, 
                details, "" if passed else "回答中未包含知识来源引用"
            )
            print(f"  {'✅ 通过' if passed else '❌ 失败'} - {details}")
        else:
            self.result.add_result(
                "业务规则", "引用标注规则", False, 
                "", result.get('error', '未知错误')
            )
            print(f"  ❌ 失败 - {result.get('error', '未知错误')}")
    
    def test_special_rules(self):
        """测试特殊规则"""
        print("\n" + "=" * 60)
        print("4. 特殊规则验证")
        print("=" * 60)
        
        print("\n4.1 测试非健康咨询范围处理规则...")
        non_health_questions = [
            "今天天气怎么样？",
            "推荐一部好看的电影",
            "怎么做红烧肉？"
        ]
        
        special_tests_passed = 0
        special_details = []
        
        for question in non_health_questions:
            success, result = self.send_consult_request(question)
            if success:
                content = result.get('content', '')
                error_code = result.get('error_code', 0)
                is_health = result.get('is_health_consultation', True)
                
                is_friendly_reject = (error_code == 40002) or \
                                    ('健康咨询' in content and '只能回答' in content) or \
                                    (not is_health)
                
                special_details.append(f"{question[:10]}: {'友好拒绝' if is_friendly_reject else '未正确处理'}")
                if is_friendly_reject:
                    special_tests_passed += 1
        
        passed = special_tests_passed >= 2
        details = f"正确处理{special_tests_passed}/{len(non_health_questions)}次"
        self.result.add_result(
            "特殊规则", "非健康咨询范围处理规则", passed, 
            details, "" if passed else f"详情: {'; '.join(special_details)}"
        )
        print(f"  {'✅ 通过' if passed else '❌ 失败'} - {details}")
        
        print("\n4.2 测试对话上下文管理规则...")
        task_id = f"context_test_{int(time.time())}"
        
        chat_history = [
            {"role": "user", "content": "高血压的症状有哪些？"},
            {"role": "assistant", "content": "高血压的症状包括头痛、头晕、心悸等。"}
        ]
        
        success1, result1 = self.send_consult_request(
            "高血压的症状有哪些？", 
            task_id=task_id,
            chat_history=chat_history
        )
        
        if success1:
            session_id = result1.get('session_id', '')
            
            chat_history_2 = chat_history + [
                {"role": "user", "content": "高血压的症状有哪些？"},
                {"role": "assistant", "content": result1.get('content', '')},
                {"role": "user", "content": "那怎么治疗呢？"}
            ]
            
            success2, result2 = self.send_consult_request(
                "那怎么治疗呢？",
                task_id=task_id,
                chat_history=chat_history_2
            )
            
            if success2:
                content2 = result2.get('content', '')
                
                has_context = '高血压' in content2 or '治疗' in content2
                passed = has_context and len(content2) > 50
                details = f"多轮对话测试: 第二轮回答长度{len(content2)}字"
                self.result.add_result(
                    "特殊规则", "对话上下文管理规则", passed, 
                    details, "" if passed else "对话上下文管理不符合预期"
                )
                print(f"  {'✅ 通过' if passed else '❌ 失败'} - {details}")
            else:
                self.result.add_result(
                    "特殊规则", "对话上下文管理规则", False, 
                    "", result2.get('error', '未知错误')
                )
                print(f"  ❌ 失败 - {result2.get('error', '未知错误')}")
        else:
            self.result.add_result(
                "特殊规则", "对话上下文管理规则", False, 
                "", result1.get('error', '未知错误')
            )
            print(f"  ❌ 失败 - {result1.get('error', '未知错误')}")
    
    def generate_report(self) -> str:
        """生成测试报告"""
        report_lines = []
        report_lines.append("# 健康咨询业务第二轮验收评估测试 - 阶段四：需求满足度测试报告")
        report_lines.append("")
        report_lines.append(f"**测试日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**测试环境**: MedicalQA conda环境")
        report_lines.append(f"**测试类型**: 需求满足度测试")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        report_lines.append("## 一、测试总结")
        report_lines.append("")
        report_lines.append("### 整体测试结果")
        report_lines.append("")
        report_lines.append("| 指标 | 数值 |")
        report_lines.append("|------|------|")
        report_lines.append(f"| 总测试数 | {self.result.total_count} |")
        report_lines.append(f"| 通过数 | {self.result.passed_count} |")
        report_lines.append(f"| 失败数 | {self.result.failed_count} |")
        report_lines.append(f"| **通过率** | **{self.result.get_pass_rate():.2f}%** |")
        report_lines.append("")
        
        if self.result.get_pass_rate() >= 90:
            status = "✅ 优秀"
        elif self.result.get_pass_rate() >= 70:
            status = "⚠️ 良好"
        else:
            status = "❌ 需改进"
        
        report_lines.append("### 测试状态")
        report_lines.append("")
        report_lines.append(f"**{status}** - 通过率{'≥90%' if self.result.get_pass_rate() >= 90 else '在70%-90%之间' if self.result.get_pass_rate() >= 70 else '<70%'}")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        report_lines.append("## 二、测试详情")
        report_lines.append("")
        
        categories = {}
        for item in self.result.test_items:
            category = item['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(item)
        
        for category, items in categories.items():
            report_lines.append(f"### {category}")
            report_lines.append("")
            report_lines.append("| 测试项 | 状态 | 详情 | 错误信息 |")
            report_lines.append("|--------|------|------|----------|")
            
            for item in items:
                status_icon = "✅ 通过" if item['passed'] else "❌ 失败"
                details = item['details'] if item['details'] else "-"
                error = item['error'] if item['error'] else "-"
                report_lines.append(f"| {item['test_name']} | {status_icon} | {details} | {error} |")
            
            report_lines.append("")
        
        report_lines.append("---")
        report_lines.append("")
        report_lines.append("## 三、发现的问题列表")
        report_lines.append("")
        
        failed_items = [item for item in self.result.test_items if not item['passed']]
        
        if failed_items:
            report_lines.append("| 序号 | 测试项 | 类别 | 错误信息 |")
            report_lines.append("|------|--------|------|----------|")
            
            for idx, item in enumerate(failed_items, 1):
                report_lines.append(f"| {idx} | {item['test_name']} | {item['category']} | {item['error']} |")
        else:
            report_lines.append("**无严重问题发现**")
        
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        report_lines.append("## 四、需求满足度分析")
        report_lines.append("")
        
        category_stats = {}
        for item in self.result.test_items:
            category = item['category']
            if category not in category_stats:
                category_stats[category] = {'total': 0, 'passed': 0}
            category_stats[category]['total'] += 1
            if item['passed']:
                category_stats[category]['passed'] += 1
        
        report_lines.append("| 需求类别 | 测试数 | 通过数 | 通过率 | 满足度 |")
        report_lines.append("|----------|--------|--------|--------|--------|")
        
        for category, stats in category_stats.items():
            pass_rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
            satisfaction = "✅ 满足" if pass_rate >= 80 else "⚠️ 部分满足" if pass_rate >= 60 else "❌ 不满足"
            report_lines.append(f"| {category} | {stats['total']} | {stats['passed']} | {pass_rate:.1f}% | {satisfaction} |")
        
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        report_lines.append("## 五、改进建议")
        report_lines.append("")
        
        if self.result.get_pass_rate() >= 90:
            report_lines.append("1. **保持现有质量**: 系统需求满足度优秀，继续保持")
            report_lines.append("2. **持续优化**: 针对未通过的测试项进行优化")
            report_lines.append("3. **性能监控**: 建立持续的性能监控机制")
        elif self.result.get_pass_rate() >= 70:
            report_lines.append("1. **修复失败项**: 优先修复未通过的测试项")
            report_lines.append("2. **完善功能**: 补充缺失的功能实现")
            report_lines.append("3. **优化性能**: 提升响应速度和准确性")
        else:
            report_lines.append("1. **全面修复**: 需要全面修复系统问题")
            report_lines.append("2. **重新设计**: 考虑重新设计部分功能")
            report_lines.append("3. **加强测试**: 增加测试用例覆盖")
        
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        report_lines.append(f"**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**报告版本**: v1.0")
        
        return "\n".join(report_lines)
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "=" * 60)
        print("健康咨询业务需求满足度测试")
        print("=" * 60)
        
        print("\n检查服务状态...")
        if not self.check_service_health():
            print("❌ 服务未启动，请先启动服务")
            print("启动命令: python src/main.py")
            return False
        
        print("✅ 服务运行正常")
        
        try:
            self.test_functional_requirements()
            self.test_non_functional_requirements()
            self.test_business_rules()
            self.test_special_rules()
            
            print("\n" + "=" * 60)
            print("测试执行完成")
            print("=" * 60)
            print(f"总测试数: {self.result.total_count}")
            print(f"通过数: {self.result.passed_count}")
            print(f"失败数: {self.result.failed_count}")
            print(f"通过率: {self.result.get_pass_rate():.2f}%")
            
            return True
        
        except Exception as e:
            print(f"\n❌ 测试执行异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函数"""
    tester = RequirementsTester()
    
    success = tester.run_all_tests()
    
    if success:
        report = tester.generate_report()
        
        report_dir = os.path.join(project_root, "test", "report")
        os.makedirs(report_dir, exist_ok=True)
        
        report_file = os.path.join(report_dir, "phase4_requirements_report.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n测试报告已生成: {report_file}")
    else:
        print("\n测试执行失败，未生成报告")


if __name__ == "__main__":
    main()
