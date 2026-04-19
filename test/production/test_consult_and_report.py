#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
健康咨询与健康报告生成业务生产测试脚本

测试内容：
1. 健康咨询业务连续请求测试（4个请求，带上下文）
2. 健康报告生成业务连续请求测试（4个请求）
3. 业务随机交替请求测试（10次随机请求）

异常处理：
- 遇到bug或系统崩溃时立即停止测试
- 记录和分析bug原因并向用户汇报
"""

import requests
import json
import time
import random
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import os


class ProductionTestRunner:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.test_results = {
            "phase1_preparation": {},
            "phase2_consult_continuous": [],
            "phase3_report_continuous": [],
            "phase4_random_alternate": [],
            "errors": [],
            "statistics": {}
        }
        self.start_time = datetime.now()
        self.test_stopped = False
        self.stop_reason = None
        
        self.consult_questions = [
            {
                "question": "头痛怎么办？",
                "chat_history": [],
                "description": "第1个咨询请求（无上下文）"
            },
            {
                "question": "那如果头痛持续很久呢？",
                "chat_history": [
                    {"role": "user", "content": "头痛怎么办？"},
                    {"role": "assistant", "content": "头痛的治疗方法包括休息、按摩、药物治疗等。如果头痛频繁或严重，建议就医检查。"}
                ],
                "description": "第2个咨询请求（带上下文）"
            },
            {
                "question": "感冒了吃什么药？",
                "chat_history": [],
                "description": "第3个咨询请求（新话题，无上下文）"
            },
            {
                "question": "有没有什么副作用？",
                "chat_history": [
                    {"role": "user", "content": "感冒了吃什么药？"},
                    {"role": "assistant", "content": "感冒可以服用感冒灵、板蓝根、布洛芬等药物，具体用药请遵医嘱。"}
                ],
                "description": "第4个咨询请求（带上下文）"
            }
        ]
        
        self.random_consult_questions = [
            "高血压怎么治疗？",
            "糖尿病的症状有哪些？",
            "失眠怎么调理？",
            "胃痛吃什么药？",
            "发烧怎么办？",
            "咳嗽吃什么好？",
            "腰痛怎么缓解？",
            "皮肤过敏怎么办？",
            "便秘怎么治疗？",
            "眼睛疲劳怎么缓解？"
        ]
        
        self.report_counter = 0

    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    def check_system_health(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                self.log("系统健康检查通过")
                return True
            else:
                self.log(f"系统健康检查失败: HTTP {response.status_code}", "ERROR")
                return False
        except Exception as e:
            self.log(f"系统健康检查异常: {str(e)}", "ERROR")
            return False

    def stop_test(self, reason: str, error_info: Dict = None):
        self.test_stopped = True
        self.stop_reason = reason
        self.log(f"测试停止: {reason}", "ERROR")
        
        if error_info:
            self.test_results["errors"].append({
                "time": datetime.now().isoformat(),
                "reason": reason,
                "error_info": error_info
            })
            
            print("\n" + "=" * 80)
            print("异常详情：")
            print(f"时间: {datetime.now().isoformat()}")
            print(f"原因: {reason}")
            if "request_data" in error_info:
                print(f"请求数据: {json.dumps(error_info['request_data'], ensure_ascii=False, indent=2)}")
            if "error_message" in error_info:
                print(f"错误信息: {error_info['error_message']}")
            if "stack_trace" in error_info:
                print(f"堆栈跟踪:\n{error_info['stack_trace']}")
            print("=" * 80 + "\n")

    def send_consult_request(self, question: str, chat_history: List = None, task_id: str = None) -> Dict:
        if self.test_stopped:
            return {"success": False, "error": "测试已停止"}

        if task_id is None:
            task_id = f"consult-test-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        if chat_history is None:
            chat_history = []

        request_data = {
            "task_id": task_id,
            "question": question,
            "chat_history": chat_history
        }

        self.log(f"发送健康咨询请求: {question[:50]}...")
        start_time = time.time()

        try:
            response = requests.post(
                f"{self.base_url}/api/v1/consult",
                json=request_data,
                stream=True,
                timeout=300
            )

            if response.status_code != 200:
                error_msg = f"HTTP状态码错误: {response.status_code}"
                self.stop_test(error_msg, {
                    "request_data": request_data,
                    "error_message": response.text
                })
                return {"success": False, "error": error_msg, "status_code": response.status_code}

            full_content = ""
            current_event = None
            sources = []
            word_count = 0

            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('event: '):
                        current_event = decoded_line[7:]
                    elif decoded_line.startswith('data: '):
                        data = decoded_line[6:]
                        if current_event == 'message':
                            try:
                                json_data = json.loads(data)
                                if 'content' in json_data:
                                    content = json_data['content']
                                    full_content += content
                            except json.JSONDecodeError:
                                pass
                        elif current_event == 'end':
                            try:
                                end_data = json.loads(data)
                                sources = end_data.get('sources', [])
                                word_count = end_data.get('word_count', 0)
                            except json.JSONDecodeError:
                                pass
                        elif current_event == 'error':
                            self.stop_test(f"收到错误事件: {data}", {
                                "request_data": request_data,
                                "error_message": data
                            })
                            return {"success": False, "error": data}

            elapsed_time = time.time() - start_time

            result = {
                "success": True,
                "task_id": task_id,
                "question": question,
                "has_context": len(chat_history) > 0,
                "answer": full_content,
                "answer_length": len(full_content),
                "word_count": word_count,
                "sources": sources,
                "elapsed_time": elapsed_time,
                "timestamp": datetime.now().isoformat()
            }

            self.log(f"健康咨询请求完成: 耗时{elapsed_time:.2f}秒, 回答长度{len(full_content)}字符")
            return result

        except Exception as e:
            error_msg = f"请求异常: {str(e)}"
            self.stop_test(error_msg, {
                "request_data": request_data,
                "error_message": str(e),
                "stack_trace": traceback.format_exc()
            })
            return {"success": False, "error": error_msg, "exception": str(e)}

    def generate_report_data(self, task_id: str) -> Dict:
        now = datetime.now()
        
        def generate_latest_data(base_value, variance, count=4):
            data = []
            for i in range(count):
                timestamp = (now - timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
                value = round(base_value + random.uniform(-variance, variance), 1)
                data.append({
                    "value": value,
                    "unit": "",
                    "time": timestamp
                })
            return data
        
        def generate_daily_stats(base_value, variance, days=30):
            data = []
            for i in range(days):
                date = (now - timedelta(days=i)).strftime("%Y-%m-%d")
                value = round(base_value + random.uniform(-variance, variance), 1)
                data.append({
                    "date": date,
                    "avg_value": value,
                    "max_value": round(value + random.uniform(0, variance), 1),
                    "min_value": round(value - random.uniform(0, variance), 1),
                    "unit": ""
                })
            return data
        
        def generate_weekly_stats(base_value, variance, weeks=12):
            data = []
            for i in range(weeks):
                start_date = (now - timedelta(weeks=i+1)).strftime("%Y-%m-%d")
                end_date = (now - timedelta(weeks=i)).strftime("%Y-%m-%d")
                value = round(base_value + random.uniform(-variance, variance), 1)
                data.append({
                    "start_date": start_date,
                    "end_date": end_date,
                    "avg_value": value,
                    "max_value": round(value + random.uniform(0, variance), 1),
                    "min_value": round(value - random.uniform(0, variance), 1),
                    "unit": ""
                })
            return data
        
        def generate_monthly_stats(base_value, variance, months=6):
            data = []
            for i in range(months):
                month = (now - timedelta(days=30*i)).strftime("%Y-%m")
                value = round(base_value + random.uniform(-variance, variance), 1)
                data.append({
                    "month": month,
                    "avg_value": value,
                    "max_value": round(value + random.uniform(0, variance), 1),
                    "min_value": round(value - random.uniform(0, variance), 1),
                    "unit": ""
                })
            return data
        
        monitoring_data = {
            "heart_rate": {
                "latest": generate_latest_data(75, 10),
                "daily_stats": generate_daily_stats(75, 5),
                "weekly_stats": generate_weekly_stats(75, 3),
                "monthly_stats": generate_monthly_stats(75, 2)
            },
            "blood_glucose": {
                "latest": generate_latest_data(5.5, 1.0),
                "daily_stats": generate_daily_stats(5.5, 0.5),
                "weekly_stats": generate_weekly_stats(5.5, 0.3),
                "monthly_stats": generate_monthly_stats(5.5, 0.2)
            },
            "perfusion_index": {
                "latest": generate_latest_data(3.5, 0.5),
                "daily_stats": generate_daily_stats(3.5, 0.3),
                "weekly_stats": generate_weekly_stats(3.5, 0.2),
                "monthly_stats": generate_monthly_stats(3.5, 0.1)
            },
            "blood_oxygen": {
                "latest": generate_latest_data(98, 1),
                "daily_stats": generate_daily_stats(98, 0.5),
                "weekly_stats": generate_weekly_stats(98, 0.3),
                "monthly_stats": generate_monthly_stats(98, 0.2)
            },
            "sleep": {
                "latest": [
                    {
                        "date": (now - timedelta(days=i)).strftime("%Y-%m-%d"),
                        "duration": round(7 + random.uniform(-1, 1), 1),
                        "quality": random.choice(["良好", "一般", "较差"]),
                        "unit": "小时"
                    }
                    for i in range(4)
                ],
                "daily_stats": generate_daily_stats(7, 0.5),
                "weekly_stats": generate_weekly_stats(7, 0.3),
                "monthly_stats": generate_monthly_stats(7, 0.2)
            },
            "blood_pressure": {
                "latest": [
                    {
                        "systolic": round(120 + random.uniform(-10, 10), 1),
                        "diastolic": round(80 + random.uniform(-5, 5), 1),
                        "unit": "mmHg",
                        "time": (now - timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
                    }
                    for i in range(4)
                ],
                "daily_stats": [
                    {
                        "date": (now - timedelta(days=i)).strftime("%Y-%m-%d"),
                        "avg_systolic": round(120 + random.uniform(-5, 5), 1),
                        "avg_diastolic": round(80 + random.uniform(-3, 3), 1),
                        "max_systolic": round(125 + random.uniform(0, 5), 1),
                        "min_systolic": round(115 + random.uniform(-5, 0), 1),
                        "unit": "mmHg"
                    }
                    for i in range(30)
                ],
                "weekly_stats": [
                    {
                        "start_date": (now - timedelta(weeks=i+1)).strftime("%Y-%m-%d"),
                        "end_date": (now - timedelta(weeks=i)).strftime("%Y-%m-%d"),
                        "avg_systolic": round(120 + random.uniform(-3, 3), 1),
                        "avg_diastolic": round(80 + random.uniform(-2, 2), 1),
                        "unit": "mmHg"
                    }
                    for i in range(12)
                ],
                "monthly_stats": [
                    {
                        "month": (now - timedelta(days=30*i)).strftime("%Y-%m"),
                        "avg_systolic": round(120 + random.uniform(-2, 2), 1),
                        "avg_diastolic": round(80 + random.uniform(-1, 1), 1),
                        "unit": "mmHg"
                    }
                    for i in range(6)
                ]
            }
        }
        
        user_profile = {
            "user_id": 1001,
            "gender": "男",
            "birth_date": "1985-03-15",
            "height": 175.0,
            "weight": 70.0,
            "past_medical_history": "2018年诊断为高血压，长期服用降压药；2020年曾患肺炎，已治愈",
            "family_history": "父亲有糖尿病史，母亲有高血压病史",
            "allergy_history": "对青霉素过敏",
            "surgical_history": "2015年做过阑尾切除术",
            "medical_compliance": "每日按时服用降压药，定期监测血压"
        }
        
        return {
            "task_id": task_id,
            "monitoring_data": monitoring_data,
            "user_profile": user_profile
        }

    def send_report_request(self, task_id: str = None) -> Dict:
        if self.test_stopped:
            return {"success": False, "error": "测试已停止"}

        if task_id is None:
            self.report_counter += 1
            task_id = f"report-test-{self.report_counter:03d}"

        request_data = self.generate_report_data(task_id)

        self.log(f"发送健康报告生成请求: {task_id}")
        start_time = time.time()

        try:
            response = requests.post(
                f"{self.base_url}/api/v1/report",
                json=request_data,
                stream=True,
                timeout=600
            )

            if response.status_code != 200:
                error_msg = f"HTTP状态码错误: {response.status_code}"
                self.stop_test(error_msg, {
                    "request_data": request_data,
                    "error_message": response.text
                })
                return {"success": False, "error": error_msg, "status_code": response.status_code}

            full_content = ""
            current_event = None
            report_metadata = {}

            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('event: '):
                        current_event = decoded_line[7:]
                    elif decoded_line.startswith('data: '):
                        data = decoded_line[6:]
                        if current_event == 'message':
                            try:
                                json_data = json.loads(data)
                                if 'content' in json_data:
                                    content = json_data['content']
                                    full_content += content
                            except json.JSONDecodeError:
                                pass
                        elif current_event == 'end':
                            try:
                                end_data = json.loads(data)
                                report_metadata = end_data
                            except json.JSONDecodeError:
                                pass
                        elif current_event == 'error':
                            self.stop_test(f"收到错误事件: {data}", {
                                "request_data": request_data,
                                "error_message": data
                            })
                            return {"success": False, "error": data}

            elapsed_time = time.time() - start_time

            result = {
                "success": True,
                "task_id": task_id,
                "report": full_content,
                "report_length": len(full_content),
                "elapsed_time": elapsed_time,
                "metadata": report_metadata,
                "timestamp": datetime.now().isoformat()
            }

            self.log(f"健康报告生成请求完成: 耗时{elapsed_time:.2f}秒, 报告长度{len(full_content)}字符")
            return result

        except Exception as e:
            error_msg = f"请求异常: {str(e)}"
            self.stop_test(error_msg, {
                "request_data": request_data,
                "error_message": str(e),
                "stack_trace": traceback.format_exc()
            })
            return {"success": False, "error": error_msg, "exception": str(e)}

    def run_phase2_consult_continuous(self):
        self.log("=" * 80)
        self.log("阶段二：健康咨询业务连续请求测试")
        self.log("=" * 80)

        for i, question_data in enumerate(self.consult_questions):
            if self.test_stopped:
                break

            self.log(f"\n执行第{i+1}个健康咨询请求: {question_data['description']}")
            result = self.send_consult_request(
                question=question_data["question"],
                chat_history=question_data["chat_history"],
                task_id=f"consult-phase2-{i+1:03d}"
            )

            if result["success"]:
                self.test_results["phase2_consult_continuous"].append(result)
            else:
                self.log(f"第{i+1}个健康咨询请求失败: {result.get('error', '未知错误')}", "ERROR")
                break

            time.sleep(2)

    def run_phase3_report_continuous(self):
        if self.test_stopped:
            return

        self.log("\n" + "=" * 80)
        self.log("阶段三：健康报告生成业务连续请求测试")
        self.log("=" * 80)

        for i in range(4):
            if self.test_stopped:
                break

            self.log(f"\n执行第{i+1}个健康报告生成请求")
            result = self.send_report_request(task_id=f"report-phase3-{i+1:03d}")

            if result["success"]:
                self.test_results["phase3_report_continuous"].append(result)
            else:
                self.log(f"第{i+1}个健康报告生成请求失败: {result.get('error', '未知错误')}", "ERROR")
                break

            time.sleep(2)

    def run_phase4_random_alternate(self):
        if self.test_stopped:
            return

        self.log("\n" + "=" * 80)
        self.log("阶段四：业务随机交替请求测试")
        self.log("=" * 80)

        random_sequence = [random.choice(["consult", "report"]) for _ in range(10)]
        self.log(f"随机生成的请求序列: {random_sequence}")

        self.test_results["random_sequence"] = random_sequence

        consult_index = 0
        report_index = 0

        for i, request_type in enumerate(random_sequence):
            if self.test_stopped:
                break

            self.log(f"\n执行第{i+1}次随机请求: {request_type}")

            if request_type == "consult":
                question = self.random_consult_questions[consult_index % len(self.random_consult_questions)]
                chat_history = []
                if consult_index > 0 and random.random() > 0.5:
                    prev_result = self.test_results["phase4_random_alternate"][-1] if self.test_results["phase4_random_alternate"] else None
                    if prev_result and prev_result.get("type") == "consult":
                        chat_history = [
                            {"role": "user", "content": prev_result.get("question", "")},
                            {"role": "assistant", "content": prev_result.get("answer", "")[:200]}
                        ]
                
                result = self.send_consult_request(
                    question=question,
                    chat_history=chat_history,
                    task_id=f"consult-phase4-{i+1:03d}"
                )
                result["type"] = "consult"
                result["question"] = question
                consult_index += 1
            else:
                result = self.send_report_request(task_id=f"report-phase4-{i+1:03d}")
                result["type"] = "report"
                report_index += 1

            if result["success"]:
                self.test_results["phase4_random_alternate"].append(result)
            else:
                self.log(f"第{i+1}次随机请求失败: {result.get('error', '未知错误')}", "ERROR")
                break

            time.sleep(2)

    def calculate_statistics(self):
        def calc_stats(results: List[Dict]) -> Dict:
            if not results:
                return {}
            
            times = [r["elapsed_time"] for r in results if r.get("success")]
            if not times:
                return {}
            
            return {
                "count": len(times),
                "avg_time": sum(times) / len(times),
                "max_time": max(times),
                "min_time": min(times),
                "total_time": sum(times)
            }

        self.test_results["statistics"] = {
            "phase2_consult": calc_stats(self.test_results["phase2_consult_continuous"]),
            "phase3_report": calc_stats(self.test_results["phase3_report_continuous"]),
            "phase4_random": calc_stats(self.test_results["phase4_random_alternate"]),
            "total_requests": (
                len(self.test_results["phase2_consult_continuous"]) +
                len(self.test_results["phase3_report_continuous"]) +
                len(self.test_results["phase4_random_alternate"])
            ),
            "total_time": (datetime.now() - self.start_time).total_seconds()
        }

    def generate_report(self) -> str:
        self.calculate_statistics()

        report_lines = []
        report_lines.append("# 健康咨询与健康报告生成业务生产测试报告")
        report_lines.append("")
        report_lines.append(f"**测试时间**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} ~ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**测试环境**: MedicalQA conda环境, 2080ti GPU")
        report_lines.append(f"**系统服务**: {self.base_url}")
        report_lines.append("")

        if self.test_stopped:
            report_lines.append("## ⚠️ 测试异常终止")
            report_lines.append("")
            report_lines.append(f"**终止原因**: {self.stop_reason}")
            report_lines.append("")
            if self.test_results["errors"]:
                report_lines.append("### 错误详情")
                report_lines.append("")
                for error in self.test_results["errors"]:
                    report_lines.append(f"- **时间**: {error['time']}")
                    report_lines.append(f"- **原因**: {error['reason']}")
                    if "error_message" in error.get("error_info", {}):
                        report_lines.append(f"- **错误信息**: {error['error_info']['error_message']}")
                    report_lines.append("")
            report_lines.append("---")
            report_lines.append("")

        report_lines.append("## 测试概述")
        report_lines.append("")
        report_lines.append("本次测试包括以下阶段：")
        report_lines.append("1. 健康咨询业务连续请求测试（4个请求，带上下文）")
        report_lines.append("2. 健康报告生成业务连续请求测试（4个请求）")
        report_lines.append("3. 业务随机交替请求测试（10次随机请求）")
        report_lines.append("")

        stats = self.test_results["statistics"]
        report_lines.append("## 测试统计")
        report_lines.append("")
        report_lines.append(f"- **总请求数**: {stats.get('total_requests', 0)}")
        report_lines.append(f"- **总测试时间**: {stats.get('total_time', 0):.2f}秒")
        report_lines.append("")

        if stats.get("phase2_consult"):
            report_lines.append("### 阶段二：健康咨询业务连续请求测试")
            report_lines.append("")
            phase2_stats = stats["phase2_consult"]
            report_lines.append(f"- **成功请求数**: {phase2_stats['count']}")
            report_lines.append(f"- **平均响应时间**: {phase2_stats['avg_time']:.2f}秒")
            report_lines.append(f"- **最大响应时间**: {phase2_stats['max_time']:.2f}秒")
            report_lines.append(f"- **最小响应时间**: {phase2_stats['min_time']:.2f}秒")
            report_lines.append("")
            report_lines.append("#### 详细结果")
            report_lines.append("")
            for i, result in enumerate(self.test_results["phase2_consult_continuous"]):
                report_lines.append(f"**请求{i+1}**: {result.get('description', result.get('question', '未知'))}")
                report_lines.append(f"- 问题: {result.get('question', 'N/A')}")
                report_lines.append(f"- 是否带上下文: {'是' if result.get('has_context') else '否'}")
                report_lines.append(f"- 响应时间: {result.get('elapsed_time', 0):.2f}秒")
                report_lines.append(f"- 回答长度: {result.get('answer_length', 0)}字符")
                report_lines.append("")

        if stats.get("phase3_report"):
            report_lines.append("### 阶段三：健康报告生成业务连续请求测试")
            report_lines.append("")
            phase3_stats = stats["phase3_report"]
            report_lines.append(f"- **成功请求数**: {phase3_stats['count']}")
            report_lines.append(f"- **平均响应时间**: {phase3_stats['avg_time']:.2f}秒")
            report_lines.append(f"- **最大响应时间**: {phase3_stats['max_time']:.2f}秒")
            report_lines.append(f"- **最小响应时间**: {phase3_stats['min_time']:.2f}秒")
            report_lines.append("")
            report_lines.append("#### 详细结果")
            report_lines.append("")
            for i, result in enumerate(self.test_results["phase3_report_continuous"]):
                report_lines.append(f"**请求{i+1}**: {result.get('task_id', '未知')}")
                report_lines.append(f"- 响应时间: {result.get('elapsed_time', 0):.2f}秒")
                report_lines.append(f"- 报告长度: {result.get('report_length', 0)}字符")
                report_lines.append("")

        if stats.get("phase4_random"):
            report_lines.append("### 阶段四：业务随机交替请求测试")
            report_lines.append("")
            phase4_stats = stats["phase4_random"]
            report_lines.append(f"- **成功请求数**: {phase4_stats['count']}")
            report_lines.append(f"- **平均响应时间**: {phase4_stats['avg_time']:.2f}秒")
            report_lines.append(f"- **最大响应时间**: {phase4_stats['max_time']:.2f}秒")
            report_lines.append(f"- **最小响应时间**: {phase4_stats['min_time']:.2f}秒")
            report_lines.append(f"- **随机序列**: {self.test_results.get('random_sequence', [])}")
            report_lines.append("")
            report_lines.append("#### 详细结果")
            report_lines.append("")
            for i, result in enumerate(self.test_results["phase4_random_alternate"]):
                report_lines.append(f"**请求{i+1}**: {result.get('type', '未知')}")
                if result.get("type") == "consult":
                    report_lines.append(f"- 问题: {result.get('question', 'N/A')}")
                    report_lines.append(f"- 回答长度: {result.get('answer_length', 0)}字符")
                else:
                    report_lines.append(f"- 任务ID: {result.get('task_id', 'N/A')}")
                    report_lines.append(f"- 报告长度: {result.get('report_length', 0)}字符")
                report_lines.append(f"- 响应时间: {result.get('elapsed_time', 0):.2f}秒")
                report_lines.append("")

        report_lines.append("## 测试结论")
        report_lines.append("")
        if self.test_stopped:
            report_lines.append("**测试状态**: ❌ 异常终止")
            report_lines.append(f"**终止原因**: {self.stop_reason}")
        else:
            report_lines.append("**测试状态**: ✅ 全部完成")
            report_lines.append("**结论**: 所有测试请求均成功完成，系统运行稳定。")
        report_lines.append("")

        return "\n".join(report_lines)

    def run(self):
        self.log("开始生产测试...")
        self.log(f"测试时间: {self.start_time}")

        if not self.check_system_health():
            self.stop_test("系统健康检查失败")
            return

        self.run_phase2_consult_continuous()
        self.run_phase3_report_continuous()
        self.run_phase4_random_alternate()

        report = self.generate_report()
        
        report_filename = f"test/report/consult_and_report/production_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.log(f"\n测试报告已生成: {report_filename}")
        print("\n" + "=" * 80)
        print(report)
        print("=" * 80)


if __name__ == "__main__":
    runner = ProductionTestRunner()
    runner.run()
