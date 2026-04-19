#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bug修复验证测试脚本
测试修复后的系统是否正常工作
"""

import requests
import json
import time

BASE_URL = "http://localhost:8001"

def test_consult_request(question: str, chat_history: list = None, task_id: str = None):
    """测试健康咨询请求"""
    if chat_history is None:
        chat_history = []
    if task_id is None:
        task_id = f"test-{int(time.time())}"
    
    request_data = {
        "task_id": task_id,
        "question": question,
        "chat_history": chat_history
    }
    
    print(f"\n{'='*60}")
    print(f"测试健康咨询请求: {question}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/consult",
            json=request_data,
            stream=True,
            timeout=120
        )
        
        if response.status_code != 200:
            print(f"❌ 请求失败: HTTP {response.status_code}")
            return False
        
        full_content = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    data = decoded_line[6:]
                    try:
                        json_data = json.loads(data)
                        if 'content' in json_data:
                            full_content += json_data['content']
                    except:
                        pass
        
        elapsed = time.time() - start_time
        print(f"✅ 请求成功")
        print(f"   耗时: {elapsed:.2f}秒")
        print(f"   回答长度: {len(full_content)}字符")
        print(f"   回答预览: {full_content[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ 请求异常: {str(e)}")
        return False

def test_report_request(task_id: str = None):
    """测试健康报告生成请求"""
    if task_id is None:
        task_id = f"report-test-{int(time.time())}"
    
    request_data = {
        "task_id": task_id,
        "monitoring_data": {
            "heart_rate": {"latest": [{"value": 75, "time": "2026-04-19 12:00:00"}]},
            "blood_pressure": {"latest": [{"systolic": 120, "diastolic": 80, "time": "2026-04-19 12:00:00"}]}
        },
        "user_profile": {
            "user_id": 1001,
            "gender": "男",
            "birth_date": "1985-03-15"
        }
    }
    
    print(f"\n{'='*60}")
    print(f"测试健康报告生成请求: {task_id}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/report",
            json=request_data,
            stream=True,
            timeout=300
        )
        
        if response.status_code != 200:
            print(f"❌ 请求失败: HTTP {response.status_code}")
            return False
        
        full_content = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    data = decoded_line[6:]
                    try:
                        json_data = json.loads(data)
                        if 'content' in json_data:
                            full_content += json_data['content']
                    except:
                        pass
        
        elapsed = time.time() - start_time
        print(f"✅ 请求成功")
        print(f"   耗时: {elapsed:.2f}秒")
        print(f"   报告长度: {len(full_content)}字符")
        return True
        
    except Exception as e:
        print(f"❌ 请求异常: {str(e)}")
        return False

def main():
    print("="*60)
    print("Bug修复验证测试")
    print("="*60)
    
    results = []
    
    # 测试1: 第一个健康咨询请求
    result1 = test_consult_request("头痛怎么办？")
    results.append(("健康咨询请求1", result1))
    time.sleep(2)
    
    # 测试2: 第二个健康咨询请求（测试自动重新初始化）
    result2 = test_consult_request("感冒了吃什么药？")
    results.append(("健康咨询请求2", result2))
    time.sleep(2)
    
    # 测试3: 第一个健康报告生成请求
    result3 = test_report_request("report-test-001")
    results.append(("健康报告请求1", result3))
    time.sleep(2)
    
    # 测试4: 第二个健康报告生成请求（测试自动重新初始化）
    result4 = test_report_request("report-test-002")
    results.append(("健康报告请求2", result4))
    
    # 打印结果汇总
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + "="*60)
    if all_passed:
        print("✅ 所有测试通过！Bug修复成功！")
    else:
        print("❌ 部分测试失败，请检查日志")
    print("="*60)

if __name__ == "__main__":
    main()
