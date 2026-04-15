# -*- coding: utf-8 -*-
"""
健康咨询API测试脚本

测试健康咨询API接口是否正常工作。
"""

import requests
import json
import time


BASE_URL = "http://localhost:8001"


def test_health_check():
    """测试健康检查接口"""
    print("\n=== 测试健康检查接口 ===")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"状态码: {response.status_code}")
        print(f"响应数据:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到服务，请确保服务已启动")
        return False
    except Exception as e:
        print(f"错误: {e}")
        return False


def test_root():
    """测试根路径"""
    print("\n=== 测试根路径 ===")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"状态码: {response.status_code}")
        print(f"响应数据:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"错误: {e}")
        return False


def test_consult_api(question: str):
    """测试健康咨询API"""
    print(f"\n=== 测试健康咨询API ===")
    print(f"问题: {question}")
    
    payload = {
        "request_id": f"test_{int(time.time())}",
        "question": question
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/consult",
            json=payload,
            timeout=30
        )
        print(f"状态码: {response.status_code}")
        print(f"响应数据:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到服务，请确保服务已启动")
        return False
    except Exception as e:
        print(f"错误: {e}")
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("MedicalQA API 测试")
    print("=" * 60)
    
    results = []
    
    results.append(("健康检查", test_health_check()))
    results.append(("根路径", test_root()))
    
    test_questions = [
        "头痛怎么办？",
        "糖尿病的症状有哪些？",
        "感冒了吃什么药？"
    ]
    
    for question in test_questions:
        results.append((f"咨询API: {question[:10]}...", test_consult_api(question)))
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    print(f"\n总计: {passed}/{total} 通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
