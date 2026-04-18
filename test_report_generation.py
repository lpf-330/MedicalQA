import requests
import json
from datetime import datetime, timedelta
import random

def generate_test_data():
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
        "task_id": "test-report-001",
        "monitoring_data": monitoring_data,
        "user_profile": user_profile
    }

def test_health_report():
    url = "http://localhost:8001/api/v1/report"
    
    test_data = generate_test_data()
    
    print("=" * 80)
    print("发送健康报告生成请求...")
    print("=" * 80)
    print(f"请求URL: {url}")
    print(f"任务ID: {test_data['task_id']}")
    print(f"监测数据指标: {list(test_data['monitoring_data'].keys())}")
    print(f"用户档案字段: {list(test_data['user_profile'].keys())}")
    print("=" * 80)
    
    try:
        response = requests.post(url, json=test_data, stream=True, timeout=300)
        
        print(f"响应状态码: {response.status_code}")
        print("=" * 80)
        
        if response.status_code == 200:
            print("开始接收流式响应...")
            print("-" * 80)
            
            full_content = ""
            current_event = None
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
                                    print(content, end='', flush=True)
                                    full_content += content
                            except json.JSONDecodeError:
                                pass
                        elif current_event == 'end':
                            print("\n" + "=" * 80)
                            print("流式传输完成")
                            break
                        elif current_event == 'error':
                            print("\n" + "=" * 80)
                            print(f"错误: {data}")
                            break
            
            print("\n" + "=" * 80)
            print(f"完整报告长度: {len(full_content)} 字符")
            print("=" * 80)
            return True
        else:
            print(f"请求失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"请求异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_health_report()
    print(f"\n测试结果: {'成功' if success else '失败'}")
