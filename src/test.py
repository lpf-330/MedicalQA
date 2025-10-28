import requests
import json

# API 服务地址
url = "http://0.0.0.0:8001/v1/medical_rag_stream"

# 要发送的问题
question_data = {
    "question": "好像有点低血压怎么办？"
}

try:
    # 发送 POST 请求，设置 stream=True 以处理流式响应
    response = requests.post(url, json=question_data, stream=True)

    # 检查响应状态码
    if response.status_code == 200:
        print("开始接收流式响应:")
        # 逐行读取响应内容
        for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
            if chunk:
                print(chunk, end='') # 打印接收到的块，不换行
        print("\n流式响应结束。")
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(f"响应内容: {response.text}")

except requests.exceptions.RequestException as e:
    print(f"请求过程中发生错误: {e}")
