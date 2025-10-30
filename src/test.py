# test_openai_rag_client.py

import requests
import json
import time
from typing import Dict, Any, List, Optional

# --- 配置 ---
API_URL_STREAM = "http://localhost:8001/v1/medical_rag_stream"
API_URL_CHAT = "http://localhost:8001/v1/chat/completions" # 原有 OpenAI 兼容端点 (如有需要)
HEADERS = {
    "Authorization": "Bearer not_needed_for_now", # 如果服务端需要，可以添加 Bearer Token
    "Content-Type": "application/json"
}

# --- 辅助函数 ---
def send_rag_request(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    stream: bool = True,
    session_id: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 1024
) -> Any:
    """
    向 RAG 服务发送请求。

    Args:
        question (str): 用户的当前问题。
        history (Optional[List[Dict[str, str]]]): 对话历史，格式为 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]。
        stream (bool): 是否使用流式响应。
        session_id (Optional[str]): 会话ID，通过 Header 传递。
        temperature (float): 生成温度。
        max_tokens (int): 最大生成 token 数。

    Returns:
        Any: 如果 stream=False，返回完整响应；如果 stream=True，返回响应对象用于迭代。
    """
    # 构造 messages 数组
    messages = history if history else []
    messages.append({"role": "user", "content": question})

    # 构造请求体 (Payload) - 符合 OpenAI ChatCompletionRequest 格式
    payload = {
        "model": "Qwen3-4B-Instruct-2507", # 使用与服务端配置一致的模型名
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
        "max_tokens": max_tokens
        # 可以添加其他 OpenAI API 支持的参数，如 top_p, frequency_penalty 等
    }

    # 准备 Headers
    request_headers = HEADERS.copy()
    if session_id:
        request_headers["X-Session-ID"] = session_id
    # 添加时间戳 (Unix timestamp in seconds)
    request_headers["X-Timestamp"] = str(int(time.time()))

    print(f"Sending request to {API_URL_STREAM}")
    print(f"Headers: {request_headers}")
    print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    print("-" * 20)

    try:
        response = requests.post(API_URL_STREAM, headers=request_headers, json=payload, stream=stream, timeout=120)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response Status Code: {e.response.status_code}")
            print(f"Response Text: {e.response.text}")
        return None

# --- 测试函数 ---
def test_non_streaming(question: str, history: Optional[List[Dict[str, str]]] = None, session_id: Optional[str] = None):
    """测试非流式响应"""
    print(f"\n--- Testing Non-Streaming for question: '{question}' ---")
    response = send_rag_request(question, history, stream=False, session_id=session_id)
    if response:
        try:
            # 非流式响应直接返回 JSON
            data = response.json()
            print("Received non-streaming response:")
            # print(json.dumps(data, indent=2, ensure_ascii=False)) # 打印完整响应
            # 提取并打印核心回答
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0].get("message", {}).get("content", "")
                print(f"Answer:\n{content}\n")
            else:
                print("No 'choices' found in response.")
        except requests.exceptions.JSONDecodeError:
            print(f"Failed to decode JSON response. Raw text: {response.text}")
        except KeyError as e:
            print(f"KeyError accessing response data: {e}. Response: {data}")

def test_streaming(question: str, history: Optional[List[Dict[str, str]]] = None, session_id: Optional[str] = None):
    """测试流式响应"""
    print(f"\n--- Testing Streaming for question: '{question}' ---")
    response = send_rag_request(question, history, stream=True, session_id=session_id)
    if response:
        print("Answer (Streaming):")
        accumulated_content = ""
        # 处理流式响应 (Server-Sent Events)
        # 每一行通常是 "data: {...}\n\n" 或 " [DONE]\n\n"
        try:
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    # print(f"Raw line: {line}") # 调试用
                    if line.startswith(" "):
                        data_str = line[len(" "):]
                        if data_str.strip() == "[DONE]":
                            print("\n[Stream Ended]")
                            break
                        try:
                            chunk_data = json.loads(data_str)
                            # 提取 content
                            choices = chunk_data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    print(content, end="", flush=True) # 实时打印，不换行
                                    accumulated_content += content
                        except json.JSONDecodeError:
                            print(f"\n[Warning: Could not decode JSON line: {data_str}]")
                    # else:
                    #     # 忽略非 data: 开头的行 (如重试、事件名等)
                    #     pass
            print("\n--- End of Stream ---\n")
        except Exception as e:
            print(f"\n[Error during streaming: {e}]\n")

# --- 主程序 ---
if __name__ == "__main__":
    # 示例 Session ID
    test_session_id = "test_session_123abc"

    # --- 测试 1: 简单问题 (非流式) ---
    simple_question = "阿司匹林片是什么？要怎么吃？"
    # test_non_streaming(simple_question, session_id=test_session_id)

    # --- 测试 2: 简单问题 (流式) ---
    test_streaming(simple_question, session_id=test_session_id)

    # --- 测试 3: 带历史的对话 (流式) ---
    # history_example = [
    #     {"role": "user", "content": "帮我查一下高血压的相关信息。"},
    #     {"role": "assistant", "content": "好的，我可以帮您查询高血压的定义、症状、治疗方法等信息。请问您具体想了解哪方面？"}
    # ]
    # follow_up_question = "我想知道它的常见并发症。"
    # test_streaming(follow_up_question, history=history_example, session_id=test_session_id)

    # --- 测试 4: 另一个简单问题 (流式) ---
    another_question = "糖尿病患者宜吃哪些食物？"
    # test_streaming(another_question, session_id=test_session_id)

    print("\n--- All tests completed. ---")
