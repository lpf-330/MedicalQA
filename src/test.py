# test_medical_rag_client.py

import requests
import json
import time
from typing import Dict, Any, List, Optional
import uuid

# --- 1. 配置 ---
API_BASE_URL = "http://localhost:8001"  # 与 rag_fastapi_service.py 中定义的端口保持一致
MEDICAL_RAG_STREAM_ENDPOINT = f"{API_BASE_URL}/v1/medical_rag_stream"
CHAT_COMPLETIONS_ENDPOINT = f"{API_BASE_URL}/v1/chat/completions" # 原有 OpenAI 兼容端点 (如有需要)

HEADERS = {
    "Authorization": "Bearer not_needed_for_now", # 如果服务端需要，可以添加 Bearer Token
    "Content-Type": "application/json"
}

# --- 2. 辅助函数 ---

def send_medical_rag_request(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    stream: bool = True,
    session_id: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    model: str = "Qwen3-4B-Instruct-2507" # 与服务端配置的模型名保持一致
) -> Any:
    """
    向 Medical RAG 服务发送请求。

    Args:
        question (str): 用户的当前问题。
        history (Optional[List[Dict[str, str]]]): 对话历史，格式为 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]。
        stream (bool): 是否使用流式响应。
        session_id (Optional[str]): 会话ID，通过 Header 传递。
        temperature (float): 生成温度。
        max_tokens (int): 最大生成 token 数。
        model (str): 模型名称。

    Returns:
        Any: 如果 stream=False，返回完整响应对象；如果 stream=True，返回响应对象用于迭代。
    """
    # --- 构造 messages 数组 ---
    messages = history if history else []
    # 将当前问题作为新的 user 消息添加到历史末尾
    messages.append({"role": "user", "content": question})

    # --- 构造请求体 (Payload) - 符合 OpenAI ChatCompletionRequest 格式 ---
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
        "max_tokens": max_tokens
        # 可以添加其他 OpenAI API 支持的参数，如 top_p, frequency_penalty 等
    }

    # --- 准备 Headers ---
    request_headers = HEADERS.copy()
    if not session_id:
        session_id = f"test_session_{uuid.uuid4().hex[:8]}" # 生成一个默认的 Session ID
    request_headers["X-Session-ID"] = session_id
    # 添加时间戳 (Unix timestamp in seconds)
    request_headers["X-Timestamp"] = str(int(time.time()))

    print(f"--- Sending Medical RAG Request ---")
    print(f"URL: {MEDICAL_RAG_STREAM_ENDPOINT}")
    print(f"Session ID: {session_id}")
    print(f"Headers: {request_headers}")
    print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    print("-" * 40)

    try:
        response = requests.post(MEDICAL_RAG_STREAM_ENDPOINT, headers=request_headers, json=payload, stream=stream, timeout=120)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return response, session_id # 返回响应对象和 session_id
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response Status Code: {e.response.status_code}")
            print(f"Response Text: {e.response.text}")
        return None, session_id

# --- 3. 测试函数 ---

def test_non_streaming(question: str, history: Optional[List[Dict[str, str]]] = None, session_id: Optional[str] = None):
    """测试非流式响应"""
    print(f"\n--- Testing Non-Streaming for question: '{question[:50]}...' ---")
    response_obj, sid = send_medical_rag_request(question, history, stream=False, session_id=session_id)
    if response_obj:
        try:
            # 非流式响应直接返回 JSON
            data = response_obj.json()
            print("Received non-streaming response:")
            # print(json.dumps(data, indent=2, ensure_ascii=False)) # 打印完整响应
            # 提取并打印核心回答
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0].get("message", {}).get("content", "")
                print(f"Answer:\n{content}\n")
            else:
                print("No 'choices' found in response.")
        except requests.exceptions.JSONDecodeError:
            print(f"Failed to decode JSON response. Raw text: {response_obj.text}")
        except KeyError as e:
            print(f"KeyError accessing response data: {e}. Response: {data}")

def test_streaming(question: str, history: Optional[List[Dict[str, str]]] = None, session_id: Optional[str] = None):
    """测试流式响应"""
    print(f"\n--- Testing Streaming for question: '{question[:50]}...' ---")
    response_obj, sid = send_medical_rag_request(question, history, stream=True, session_id=session_id)
    if response_obj:
        print("Answer (Streaming):")
        accumulated_content = ""
        # 处理流式响应 (Server-Sent Events)
        # 每一行通常是 " ...\n\n" 或 " [DONE]\n\n"
        try:
            for line in response_obj.iter_lines(decode_unicode=True):
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
                                finish_reason = choices[0].get("finish_reason", None)
                                if content or finish_reason: # 只发送有内容的部分
                                    print(content, end="", flush=True) # 实时打印，不换行
                                    accumulated_content += content
                        except json.JSONDecodeError:
                            print(f"\n[Warning: Could not decode JSON line: {data_str[:100]}...]")
                    # else:
                    #     # 忽略非 data: 开头的行 (如 retry:, event:, id: 等)
                    #     pass
            print("\n--- End of Stream ---\n")
            # print(f"Accumulated Content:\n{accumulated_content}\n") # 打印累积的完整内容
        except Exception as e:
            print(f"\n[Error during streaming: {e}]\n")
    else:
        print(f"Failed to get streaming response for session {sid}.")

# --- 4. 主程序 ---
if __name__ == "__main__":
    # --- 示例 Session ID ---
    test_session_id = f"test_session_{uuid.uuid4().hex[:8]}"

    # --- 测试 1: 简单问题 (非流式) ---
    # simple_question = "高血压的常见症状有哪些？"
    # test_non_streaming(simple_question, session_id=test_session_id)

    # --- 测试 2: 简单问题 (流式) ---
    # test_streaming(simple_question, session_id=test_session_id)

    # --- 测试 3: 带历史的对话 (流式) ---
    history_example = [
        {"role": "user", "content": "帮我查一下高血压的相关信息。"},
        {"role": "assistant", "content": "好的，我可以帮您查询高血压的定义、症状、治疗方法等信息。请问您具体想了解哪方面？"}
    ]
    follow_up_question = "我想知道它有哪些具体表现。"
    test_streaming(follow_up_question, history=history_example, session_id=test_session_id)

    # --- 测试 4: 另一个简单问题 (流式) ---
    # another_question = "糖尿病患者宜吃哪些食物？"
    # test_streaming(another_question, session_id=test_session_id)

    print("\n--- All tests completed. ---")
