# -*- coding: utf-8 -*-
"""
vLLM 客户端 - 美观版
启动命令: python vllm_client.py
"""

import requests
import json
import time
import sys
import os
import logging
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import WordCompleter
from colorama import init, Fore, Style as ColoramaStyle
import re

# 初始化
init(autoreset=True)
console = Console()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# 重要修改：使用127.0.0.1替代localhost (针对WSL2)
API_URL = "http://localhost:8000/v1/chat/completions"
HEALTH_URL = "http://localhost:8000/health"
MAX_RETRIES = 5
RETRY_DELAY = 10  # 秒

class VLLMClient:
    def __init__(self):
        self.console = Console()
        self.history = []
        self.style = Style.from_dict({
            'prompt': '#ansicyan bold',
            'input': '#ansigreen',
        })
        self.completer = WordCompleter(['clear', 'exit', 'quit', 'help', 'history'])
        
    def check_service_ready(self):
    #"""检查服务是否已启动 - 简化版"""
    # 临时简化，假设服务总是可用
    #	print("[黄色]跳过服务检查，假设服务已就绪...[/黄色]")
    #	return True

    # --- 原有代码注释掉 ---
     with self.console.status("[bold cyan]正在检查vLLM服务状态...", spinner="dots") as status:
         for i in range(MAX_RETRIES):
             try:
                 response = requests.get(HEALTH_URL, timeout=5)
                 if response.status_code == 200 and '"status":"ok"' in response.text:
                     return True
             except requests.exceptions.RequestException:
                 pass
    
             if i < MAX_RETRIES - 1:
                 status.update(f"[yellow]服务未就绪，等待 {RETRY_DELAY} 秒后重试 ({i+1}/{MAX_RETRIES})...")
                 time.sleep(RETRY_DELAY)
     return False

    def call_vllm_api(self, user_input, stream=False):
        """调用vLLM API获取响应"""
        headers = {"Content-Type": "application/json"}

        messages=[
            {"role": "system", "content": "你是一个有用的助手。"}
        ]

        for item in self.history:
            messages.append({"role": "user", "content": item['user']})
            messages.append({"role": "assistant", "content": item['assistant']})

        messages.append({"role": "user", "content": user_input})

        payload = {
            "model": "Qwen3-4B-Instruct-2507",
            "messages": messages,
            "temperature": 0.8,
            "max_tokens": 1024,
            "stream": stream # 添加 stream 参数
        }
        
        try:
            if stream:
                # 流式请求使用 requests.post 并处理响应流
                response = requests.post(API_URL, headers=headers, json=payload, timeout=120, stream=True)
                response.raise_for_status()
                return response # 返回原始响应对象用于流式处理
            else:
                # 非流式请求保持原样
                with self.console.status("[bold cyan]AI正在思考中...", spinner="dots") as status:
                    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
                    response.raise_for_status()
                    return response.json()
        except requests.exceptions.RequestException as e:
            self.console.print(f"[bold red]API请求失败: {str(e)}[/bold red]")
            if hasattr(e, 'response') and e.response is not None:
                self.console.print(f"[bold red]响应内容: {e.response.text}[/bold red]")
            return None
    
    def display_response(self, user_input, response):
        """显示API响应"""
        if not response or 'choices' not in response or not response['choices']:
            self.console.print("[bold red]无效的API响应[/bold red]")
            return

        assistant_message = response['choices'][0]['message']['content']

        # 添加到历史记录
        self.history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user_input,
            "assistant": assistant_message
        })

        # --- 改进的显示方式 ---
    
        # 1. 简化对话历史显示 (使用普通文本而非表格)
        self.console.print("\n[bold blue]== 对话历史 ==[/bold blue]")
        # 显示最近几条 (例如 3 条)
        recent_history = self.history[-3:] if len(self.history) >= 3 else self.history
        for item in recent_history:
            if item["user"] == user_input: # 当前用户输入
                self.console.print(f"[bold cyan]👤 用户:[/bold cyan] {item['user']}")
            else: # 历史记录
                # 可以考虑截断过长的历史记录
                user_msg = item['user']
                if len(user_msg) > 80:
                    user_msg = user_msg[:77] + "..."
                    self.console.print(f"[dim]👤 用户:[/dim] {user_msg}")
            
        # 显示 AI 的回复标识
        self.console.print("[bold green]🤖 AI:[/bold green]")

        # 2. 以Markdown格式显示AI回复
        self.console.print(Panel(
            Markdown(assistant_message),
            border_style="green",
            padding=(1, 2)
        ))
    
        # 3. 显示使用统计
        usage = response.get('usage', {})
        if usage:
            self.console.print("[bold magenta]使用统计:[/bold magenta]")
            usage_text = (f"提示词: {usage.get('prompt_tokens', 0)} tokens | "
                          f"生成: {usage.get('completion_tokens', 0)} tokens | "
                          f"总计: {usage.get('total_tokens', 0)} tokens")
            self.console.print(usage_text)

        # --- 改进结束 ---

    def display_streaming_response(self, user_input, response):
        """流式显示API响应"""
        if not response:
            self.console.print("[bold red]流式API请求失败[/bold red]")
            return
        
        # 简化对话历史显示
        self.console.print("\n[bold blue]== 对话历史 ==[/bold blue]")
        recent_history = self.history[-2:] if len(self.history) >= 2 else self.history
        for item in recent_history:
            user_msg = item['user']
            if len(user_msg) > 80:
                user_msg = user_msg[:77] + "..."
            self.console.print(f"[dim]👤 用户:[/dim] {user_msg}")
        self.console.print(f"[bold cyan]👤 用户:[/bold cyan] {user_input}")
        self.console.print("[bold green]🤖 AI:[/bold green] ", end="") # 不换行

        # 准备接收流式数据
        full_response = ""
        in_code_block = False
        
        # 逐行读取服务器发送的事件流
        try:
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    
                    # SSE 数据以 "data: " 开头
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:] # 去掉 "data: " 前缀
                        
                        # 检查是否是流结束标记 [DONE]
                        if data_str.strip() == "[DONE]":
                            break
                        
                        try:
                            # 解析 JSON 数据
                            data = json.loads(data_str)
                            
                            # 提取文本内容
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            
                            if content:
                                full_response += content
                                # 直接打印内容，不换行
                                self.console.print(content, end="", markup=False) 
                                # 确保内容立即显示
                                self.console.file.flush() 
                                
                        except json.JSONDecodeError:
                            # 如果不是有效的 JSON，可能是其他 SSE 事件，忽略
                            pass
                            
        except Exception as e:
            self.console.print(f"\n[bold red]流式接收出错: {e}[/bold red]")
        
        # 流结束后换行
        self.console.print("")

        self.history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user_input,
            "assistant": full_response # 使用流式接收到的完整回复
        })

        # 更新历史记录中的完整回复
        if self.history:
            self.history[-1]["assistant"] = full_response
            
        # 可选：显示使用统计（如果流的最后一条消息包含）
        # 这通常比较复杂，因为使用统计在流结束后才返回
        # 可以考虑在非流式请求中获取，或者服务端特殊处理
        

    def show_help(self):
        """显示帮助信息"""
        help_text = """
        [bold cyan]vLLM 客户端帮助[/bold cyan]
        
        [bold yellow]基本命令:[/bold yellow]
        • [green]输入问题[/green] - 与AI进行对话
        • [green]clear[/green] - 清空屏幕
        • [green]exit[/green] 或 [green]quit[/green] - 退出程序
        • [green]history[/green] - 查看对话历史
        • [green]help[/green] - 显示此帮助信息
        
        [bold yellow]提示:[/bold yellow]
        • 支持Markdown格式的输出（粗体、斜体、代码块等）
        • 对话历史会自动保存在内存中
        """
        self.console.print(Panel(Markdown(help_text), title="帮助", border_style="cyan", padding=(1, 2)))
    
    def show_history(self):
        """显示对话历史"""
        if not self.history:
            self.console.print("[yellow]没有对话历史记录[/yellow]")
            return
        
        history_table = Table(show_header=True, header_style="bold magenta")
        history_table.add_column("时间", style="cyan", width=20)
        history_table.add_column("用户消息", style="green")
        history_table.add_column("AI回复", style="blue")
        
        for item in self.history[-5:]:  # 只显示最近5条
            user_msg = item['user'][:50] + "..." if len(item['user']) > 50 else item['user']
            assistant_msg = item['assistant'][:50] + "..." if len(item['assistant']) > 50 else item['assistant']
            history_table.add_row(item['timestamp'], user_msg, assistant_msg)
        
        self.console.print("\n[bold magenta]最近对话历史 (最近5条):[/bold magenta]")
        self.console.print(history_table)
    
    def main(self):
        """主函数"""
        # 显示欢迎信息
        welcome_panel = Panel(
            "[bold cyan]欢迎使用vLLM客户端[/bold cyan]\n\n"
            "这是一个与您的AI模型进行交互的终端界面\n"
            "输入您的问题，AI将为您解答\n\n"
            "[bold yellow]提示:[/bold yellow] 输入 'help' 查看帮助信息",
            title="[bold green]vLLM 客户端[/bold green]",
            border_style="blue",
            padding=(2, 4)
        )
        self.console.print(welcome_panel)
        
        # 检查服务是否就绪
        if not self.check_service_ready():
            self.console.print("[bold red]错误: 无法连接到vLLM服务[/bold red]")
            self.console.print("[yellow]请先运行 'python vllm_serve.py' 启动服务[/yellow]")
            sys.exit(1)
        
        self.console.print("\n[bold green]✓ 服务连接成功! 可以开始对话了[/bold green]")
        
        # 创建输入历史
        history = InMemoryHistory()
        
        # 主循环
        while True:
            try:
                # 获取用户输入
                user_input = prompt(
                    '\n[user:]您:',
                    style=self.style,
                    history=history,
                    completer=self.completer,
                    complete_while_typing=True
                ).strip()
                
                # 处理特殊命令
                if user_input.lower() in ['exit', 'quit']:
                    self.console.print("[cyan]\n再见! 感谢使用vLLM客户端[/cyan]")
                    break
                elif user_input.lower() == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    self.console.print(welcome_panel)
                    self.console.print("\n[bold green]✓ 服务连接成功! 可以开始对话了[/bold green]")
                    continue
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                elif user_input.lower() == 'history':
                    self.show_history()
                    continue
                
                if not user_input:
                    continue
                
                # 调用API
                # --- 修改调用方式 ---
                # 询问用户是否需要流式输出 (可选)
                # use_streaming = input("是否使用流式输出? (y/N): ").strip().lower() == 'y'
                use_streaming = True # 默认使用流式
                
                if use_streaming:
                    response = self.call_vllm_api(user_input, stream=True)
                    if response:
                        self.display_streaming_response(user_input, response)
                else:
                    response = self.call_vllm_api(user_input, stream=False)
                    if response:
                        self.display_response(user_input, response)
                # --- 修改结束 ---
                
            except KeyboardInterrupt:
                self.console.print("\n[cyan]检测到Ctrl+C，输入 'exit' 退出程序[/cyan]")
            except Exception as e:
                self.console.print(f"[bold red]发生错误: {str(e)}[/bold red]")

if __name__ == "__main__":
    # === 诊断代码开始 ===
    import requests
    import sys

# 测试连接
    test_urls = [
        "http://localhost:8000/health",
        "http://127.0.0.1:8000/health",
    # 如果之前获取过WSL2的IP，也可以在这里测试
    # f"http://<YOUR_WSL2_IP>:8000/health"
    ]

    print("=== 网络连接诊断 ===")
    for url in test_urls:
        try:
            print(f"尝试连接: {url} ...")
            response = requests.get(url, timeout=5)
            print(f"  -> 成功! 状态码: {response.status_code}, 响应: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"  -> 失败! 错误: {e}")
        except Exception as e:
            print(f"  -> 发生未预期错误: {e}")

    print("=== 诊断结束 ===\n")
    #sys.exit(0) # 取消注释这行可以只运行诊断部分
# === 诊断代码结束 ===

    client = VLLMClient()
    client.main()
