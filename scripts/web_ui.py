import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 1. 加载模型（与Transformers方案一致，复用优化配置）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model_path = "/home/project/qwen3-4B-instruct/base_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="xformers"  # xformers优化显存
)

# 2. 多轮对话函数（适配gradio 5.x新API）
def qwen_chat(message, history):
    # 构建对话历史（保留之前的问答，实现多轮记忆）
    prompt = ""
    for user_msg, assistant_msg in history:
        prompt += f"<s>[INST] {user_msg} [/INST] {assistant_msg}</s>"
    # 添加当前用户输入
    prompt += f"<s>[INST] {message} [/INST]"
    
    # 生成回答（优化显存占用）
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.05,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解析结果，去除prompt前缀
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()

# 3. 创建Web界面（gradio 5.44.1特性：响应式布局+主题）
with gr.Blocks(
    title="Qwen3-4B-Instruct 聊天界面",
    theme=gr.themes.Soft(),  # 柔和主题，长时间使用不刺眼
    css=".gradio-container {max-width: 1200px !important;}"  # 宽屏优化
) as demo:
    gr.Markdown("# Qwen3-4B-Instruct 可视化对话")
    # 对话记录组件（支持复制、滚动）
    chatbot = gr.Chatbot(
        label="对话记录",
        height=500,
        show_copy_button=True,  # 复制回答按钮
        show_share_button=True  # 分享对话按钮（可选）
    )
    # 输入组件（支持回车发送）
    msg = gr.Textbox(
        label="输入问题",
        placeholder="请输入你的问题...",
        width=1000,
        lines=2
    )
    # 按钮组件（清除对话+发送）
    with gr.Row():
        clear_btn = gr.Button("清除对话", variant="secondary")
        submit_btn = gr.Button("发送", variant="primary")
    
    # 事件绑定（适配gradio 5.x API）
    # 回车发送：发送后清空输入框
    msg.submit(qwen_chat, [msg, chatbot], chatbot).then(
        lambda: gr.Textbox(value=""), outputs=msg
    )
    # 按钮发送：同上
    submit_btn.click(qwen_chat, [msg, chatbot], chatbot).then(
        lambda: gr.Textbox(value=""), outputs=msg
    )
    # 清除对话：清空聊天记录
    clear_btn.click(lambda: None, outputs=chatbot, queue=False)

# 4. 启动Web服务（Windows可访问）
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,        # 与Windows端口转发一致
        server_flags=["--timeout 120"]  # 延长超时，避免长回答中断
    )
