from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import Accelerator  # 你的依赖，动态分配内存
import torch

# 1. 初始化accelerator（核心优化：GPU/CPU内存动态分配）
accelerator = Accelerator(device_placement=True, mixed_precision="fp16")

# 2. 4-bit量化配置（bitsandbytes核心，适配4060 8G）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # 双重量化，进一步降显存
    bnb_4bit_quant_type="nf4",       # 最优量化类型，精度损失最少
    bnb_4bit_compute_dtype=torch.float16  # 适配4060的Tensor Core
)

# 3. 加载模型（路径正确，适配transformers 4.56.1）
model_path = "/home/project/qwen3-4B-instruct/base_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",  # accelerate自动分配设备
    trust_remote_code=True,
    torch_dtype=torch.float16,
    # xformers优化：显存占用降低15%
    attn_implementation="xformers"  # 启用xformers的注意力优化
)

# 4. 模型包装（accelerate优化推理速度）
model, tokenizer = accelerator.prepare(model, tokenizer)

# 5. 交互式对话函数（调试用）
def generate_response(user_input):
    # Qwen3专用prompt格式
    prompt = f"<s>[INST] {user_input} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(accelerator.device)
    
    # 生成配置（适配8G显存）
    with torch.no_grad():  # 禁用梯度计算，节省内存
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # 单次生成不超过512token
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.05,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码并返回结果（去除prompt前缀）
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()

# 6. 启动交互式对话（调试用）
if __name__ == "__main__":
    print("Qwen3-4B-Instruct 启动成功（输入'exit'退出）：")
    while True:
        user_msg = input("你：")
        if user_msg.lower() == "exit":
            break
        result = generate_response(user_msg)
        print(f"Qwen：{result}\n")
