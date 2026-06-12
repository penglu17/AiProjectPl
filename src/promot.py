# ==============================================
# 1. 导入依赖库
# ==============================================
import gradio as gr
import os
from dotenv import load_dotenv
from openai import OpenAI
from langsmith import traceable

# ==============================================
# 2. 加载环境变量配置
# ==============================================
load_dotenv()

api_key = os.getenv("QIANWEN_API_KEY")
base_url = os.getenv("QWEN_API_URL")
use_ai = bool(api_key and base_url)

# 初始化客户端
client = None
if use_ai:
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

# ==============================================
# 3. LLM 调用 & 追踪
# ==============================================
@traceable(run_type="llm", name="Qwen Chat")
def call_llm(messages):
    response = client.chat.completions.create(
        model="qwen-max",
        messages=messages,
        stream=False
    )
    return response.choices[0].message.content.strip()

# ==============================================
# 4. 核心聊天逻辑
# ==============================================
def chat_response(message, history, instruction, context):
    if not use_ai:
        return "未配置AI密钥，请检查 .env 文件"

    ctx_content = context.strip()
    # 规则1：背景为空，直接返回不知道
    if not ctx_content:
        return "不知道"

    messages = []
    ins_content = instruction.strip()

    # 拼接自定义指令 + 强制规则：仅根据背景回答，找不到就说不知道
    base_rule = "你只能依据提供的参考背景回答问题。如果背景中没有对应答案，请严格只回复：不知道。禁止编造内容、禁止额外解释。"
    if ins_content:
        final_instruction = f"{ins_content}\n{base_rule}"
    else:
        final_instruction = base_rule

    messages.append({"role": "system", "content": final_instruction})
    messages.append({"role": "user", "content": f"参考背景信息：{ctx_content}"})

    # 拼接历史对话
    for turn in history:
        if isinstance(turn, dict):
            messages.append({"role": turn["role"], "content": turn["content"]})
        else:
            user_msg, bot_msg = turn
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})

    # 当前提问
    messages.append({"role": "user", "content": message.strip()})

    try:
        res = call_llm(messages)
        return res
    except Exception as e:
        print("详细错误：", e)
        return f"请求异常：{str(e)}"

# ==============================================
# 5. 界面布局
# ==============================================
with gr.Blocks(title="基于上下文问答工具") as demo:
    gr.Markdown("## 🤖 限定上下文问答\n1. 未填写背景 → 直接回复不知道\n2. 有背景但查不到答案 → 也回复不知道\n👉 示例背景：我叫小明，今年已经20岁啦，爱好是看书和打球")

    with gr.Row():
        instruction = gr.Textbox(
            label="📝 Instruction 额外指令",
            placeholder="自定义角色/语气（选填）",
            lines=3
        )
        context = gr.Textbox(
            label="📚 Context 参考背景",
            placeholder="例如：我叫小明，今年已经20岁啦，爱好是看书和打球",
            lines=3
        )

    chat_ui = gr.ChatInterface(
        fn=chat_response,
        additional_inputs=[instruction, context],
        description="输入问题提问，示例：你今年几岁了 / 你喜欢什么",
        multimodal=False
    )

# ==============================================
# 6. 启动服务
# ==============================================
if __name__ == "__main__":
    demo.launch()