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

# 初始化客户端（不使用 wrap_openai）
client = None
if use_ai:
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

# ==============================================
# 3. 加追踪装饰器 + 聊天逻辑
# ==============================================
@traceable(run_type="llm", name="Qwen Chat")
def call_llm(messages):
    """单独抽离调用函数，用于 LangSmith 追踪"""
    response = client.chat.completions.create(
        model="qwen-max",
        messages=messages,
        stream=False
    )
    return response.choices[0].message.content

def chat_response(message, history):
    if not use_ai:
        return "未配置AI密钥，请检查 .env 文件"

    # 统一处理历史格式——兼容 dict 列表和 tuple 列表
    messages = []
    for turn in history:
        if isinstance(turn, dict):
            # Gradio 5 格式
            messages.append({"role": turn["role"], "content": turn["content"]})
        else:
            # Gradio 4 及以下格式 (user, bot) 元组
            user_msg, bot_msg = turn
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": message})

    try:
        return call_llm(messages)
    except Exception as e:
        print("详细错误：", e)
        return f"请求异常：{str(e)}"
# ==============================================
# 4. 界面
# ==============================================
demo = gr.ChatInterface(
    fn=chat_response,
    title="纯文本聊天机器人",
    description="支持多轮对话，已接入 LangSmith 追踪",
    multimodal=False  # 关闭图片功能
)

# ==============================================
# 5. 启动
# ==============================================
if __name__ == "__main__":
    demo.launch()