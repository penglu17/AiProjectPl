import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# ==============================================
# 1. 加载环境变量并初始化 AI 模型
# ==============================================
load_dotenv()

api_key = os.getenv("QIANWEN_API_KEY")
api_base = os.getenv("QWEN_API_URL")

# 校验环境变量
if not api_key or not api_base:
    raise ValueError("请在 .env 文件中正确配置 QIANWEN_API_KEY 和 QWEN_API_URL")

# 初始化 LangChain 的 ChatOpenAI（自动接入 LangSmith 监听）
model = ChatOpenAI(
    model="qwen-plus",
    api_key=api_key,
    base_url=api_base,
    temperature=0.3,
    streaming=True  # 开启流式输出，提升 Gradio 体验
)

# ==============================================
# 2. 核心业务逻辑函数
# ==============================================
def analyze_and_generate_email(customer_info: str) -> str:
    """
    接收客户信息，返回分析报告和销售邮件
    """
    prompt = f"""
    根据以下客户信息完成分析并撰写销售邮件：
    客户信息：{customer_info}

    工作要求：
    1. 梳理客户基本情况、核心痛点与潜在需求
    2. 结合痛点设计对应的产品/服务价值点
    3. 撰写一封正式、有说服力的B2B销售外联邮件
    邮件格式规范：标题、称呼、正文、落款齐全。

    最终输出：客户分析报告 + 完整销售邮件
    """
    
    # 使用 LangChain 标准调用方式
    response = model.invoke([{"role": "user", "content": prompt}])
    return response.content

# ==============================================
# 3. Gradio 聊天交互处理函数
# ==============================================
def chat_respond(message, history):
    """
    处理 Gradio 聊天框的输入，并支持流式输出
    """
    # 如果用户输入包含客户信息特征，则走分析流程；否则作为普通对话
    # 这里为了演示，我们将用户的输入直接作为客户信息传给分析函数
    # 实际场景中你可以加一个路由判断
    try:
        # 使用流式输出 (astream)
        partial_message = ""
        for chunk in model.stream([{"role": "user", "content": message}]):
            if chunk.content:
                partial_message += chunk.content
                yield partial_message
    except Exception as e:
        yield f"发生错误：{str(e)}"

# ==============================================
# 4. 创建并启动 Gradio 界面
# ==============================================
demo = gr.ChatInterface(
    fn=chat_respond,
    title="🤖 AI 销售情报分析助手",
    description="输入目标客户信息，AI 将自动分析痛点并生成 B2B 销售邮件。运行过程由 LangSmith 实时监听。",
    examples=[
        "目标客户：某中小型电商运营公司。主营业务：电商店铺代运营、直播带货托管。当前痛点：团队人手不足，数据分析效率低，缺少自动化工具降本增效。对接人：市场负责人。",
        "目标客户：一家传统制造企业。主营业务：汽车零部件生产。当前痛点：供应链上下游信息不透明，库存积压严重。对接人：供应链总监。"
    ]
)

if __name__ == "__main__":
    demo.launch()