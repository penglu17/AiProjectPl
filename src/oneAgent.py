
import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool  # 注意：tool 装饰器现在从 langchain_core.tools 导入

# ==============================================
# 1. 加载环境变量并初始化 AI 模型
# ==============================================
load_dotenv()

api_key = os.getenv("QIANWEN_API_KEY")
api_base = os.getenv("QWEN_API_URL")
print("环境变量加载完成，API Key:", api_key[:5] + "..." if api_key else "未找到")
print("环境变量加载完成，API Base:", api_base if api_base else "未找到")
if not api_key or not api_base:
    raise ValueError("请在 .env 文件中正确配置 QIANWEN_API_KEY 和 QWEN_API_URL")

model = ChatOpenAI(
    model="qwen-plus",
    api_key=api_key,
    base_url=api_base,
    temperature=0.3,
)

# ==============================================
# 2. 定义自定义工具：保存邮件/文档
# ==============================================
@tool
def save_email_or_document(content: str, filename: str) -> str:
    """
    当需要保存生成的销售邮件、分析报告或任何文档时调用此工具。
    
    Args:
        content (str): 需要保存的邮件或文档的完整文本内容。
        filename (str): 保存的文件名（需包含后缀，如 'sales_email.txt' 或 'report.md'）。
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return f"文件 '{filename}' 已成功保存到本地！"
    except Exception as e:
        return f"保存文件时发生错误: {str(e)}"

# 将工具注册到列表中
tools = [save_email_or_document]

# ==============================================
# 3. 构建新版 Agent
# ==============================================
# 定义系统提示词
system_prompt = (
    "你是一个专业的 B2B 销售情报分析助手。你的工作流程是：\n"
    "1. 分析客户信息，梳理痛点与需求。\n"
    "2. 撰写正式、有说服力的 B2B 销售外联邮件。\n"
    "3. 必须使用 `save_email_or_document` 工具将生成的完整内容（包含分析报告和邮件）保存为文件。\n"
    "4. 最后向用户汇报保存结果并展示邮件内容。"
)

# 使用新版 API 初始化 Agent（无需再手动创建 PromptTemplate 和 AgentExecutor）
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
)

# ==============================================
# 4. 接入 Gradio 可视化界面
# ==============================================
def chat_respond(message, history):
    """
    处理 Gradio 聊天框的输入，支持流式输出
    """
    partial_message = ""
    try:
        # 新版 Agent 的流式处理
        for chunk in agent.stream({"messages": [{"role": "user", "content": message}]}):
            # 新版返回的是包含 'messages' 的状态字典，提取 AI 的输出
            if "messages" in chunk:
                for msg in chunk["messages"]:
                    # 严格检查：确保 msg 有 content 属性，且是字符串，且不为空
                    if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content:
                        partial_message += msg.content
                        # 每次拼接后，必须 yield 当前累积的完整消息
                        yield partial_message
        
        # 【关键防御】：如果循环结束但 partial_message 依然为空（比如AI只返回了工具调用没返回文本）
        # 必须 yield 一个默认提示，防止 Gradio 抛出 StopAsyncIteration
        if not partial_message:
            yield "AI 已完成工具调用，但未生成最终文本回复。"
            
    except Exception as e:
        # 捕获其他可能的异常，并在控制台打印真实错误
        import traceback
        print(f"\n🚨 Agent 运行发生错误: {type(e).__name__} - {str(e)}")
        traceback.print_exc()
        yield f"发生错误：{str(e)}"
demo = gr.ChatInterface(
    fn=chat_respond,
    title="🤖 AI 销售情报分析助手 (带工具)",
    description="输入目标客户信息，AI 将自动分析痛点、生成 B2B 销售邮件，并自动保存为本地文件。",
    examples=[
        "目标客户：某中小型电商运营公司。主营业务：电商店铺代运营、直播带货托管。当前痛点：团队人手不足，数据分析效率低，缺少自动化工具降本增效。对接人：市场负责人。请帮我写封开发信并保存。",
        "目标客户：一家传统制造企业。主营业务：汽车零部件生产。当前痛点：供应链上下游信息不透明，库存积压严重。对接人：供应链总监。请生成跟进邮件并保存为 report.md。"
    ]
)

if __name__ == "__main__":
    print("🚀 正在启动 AI 销售情报分析助手...")
    demo.launch()

