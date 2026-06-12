import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from typing import TypedDict

# ==============================================
# 1. 初始化模型与工具
# ==============================================
load_dotenv()
model = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("QIANWEN_API_KEY"),
    base_url=os.getenv("QWEN_API_URL"),
    temperature=0.3,
)

# 定义文件保存工具
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

# ==============================================
# 2. 定义共享状态 (State)
# ==============================================
class AgentState(TypedDict):
    user_input: str          # 用户原始输入
    analysis: str            # Agent 1 的分析结果
    final_email: str         # Agent 2 的最终邮件
    save_status: str         # 记录文件保存的状态

# ==============================================
# 3. 定义 Agent 1: 情报分析师
# ==============================================
def analyst_agent(state: AgentState):
    print("🔍 [情报分析师] 正在分析客户痛点...")
    prompt = [
        SystemMessage(content="你是一个B2B销售情报分析师。请根据用户提供的客户信息，提炼出核心痛点、业务需求和对接人画像。输出格式要清晰结构化。"),
        HumanMessage(content=state["user_input"])
    ]
    response = model.invoke(prompt)
    return {"analysis": response.content}

# ==============================================
# 4. 定义 Agent 2: 销售文案专家 (带工具)
# ==============================================
def writer_agent(state: AgentState):
    print("✍️ [销售文案专家] 正在撰写开发信并保存文件...")
    prompt = [
        SystemMessage(content=(
            "你是一个顶尖的B2B销售文案专家。请根据【情报分析师】提供的分析结果，撰写一封正式、有说服力的销售外联邮件。\n"
            "撰写完成后，**必须**调用 `save_email_or_document` 工具，将邮件内容保存为 'sales_email.md'。"
        )),
        HumanMessage(content=f"客户分析情报：\n{state['analysis']}\n\n请基于以上情报撰写邮件并保存。")
    ]
    
    # 将工具绑定到模型，并强制模型调用工具
    model_with_tools = model.bind_tools([save_email_or_document])
    response = model_with_tools.invoke(prompt)
    
    # 处理工具调用
    save_result = "未触发文件保存"
    final_email_content = response.content
    
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "save_email_or_document":
                # 执行保存工具
                tool_result = save_email_or_document.invoke(tool_call["args"])
                save_result = tool_result
                # 如果工具参数里有内容，更新最终邮件内容
                final_email_content = tool_call["args"].get("content", final_email_content)
                
    return {
        "final_email": final_email_content, 
        "save_status": save_result
    }

# ==============================================
# 5. 构建 LangGraph 工作流
# ==============================================
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("analyst", analyst_agent)
workflow.add_node("writer", writer_agent)

# 定义边 (分析师 -> 文案专家 -> 结束)
# set_entry_point("节点名") 的作用完全等价于 add_edge(START, "节点名")。它只是为了简化代码、提高可读性而存在的。
workflow.set_entry_point("analyst")
workflow.add_edge("analyst", "writer")
workflow.add_edge("writer", END)

# 编译图
app = workflow.compile()

# ==============================================
# 6. 接入 Gradio 界面
# ==============================================
def chat_respond(message, history):
    try:
        result = app.invoke({"user_input": message})
        yield (
            f"✅ **情报分析已完成，销售邮件已生成！**\n\n"
            f"📄 **文件保存状态：** {result.get('save_status', '无')}\n\n"
            f"---\n\n"
            f"{result['final_email']}"
        )
    except Exception as e:
        yield f"发生错误：{str(e)}"

demo = gr.ChatInterface(
    fn=chat_respond,
    title="🤖 多Agent协作：B2B销售情报与文案团队",
    description="输入客户信息，情报分析师和销售文案专家将接力为你完成工作，并自动保存邮件到本地。",
    examples=[
        "目标客户：某中小型电商运营公司。主营业务：电商店铺代运营。痛点：团队人手不足，数据分析效率低。对接人：市场负责人。"
    ]
)

if __name__ == "__main__":
    print("🚀 正在启动多 Agent 协作系统...")
    demo.launch()