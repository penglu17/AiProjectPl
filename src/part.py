# ==============================================
# 1. 导入依赖库
# ==============================================
import gradio as gr                    # 导入 Gradio 库，用于创建可视化界面
from langchain_openai import ChatOpenAI  # 导入 LangChain 的 OpenAI 兼容模型接口
import os                              # 导入操作系统模块，用于访问环境变量
from dotenv import load_dotenv          # 导入 dotenv 库，用于加载 .env 文件



# ==============================================
# 2. 加载环境变量配置
# ==============================================
load_dotenv()                          # 从 .env 文件加载环境变量到系统中



# ==============================================
# 3. 定义天气查询工具函数
# ==============================================
def get_weather(city: str) -> tuple[str, int]:
    """
    查询指定城市的当前天气（模拟实现）
    
    参数:
        city (str): 要查询天气的城市名称
        
    返回:
        tuple[str, int]: (天气信息字符串, 城市名称长度)
    """
    return f"{city} 今日天气：晴，气温22°C，湿度60%（城市名长度：{len(city)}）", len(city)  # 返回模拟的天气信息



# ==============================================
# 4. 校验环境变量并初始化 AI 模型
# ==============================================
api_key = os.getenv("QIANWEN_API_KEY")   # 从环境变量获取千问 API 密钥
base_url = os.getenv("QWEN_API_URL")     # 从环境变量获取千问 API 地址
print(api_key, base_url)                 # 打印环境变量值（调试用）


# 检查 API 密钥和地址是否有效
if not api_key or not base_url:
    print("警告：环境变量未设置，将使用模拟模式")  # 提示用户使用模拟模式
    use_ai = False                                    # 标记不使用 AI
else:
    use_ai = True                                     # 标记使用 AI
    # 初始化千问大语言模型
    model = ChatOpenAI(
        model="qwen-turbo",           # 指定使用的模型名称
        api_key=api_key,              # 设置 API 密钥
        streaming=False,              # 禁用流式输出
        base_url=base_url             # 设置 API 访问地址
    )



# ==============================================
# 5. 核心聊天函数 - 处理用户请求
# ==============================================
def chat_with_agent(user_input):
    """
    处理用户输入并返回响应
    
    参数:
        user_input (str): 用户输入的问题
        
    返回:
        tuple[str, int]: (AI 回复, 查询城市数量)
    """
    print(f"收到请求: {user_input}")   # 打印用户请求（调试用）
    
    try:
        # 判断用户是否询问天气相关问题
        if "天气" in user_input:
            # 定义支持查询的城市列表
            cities = ["北京", "上海", "广州", "深圳", "杭州", "成都", "南京", "武汉", 
                      "无锡", "苏州", "宁波", "长沙", "重庆", "西安", "天津", "青岛"]
            
            # 从用户输入中提取匹配的城市
            found_cities = [city for city in cities if city in user_input]
            
            if found_cities:
                # 调用天气工具获取每个城市的天气信息
                weather_results = [get_weather(city)[0] for city in found_cities]
                weather_text = "\n".join(weather_results)  # 拼接天气信息
                city_count = len(found_cities)  # 查询的城市数量
                city_len_info = [get_weather(city)[1] for city in found_cities]
                if use_ai:
                    # 使用 AI 模型将天气信息整理成自然语言回复
                    prompt = f"请将以下天气信息用自然友好的语言回复用户：\n{weather_text}\n\n用户问题：{user_input}"
                    response = model.invoke([{"role": "user", "content": prompt}])
                    return response.content, city_count,sum(city_len_info)  # 返回 AI 回复和城市数量
                else:
                    return weather_text, city_count,sum([get_weather(city)[1] for city in found_cities])  # 直接返回天气信息和城市数量
            else:
                # 用户提到天气但没有指定城市
                if use_ai:
                    # 使用 AI 友好地追问用户具体城市
                    response = model.invoke([{"role": "user", "content": f"用户问：{user_input}，请友好地询问用户想查询哪个城市的天气"}])
                    return response.content, 0
                else:
                    return "请问你想查询哪个城市的天气呢？", 0  # 模拟追问
        else:
            # 用户询问非天气问题
            if use_ai:
                # 直接调用 AI 模型回答
                response = model.invoke([{"role": "user", "content": user_input}])
                return response.content, 0
            else:
                return "这是一个天气查询助手，请输入天气相关的问题（需要配置 AI 环境变量才能回答其他问题）", 0
    
    except Exception as e:
        # 捕获并处理异常
        print(f"错误: {str(e)}")       # 打印错误信息（调试用）
        return f"发生错误：{str(e)}", 0    # 返回错误信息给用户



# ==============================================
# 6. 创建 Gradio 可视化界面
# ==============================================
demo = gr.Interface(
    fn=chat_with_agent,                          # 指定处理用户输入的函数
    inputs=gr.Textbox(                           # 输入组件：文本框
        label="输入问题",                         # 标签显示文字
        placeholder="例如：无锡今天天气怎么样？"   # 占位提示文字
    ),
    outputs=[                                    # 输出组件：多个输出
        gr.Textbox(label="AI 回复"),              # 第一个输出：文本
        gr.Number(label="查询城市数量"),            # 第二个输出：数字
        gr.Number(label="城市名总长度"),   # 新增输出：数字
    ],
    title="AI Agent 天气查询助手"                 # 界面标题
)



# ==============================================
# 7. 启动 Gradio 服务
# ==============================================
demo.launch()  # 启动界面，默认在 http://localhost:7860 运行