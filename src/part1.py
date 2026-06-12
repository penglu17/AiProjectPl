# ==============================================
# 1. 导入依赖库
# ==============================================
import gradio as gr
import cv2
import numpy as np
from PIL import Image
import os
import base64
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from langsmith.wrappers import wrap_openai

# ==============================================
# 2. 加载环境变量配置
# ==============================================
load_dotenv()

# ==============================================
# 3. 定义图像转铅笔画工具函数
# ==============================================
def image_to_pencil_drawing(input_img: Image.Image) -> tuple[Image.Image, int, int]:
    img = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)
    height, width = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, (21, 21), 0)
    pencil = cv2.divide(gray, gauss, scale=256.0)
    pencil = cv2.normalize(pencil, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    result_img = Image.fromarray(pencil)
    return result_img, width, height

# ==============================================
# 4. 加载配置 & 初始化可追踪的 OpenAI 客户端
# ==============================================
api_key = os.getenv("QIANWEN_API_KEY")
base_url = os.getenv("QWEN_API_URL")
use_ai = bool(api_key and base_url)

# 包装 OpenAI 客户端，自动接入 LangSmith 追踪
client = None
if use_ai:
    raw_client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    # 关键：wrap_openai 开启追踪
    client = wrap_openai(raw_client)

# ==============================================
# 5. AI 图文描述函数（多模态 + 可追踪）
# ==============================================
def get_image_description(pil_img, prompt_text):
    # 图片转 base64
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # 多模态消息格式（千问/通义兼容）
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                }
            ]
        }
    ]

    # 调用模型（会被 LangSmith 自动捕获）
    try:
        response = client.chat.completions.create(
            model="qwen-max",  # 可切换 qwen-turbo / qwen-max
            messages=messages,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI 调用异常：{str(e)}"

# ==============================================
# 6. 核心处理函数
# ==============================================
def process_image(input_img):
    print("收到图片，开始转换铅笔画...")
    try:
        pencil_img, img_w, img_h = image_to_pencil_drawing(input_img)

        if use_ai:
            ai_desc = get_image_description(pencil_img, "请简单描述这张铅笔画图片的内容，语言简洁自然")
        else:
            ai_desc = "未配置AI密钥，仅完成图像转换"

        return pencil_img, img_w, img_h, ai_desc

    except Exception as e:
        print(f"错误：{str(e)}")
        return None, 0, 0, f"图片处理失败：{str(e)}"

# ==============================================
# 7. 创建 Gradio 界面
# ==============================================
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="上传原图", format="png"),
    outputs=[
        gr.Image(label="铅笔画效果图1"),
        gr.Number(label="图像宽度(px)"),
        gr.Number(label="图像高度(px)"),
        gr.Textbox(label="AI 图文描述")
    ],
    title="图像转铅笔画工具",
    description="上传任意图片，一键生成素描铅笔画风格图像，支持AI内容描述1.0版本"
)

# ==============================================
# 8. 启动服务
# ==============================================
if __name__ == "__main__":
    demo.launch()