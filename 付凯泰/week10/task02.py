import os
import io
import base64
import fitz  # PyMuPDF
from openai import OpenAI


def pdf_first_page_to_base64(pdf_path: str, zoom: float = 2.0) -> str:
    """
    把本地 PDF 的第一页渲染成 PNG，并转成 base64 字符串
    :param pdf_path: 本地 PDF 路径
    :param zoom: 渲染清晰度，2.0 一般够用；文字很多可调到 3.0
    :return: PNG 图片的 base64 字符串
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)  # 第 1 页，索引从 0 开始
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    png_bytes = pix.tobytes("png")
    doc.close()

    return base64.b64encode(png_bytes).decode("utf-8")


def parse_pdf_first_page_with_qwenvl(pdf_path: str):
    """
    使用阿里云百炼的 Qwen-VL 解析本地 PDF 的第一页
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("未检测到环境变量 DASHSCOPE_API_KEY，请先配置 API Key。")

    # 1) 本地 PDF 第 1 页 -> base64 PNG
    base64_image = pdf_first_page_to_base64(pdf_path, zoom=2.0)

    # 2) 初始化阿里云百炼 OpenAI 兼容客户端
    client = OpenAI(
        api_key='sk-b220c23d00d84424a07f2cf000e48920',
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # 3) 调用 Qwen-VL
    response = client.chat.completions.create(
        model="qwen3-vl-plus",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            "请解析这份 PDF 的第一页，并按下面格式输出：\n"
                            "1. 页面主题/标题\n"
                            "2. 主要内容概述\n"
                            "3. 提取页面中能识别到的关键文字\n"
                            "4. 如果有表格、图片、公式或结构化信息，请单独说明\n"
                            "5. 用中文回答"
                        )
                    }
                ]
            }
        ]
    )

    print("模型解析结果：\n")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    pdf_path = r"D:\桌面\笔记\软件学院-软件工程-基于SpringBoot的旅游网站的设计与实现.pdf"   
    parse_pdf_first_page_with_qwenvl(pdf_path)
