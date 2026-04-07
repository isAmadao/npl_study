import base64
import os
from io import BytesIO

import fitz
from PIL import Image
from openai import OpenAI
open_client = OpenAI(
    api_key="sk-8fb3abb209d34b1a89932c3ced430028",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# ========== PDF 第一页转图片（Base64格式）==========
def pdf_first_page_to_base64(pdf_path: str, dpi: int = 150) -> str:
    """
    将 PDF 第一页转为 Base64 编码的图片
    dpi 参数控制分辨率，越高越清晰但 token 消耗越大
    """
    # 打开 PDF
    doc = fitz.open(pdf_path)

    # 加载第一页（索引为0）
    page = doc.load_page(0)

    # 设置缩放矩阵，控制输出图片分辨率
    # dpi=150 时，矩阵系数约为 150/72 ≈ 2.08
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    # 渲染为 pixmap
    pix = page.get_pixmap(matrix=mat)

    # 转换为 PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # 转为 Base64
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    doc.close()
    return img_base64


# ========== 调用 Qwen-VL 进行解析 ==========
def parse_pdf_with_qwen_vl(
        pdf_path: str,
        prompt: str = "请详细描述这张图片中的内容，包括文字、表格、图表等所有信息。",
        model: str = "qwen-vl-max",
        dpi: int = 150,
) -> str:
    """
    使用 Qwen-VL 解析 PDF 第一页
    """
    # 获取图片 Base64
    img_base64 = pdf_first_page_to_base64(pdf_path, dpi)

    # 构造请求消息（支持 OpenAI 兼容格式）
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url":{"url": f"data:image/png;base64,{img_base64}"}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    try:
        # 调用 DashScope API
        response = open_client.chat.completions.create(
            model=model,
            messages=messages,
        )
        # 成功时直接提取内容
        result_text = response.choices[0].message.content
        return result_text
    except Exception as e:
        raise Exception(f"API 调用失败: {e}")


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 替换为你的 PDF 文件路径
    pdf_file = "./llm.pdf"

    if not os.path.exists(pdf_file):
        print(f"文件不存在: {pdf_file}")
    else:
        try:
            # 通用解析
            result = parse_pdf_with_qwen_vl(
                pdf_file,
                prompt="将图⽚内容提取为 Markdown 格式，输出关键信息。"
            )
            print("=== 解析结果 ===")
            print(result)
        except Exception as e:
            print(f"解析失败: {e}")