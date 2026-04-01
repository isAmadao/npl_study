

import base64
import os
import sys
import fitz
from openai import OpenAI



def pdf_first_page_to_base64(pdf_path: str, dpi: int = 150) -> str:
    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return base64.b64encode(img_bytes).decode("utf-8")


def parse_pdf_with_qwen_vl(pdf_path: str, prompt: str = "请详细描述这一页的内容") -> str:
    api_key = API_KEY
    if not api_key:
        raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")


    img_b64 = pdf_first_page_to_base64(pdf_path)
    image_url = f"data:image/png;base64,{img_b64}"


    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    response = client.chat.completions.create(
        model="qwen-vl-plus",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    default_pdf = os.path.join(
        os.path.dirname(__file__),
        "../../Week10-多模态大模型.pdf",
    )
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else default_pdf

    if not os.path.exists(pdf_file):
        print(f"文件不存在：{pdf_file}")
        sys.exit(1)

    result = parse_pdf_with_qwen_vl(
        pdf_file,
        prompt="请提取并结构化这一页中的所有文字内容，包括标题、正文、图注等。",
    )

    print("\n===== Qwen-VL 解析结果 =====")
    print(result)
