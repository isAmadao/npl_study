#作业2: 使用云端的Qwen-VL 对本地的pdf（任意pdf的第一页） 进行解析，写一下这个代码；
# https://help.aliyun.com/zh/model-studio/visual-reasoning
import os
import fitz


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL_NAME = "qwen-vl-max"
from dashscope import MultiModalConversation
import dashscope

# ⚠️ 在这里填入你的 API Key
dashscope.api_key =OPENAI_API_KEY


def analyze_game_image(image_path, prompt=None):
    if not os.path.exists(image_path):
        print(f"❌ 文件不存在：{image_path}")
        return None

    if prompt is None:
        prompt = """
        请详细描述这张图片？
        """

    print(f"🎮 分析：{os.path.basename(image_path)}")

    messages = [{
        'role': 'user',
        'content': [
            {'image': f'file://{image_path}'},
            {'text': prompt}
        ]
    }]

    try:
        response = MultiModalConversation.call(
            model='qwen-vl-max',
            messages=messages
        )

        result = response.output.choices[0].message.content
        print("\n" + "=" * 70)
        print(result)
        print("=" * 70)

        return result

    except Exception as e:
        print(f"❌ 失败：{e}")
        return None

#测试发现 qqwn-vl 没有直接支持上传pdf的操作 ，改为 现将pdf转图片 再分析
def analyze_pdf_first_page(pdf_path, prompt=None):
    # ========== . 检查文件 ==========
    if not os.path.exists(pdf_path):
        print(f"❌ 文件不存在：{pdf_path}")
        return None

    # ==========  PDF 转图片 ==========
    print(f"📄 正在解析：{os.path.basename(pdf_path)}")

    pdf = fitz.open(pdf_path)

    if len(pdf) == 0:
        print("❌ PDF 是空的")
        pdf.close()
        return None

    # 获取第 1 页
    page = pdf[0]

    # 渲染成图片（高清）
    mat = fitz.Matrix(2, 2)  # 2 倍缩放
    pix = page.get_pixmap(matrix=mat)

    # 保存临时文件
    image_path = "temp_pdf_page1.png"
    pix.save(image_path)
    print(f"✅ 已将第 1 页转为图片：{image_path}\n")
    pdf.close()
    return analyze_game_image(image_path)


if __name__ == "__main__":
    image_path = "./human.jpg"
    pdf_path = "Week10-多模态大模型.pdf"
    analyze_game_image(image_path)
    analyze_pdf_first_page(pdf_path)


