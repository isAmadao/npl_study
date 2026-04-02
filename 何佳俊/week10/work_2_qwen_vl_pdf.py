"""
作业 2: 使用 Qwen-VL 解析本地 PDF 文件
功能：读取本地 PDF 的第一页，使用阿里云 Qwen-VL 大模型进行内容分析
"""
import fitz  # PyMuPDF

# ========== 配置区域 ==========
# PDF 文件路径
PDF_PATH = r"C:\Users\Administrator.DESKTOP-K90E5CL\Desktop\AiLearn\aiLearnWork\Week10\homework\work2\Week10-多模态大模型.pdf"

# 输出的图片路径
OUTPUT_IMAGE_PATH = r"C:\Users\Administrator.DESKTOP-K90E5CL\Desktop\AiLearn\aiLearnWork\Week10\homework\work2\page1.png"

ALIYUN_API_KEY = "sk-19f96f86d70294228bbb74d78d33859df23"

# 分析提示词
PROMPT = """请详细分析这张图片的内容：
1. 这是什么类型的文档？（如：合同、报告、论文、通知等）
2. 文档的主要内容是什么？
3. 提取关键信息（如：标题、日期、人名、机构名等）
4. 总结文档的核心要点
请用中文回答。"""
# ===========================


def pdf_to_image(pdf_path, output_image_path, page_number=0):
    """
    将 PDF 的指定页面转换为图片
    
    Args:
        pdf_path: PDF 文件路径
        output_image_path: 输出图片路径
        page_number: 页码（从 0 开始）
    
    Returns:
        str: 生成的图片路径
    """
    print(f"\n正在打开 PDF 文件：{pdf_path}")
    
    # 打开 PDF
    doc = fitz.open(pdf_path)
    
    print(f"PDF 共有 {len(doc)} 页，正在处理第 {page_number + 1} 页...")
    
    # 获取指定页面
    page = doc[page_number]
    
    # 设置缩放比例（提高图片质量）
    zoom = 2.0
    mat = fitz.Matrix(zoom, zoom)
    
    # 渲染页面为图片
    pix = page.get_pixmap(matrix=mat)
    
    # 保存图片
    print(f"正在保存图片到：{output_image_path}")
    pix.save(output_image_path)
    
    # 关闭 PDF
    doc.close()
    
    print(f"✓ PDF 第 {page_number + 1} 页已转换为图片")
    print(f"  图片尺寸：{pix.width} x {pix.height}")
    
    return output_image_path


def analyze_with_qwen_vl(image_path, api_key, prompt):
    """
    使用 Qwen-VL 分析图片内容
    
    Args:
        image_path: 图片路径
        api_key: 阿里云 API Key
        prompt: 分析提示词
    
    Returns:
        str: 模型返回的分析结果
    """
    import dashscope
    from dashscope import MultiModalConversation
    
    # 设置 API Key
    dashscope.api_key = api_key
    
    print(f"\n正在调用 Qwen-VL 分析图片...")
    print(f"  图片路径：{image_path}")

    # 调用多模态对话 API
    response = MultiModalConversation.call(
        model='qwen-vl-max-latest',
        messages=[{
            'role': 'user',
            'content': [
                {'image': f'file://{image_path}'},
                {'text': prompt}
            ]
        }]
    )

    # 检查响应状态
    if response.status_code == 200:
        content = response.output.choices[0].message.content[0]['text']
        return content
    else:
        error_msg = f"API 调用失败：{response.code} - {response.message}"
        print(f"\n❌ {error_msg}")
        return error_msg


def main():
    print("=" * 60)
    print("作业 2: Qwen-VL 解析 PDF")
    print("=" * 60)
    
    # 步骤 1: PDF 转图片
    image_path = pdf_to_image(PDF_PATH, OUTPUT_IMAGE_PATH, page_number=5)
    
    # 步骤 2: 调用 Qwen-VL 分析
    result = analyze_with_qwen_vl(image_path, ALIYUN_API_KEY, PROMPT)
    
    # 步骤 4: 显示结果
    print("\n" + "=" * 60)
    print("Qwen-VL 分析结果")
    print("=" * 60)
    print(result)
    print("=" * 60)
    print("✅ 作业 2 完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
