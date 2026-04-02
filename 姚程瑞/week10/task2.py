import base64
import os
from openai import OpenAI
from PyPDF2 import PdfReader
import io

API_KEY = "sk-95f2c1b4272f4b1895975ec05d5ae935"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n\n"
    return text

def parse_pdf_with_qwen_vl(pdf_path):
    print(f"正在解析PDF: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print(f"文件不存在: {pdf_path}")
        return
    
    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        print(f"PDF共有 {num_pages} 页")
        
        all_text = ""
        
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            
            if page_text.strip():
                print(f"处理第 {page_num+1} 页（纯文本提取）...")
                
                response = client.chat.completions.create(
                    model="qwen-vl-plus",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"请优化以下PDF第{page_num+1}页的文本内容，使其更易读：\n\n{page_text[:3000]}"
                                }
                            ]
                        }
                    ],
                    stream=False
                )
                
                optimized_text = response.choices[0].message.content
                all_text += f"\n\n========== 第 {page_num+1} 页 ==========\n"
                all_text += optimized_text
                print(f"第 {page_num+1} 页处理完成")
            else:
                print(f"第 {page_num+1} 页无法提取文本（可能是扫描件）")
        
        output_file = pdf_path.replace('.pdf', '_parsed.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(all_text)
        
        print(f"\n解析完成！结果已保存到: {output_file}")
        return all_text
        
    except Exception as e:
        print(f"错误: {e}")
        return None

if __name__ == "__main__":
    pdf_path = r"d:\GZU\badou\作业\十\amd-rocm-6-brief.pdf"
    result = parse_pdf_with_qwen_vl(pdf_path)
