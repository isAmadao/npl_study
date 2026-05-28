#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF 文档解析对比工具
对比 MinerU 和 pdfplumber 的解析效果
"""

import os
import pdfplumber
from pathlib import Path


def parse_with_pdfplumber(pdf_path: str, output_dir: str = "output_pdfplumber") -> dict:
    """
    使用 pdfplumber 解析 PDF

    Args:
        pdf_path: PDF 文件路径
        output_dir: 输出目录

    Returns:
        解析结果字典
    """
    result = {
        "success": False,
        "text": "",
        "tables": [],
        "images": [],
        "pages": 0,
        "error": None
    }

    try:
        # 创建输出目录
        Path(output_dir).mkdir(exist_ok=True)

        with pdfplumber.open(pdf_path) as pdf:
            result["pages"] = len(pdf.pages)

            all_text = []
            all_tables = []

            for i, page in enumerate(pdf.pages):
                print(f"  正在解析第 {i+1} 页...")

                # 提取文本
                text = page.extract_text() or ""
                all_text.append(f"--- Page {i+1} ---\n{text}")

                # 提取表格
                tables = page.extract_tables()
                if tables:
                    all_tables.append({
                        "page": i+1,
                        "tables": tables
                    })

                # 尝试提取图片（pdfplumber 不直接提取图片，只能获取位置）
                images = page.images
                if images:
                    result["images"].append({
                        "page": i+1,
                        "count": len(images),
                        "images": images
                    })

            result["text"] = "\n\n".join(all_text)
            result["tables"] = all_tables
            result["success"] = True

            # 保存结果
            text_output_path = os.path.join(output_dir, f"{Path(pdf_path).stem}.txt")
            with open(text_output_path, "w", encoding="utf-8") as f:
                f.write(result["text"])

            print(f"  pdfplumber 解析完成！文本已保存到: {text_output_path}")

    except Exception as e:
        result["error"] = str(e)
        print(f"  pdfplumber 解析失败: {e}")

    return result


def parse_with_mineru_info() -> dict:
    """
    提供 MinerU 功能说明和使用指南

    Returns:
        MinerU 信息字典
    """
    return {
        "name": "MinerU",
        "version": "2.5",
        "features": [
            "复杂布局解析（单栏/多栏）",
            "智能去除页眉页脚脚注页码",
            "保留文档结构（标题、段落、列表）",
            "提取图片和图片描述",
            "识别并转换表格为HTML/Markdown格式",
            "识别并转换公式为LaTeX格式",
            "自动检测扫描版PDF并启用OCR（支持109种语言）",
            "支持多种输出格式（Markdown/JSON/中间格式）",
            "可视化布局验证",
            "GPU加速支持（CUDA/NPU/MPS）"
        ],
        "installation": [
            "pip install uv -i https://mirrors.aliyun.com/pypi/simple",
            "uv pip install -U 'mineru[core]' -i https://mirrors.aliyun.com/pypi/simple"
        ],
        "usage": {
            "cli": "mineru -p <input_path> -o <output_path>",
            "api": "mineru-api --host 0.0.0.0 --port 8000"
        }
    }


def compare_parsers(pdfplumber_result: dict, mineru_info: dict) -> str:
    """
    对比两种解析器的效果（对比内容已移至外部 md 文档）

    Args:
        pdfplumber_result: pdfplumber 解析结果
        mineru_info: MinerU 信息

    Returns:
        简化的对比说明
    """
    return """
# MinerU vs pdfplumber 对比分析

## 概述

详细对比分析请查看外部文档：`MinerU_vs_pdfplumber_详细对比.md`

## 简要对比

| 功能维度 | pdfplumber | MinerU |
|---------|-----------|--------|
| 文本提取 | ✅ 基础支持 | ✅ 智能排版（阅读顺序） |
| 表格提取 | ✅ 基础提取 | ✅ 智能识别 + HTML/Markdown 转换 |
| 公式识别 | ❌ 不支持 | ✅ LaTeX 转换 |
| 图片提取 | ⚠️ 仅位置信息 | ✅ 完整提取 + 描述 |
| 复杂布局 | ❌ 处理困难 | ✅ 支持单栏/多栏混合排版 |
| OCR 支持 | ❌ 无 | ✅ 109种语言 |
"""


def main():
    """主函数"""
    pdf_dir = "模型论文"
    sample_pdf = "2509-MinerU2.5.pdf"
    pdf_path = os.path.join(pdf_dir, sample_pdf)

    if not os.path.exists(pdf_path):
        print(f"PDF 文件不存在: {pdf_path}")
        # 尝试找其他 PDF
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
        if pdf_files:
            sample_pdf = pdf_files[0]
            pdf_path = os.path.join(pdf_dir, sample_pdf)
            print(f"使用替代文件: {pdf_path}")
        else:
            print("未找到 PDF 文件！")
            return

    print("="*60)
    print(f"开始解析 PDF: {sample_pdf}")
    print("="*60)

    # 1. 使用 pdfplumber 解析
    print("\n[1/2] 使用 pdfplumber 解析...")
    pdfplumber_result = parse_with_pdfplumber(pdf_path, "output_pdfplumber")

    # 2. 获取 MinerU 信息
    print("\n[2/2] MinerU 功能说明...")
    mineru_info = parse_with_mineru_info()

    # 3. 生成对比报告
    print("\n[3/3] 生成对比分析...")
    comparison_report = compare_parsers(pdfplumber_result, mineru_info)

    # 4. 保存报告
    report_path = "mineru_vs_pdfplumber_comparison.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(comparison_report)

    print(f"\n[OK] 对比分析报告已保存到: {report_path}")
    print("\n" + "="*60)
    print("核心差异总结:")
    print("="*60)
    print("- MinerU 是完整的智能解析系统，pdfplumber 是轻量级工具")
    print("- MinerU 擅长复杂布局、公式、图片、扫描件处理")
    print("- 对于 RAG 系统，MinerU 能提供更高质量的文档解析")
    print("- 详细对比分析请查看: MinerU_vs_pdfplumber_详细对比.md")
    print("="*60)


if __name__ == "__main__":
    main()
