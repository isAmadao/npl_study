from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

INPUT_PATH = _ROOT / "data" / "sample.png"
QWEN_VL_MODEL = "qwen3-vl-plus"
_IMAGE_SUFFIX = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}



def _bytes_to_data_url(raw: bytes, mime: str) -> str:
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def input_to_data_url(path: Path) -> str:
    """
    支持：
    - .pdf：需已安装 pymupdf
    - 常见图片：直接读文件，不依赖 pymupdf
    """
    suf = path.suffix.lower()
    if suf in _IMAGE_SUFFIX:
        mime, _ = mimetypes.guess_type(str(path))
        if not mime:
            mime = "image/png" if suf == ".png" else "image/jpeg"
        raw = path.read_bytes()
        return _bytes_to_data_url(raw, mime)


    raise ValueError(f"不支持的文件类型: {path.suffix}，请使用 .pdf 或图片（png/jpg/webp 等）")


def parse_page_with_qwen_vl(
    input_path: Path,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = QWEN_VL_MODEL,
    user_prompt: str | None = None,
    stream: bool = False,
    enable_thinking: bool = False,
) -> str:
    from openai import OpenAI

    key = api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise EnvironmentError(
            "请设置环境变量 DASHSCOPE_API_KEY（阿里云百炼 API Key）"
        )

    url = base_url or os.environ.get(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    data_url = input_to_data_url(input_path)
    text = user_prompt or (
        "这是文档第一页的图像（可能来自 PDF 首页截图或导出图）。请用中文完成：\n"
        "1. 简要描述版面结构（标题、段落、表格、图片等）；\n"
        "2. 尽可能提取可见正文文字（保持阅读顺序）；\n"
        "3. 若有表格，说明表头与大致内容。"
    )

    client = OpenAI(api_key=key, base_url=url)

    kwargs = dict(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": text},
                ],
            }
        ],
        stream=stream,
    )

    if enable_thinking:
        kwargs["extra_body"] = {"enable_thinking": True}

    if stream:
        parts: list[str] = []
        reasoning_parts: list[str] = []
        is_answering = False
        completion = client.chat.completions.create(**kwargs)
        for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            rc = getattr(delta, "reasoning_content", None)
            if rc:
                reasoning_parts.append(rc)
                print(rc, end="", flush=True)
            else:
                c = delta.content or ""
                if c and not is_answering and reasoning_parts:
                    print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                    is_answering = True
                if c:
                    print(c, end="", flush=True)
                    parts.append(c)
        print()
        return "".join(parts)

    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    content = msg.content
    if not content:
        return ""
    return content


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"未找到文件: {INPUT_PATH}\n"
            "请放入 PDF，或放入第一页导出的 PNG/JPG，并修改脚本中的 INPUT_PATH。"
        )

    print("输入:", INPUT_PATH)
    print("模型:", QWEN_VL_MODEL)
    print("-" * 60)

    answer = parse_page_with_qwen_vl(
        INPUT_PATH,
        stream=False,
        enable_thinking=False,
    )
    print(answer)


if __name__ == "__main__":
    main()
