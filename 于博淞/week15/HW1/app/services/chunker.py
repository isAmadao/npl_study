"""
Markdown chunker for MinerU output.

Strategy:
  1. Split the document into pages via MinerU page markers.
  2. Within each page, split into atomic blocks (paragraph / table / code block).
     Tables and code blocks are never broken mid-block.
     Markdown image refs are stripped (images are indexed separately by the worker).
  3. Greedily accumulate blocks into chunks up to CHUNK_SIZE chars.
  4. A single block that exceeds CHUNK_SIZE is hard-split with sentence-boundary
     awareness and CHUNK_OVERLAP chars of overlap.
  5. Chunks shorter than MIN_CHUNK_LEN are discarded.
"""
from __future__ import annotations

import re

CHUNK_SIZE = 800      # target max chars per chunk  (~320 tokens for Chinese)
CHUNK_OVERLAP = 150   # overlap between consecutive window-split chunks
MIN_CHUNK_LEN = 30    # discard chunks shorter than this

# MinerU embeds page markers in several formats depending on version
_PAGE_RE = re.compile(
    r"<!--\s*(?:page[_\s]break:\s*|page[_\s])?(\d+)\s*-->",
    re.IGNORECASE,
)
# Strip markdown image syntax — images are indexed as separate vectors
_IMG_RE = re.compile(r"!\[.*?\]\([^)]*\)")


# ── public API ────────────────────────────────────────────────────────────────

def chunk_markdown(md_path: str) -> list[dict]:
    """
    Parse a MinerU markdown file into page-aware, overlap-capable text chunks.

    Returns:
        list of {"page_num": int, "chunk_index": int, "content": str}
    """
    with open(md_path, encoding="utf-8") as f:
        raw = f.read()

    pages = _split_pages(raw)

    result: list[dict] = []
    idx = 0
    for page_num, page_text in pages:
        blocks = _split_blocks(page_text)
        for chunk in _aggregate(blocks):
            result.append({"page_num": page_num, "chunk_index": idx, "content": chunk})
            idx += 1

    return result


# ── page splitting ────────────────────────────────────────────────────────────

def _split_pages(text: str) -> list[tuple[int, str]]:
    """Return [(page_num, content), ...]."""
    markers = [(m.start(), m.end(), int(m.group(1))) for m in _PAGE_RE.finditer(text)]

    if not markers:
        return [(1, text.strip())]

    pages: list[tuple[int, str]] = []
    current_page = 1
    prev_end = 0

    for start, end, page_num in markers:
        segment = text[prev_end:start].strip()
        if segment:
            pages.append((current_page, segment))
        current_page = page_num
        prev_end = end

    tail = text[prev_end:].strip()
    if tail:
        pages.append((current_page, tail))

    return pages


# ── block splitting ───────────────────────────────────────────────────────────

def _split_blocks(text: str) -> list[str]:
    """
    Split page text into atomic blocks.

    Rules:
    - Code fences (``` ... ```) are kept as one block.
    - Table runs (consecutive lines containing |) are kept as one block.
    - Otherwise, blank lines delimit blocks.
    - Markdown image refs are removed from text blocks.
    """
    blocks: list[str] = []
    buf: list[str] = []
    in_code = False
    in_table = False

    for line in text.splitlines():
        # ── code block toggle ──
        if line.startswith("```"):
            if not in_code:
                _flush(buf, blocks)
                in_code = True
                buf = [line]
            else:
                buf.append(line)
                blocks.append("\n".join(buf))
                buf = []
                in_code = False
            continue

        if in_code:
            buf.append(line)
            continue

        # ── table detection ──
        is_table = bool(re.match(r"\s*\|", line))
        if is_table and not in_table:
            _flush(buf, blocks)
            in_table = True
            buf = [line]
            continue
        if not is_table and in_table:
            _flush(buf, blocks)
            in_table = False

        if in_table:
            buf.append(line)
            continue

        # ── normal text ──
        clean = _IMG_RE.sub("", line).strip()

        if clean:
            buf.append(clean)
        else:
            # blank line = paragraph boundary
            _flush(buf, blocks)

    _flush(buf, blocks)
    return blocks


def _flush(buf: list[str], blocks: list[str]) -> None:
    text = "\n".join(buf).strip()
    if len(text) >= MIN_CHUNK_LEN:
        blocks.append(text)
    buf.clear()


# ── aggregation & sliding window ──────────────────────────────────────────────

def _aggregate(blocks: list[str]) -> list[str]:
    """
    Greedily merge blocks into chunks ≤ CHUNK_SIZE.
    Blocks larger than CHUNK_SIZE are hard-split with overlap.
    """
    chunks: list[str] = []
    buf = ""

    for block in blocks:
        if len(block) > CHUNK_SIZE:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(_sliding_window(block))
            continue

        if buf and len(buf) + 2 + len(block) > CHUNK_SIZE:
            chunks.append(buf)
            buf = block
        else:
            buf = (buf + "\n\n" + block).strip() if buf else block

    if len(buf) >= MIN_CHUNK_LEN:
        chunks.append(buf)

    return chunks


def _sliding_window(text: str) -> list[str]:
    """Hard-split a long block into overlapping windows."""
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        if end >= len(text):
            tail = text[start:].strip()
            if len(tail) >= MIN_CHUNK_LEN:
                chunks.append(tail)
            break

        # Try to break at a sentence boundary
        for boundary in "。！？\n":
            pos = text.rfind(boundary, start + CHUNK_OVERLAP, end)
            if pos != -1:
                end = pos + 1
                break

        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_LEN:
            chunks.append(chunk)

        start = end - CHUNK_OVERLAP

    return chunks
