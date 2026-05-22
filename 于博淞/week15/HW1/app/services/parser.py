"""MinerU PDF parser wrapper."""
import logging
import os
import subprocess
from pathlib import Path

from app.core.config import settings

log = logging.getLogger(__name__)


def parse_pdf(doc_id: int, file_path: str) -> dict:
    """
    Parse a PDF with MinerU (magic-pdf CLI).

    Returns:
        {"markdown_path": str, "image_dir": str}

    MinerU writes output under:
        <out_dir>/<method>/<pdf_stem>/<pdf_stem>.md
        <out_dir>/<method>/<pdf_stem>/images/

    The method directory ("auto", "txt", "ocr") is determined by MinerU.
    We glob for the first .md file rather than hardcoding the path.
    """
    out_dir = Path(settings.parsed_storage_path) / str(doc_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("[doc=%d] Running MinerU on %s", doc_id, file_path)
    proc = subprocess.run(
        ["magic-pdf", "-p", file_path, "-o", str(out_dir), "-m", "auto"],
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        log.error("[doc=%d] MinerU stderr:\n%s", doc_id, proc.stderr[-2000:])
        raise RuntimeError(
            f"MinerU exited with code {proc.returncode}. "
            f"Stderr tail: {proc.stderr[-500:]!r}"
        )

    # Locate the produced markdown — glob is robust across MinerU versions
    md_files = sorted(out_dir.rglob("*.md"))
    if not md_files:
        raise FileNotFoundError(
            f"MinerU produced no .md file under {out_dir}. "
            f"Stdout: {proc.stdout[-500:]!r}"
        )

    md_path = md_files[0]   # there should be exactly one
    img_dir = md_path.parent / "images"

    log.info("[doc=%d] Parsed → %s (images: %s)", doc_id, md_path, img_dir)
    return {"markdown_path": str(md_path), "image_dir": str(img_dir)}
