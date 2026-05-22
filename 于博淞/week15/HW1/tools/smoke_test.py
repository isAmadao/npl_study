"""
Quick end-to-end smoke test for the upload flow.

Usage (API must be running):
    python tools/smoke_test.py --pdf path/to/sample.pdf

Steps:
  1. Create a knowledge base (or reuse existing)
  2. Upload the PDF
  3. Poll until processing is done (or error)
  4. Print result
"""
import argparse
import sys
import time

import httpx

BASE = "http://localhost:8000/api/v1"


def create_kb(client: httpx.Client, name: str = "smoke-test-kb") -> int:
    r = client.post(f"{BASE}/kb", json={"name": name})
    if r.status_code == 409:
        # Already exists — find its id
        kbs = client.get(f"{BASE}/kb").json()
        return next(k["id"] for k in kbs if k["name"] == name)
    r.raise_for_status()
    return r.json()["id"]


def upload_pdf(client: httpx.Client, kb_id: int, pdf_path: str) -> int:
    with open(pdf_path, "rb") as f:
        r = client.post(
            f"{BASE}/upload/document",
            data={"kb_id": kb_id},
            files={"file": (pdf_path.split("/")[-1], f, "application/pdf")},
            timeout=30,
        )
    r.raise_for_status()
    data = r.json()
    print(f"Uploaded: doc_id={data['doc_id']}  status={data['status']}")
    return data["doc_id"]


def poll_status(client: httpx.Client, doc_id: int, timeout: int = 300) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"{BASE}/upload/document/{doc_id}/status")
        r.raise_for_status()
        status = r.json()["status"]
        print(f"  [{time.strftime('%H:%M:%S')}] status = {status}")
        if status in ("done", "error"):
            return status
        time.sleep(5)
    return "timeout"


def test_chat(client: httpx.Client, kb_id: int, query: str) -> None:
    print(f"\nQuery: {query}")
    r = client.post(
        f"{BASE}/chat",
        json={"kb_id": kb_id, "query": query, "top_k": 5},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    print(f"Answer: {data['answer'][:200]}…")
    print(f"Sources ({len(data['sources'])}):")
    for s in data["sources"]:
        print(f"  [{s['chunk_type']}] {s['filename']} p.{s['page_num']} score={s['score']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to a PDF file")
    parser.add_argument("--kb", default="smoke-test-kb", help="Knowledge base name")
    parser.add_argument("--query", default="请总结文档的主要内容", help="Test query after upload")
    args = parser.parse_args()

    with httpx.Client() as client:
        kb_id = create_kb(client, args.kb)
        print(f"Knowledge base id: {kb_id}")

        doc_id = upload_pdf(client, kb_id, args.pdf)

        print("Polling processing status (worker must be running)…")
        final = poll_status(client, doc_id)

        if final == "error":
            print("FAILED — check worker logs", file=sys.stderr)
            sys.exit(1)
        elif final == "timeout":
            print("TIMEOUT — worker may still be running", file=sys.stderr)
            sys.exit(2)

        print("SUCCESS — document indexed")
        test_chat(client, kb_id, args.query)


if __name__ == "__main__":
    main()
