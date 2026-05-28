"""
Evaluation script.

Usage:
    python eval/evaluate.py --queries eval/queries.json --output eval/results.json

queries.json format:
[
  {
    "query": "...",
    "kb_id": 1,
    "expected_filename": "doc.pdf",
    "expected_page": 3,
    "expected_answer": "..."
  },
  ...
]

Scoring per question (total 1.0):
  - filename match : 0.25
  - page match     : 0.25
  - Jaccard sim    : 0.50
"""
import argparse
import json

import httpx

API_BASE = "http://localhost:8000/api/v1"


def jaccard(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def score_one(result: dict, expected: dict) -> dict:
    sources = result.get("sources", [])
    filenames = [s["filename"] for s in sources]
    pages = [s["page_num"] for s in sources]

    fn_match = float(expected["expected_filename"] in filenames) * 0.25
    pg_match = float(expected["expected_page"] in pages) * 0.25
    jac = jaccard(result.get("answer", ""), expected["expected_answer"]) * 0.50

    return {
        "query": expected["query"],
        "filename_score": fn_match,
        "page_score": pg_match,
        "jaccard_score": jac,
        "total": fn_match + pg_match + jac,
        "answer": result.get("answer", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.queries, encoding="utf-8") as f:
        queries = json.load(f)

    scores = []
    with httpx.Client(timeout=120) as client:
        for q in queries:
            resp = client.post(
                f"{API_BASE}/chat",
                json={"kb_id": q["kb_id"], "query": q["query"], "top_k": 5},
            )
            resp.raise_for_status()
            s = score_one(resp.json(), q)
            scores.append(s)
            print(f"[{s['total']:.2f}] {q['query'][:60]}")

    avg = sum(s["total"] for s in scores) / len(scores) if scores else 0
    print(f"\nAverage score: {avg:.4f} over {len(scores)} queries")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"average": avg, "results": scores}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
