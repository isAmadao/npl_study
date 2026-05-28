# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multimodal RAG (Retrieval-Augmented Generation) system for PDF knowledge bases. Users upload PDF documents; the system parses them into text chunks and images, embeds everything into a vector store, then answers natural-language questions by retrieving relevant text+image context and passing it to a multimodal LLM.

## Commands

```bash
# Start infrastructure (Kafka + Milvus) — first time only
docker compose up -d

# Install Python dependencies
pip install -r requirements.txt

# Start FastAPI dev server
uvicorn app.main:app --reload --port 8000

# Start document processing worker (separate terminal)
python -m app.workers.document_worker

# End-to-end smoke test (API + worker both running)
python tools/smoke_test.py --pdf path/to/sample.pdf

# Run evaluation against test queries
python eval/evaluate.py --queries eval/queries.json --output eval/results.json
```

## Architecture

The system has three runtime processes:

**1. Web API** (`app/main.py`) — FastAPI, handles:
- `POST /upload/document` — saves PDF locally, publishes parse job to Kafka topic, returns immediately
- `POST /chat` — embeds user query via CLIP+BGE, retrieves top-k chunks/images from Milvus, calls Qwen-VL with retrieved context, returns answer with source citations (filename + page)

**2. Document Worker** (`app/workers/document_worker.py`) — Kafka consumer, handles:
- Consumes parse jobs from Kafka
- Calls MinerU to parse PDF → markdown + image files (slow, GPU-bound, ~1 min/file)
- Splits markdown into chunks, embeds with BGE (text) and CLIP (images)
- Upserts embeddings into Milvus; records metadata (filename, page, chunk index) in SQLite

**3. Milvus** — external vector database, two collections:
- `text_chunks` — BGE embeddings of markdown chunks
- `image_chunks` — CLIP embeddings of extracted images

## Why Kafka

MinerU takes ~1 minute per PDF and requires GPU. Upload endpoints must return in <30s. Kafka decouples the web tier (producer, high concurrency) from the offline worker (consumer, slow processing). Direct HTTP to MinerU is not viable.

## Models & Their Roles

| Model | Role | When called |
|---|---|---|
| MinerU / DeepSeek-OCR | PDF → markdown + images | Worker, offline |
| BGE | Text chunk embedding | Worker (index) + API (query) |
| CLIP | Image embedding + text→image retrieval | Worker (index) + API (query) |
| Qwen-VL | Multimodal QA — final answer generation | API, per chat request |

## Data Storage Layout

```
data/
├── pdfs/          # Raw uploaded PDFs
├── parsed/        # MinerU output: {doc_id}/output.md + images/
└── chunks/        # (optional cache) chunk JSON before embedding
```

SQLite stores document metadata and chunk records (doc_id, filename, page_num, chunk_index, milvus_id). Milvus stores the actual vectors.

## Key Conventions

- **Async parse flow**: upload endpoint only writes file + publishes to Kafka. Status polling goes through SQLite (`documents.status`: `pending → processing → done → error`).
- **Citations**: every Milvus record carries `filename` and `page_num` as scalar fields so the chat endpoint can return source attribution without extra DB lookups.
- **Retrieval fusion**: query is embedded with both BGE (text retrieval) and CLIP (image retrieval); results are merged and ranked before being passed to Qwen-VL.
- **Evaluation scoring**: each answer is scored on page match (0.25), filename match (0.25), and Jaccard similarity of answer text (0.5).

## Environment Variables

```env
DATABASE_URL=sqlite:///./mindwise.db
MILVUS_HOST=localhost
MILVUS_PORT=19530
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_PARSE=document_parse
QWEN_VL_API_KEY=
QWEN_VL_BASE_URL=
PDF_STORAGE_PATH=./data/pdfs
PARSED_STORAGE_PATH=./data/parsed
```
