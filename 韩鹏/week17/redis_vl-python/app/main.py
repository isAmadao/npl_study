"""FastAPI application entrypoint."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.modules import EmbeddingsCache, SemanticCache, SemanticMessageHistory, SemanticRouter
from app.redis_client import get_redis_client
from app.vectorizer import Vectorizer

APP_VERSION = "0.6.0"
FLOW_STEPS = ["embeddings_cache", "semantic_cache", "semantic_router", "message_history"]

app = FastAPI(title="RedisVL Python Demo", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
vectorizer = Vectorizer()


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


DEFAULT_ROUTER_ROUTES: list[dict[str, object]] = [
    {
        "name": "技术问答",
        "references": [
            "Python 列表怎么排序",
            "Python 怎么用",
            "什么是 Redis",
            "如何配置 Docker",
            "代码怎么写",
            "API 调用方式",
        ],
        "distance_threshold": 0.3,
    },
    {
        "name": "日常闲聊",
        "references": [
            "你好啊今天怎么样",
            "今天天气不错",
            "你好吗",
            "周末去哪里玩",
            "推荐一部电影",
        ],
        "distance_threshold": 0.35,
    },
    {
        "name": "数学计算",
        "references": [
            "3.14 乘以 2",
            "1+1 等于几",
            "根号2是多少",
            "计算圆的面积",
            "求解方程",
        ],
        "distance_threshold": 0.3,
    },
]


def _mock_llm_response(prompt: str) -> str:
    return f"这是一个 mock 回复：你问的是“{prompt}”。后续版本会替换为真实 LLM 输出。"


def _normalize_session_tag(session_id: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in session_id.strip())
    return cleaned[:64] or "default"


def _require_session_id(session_id: str) -> tuple[str, str]:
    cleaned_session_id = session_id.strip()
    if not cleaned_session_id:
        raise HTTPException(status_code=400, detail="Session ID must not be empty.")
    return cleaned_session_id, _normalize_session_tag(cleaned_session_id)


def _build_message_history_result(
    message_history: SemanticMessageHistory,
    prompt: str,
    response: str,
    vector: list[float],
) -> dict[str, object]:
    relevant_messages = message_history.get_relevant(prompt, top_k=3, query_vector=vector)
    message_history.store(prompt, response, vector=vector)
    recent_messages = message_history.get_recent(top_k=5)
    return {
        "stored": {"prompt": prompt, "response": response},
        "relevant_messages": relevant_messages[:3],
        "count": len(relevant_messages),
        "recent_messages": recent_messages,
        "recent_count": len(recent_messages),
    }


@app.get("/api/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/session/history")
async def get_session_history(
    session_id: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=20),
) -> dict[str, object]:
    cleaned_session_id, session_tag = _require_session_id(session_id)
    redis_client = None
    try:
        redis_client = get_redis_client()
        message_history = SemanticMessageHistory(
            redis_client=redis_client,
            vectorizer=vectorizer,
            session_tag=session_tag,
        )
        recent_messages = message_history.get_recent(top_k=top_k)
        return {
            "session_id": cleaned_session_id,
            "normalized_session_id": session_tag,
            "recent_messages": recent_messages,
            "count": len(recent_messages),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if redis_client is not None:
            redis_client.close()


@app.delete("/api/session/history")
async def clear_session_history(session_id: str = Query(..., min_length=1)) -> dict[str, object]:
    cleaned_session_id, session_tag = _require_session_id(session_id)
    redis_client = None
    try:
        redis_client = get_redis_client()
        message_history = SemanticMessageHistory(
            redis_client=redis_client,
            vectorizer=vectorizer,
            session_tag=session_tag,
        )
        cleared = message_history.clear()
        return {
            "session_id": cleaned_session_id,
            "normalized_session_id": session_tag,
            "cleared": cleared,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if redis_client is not None:
            redis_client.close()


@app.delete("/api/cache/semantic")
async def clear_semantic_cache() -> dict[str, object]:
    redis_client = None
    try:
        redis_client = get_redis_client()
        semantic_cache = SemanticCache(redis_client=redis_client, vectorizer=vectorizer)
        return {"cleared": semantic_cache.clear()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if redis_client is not None:
            redis_client.close()


@app.post("/api/chat")
async def chat(payload: ChatRequest) -> dict[str, object]:
    started_at = time.perf_counter()
    prompt = payload.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    cleaned_session_id, session_tag = _require_session_id(payload.session_id)
    redis_client = None
    message_history_result: dict[str, object] | None = None
    router_result: dict[str, object] | None = None
    try:
        redis_client = get_redis_client()

        embeddings_cache = EmbeddingsCache(redis_client=redis_client)
        cached_vector = embeddings_cache.get([prompt])[prompt]
        embed_hit = cached_vector is not None
        vector = cached_vector if cached_vector is not None else vectorizer.embed(prompt)
        if not embed_hit:
            embeddings_cache.set([prompt], [vector])

        embeddings_cache_result = {
            "hit": embed_hit,
            "status": "hit" if embed_hit else "miss",
            "vector_dim": len(vector),
            "vector_preview": [round(value, 6) for value in vector[:5]],
        }

        semantic_cache = SemanticCache(redis_client=redis_client, vectorizer=vectorizer)
        semantic_hit_result = semantic_cache.check(prompt, query_vector=vector)
        if semantic_hit_result is not None:
            final_response = semantic_hit_result["response"]
            semantic_cache_result = {
                "hit": True,
                "response": final_response,
                "distance": round(float(semantic_hit_result["distance"]), 6),
                "cached_prompt": semantic_hit_result["cached_prompt"],
            }
        else:
            final_response = _mock_llm_response(prompt)
            semantic_cache.store(
                prompt=prompt,
                response=final_response,
                metadata={"session_id": cleaned_session_id},
                vector=vector,
            )
            semantic_cache_result = {
                "hit": False,
                "message": "cache miss",
                "response": final_response,
            }

        semantic_router = SemanticRouter(redis_client=redis_client, vectorizer=vectorizer)
        if DEFAULT_ROUTER_ROUTES:
            semantic_router.sync_routes(DEFAULT_ROUTER_ROUTES)
        router_match = semantic_router.route(prompt, query_vector=vector)
        if router_match is None:
            router_result = {
                "matched": False,
                "message": "未匹配到任何路由",
            }
        else:
            router_result = {
                "matched": True,
                "route_name": router_match["route_name"],
                "distance": round(float(router_match["distance"]), 6),
                "matched_reference": router_match["matched_reference"],
                "threshold": round(float(router_match["threshold"]), 6),
            }

        message_history = SemanticMessageHistory(
            redis_client=redis_client,
            vectorizer=vectorizer,
            session_tag=session_tag,
        )
        message_history_result = _build_message_history_result(
            message_history=message_history,
            prompt=prompt,
            response=final_response,
            vector=vector,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if redis_client is not None:
            redis_client.close()

    response_time_ms = round((time.perf_counter() - started_at) * 1000, 2)
    return {
        "session_id": cleaned_session_id,
        "normalized_session_id": session_tag,
        "prompt": prompt,
        "embeddings_cache": embeddings_cache_result,
        "semantic_cache": semantic_cache_result,
        "message_history": message_history_result,
        "router": router_result or {"matched": False, "message": "未匹配到任何路由"},
        "timing": {
            "response_time_ms": response_time_ms,
            "flow": FLOW_STEPS,
        },
    }
