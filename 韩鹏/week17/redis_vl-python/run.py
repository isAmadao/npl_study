"""Local startup script for the V0.1.0 scaffold."""

from __future__ import annotations

import sys

import uvicorn

from app.config import get_settings
from app.redis_client import get_redis_client


def main() -> int:
    settings = get_settings()
    print(f"[INFO] Checking Redis connectivity: {settings.redis_host}:{settings.redis_port}")
    try:
        get_redis_client()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        print("[HINT] Run `docker compose up -d` and retry.")
        return 1

    print("[INFO] Redis is reachable.")
    print(f"[INFO] Starting FastAPI at http://localhost:{settings.app_port}")
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
