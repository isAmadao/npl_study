from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_mvp.db import init_db
from rag_mvp.services.ingest_service import IngestionService
from rag_mvp.services.queue_service import TaskQueue


def process_one_task() -> bool:
    queue = TaskQueue()
    task = queue.get_next_task()
    if task is None:
        return False
    IngestionService().process_file(task.file_id)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="图文知识库异步处理 worker")
    parser.add_argument("--once", action="store_true", help="只处理一条任务")
    args = parser.parse_args()

    init_db()

    if args.once:
        processed = process_one_task()
        print("processed=1" if processed else "processed=0")
        return

    while True:
        processed = process_one_task()
        if not processed:
            time.sleep(3)


if __name__ == "__main__":
    main()

