"""
SQLite 元数据模型：知识库、文档、chunk 三张核心表。
"""
import sqlite3
import os
from datetime import datetime
from contextlib import contextmanager

import config


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS knowledge_bases (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL UNIQUE,
                description TEXT    DEFAULT '',
                created_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                kb_id       INTEGER NOT NULL,
                filename    TEXT    NOT NULL,
                filepath    TEXT    NOT NULL,
                status      TEXT    NOT NULL DEFAULT 'pending',
                page_count  INTEGER DEFAULT 0,
                error_msg   TEXT    DEFAULT '',
                created_at  TEXT    NOT NULL,
                updated_at  TEXT    NOT NULL,
                FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id)
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id      INTEGER NOT NULL,
                kb_id       INTEGER NOT NULL,
                chunk_type  TEXT    NOT NULL,
                content     TEXT    DEFAULT '',
                file_path   TEXT    DEFAULT '',
                page_num    INTEGER DEFAULT 0,
                chunk_index INTEGER DEFAULT 0,
                milvus_id   INTEGER DEFAULT 0,
                created_at  TEXT    NOT NULL,
                FOREIGN KEY (doc_id)  REFERENCES documents(id),
                FOREIGN KEY (kb_id)   REFERENCES knowledge_bases(id)
            );

            CREATE INDEX IF NOT EXISTS idx_documents_kb_id   ON documents(kb_id);
            CREATE INDEX IF NOT EXISTS idx_documents_status  ON documents(status);
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id     ON chunks(doc_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_kb_id      ON chunks(kb_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type ON chunks(chunk_type);
        """)