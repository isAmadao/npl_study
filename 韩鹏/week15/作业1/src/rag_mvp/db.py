from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from rag_mvp.config import settings


Base = declarative_base()


def create_engine_for_path(db_path: Path):
    return create_engine(
        f"sqlite:///{db_path.as_posix()}",
        connect_args={"check_same_thread": False},
        future=True,
    )


engine = create_engine_for_path(settings.db_path)
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)


def init_db(custom_engine=None) -> None:
    from rag_mvp import models  # noqa: F401

    settings.ensure_directories()
    Base.metadata.create_all(bind=custom_engine or engine)
