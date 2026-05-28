from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

_connect_args = {"check_same_thread": False} if "sqlite" in settings.database_url else {}

engine = create_engine(settings.database_url, connect_args=_connect_args)


class Base(DeclarativeBase):
    pass
