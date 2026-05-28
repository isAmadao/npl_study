from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from ..core.config import DB_PATH


class Base(DeclarativeBase):
    pass


class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    original_name = Column(String(512), nullable=False)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(1000), nullable=False)
    filestate = Column(String(20), nullable=False, default="uploaded")
    error_message = Column(String(2000), nullable=True)
    created_at = Column(DateTime, default=datetime.now)


engine = create_engine(f"sqlite:///{DB_PATH}")
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)
