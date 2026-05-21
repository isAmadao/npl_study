# file_manager.py
import os
import hashlib
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class File(Base):
    """文件表 - 核心文件管理系统"""
    __tablename__ = 'files'
    
    # 主键和基本信息
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)  # 文件名
    filepath = Column(String(1000), nullable=False)  # 完整路径
    filestate = Column(String(20), nullable=False)  # 处理状态

    # 索引
    __table_args__ = (
        {'sqlite_autoincrement': True}
    )

db_path = os.path.join(os.getcwd(), 'db.db')
engine = create_engine(f'sqlite:///{db_path}')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
