from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FileTask:
    file_id: int
    file_path: str
    original_name: str
    suffix: str

    def to_dict(self) -> dict:
        return {
            "file_id": self.file_id,
            "file_path": self.file_path,
            "original_name": self.original_name,
            "suffix": self.suffix,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FileTask":
        return cls(
            file_id=int(data["file_id"]),
            file_path=str(data["file_path"]),
            original_name=str(data["original_name"]),
            suffix=str(data["suffix"]),
        )


@dataclass
class DispatchResult:
    backend: str
    message: str = ""


@dataclass
class ParsedImage:
    image_path: str
    page_no: int
    description: str


@dataclass
class ParsedDocument:
    text_pages: list[tuple[int, str]]
    images: list[ParsedImage]


@dataclass
class VectorRecord:
    record_id: str
    file_id: int
    file_name: str
    source_path: str
    chunk_type: str
    page_no: int
    content: str
    image_path: str | None
    embedding: list[float]

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "file_id": self.file_id,
            "file_name": self.file_name,
            "source_path": self.source_path,
            "chunk_type": self.chunk_type,
            "page_no": self.page_no,
            "content": self.content,
            "image_path": self.image_path,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VectorRecord":
        return cls(
            record_id=str(data["record_id"]),
            file_id=int(data["file_id"]),
            file_name=str(data["file_name"]),
            source_path=str(data["source_path"]),
            chunk_type=str(data["chunk_type"]),
            page_no=int(data.get("page_no", 0)),
            content=str(data.get("content", "")),
            image_path=data.get("image_path"),
            embedding=list(data["embedding"]),
        )


@dataclass
class SearchResult:
    record_id: str
    file_id: int
    file_name: str
    source_path: str
    chunk_type: str
    page_no: int
    content: str
    image_path: str | None
    score: float


@dataclass
class ChatReply:
    answer: str
    references: list[SearchResult]

