from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

ExtractionSource = Literal["native", "ocr", "mixed", "filename_only"]


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    text: str
    source: ExtractionSource = "native"


@dataclass(frozen=True, slots=True)
class FileChunk:
    id: str
    file_path: str
    file_name: str
    chunk_index: int
    text: str
    created_at: datetime
    modified_at: datetime
    extraction_source: ExtractionSource = "native"

    @staticmethod
    def make_id(file_path: str, chunk_index: int) -> str:
        digest = hashlib.sha256(f"{file_path}::{chunk_index}".encode()).hexdigest()
        return digest[:32]


@dataclass(frozen=True, slots=True)
class QueryHit:
    chunk_id: str
    file_path: str
    file_name: str
    chunk_index: int
    text: str
    distance: float
    modified_at: datetime | None = None
    extraction_source: ExtractionSource = "native"


# --- streaming chat: discriminated union ---


@dataclass(frozen=True, slots=True)
class ChatStreamChunk:
    pass


@dataclass(frozen=True, slots=True)
class TokenChunk(ChatStreamChunk):
    text: str = ""


@dataclass(frozen=True, slots=True)
class SourcesChunk(ChatStreamChunk):
    paths: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class StatusChunk(ChatStreamChunk):
    message: str = ""


@dataclass(frozen=True, slots=True)
class ChatResult:
    answer: str
    sources: tuple[str, ...]
