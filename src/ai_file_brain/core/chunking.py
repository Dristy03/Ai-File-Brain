from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextChunk:
    text: str
    chunk_index: int
    start: int
    end: int


class ChunkingService:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 400) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[TextChunk]:
        if not text:
            return []

        chunks: list[TextChunk] = []
        n = len(text)
        start = 0
        index = 0

        while start < n:
            end = min(start + self._chunk_size, n)

            if end < n:
                # Prefer to end on whitespace within the last 20% of the window.
                window_start = start + int(self._chunk_size * 0.8)
                pivot = self._last_whitespace(text, window_start, end)
                if pivot > start:
                    end = pivot

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(TextChunk(text=chunk_text, chunk_index=index, start=start, end=end))
                index += 1

            if end >= n:
                break

            next_start = end - self._chunk_overlap
            start = next_start if next_start > start else end

        return chunks

    @staticmethod
    def _last_whitespace(text: str, lo: int, hi: int) -> int:
        for i in range(hi - 1, lo - 1, -1):
            if text[i].isspace():
                return i + 1
        return -1
