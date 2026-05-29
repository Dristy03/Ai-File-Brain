from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from ollama import AsyncClient

logger = logging.getLogger(__name__)

# Cap on inputs per Ollama embed request. Keeps a single request bounded for
# files that split into hundreds of chunks, while still collapsing N per-chunk
# round-trips into ~N/EMBED_BATCH_SIZE.
EMBED_BATCH_SIZE = 64


@runtime_checkable
class EmbeddingService(Protocol):
    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbeddingService:
    def __init__(self, client: AsyncClient, model: str) -> None:
        self._client = client
        self._model = model

    async def embed(self, text: str) -> list[float]:
        if not text:
            return []
        result = await self.embed_batch([text])
        return result[0] if result else []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts, returning one vector per input in the same order.

        Callers rely on the 1:1 ordering to pair vectors back to their chunks.
        """
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            group = texts[i : i + EMBED_BATCH_SIZE]
            resp = await self._client.embed(model=self._model, input=group)
            out.extend(list(vec) for vec in resp["embeddings"])
        return out
