from __future__ import annotations

import asyncio
import logging
from typing import Protocol, runtime_checkable

from ollama import AsyncClient

logger = logging.getLogger(__name__)

# Cap on inputs per Ollama embed request. Keeps a single request bounded for
# files that split into hundreds of chunks, while still collapsing N per-chunk
# round-trips into ~N/EMBED_BATCH_SIZE.
EMBED_BATCH_SIZE = 64

# Ollama occasionally returns an empty / short embeddings list — seen right after
# the model loads or while the server is busy (e.g. mid-generation). A zero-length
# query embedding makes the vector search find nothing, which surfaces as a bogus
# "I couldn't find any relevant content". Retry a few times with backoff so a
# transient empty response doesn't poison a query or an indexing batch.
EMBED_MAX_ATTEMPTS = 3
EMBED_RETRY_BACKOFF_SECONDS = (0.5, 1.5, 3.0)


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
            out.extend(await self._embed_group(group))
        return out

    async def _embed_group(self, group: list[str]) -> list[list[float]]:
        """Embed one batch, retrying when Ollama returns an empty/short result.

        A complete result is one non-empty vector per input. Anything less is
        treated as a transient failure and retried; after the last attempt we
        return whatever came back so the caller still gets 1:1 length where
        possible (callers tolerate empty vectors by skipping those chunks).
        """
        vecs: list[list[float]] = []
        for attempt in range(EMBED_MAX_ATTEMPTS):
            resp = await self._client.embed(model=self._model, input=group)
            vecs = [list(vec) for vec in (resp.get("embeddings") or [])]
            if len(vecs) == len(group) and all(vecs):
                return vecs
            if attempt < EMBED_MAX_ATTEMPTS - 1:
                logger.warning(
                    "Embedding response incomplete (%d/%d non-empty); retrying",
                    sum(1 for v in vecs if v),
                    len(group),
                )
                await asyncio.sleep(EMBED_RETRY_BACKOFF_SECONDS[attempt])
        logger.warning(
            "Embedding still incomplete after %d attempts for %d inputs",
            EMBED_MAX_ATTEMPTS,
            len(group),
        )
        return vecs
