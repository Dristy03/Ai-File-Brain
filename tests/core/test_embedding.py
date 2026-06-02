import pytest

import ai_file_brain.core.embedding as em
from ai_file_brain.core.embedding import OllamaEmbeddingService


class FlakyClient:
    """Returns an empty embeddings list for the first ``fail_times`` calls, then
    a valid one — mimics Ollama's transient empty responses under load/cold."""

    def __init__(self, fail_times: int) -> None:
        self.calls = 0
        self.fail_times = fail_times

    async def embed(self, model, input):
        self.calls += 1
        if self.calls <= self.fail_times:
            return {"embeddings": []}
        return {"embeddings": [[0.1, 0.2, 0.3] for _ in input]}


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(em.asyncio, "sleep", _noop)


@pytest.mark.asyncio
async def test_embed_retries_until_valid():
    client = FlakyClient(fail_times=2)
    svc = OllamaEmbeddingService(client, "m")

    out = await svc.embed("hello")

    assert out == [0.1, 0.2, 0.3]
    assert client.calls == 3  # two empty responses, then success


@pytest.mark.asyncio
async def test_embed_gives_up_after_max_attempts():
    client = FlakyClient(fail_times=99)
    svc = OllamaEmbeddingService(client, "m")

    out = await svc.embed("hello")

    assert out == []  # exhausted retries -> empty (caller tolerates)
    assert client.calls == em.EMBED_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_embed_batch_no_retry_when_complete():
    client = FlakyClient(fail_times=0)
    svc = OllamaEmbeddingService(client, "m")

    out = await svc.embed_batch(["a", "b"])

    assert out == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert client.calls == 1  # single request, no retries needed
