import pytest

from ai_file_brain.app.services.health_check_service import HealthCheckService
from ai_file_brain.app.view_models.status_bar_vm import StatusBarViewModel
from ai_file_brain.config import AiFileBrainSettings


class FakeOllama:
    def __init__(self, ok: bool, models=("nomic-embed-text", "llama3.2")):
        self.ok = ok
        self.models = list(models)

    async def list(self):
        if not self.ok:
            raise RuntimeError("nope")
        # Ollama reports tagged names like "llama3.2:latest".
        return {"models": [{"model": f"{m}:latest"} for m in self.models]}


class FakeRepo:
    def __init__(self, ok: bool, count: int = 7):
        self.ok = ok
        self._count = count

    async def heartbeat(self) -> bool:
        return self.ok

    async def count(self) -> int:
        if not self.ok:
            raise RuntimeError("offline")
        return self._count


def _settings() -> AiFileBrainSettings:
    return AiFileBrainSettings(
        watch_folder="C:/x",
        ollama_url="http://x",
        chroma_path="./tmp",
    )


@pytest.mark.asyncio
async def test_healthy_probe_sets_flags(qtbot):
    status = StatusBarViewModel()
    svc = HealthCheckService(FakeOllama(True), FakeRepo(True), status, _settings())
    await svc.probe_once()
    assert status.ollama_healthy is True
    assert status.chroma_healthy is True
    assert status.chunk_count == 7
    assert status.watch_folder == "C:/x"


@pytest.mark.asyncio
async def test_unhealthy_probe_clears_flags(qtbot):
    status = StatusBarViewModel()
    svc = HealthCheckService(FakeOllama(False), FakeRepo(False), status, _settings())
    await svc.probe_once()
    assert status.ollama_healthy is False
    assert status.chroma_healthy is False


@pytest.mark.asyncio
async def test_probe_marks_ollama_checked(qtbot):
    status = StatusBarViewModel()
    assert status.ollama_checked is False
    svc = HealthCheckService(FakeOllama(True), FakeRepo(True), status, _settings())
    await svc.probe_once()
    assert status.ollama_checked is True


@pytest.mark.asyncio
async def test_recovery_fires_on_down_then_up(qtbot):
    """Ollama down -> up must trigger the recovery callback exactly once."""
    import asyncio

    calls: list[int] = []

    async def _on_recovered() -> None:
        calls.append(1)

    ollama = FakeOllama(False)
    status = StatusBarViewModel()
    svc = HealthCheckService(
        ollama, FakeRepo(True), status, _settings(), on_ollama_recovered=_on_recovered
    )

    await svc.probe_once()  # down — no recovery
    assert calls == []

    ollama.ok = True
    await svc.probe_once()  # up after down — recovery fires
    await asyncio.sleep(0)  # let the spawned recovery task run
    assert calls == [1]

    await svc.probe_once()  # still up — must not fire again
    await asyncio.sleep(0)
    assert calls == [1]


@pytest.mark.asyncio
async def test_recovery_does_not_fire_on_first_healthy_probe(qtbot):
    """A clean start with Ollama already up must NOT trigger recovery — the
    initial scan already covers indexing."""
    import asyncio

    calls: list[int] = []

    async def _on_recovered() -> None:
        calls.append(1)

    status = StatusBarViewModel()
    svc = HealthCheckService(
        FakeOllama(True), FakeRepo(True), status, _settings(), on_ollama_recovered=_on_recovered
    )

    await svc.probe_once()
    await asyncio.sleep(0)
    assert calls == []


@pytest.mark.asyncio
async def test_missing_model_reported(qtbot):
    """Ollama up but the embedding model not installed -> reported in status."""
    status = StatusBarViewModel()
    # Only the chat model is installed; embedding model (nomic-embed-text) missing.
    svc = HealthCheckService(
        FakeOllama(True, models=["llama3.2"]), FakeRepo(True), status, _settings()
    )
    await svc.probe_once()
    assert status.ollama_healthy is True
    assert "nomic-embed-text" in status.missing_models


@pytest.mark.asyncio
async def test_recovery_waits_for_embedding_model(qtbot):
    """Recovery must fire when the embedding model becomes available, not merely
    when Ollama is reachable."""
    import asyncio

    calls: list[int] = []

    async def _on_recovered() -> None:
        calls.append(1)

    ollama = FakeOllama(True, models=["llama3.2"])  # embedding model missing
    status = StatusBarViewModel()
    svc = HealthCheckService(
        ollama, FakeRepo(True), status, _settings(), on_ollama_recovered=_on_recovered
    )

    await svc.probe_once()  # up but can't embed -> no recovery
    await asyncio.sleep(0)
    assert calls == []
    assert "nomic-embed-text" in status.missing_models

    ollama.models = ["llama3.2", "nomic-embed-text"]  # model finished pulling
    await svc.probe_once()
    await asyncio.sleep(0)
    assert status.missing_models == ()
    assert calls == [1]
