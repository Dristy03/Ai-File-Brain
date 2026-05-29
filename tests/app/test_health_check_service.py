import pytest

from ai_file_brain.app.services.health_check_service import HealthCheckService
from ai_file_brain.app.view_models.status_bar_vm import StatusBarViewModel
from ai_file_brain.config import AiFileBrainSettings


class FakeOllama:
    def __init__(self, ok: bool):
        self.ok = ok

    async def list(self):
        if not self.ok:
            raise RuntimeError("nope")
        return {"models": []}


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
