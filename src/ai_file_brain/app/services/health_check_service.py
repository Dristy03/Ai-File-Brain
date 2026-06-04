from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from ollama import AsyncClient

from ai_file_brain.app.view_models.status_bar_vm import StatusBarViewModel
from ai_file_brain.config import AiFileBrainSettings
from ai_file_brain.core.storage import VectorRepository

logger = logging.getLogger(__name__)


class HealthCheckService:
    INTERVAL_SECONDS = 10.0

    def __init__(
        self,
        ollama: AsyncClient,
        repo: VectorRepository,
        status: StatusBarViewModel,
        settings: AiFileBrainSettings,
        on_ollama_recovered: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._ollama = ollama
        self._repo = repo
        self._status = status
        self._settings = settings
        self._task: asyncio.Task | None = None
        self._stopped = False
        # Fired (once) when embeddings become possible again — i.e. Ollama is up
        # AND the embedding model is installed — so the watcher can re-index files
        # that failed to embed while it was unreachable or the model was missing.
        self._on_ollama_recovered = on_ollama_recovered
        self._embed_was_ready: bool | None = None
        self._recovery_tasks: set[asyncio.Task] = set()
        self._status.watch_folder = settings.watch_folder

    def start(self) -> None:
        if self._task is not None:
            return
        self._stopped = False
        self._task = asyncio.ensure_future(self._loop())

    def stop(self) -> None:
        self._stopped = True
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None

    async def probe_once(self) -> None:
        await self._probe_ollama()
        await self._probe_chroma()

    async def _loop(self) -> None:
        try:
            await self.probe_once()
            while not self._stopped:
                await asyncio.sleep(self.INTERVAL_SECONDS)
                if self._stopped:
                    return
                await self.probe_once()
        except asyncio.CancelledError:
            return

    async def _probe_ollama(self) -> None:
        installed: list[str] = []
        try:
            resp = await self._ollama.list()
            installed = _installed_model_names(resp)
            healthy = True
        except Exception as ex:
            logger.debug("Ollama probe failed: %s", ex)
            healthy = False
        self._status.ollama_healthy = healthy
        self._status.ollama_checked = True

        # One list() call also tells us whether the models we need are installed,
        # so a fresh machine gets a clear "run: ollama pull <model>" prompt instead
        # of silent embedding/chat failures.
        required = (self._settings.embedding_model, self._settings.chat_model)
        missing = (
            tuple(m for m in required if not _is_installed(m, installed)) if healthy else ()
        )
        self._status.missing_models = missing

        # "Embed-ready" = Ollama up AND embedding model present. A false->true
        # transition (Ollama came back, or the model finished pulling) kicks a
        # re-index of anything that couldn't embed before. Skip the first probe
        # (was None) — startup runs its own scan.
        embed_ready = healthy and _is_installed(self._settings.embedding_model, installed)
        if embed_ready and self._embed_was_ready is False and self._on_ollama_recovered is not None:
            logger.info("Embeddings available again; triggering re-index of pending files")
            self._spawn_recovery()
        self._embed_was_ready = embed_ready

    def _spawn_recovery(self) -> None:
        task = asyncio.ensure_future(self._safe_recover())
        self._recovery_tasks.add(task)
        task.add_done_callback(self._recovery_tasks.discard)

    async def _safe_recover(self) -> None:
        try:
            await self._on_ollama_recovered()
        except Exception:
            logger.exception("Ollama-recovery re-index failed")

    async def _probe_chroma(self) -> None:
        try:
            ok = await self._repo.heartbeat()
            self._status.chroma_healthy = ok
            if ok:
                try:
                    self._status.chunk_count = await self._repo.count()
                except Exception as ex:
                    logger.debug("Chroma count failed: %s", ex)
        except Exception as ex:
            logger.debug("Chroma probe failed: %s", ex)
            self._status.chroma_healthy = False


def _installed_model_names(resp) -> list[str]:
    """Pull model names out of an Ollama ``list()`` response, tolerating both the
    object form (``resp.models[i].model``) and the dict form
    (``resp["models"][i]["model"|"name"]``) across client versions."""
    models = getattr(resp, "models", None)
    if models is None and isinstance(resp, dict):
        models = resp.get("models")
    names: list[str] = []
    for m in models or []:
        name = getattr(m, "model", None) or getattr(m, "name", None)
        if name is None and isinstance(m, dict):
            name = m.get("model") or m.get("name")
        if name:
            names.append(str(name))
    return names


def _is_installed(model: str, installed: list[str]) -> bool:
    """Whether a configured model (e.g. ``llama3.2``) is present, matching Ollama's
    tagged names (``llama3.2:latest``) as well as exact matches."""
    return any(name == model or name.startswith(model + ":") for name in installed)
