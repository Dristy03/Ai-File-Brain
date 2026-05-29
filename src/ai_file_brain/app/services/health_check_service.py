from __future__ import annotations

import asyncio
import logging

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
    ) -> None:
        self._ollama = ollama
        self._repo = repo
        self._status = status
        self._settings = settings
        self._task: asyncio.Task | None = None
        self._stopped = False
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
        try:
            await self._ollama.list()
            self._status.ollama_healthy = True
        except Exception as ex:
            logger.debug("Ollama probe failed: %s", ex)
            self._status.ollama_healthy = False

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
