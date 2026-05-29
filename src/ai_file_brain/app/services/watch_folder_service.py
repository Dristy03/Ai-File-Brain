from __future__ import annotations

import logging
from pathlib import Path

from ai_file_brain.app.view_models.status_bar_vm import StatusBarViewModel
from ai_file_brain.config import AiFileBrainSettings, save_user_overrides
from ai_file_brain.core.watcher import FileWatcherService

logger = logging.getLogger(__name__)


class WatchFolderService:
    """Hot-swap the folder the watcher is observing, persist the choice."""

    def __init__(
        self,
        settings: AiFileBrainSettings,
        watcher: FileWatcherService,
        status_vm: StatusBarViewModel,
    ) -> None:
        self._settings = settings
        self._watcher = watcher
        self._status_vm = status_vm

    async def change_to(self, new_path: str) -> str:
        resolved = Path(new_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Folder does not exist: {resolved}")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Not a directory: {resolved}")

        canonical = str(resolved)
        current_canonical = str(Path(self._settings.watch_folder).expanduser().resolve())
        if canonical == current_canonical:
            logger.info("Watch folder unchanged: %s", canonical)
            return canonical

        logger.info("Switching watch folder %s -> %s", current_canonical, canonical)
        await self._watcher.stop()

        self._settings.watch_folder = canonical
        save_user_overrides({"watch_folder": canonical})
        self._status_vm.watch_folder = canonical

        await self._watcher.start()
        return canonical
