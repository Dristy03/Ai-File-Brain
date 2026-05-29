from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from ollama import AsyncClient
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from ai_file_brain.app.models.chat_turn import ChatTurn  # noqa: F401  (Qt meta)
from ai_file_brain.app.services.health_check_service import HealthCheckService
from ai_file_brain.app.services.indexing_activity_service import IndexingActivityService
from ai_file_brain.app.services.tray_icon_service import TrayIconService
from ai_file_brain.app.services.watch_folder_service import WatchFolderService
from ai_file_brain.app.view_models.main_window_vm import MainWindowViewModel
from ai_file_brain.app.view_models.status_bar_vm import StatusBarViewModel
from ai_file_brain.app.views.main_window import MainWindow
from ai_file_brain.config import AiFileBrainSettings
from ai_file_brain.core.chat import ChatService
from ai_file_brain.core.chunking import ChunkingService
from ai_file_brain.core.embedding import OllamaEmbeddingService
from ai_file_brain.core.storage import ChromaVectorRepository
from ai_file_brain.core.watcher import FileWatcherService, IndexingPipeline, IndexingProgress

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: AiFileBrainSettings
    qapp: QApplication
    ollama: AsyncClient
    embedder: OllamaEmbeddingService
    vector_repo: ChromaVectorRepository
    chat: ChatService
    chunker: ChunkingService
    pipeline: IndexingPipeline
    watcher: FileWatcherService
    status_vm: StatusBarViewModel
    main_window_vm: MainWindowViewModel
    main_window: MainWindow
    health_check: HealthCheckService
    tray: TrayIconService
    watch_folder_service: WatchFolderService
    activity_service: IndexingActivityService

    async def startup(self) -> None:
        try:
            await self.vector_repo.initialize()
        except Exception as ex:
            logger.exception("ChromaDB initialization failed")
            QMessageBox.critical(
                None,
                "AI File Brain",
                f"Could not open the local vector store:\n\n{ex}\n\n"
                f"Path: {self.settings.chroma_path_resolved()}",
            )
            self.qapp.quit()
            return

        self.health_check.start()
        await self.watcher.start()
        self.tray.attach()
        self.main_window.show_and_raise()

    async def shutdown(self) -> None:
        try:
            await self.watcher.stop()
        except Exception:
            logger.exception("Watcher stop failed")
        self.health_check.stop()
        self.tray.detach()


def build_container(settings: AiFileBrainSettings, qapp: QApplication, icon: QIcon) -> Container:
    ollama = AsyncClient(host=settings.ollama_url)
    embedder = OllamaEmbeddingService(ollama, settings.embedding_model)
    vector_repo = ChromaVectorRepository(settings)

    chunker = ChunkingService(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chat = ChatService(settings, embedder, vector_repo, ollama)
    pipeline = IndexingPipeline(chunker, embedder, vector_repo, settings)

    status_vm = StatusBarViewModel()
    main_window_vm = MainWindowViewModel(chat)
    main_window = MainWindow(main_window_vm, status_vm)
    health_check = HealthCheckService(ollama, vector_repo, status_vm, settings)
    activity_service = IndexingActivityService(status_vm)

    def _progress(p: IndexingProgress) -> None:
        if p.state == "indexed":
            try:
                # Optimistic local nudge; the 10s health probe will reconcile.
                status_vm.chunk_count = status_vm.chunk_count + _maybe_chunk_delta(p.detail)
            except Exception:
                pass
        activity_service.on_progress(p)

    watcher = FileWatcherService(settings, pipeline, vector_repo, progress=_progress)
    watch_folder_service = WatchFolderService(settings, watcher, status_vm)

    tray_icon_service: TrayIconService | None = None

    def _toggle() -> None:
        main_window.toggle_visibility()

    def _show() -> None:
        main_window.show_and_raise()

    def _change_folder() -> None:
        parent = main_window if main_window.isVisible() else None
        chosen = QFileDialog.getExistingDirectory(
            parent, "Choose folder to watch", settings.watch_folder
        )
        if not chosen:
            return
        asyncio.ensure_future(_apply_folder_change(chosen))

    async def _apply_folder_change(path: str) -> None:
        try:
            await watch_folder_service.change_to(path)
        except Exception as ex:
            logger.exception("Failed to change watch folder")
            QMessageBox.warning(
                main_window,
                "AI File Brain",
                f"Couldn't switch watch folder:\n\n{ex}",
            )

    def _quit() -> None:
        main_window.mark_quitting()
        asyncio.ensure_future(_graceful_quit(qapp))

    async def _graceful_quit(app: QApplication) -> None:
        try:
            if tray_icon_service is not None:
                tray_icon_service.detach()
            health_check.stop()
            await watcher.stop()
        finally:
            app.quit()

    tray_icon_service = TrayIconService(
        icon, _toggle, _show, _change_folder, _quit, status_vm
    )
    main_window.set_change_folder_handler(_change_folder)

    return Container(
        settings=settings,
        qapp=qapp,
        ollama=ollama,
        embedder=embedder,
        vector_repo=vector_repo,
        chat=chat,
        chunker=chunker,
        pipeline=pipeline,
        watcher=watcher,
        status_vm=status_vm,
        main_window_vm=main_window_vm,
        main_window=main_window,
        health_check=health_check,
        tray=tray_icon_service,
        watch_folder_service=watch_folder_service,
        activity_service=activity_service,
    )


def _maybe_chunk_delta(detail: str) -> int:
    # detail format from progress: "<n> chunks"
    try:
        head = detail.split(" ", 1)[0]
        return int(head)
    except (ValueError, IndexError):
        return 0
