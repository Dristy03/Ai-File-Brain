from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from watchdog.events import (
    DirDeletedEvent,
    DirMovedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from ai_file_brain.config import AiFileBrainSettings
from ai_file_brain.core.chunking import ChunkingService
from ai_file_brain.core.embedding import EmbeddingService
from ai_file_brain.core.exclusions import (
    classify_path,
    prune_excluded_dirs,
)
from ai_file_brain.core.extraction import get_extractor, is_supported
from ai_file_brain.core.models import FileChunk
from ai_file_brain.core.storage import VectorRepository, _under_watch_folder

logger = logging.getLogger(__name__)

DEBOUNCE_SECONDS = 0.5
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = (1.0, 2.0, 4.0)

# Bulk-scan throttling. The threaded directory walk ships files to the indexing
# workers in batches through a bounded queue; when the queue is full the walk
# blocks (back-pressure) instead of piling millions of paths into memory.
SCAN_BATCH_SIZE = 200
INDEX_QUEUE_MAXSIZE = 1000


ProgressCallback = Callable[["IndexingProgress"], None]

_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SEPARATOR_RE = re.compile(r"[_\-.]+")


def _filename_to_text(file_name: str) -> str:
    """Turn a filename into clean words for embedding.

    Drops the extension and splits separators / camelCase so the embedding
    reflects the *meaning* of the name:
    ``"Office_Attendance_May.xlsx"`` -> ``"Office Attendance May"``. Falls back
    to the raw name when there's nothing to split.
    """
    stem = os.path.splitext(file_name)[0]
    spaced = _SEPARATOR_RE.sub(" ", stem)
    spaced = _CAMEL_BOUNDARY_RE.sub(" ", spaced)
    spaced = " ".join(spaced.split())
    return spaced or file_name


@dataclass(frozen=True, slots=True)
class IndexingProgress:
    file_path: str
    state: str  # "indexing" | "indexed" | "deleted" | "error"
    detail: str = ""


class IndexingPipeline:
    def __init__(
        self,
        chunker: ChunkingService,
        embedder: EmbeddingService,
        repo: VectorRepository,
        settings: AiFileBrainSettings,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._repo = repo
        self._settings = settings
        self._content_exts = frozenset(e.lower() for e in settings.content_extensions)

    def _is_content(self, file_path: str) -> bool:
        """A file is content-indexed only if it's in ``content_extensions`` AND
        actually has a registered extractor; otherwise it falls back to a stub."""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self._content_exts and is_supported(file_path)

    async def index_file(self, file_path: str) -> int:
        for attempt, backoff in enumerate(RETRY_BACKOFF_SECONDS, start=1):
            try:
                if self._is_content(file_path):
                    return await self._index_supported_once(file_path)
                # Name-only tier (code, images this phase, .xlsx/.mp4/.zip …):
                # store a tiny filename-only stub so substring / name search can
                # still find it. Excluded from semantic content search via a
                # metadata filter in the repo layer.
                return await self._index_filename_only_once(file_path)
            except FileNotFoundError:
                logger.info("File disappeared before indexing: %s", file_path)
                return 0
            except (PermissionError, OSError) as ex:
                if attempt >= MAX_RETRIES:
                    logger.warning(
                        "Giving up indexing %s after %d attempts: %s",
                        file_path,
                        attempt,
                        ex,
                    )
                    return 0
                logger.debug(
                    "Indexing %s failed (attempt %d/%d): %s; retrying in %.1fs",
                    file_path,
                    attempt,
                    MAX_RETRIES,
                    ex,
                    backoff,
                )
                await asyncio.sleep(backoff)
        return 0

    async def _index_filename_only_once(self, file_path: str) -> int:
        try:
            stat = os.stat(file_path)
        except OSError:
            return 0
        file_name = os.path.basename(file_path)
        if not file_name:
            return 0
        modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        created = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)

        # Embed the *meaning* of the filename (extension dropped, separators and
        # camelCase split into words) so a conceptual question can reach this
        # file semantically via query_filename_only — e.g. "office timings"
        # finding "Office_Attendance.xlsx". The stored chunk text stays the raw
        # filename for display / substring matching.
        embedding = await self._embedder.embed(_filename_to_text(file_name))
        if not embedding:
            return 0

        chunk = FileChunk(
            id=FileChunk.make_id(file_path, 0),
            file_path=file_path,
            file_name=file_name,
            chunk_index=0,
            text=file_name,
            created_at=created,
            modified_at=modified,
            extraction_source="filename_only",
        )
        await self._repo.delete_by_path(file_path)
        await self._repo.upsert_batch([chunk], [embedding])
        return 1

    async def _index_supported_once(self, file_path: str) -> int:
        extractor = get_extractor(file_path)

        try:
            size = os.path.getsize(file_path)
        except OSError:
            return 0
        if size > self._settings.max_file_size_bytes:
            logger.info(
                "Skipping %s: %d bytes exceeds max_file_size_bytes (%d)",
                file_path,
                size,
                self._settings.max_file_size_bytes,
            )
            await self._repo.delete_by_path(file_path)
            return 0

        result = await extractor.extract(file_path)
        if not result.text.strip():
            logger.info("No extractable text in %s; clearing prior chunks", file_path)
            await self._repo.delete_by_path(file_path)
            return 0

        text_chunks = self._chunker.chunk(result.text)
        if not text_chunks:
            await self._repo.delete_by_path(file_path)
            return 0

        stat = os.stat(file_path)
        modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        created = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
        file_name = os.path.basename(file_path)

        await self._repo.delete_by_path(file_path)

        # Fold the filename's *meaning* into every chunk's embedding input (not
        # its stored text) so a conceptual question reaches the file by its name
        # even when the body never says the query words — e.g. "office timings"
        # finding "Office_Attendance.docx". The filename is short relative to a
        # chunk, so content relevance is barely perturbed. The raw content is
        # what's stored and shown to the LLM (which already sees the filename in
        # the per-file header).
        name_text = _filename_to_text(file_name)
        embed_inputs = [
            f"{name_text}\n{tc.text}" if name_text else tc.text for tc in text_chunks
        ]
        vectors = await self._embedder.embed_batch(embed_inputs)

        chunks: list[FileChunk] = []
        embeddings: list[list[float]] = []
        for tc, embedding in zip(text_chunks, vectors, strict=True):
            if not embedding:
                continue
            chunks.append(
                FileChunk(
                    id=FileChunk.make_id(file_path, tc.chunk_index),
                    file_path=file_path,
                    file_name=file_name,
                    chunk_index=tc.chunk_index,
                    text=tc.text,
                    created_at=created,
                    modified_at=modified,
                    extraction_source=result.source,
                )
            )
            embeddings.append(embedding)

        if chunks:
            await self._repo.upsert_batch(chunks, embeddings)
        return len(chunks)


class FileWatcherService:
    def __init__(
        self,
        settings: AiFileBrainSettings,
        pipeline: IndexingPipeline,
        repo: VectorRepository,
        progress: ProgressCallback | None = None,
    ) -> None:
        self._settings = settings
        self._pipeline = pipeline
        self._repo = repo
        self._progress = progress
        self._loop: asyncio.AbstractEventLoop | None = None
        self._observer: Observer | None = None
        self._handler: _Handler | None = None
        self._debouncers: dict[str, asyncio.TimerHandle] = {}
        self._tasks: set[asyncio.Task] = set()
        self._running = False
        # Bounded work queue + fixed worker pool drain bulk scans so a giant
        # watch root can't spawn unbounded indexing tasks. Live file events stay
        # on the direct task path (they're naturally low-rate).
        self._index_queue: asyncio.Queue[str] | None = None
        self._workers: list[asyncio.Task] = []
        self._concurrency = max(1, settings.max_concurrent_indexing)
        # Precompute the tier sets once; consulted per file during scans/events.
        self._content_exts = frozenset(e.lower() for e in settings.content_extensions)
        self._name_only_exts = frozenset(e.lower() for e in settings.name_only_extensions)

    async def start(self) -> None:
        if self._running:
            return
        self._loop = asyncio.get_running_loop()
        Path(self._settings.watch_folder).mkdir(parents=True, exist_ok=True)
        self._observer = Observer()
        self._handler = _Handler(self)
        self._observer.schedule(self._handler, self._settings.watch_folder, recursive=True)
        self._observer.start()
        self._running = True
        # Start the bounded indexing worker pool before any scan can enqueue work.
        self._index_queue = asyncio.Queue(maxsize=INDEX_QUEUE_MAXSIZE)
        self._workers = [
            self._loop.create_task(self._index_worker()) for _ in range(self._concurrency)
        ]
        logger.info(
            "Watching %s (indexing concurrency=%d)",
            self._settings.watch_folder,
            self._concurrency,
        )
        # Run the initial scan in the background so a large watch root (e.g. D:\)
        # doesn't block the UI from coming up. Tracked in _tasks so stop() cancels it.
        self._schedule_task(self._initial_scan())
        # Files can disappear while the app is closed — watchdog only sees events
        # while running. Reconcile on startup so anything deleted offline is
        # dropped from the index instead of lingering in retrieval forever.
        self._schedule_task(self._reconcile_index())

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for handle in list(self._debouncers.values()):
            handle.cancel()
        self._debouncers.clear()
        for task in list(self._tasks):
            task.cancel()
        for worker in self._workers:
            worker.cancel()
        self._workers.clear()
        self._index_queue = None
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=2.0)
            self._observer = None

    # called from watchdog thread
    def _on_event(self, kind: str, src: str, dst: str | None = None) -> None:
        loop = self._loop
        if not loop or not self._running:
            return
        loop.call_soon_threadsafe(self._handle_event, kind, src, dst)

    def _handle_event(self, kind: str, src: str, dst: str | None) -> None:
        if kind == "deleted":
            # Always emit a delete — even excluded paths may have stale chunks
            # from a prior run with different settings, and the repo's delete is
            # a no-op when there's nothing to remove.
            self._schedule_task(self._handle_delete(src))
            return
        if kind == "dir_deleted":
            # A whole directory was removed. Watchdog reports this as one event,
            # not one-per-contained-file, so purge every indexed file under it.
            self._schedule_task(self._handle_dir_delete(src))
            return
        if kind == "moved":
            if dst and self._should_track(dst):
                self._schedule_task(self._handle_rename(src, dst))
            else:
                # Source moved out of a tracked location → treat as delete.
                self._schedule_task(self._handle_delete(src))
            return
        if kind == "dir_moved":
            # Directory relocated: drop the old subtree's chunks, then re-index
            # the new location if it's still inside the watch folder.
            self._schedule_task(self._handle_dir_move(src, dst))
            return
        # created / modified
        if self._should_track(src):
            self._debounce_index(src)

    def _should_track(self, path: str) -> bool:
        """Track a file only if it falls in a configured indexing tier
        (``content_extensions`` or ``name_only_extensions``) and isn't excluded.
        Everything else — config/data files, unknown binaries — is ignored."""
        return self._classify(path) is not None

    def _classify(self, path: str) -> str | None:
        return classify_path(
            path,
            self._content_exts,
            self._name_only_exts,
            self._settings.excluded_dir_names,
            self._settings.excluded_extensions,
        )

    def _debounce_index(self, file_path: str) -> None:
        loop = self._loop
        if not loop:
            return
        existing = self._debouncers.pop(file_path, None)
        if existing:
            existing.cancel()
        handle = loop.call_later(
            DEBOUNCE_SECONDS,
            lambda p=file_path: self._fire_debounced(p),
        )
        self._debouncers[file_path] = handle

    def _fire_debounced(self, file_path: str) -> None:
        self._debouncers.pop(file_path, None)
        self._schedule_task(self._index(file_path))

    def _schedule_task(self, coro: Awaitable[None]) -> None:
        loop = self._loop
        if not loop:
            return
        task = loop.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _index(self, file_path: str) -> None:
        self._notify(IndexingProgress(file_path, "indexing"))
        try:
            count = await self._pipeline.index_file(file_path)
            self._notify(IndexingProgress(file_path, "indexed", f"{count} chunks"))
        except Exception as ex:
            logger.exception("Indexing failed for %s", file_path)
            self._notify(IndexingProgress(file_path, "error", str(ex)))

    async def _handle_delete(self, file_path: str) -> None:
        try:
            await self._repo.delete_by_path(file_path)
            self._notify(IndexingProgress(file_path, "deleted"))
        except Exception as ex:
            logger.warning("Delete-from-index failed for %s: %s", file_path, ex)

    async def _handle_rename(self, old_path: str, new_path: str) -> None:
        try:
            await self._repo.delete_by_path(old_path)
        except Exception as ex:
            logger.warning("Delete-on-rename failed for %s: %s", old_path, ex)
        await self._index(new_path)

    async def _handle_dir_delete(self, dir_path: str) -> None:
        try:
            await self._repo.delete_under_dir(dir_path)
            self._notify(IndexingProgress(dir_path, "deleted"))
        except Exception as ex:
            logger.warning("Delete-subtree-from-index failed for %s: %s", dir_path, ex)

    async def _handle_dir_move(self, old_dir: str, new_dir: str | None) -> None:
        try:
            await self._repo.delete_under_dir(old_dir)
        except Exception as ex:
            logger.warning("Delete-subtree-on-move failed for %s: %s", old_dir, ex)
        # Moved within the watch folder → re-index the files at their new home.
        # Moved outside → nothing to re-index; the purge above is the whole job.
        if new_dir and _under_watch_folder(new_dir, self._settings.watch_folder):
            await self._scan_and_index(new_dir)

    async def _index_worker(self) -> None:
        """Drain the bulk-scan queue, indexing at most ``_concurrency`` files at
        once. Runs for the life of the watcher; cancelled in ``stop()``."""
        queue = self._index_queue
        if queue is None:
            return
        while True:
            file_path = await queue.get()
            try:
                await self._index(file_path)
            except Exception:
                logger.exception("Worker indexing failed for %s", file_path)
            finally:
                queue.task_done()

    async def _enqueue_scanned(self, paths: list[str]) -> None:
        """Put a batch of scanned paths onto the bounded work queue, skipping
        ones already indexed. ``queue.put`` blocks when the queue is full, which
        is what throttles the producer walk."""
        queue = self._index_queue
        if queue is None:
            return
        for file_path in paths:
            try:
                if await self._repo.has_path(file_path):
                    continue
            except Exception as ex:
                logger.warning("has_path check failed for %s: %s", file_path, ex)
            await queue.put(file_path)

    def _produce_to_queue(self, root: str) -> None:
        """Walk ``root`` in a worker thread, pruning excluded subtrees, and ship
        trackable files to the indexing queue in batches. Runs off the event
        loop so a giant tree (e.g. C:\\) never blocks the UI; back-pressure from
        the bounded queue keeps memory flat. Called via ``asyncio.to_thread``."""
        loop = self._loop
        if loop is None:
            return

        def ship(batch: list[str]) -> None:
            # Hop back onto the event loop to enqueue, and block this thread until
            # it's done so a full queue throttles the walk instead of buffering.
            asyncio.run_coroutine_threadsafe(self._enqueue_scanned(batch), loop).result()

        batch: list[str] = []
        for file_path in self._walk_trackable(root):
            batch.append(file_path)
            if len(batch) >= SCAN_BATCH_SIZE:
                if not self._running:
                    return
                ship(batch)
                batch = []
        if batch and self._running:
            ship(batch)

    def _walk_trackable(self, root: str):
        """Yield indexable file paths under ``root``, pruning excluded directories
        *before descending* so we never enumerate the files inside them. Only
        files in a configured indexing tier are yielded."""
        excluded_dirs = self._settings.excluded_dir_names

        def _on_error(err: OSError) -> None:
            logger.debug("Skipping unreadable path during scan: %s", err)

        for dirpath, dirnames, filenames in os.walk(root, topdown=True, onerror=_on_error):
            dirnames[:] = prune_excluded_dirs(dirnames, excluded_dirs)
            for name in filenames:
                file_path = os.path.join(dirpath, name)
                if self._classify(file_path) is not None:
                    yield file_path

    async def _scan_and_index(self, root: str) -> None:
        try:
            await asyncio.to_thread(self._produce_to_queue, root)
        except OSError as ex:
            logger.warning("Scan of %s failed: %s", root, ex)

    async def _initial_scan(self) -> None:
        await self._scan_and_index(self._settings.watch_folder)

    async def _reconcile_index(self) -> None:
        """Drop index entries for watched files that no longer exist on disk.

        The watcher only sees deletions that happen while it's running; a file
        removed while the app was closed would otherwise linger in the index and
        keep surfacing in retrieval. On startup we list every indexed path under
        the current watch folder and delete the ones that are gone. Paths from a
        previously-watched folder are left alone — those are intentionally
        retained so switching back is instant (see storage scoping).
        """
        try:
            indexed = await self._repo.all_file_paths()
        except Exception as ex:
            logger.warning("Index reconcile failed to list paths: %s", ex)
            return
        folder = self._settings.watch_folder

        def _missing() -> list[str]:
            return [
                p
                for p in indexed
                if _under_watch_folder(p, folder) and not os.path.exists(p)
            ]

        missing = await asyncio.to_thread(_missing)
        for file_path in missing:
            logger.info("Reconcile: dropping vanished file from index: %s", file_path)
            self._schedule_task(self._handle_delete(file_path))

    def _notify(self, progress: IndexingProgress) -> None:
        if self._progress is None:
            return
        try:
            self._progress(progress)
        except Exception:
            logger.exception("Progress callback raised")


class _Handler(FileSystemEventHandler):
    def __init__(self, parent: FileWatcherService) -> None:
        super().__init__()
        self._parent = parent

    def on_created(self, event):
        if isinstance(event, FileCreatedEvent):
            self._parent._on_event("created", event.src_path)

    def on_modified(self, event):
        if isinstance(event, FileModifiedEvent):
            self._parent._on_event("modified", event.src_path)

    def on_deleted(self, event):
        if isinstance(event, DirDeletedEvent):
            self._parent._on_event("dir_deleted", event.src_path)
        elif isinstance(event, FileDeletedEvent):
            self._parent._on_event("deleted", event.src_path)

    def on_moved(self, event):
        if isinstance(event, DirMovedEvent):
            self._parent._on_event("dir_moved", event.src_path, event.dest_path)
        elif isinstance(event, FileMovedEvent):
            self._parent._on_event("moved", event.src_path, event.dest_path)
