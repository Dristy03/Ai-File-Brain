from __future__ import annotations

import asyncio
import logging
import os
import re
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from ai_file_brain.core.models import FileChunk, QueryHit

logger = logging.getLogger(__name__)

# Lives beside the Chroma store (settings.chroma_path_resolved()).
DB_FILENAME = "files-meta.db"

# Backfilled chunks whose Chroma metadata is missing a parseable modified_at get
# this sentinel so they still sort/serialize. They'll read as "older than disk"
# and simply get re-indexed on the next scan, which refreshes the real value.
_EPOCH = datetime.fromtimestamp(0, tz=UTC)

_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _under_watch_folder(file_path: str, watch_folder: str) -> bool:
    """True if ``file_path`` lives under ``watch_folder``.

    Case-insensitive on Windows (via os.path.normcase). An empty/unset
    watch_folder disables scoping.
    """
    if not watch_folder:
        return True
    a = os.path.normcase(os.path.normpath(file_path))
    b = os.path.normcase(os.path.normpath(watch_folder))
    prefix = b if b.endswith(os.sep) else b + os.sep
    return a.startswith(prefix)


def _filename_tokens(file_name: str) -> list[str]:
    """Split a filename into lowercase word tokens on separators and camelCase.

    'Microsoft.NET.Native.Runtime' -> ['microsoft','net','native','runtime'];
    'mcfcoreinstaller.zip' -> ['mcfcoreinstaller','zip']. Powers token-prefix
    filename matching: 'time' won't match inside 'runtime', but 'mcf' still
    matches the start of 'mcfcoreinstaller'.
    """
    spaced = _CAMEL_BOUNDARY_RE.sub(" ", file_name).lower()
    return [t for t in _NON_ALNUM_RE.split(spaced) if t]


def _safe_extraction_source(raw):
    """Coerce a metadata value to a valid ExtractionSource, defaulting to native
    for old chunks that predate the field or for any unrecognized value."""
    if raw in ("native", "ocr", "mixed", "filename_only"):
        return raw
    return "native"


def _parse_modified(iso) -> datetime | None:
    if not isinstance(iso, str):
        return None
    try:
        return datetime.fromisoformat(iso)
    except ValueError:
        return None


class FileMetadataIndex:
    """SQLite sidecar holding one representative row per indexed file.

    Chroma stays the source of truth for vectors; this is a *derived* index that
    lets metadata-only queries — ``most_recent``, ``query_by_filename_substrings``,
    ``all_file_paths``, ``path_mtimes`` — avoid the O(N) full-collection scans
    (``col.get`` of every chunk) they used to do. One row per file: the lowest
    ``chunk_index`` chunk, which carries enough (text, name, mtime) to rebuild a
    :class:`QueryHit` for display without ever touching Chroma.

    A single sqlite3 connection is shared across threads (bulk-scan workers
    upsert concurrently via ``asyncio.to_thread``) and serialized with a
    ``threading.Lock``; WAL keeps readers from blocking the writer. All public
    methods are async and run their sync body off the event loop.
    """

    def __init__(self, db_dir: Path | str) -> None:
        self._db_path = Path(db_dir) / DB_FILENAME
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    # --- lifecycle ---

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    def _initialize_sync(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Opening metadata sidecar at %s", self._db_path)
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        with self._lock:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    file_path TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    modified_at TEXT,
                    extraction_source TEXT NOT NULL DEFAULT 'native'
                )
                """
            )
            conn.commit()
        self._conn = conn

    def _require(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("FileMetadataIndex.initialize() was not called")
        return self._conn

    # --- mutation ---

    async def upsert_files(self, chunks: list[FileChunk]) -> None:
        """Store one representative row per file from ``chunks``.

        Each file's representative is its lowest-``chunk_index`` chunk, so the
        stored text is the start of the file. ``file_path`` is the primary key,
        so re-indexing a file overwrites its row in place.
        """
        if not chunks:
            return
        await asyncio.to_thread(self._upsert_files_sync, chunks)

    def _upsert_files_sync(self, chunks: list[FileChunk]) -> None:
        best: dict[str, FileChunk] = {}
        for c in chunks:
            existing = best.get(c.file_path)
            if existing is None or c.chunk_index < existing.chunk_index:
                best[c.file_path] = c
        rows = [
            (
                c.file_path,
                c.file_name,
                c.id,
                c.chunk_index,
                c.text,
                c.modified_at.isoformat() if c.modified_at else None,
                c.extraction_source,
            )
            for c in best.values()
        ]
        conn = self._require()
        with self._lock:
            conn.executemany(
                """
                INSERT OR REPLACE INTO files
                    (file_path, file_name, chunk_id, chunk_index, text,
                     modified_at, extraction_source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    async def delete_by_path(self, file_path: str) -> None:
        await asyncio.to_thread(self._delete_by_path_sync, file_path)

    def _delete_by_path_sync(self, file_path: str) -> None:
        conn = self._require()
        with self._lock:
            conn.execute("DELETE FROM files WHERE file_path = ?", (file_path,))
            conn.commit()

    async def delete_under_dir(self, dir_path: str) -> list[str]:
        """Delete every file row living under ``dir_path`` (recursive) and return
        the deleted paths so the caller can purge the matching Chroma chunks.

        Replaces the old all-of-Chroma scan: the sidecar already holds every
        distinct path, so prefix-matching here is cheap.
        """
        return await asyncio.to_thread(self._delete_under_dir_sync, dir_path)

    def _delete_under_dir_sync(self, dir_path: str) -> list[str]:
        norm = os.path.normcase(os.path.normpath(dir_path))
        prefix = norm if norm.endswith(os.sep) else norm + os.sep
        conn = self._require()
        with self._lock:
            rows = conn.execute("SELECT file_path FROM files").fetchall()
            victims = [
                p
                for (p,) in rows
                if os.path.normcase(os.path.normpath(p)).startswith(prefix)
            ]
            if victims:
                conn.executemany(
                    "DELETE FROM files WHERE file_path = ?", [(v,) for v in victims]
                )
                conn.commit()
        return victims

    # --- reads ---

    async def is_empty(self) -> bool:
        return await asyncio.to_thread(self._is_empty_sync)

    def _is_empty_sync(self) -> bool:
        conn = self._require()
        with self._lock:
            row = conn.execute("SELECT 1 FROM files LIMIT 1").fetchone()
        return row is None

    async def has_path(self, file_path: str) -> bool:
        return await asyncio.to_thread(self._has_path_sync, file_path)

    def _has_path_sync(self, file_path: str) -> bool:
        conn = self._require()
        with self._lock:
            row = conn.execute(
                "SELECT 1 FROM files WHERE file_path = ? LIMIT 1", (file_path,)
            ).fetchone()
        return row is not None

    async def all_file_paths(self) -> set[str]:
        return await asyncio.to_thread(self._all_file_paths_sync)

    def _all_file_paths_sync(self) -> set[str]:
        conn = self._require()
        with self._lock:
            rows = conn.execute("SELECT file_path FROM files").fetchall()
        return {p for (p,) in rows if p}

    async def path_mtimes(self) -> dict[str, datetime]:
        return await asyncio.to_thread(self._path_mtimes_sync)

    def _path_mtimes_sync(self) -> dict[str, datetime]:
        conn = self._require()
        with self._lock:
            rows = conn.execute("SELECT file_path, modified_at FROM files").fetchall()
        out: dict[str, datetime] = {}
        for path, iso in rows:
            mtime = _parse_modified(iso)
            if path and mtime is not None:
                out[path] = mtime
        return out

    async def most_recent(self, n: int, watch_folder: str) -> list[QueryHit]:
        if n <= 0:
            return []
        return await asyncio.to_thread(self._most_recent_sync, n, watch_folder)

    def _most_recent_sync(self, n: int, watch_folder: str) -> list[QueryHit]:
        conn = self._require()
        with self._lock:
            # ISO 8601 strings (all UTC) sort lexicographically by instant, so
            # ORDER BY on the stored string gives true newest-first.
            rows = conn.execute(
                """
                SELECT file_path, file_name, chunk_id, chunk_index, text,
                       modified_at, extraction_source
                FROM files
                WHERE modified_at IS NOT NULL
                ORDER BY modified_at DESC
                """
            ).fetchall()
        hits: list[QueryHit] = []
        for row in rows:
            if not _under_watch_folder(row[0], watch_folder):
                continue
            hits.append(_row_to_hit(row))
            if len(hits) >= n:
                break
        return hits

    async def query_by_filename_substrings(
        self, substrings: list[str], n: int, watch_folder: str
    ) -> list[QueryHit]:
        if not substrings or n <= 0:
            return []
        needles = [s.lower() for s in substrings if s]
        if not needles:
            return []
        return await asyncio.to_thread(
            self._query_by_filename_substrings_sync, needles, n, watch_folder
        )

    def _query_by_filename_substrings_sync(
        self, needles: list[str], n: int, watch_folder: str
    ) -> list[QueryHit]:
        conn = self._require()
        with self._lock:
            rows = conn.execute(
                """
                SELECT file_path, file_name, chunk_id, chunk_index, text,
                       modified_at, extraction_source
                FROM files
                """
            ).fetchall()
        hits: list[QueryHit] = []
        for row in rows:
            file_path, file_name = row[0], row[1]
            if not file_name or not file_path:
                continue
            if not _under_watch_folder(file_path, watch_folder):
                continue
            tokens = _filename_tokens(file_name)
            if not any(tok.startswith(needle) for tok in tokens for needle in needles):
                continue
            hits.append(_row_to_hit(row))
            if len(hits) >= n:
                break
        return hits


def _row_to_hit(row) -> QueryHit:
    file_path, file_name, chunk_id, chunk_index, text, modified_iso, source = row
    return QueryHit(
        chunk_id=chunk_id,
        file_path=file_path,
        file_name=file_name,
        chunk_index=int(chunk_index or 0),
        text=text or "",
        distance=0.0,
        modified_at=_parse_modified(modified_iso),
        extraction_source=_safe_extraction_source(source),
    )


def chunks_from_chroma_get(result: dict) -> list[FileChunk]:
    """Rebuild FileChunks from a Chroma ``col.get`` result, for the one-time
    sidecar backfill. ``created_at`` is unused by the sidecar, so we mirror
    ``modified_at`` into it rather than carry a separate field."""
    ids = result.get("ids") or []
    documents = result.get("documents") or [""] * len(ids)
    metadatas = result.get("metadatas") or [{}] * len(ids)
    chunks: list[FileChunk] = []
    for i, chunk_id in enumerate(ids):
        meta = metadatas[i] or {}
        file_path = str(meta.get("file_path", ""))
        if not file_path:
            continue
        modified_at = _parse_modified(meta.get("modified_at")) or _EPOCH
        chunks.append(
            FileChunk(
                id=chunk_id,
                file_path=file_path,
                file_name=str(meta.get("file_name", "")),
                chunk_index=int(meta.get("chunk_index", 0) or 0),
                text=documents[i] or "",
                created_at=modified_at,
                modified_at=modified_at,
                extraction_source=_safe_extraction_source(meta.get("extraction_source")),
            )
        )
    return chunks
