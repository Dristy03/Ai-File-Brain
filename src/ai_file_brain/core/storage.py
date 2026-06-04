from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Protocol, runtime_checkable

from ai_file_brain.config import AiFileBrainSettings
from ai_file_brain.core.metadata_index import (
    FileMetadataIndex,
    _safe_extraction_source,
    _under_watch_folder,
    chunks_from_chroma_get,
)
from ai_file_brain.core.models import FileChunk, QueryHit

# Re-exported so watcher.py / chat.py keep importing the scoping helper from here.
__all__ = ["ChromaVectorRepository", "VectorRepository", "_under_watch_folder"]

logger = logging.getLogger(__name__)

COLLECTION_NAME = "ai-file-brain"

# Filename-only chunks (extension types we can't extract: .zip, .exe, .mp4 …) are
# excluded from semantic search so they don't compete with content-bearing
# chunks. They still show up via query_by_filename_substrings and most_recent.
_NOT_FILENAME_ONLY_CLAUSE = {"extraction_source": {"$ne": "filename_only"}}

# When the watch folder changes, chunks from the previous folder stay in the DB
# (so switching back is instant) but every query is post-filtered to chunks
# whose file_path lives under the *current* watch folder. We over-fetch from
# Chroma by this factor so that filter still yields top_k results when most
# nearest neighbours are from an old folder.
_QUERY_OVER_FETCH = 10
_QUERY_OVER_FETCH_MAX = 200


@runtime_checkable
class VectorRepository(Protocol):
    async def initialize(self) -> None: ...
    async def upsert(self, chunk: FileChunk, embedding: list[float]) -> None: ...
    async def upsert_batch(
        self, chunks: list[FileChunk], embeddings: list[list[float]]
    ) -> None: ...
    async def delete_by_path(self, file_path: str) -> None: ...
    async def delete_under_dir(self, dir_path: str) -> None: ...
    async def all_file_paths(self) -> set[str]: ...
    async def path_mtimes(self) -> dict[str, datetime]: ...
    async def query(
        self,
        embedding: list[float],
        top_k: int,
        modified_at_range: tuple[datetime, datetime] | None = None,
    ) -> list[QueryHit]: ...
    async def query_filename_only(
        self,
        embedding: list[float],
        n: int,
        modified_at_range: tuple[datetime, datetime] | None = None,
    ) -> list[QueryHit]: ...
    async def most_recent(self, n: int) -> list[QueryHit]: ...
    async def query_by_filename_substrings(
        self, substrings: list[str], n: int
    ) -> list[QueryHit]: ...
    async def has_path(self, file_path: str) -> bool: ...
    async def count(self) -> int: ...
    async def heartbeat(self) -> bool: ...


class ChromaVectorRepository:
    def __init__(self, settings: AiFileBrainSettings) -> None:
        self._settings = settings
        self._client = None
        self._collection = None
        # SQLite sidecar of per-file metadata, kept beside the Chroma store. It
        # backs the metadata-only queries (most_recent, filename search, path
        # listing, mtimes) so they no longer scan every chunk in Chroma.
        self._meta = FileMetadataIndex(settings.chroma_path_resolved())

    async def initialize(self) -> None:
        await asyncio.to_thread(self._init_sync)
        await self._meta.initialize()
        await self._backfill_sidecar_if_needed()

    async def _backfill_sidecar_if_needed(self) -> None:
        """One-time populate of the sidecar from Chroma.

        Handles upgrades from a DB that predates the sidecar (and any case where
        the sidecar file was deleted): if it's empty but Chroma has chunks, do a
        single full scan to seed it, then never again. Steady-state writes keep
        the two in sync from there.
        """
        if not await self._meta.is_empty():
            return
        chunk_count = await self.count()
        if chunk_count == 0:
            return
        logger.info(
            "Backfilling metadata sidecar from Chroma (%d chunks)…", chunk_count
        )
        col = self._require()
        result = await asyncio.to_thread(col.get, include=["metadatas", "documents"])
        chunks = chunks_from_chroma_get(result)
        await self._meta.upsert_files(chunks)
        logger.info("Metadata sidecar backfill complete")

    def _init_sync(self) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        path = self._settings.chroma_path_resolved()
        path.mkdir(parents=True, exist_ok=True)
        logger.info("Opening ChromaDB at %s", path)

        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' ready", COLLECTION_NAME)

    def _require(self):
        if self._collection is None:
            raise RuntimeError("ChromaVectorRepository.initialize() was not called")
        return self._collection

    async def upsert(self, chunk: FileChunk, embedding: list[float]) -> None:
        await self.upsert_batch([chunk], [embedding])

    async def upsert_batch(
        self, chunks: list[FileChunk], embeddings: list[list[float]]
    ) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be same length")
        col = self._require()

        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "file_path": c.file_path,
                "file_name": c.file_name,
                "chunk_index": c.chunk_index,
                "created_at": c.created_at.isoformat(),
                "modified_at": c.modified_at.isoformat(),
                "extraction_source": c.extraction_source,
            }
            for c in chunks
        ]

        await asyncio.to_thread(
            col.upsert,
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        # Mirror the per-file representative row into the sidecar.
        await self._meta.upsert_files(chunks)

    async def delete_by_path(self, file_path: str) -> None:
        col = self._require()
        await asyncio.to_thread(col.delete, where={"file_path": file_path})
        await self._meta.delete_by_path(file_path)

    async def delete_under_dir(self, dir_path: str) -> None:
        """Delete every chunk whose file lives under ``dir_path`` (recursive).

        Used when a whole directory is removed from, or moved out of, the watch
        tree — watchdog reports that as a single directory event, not one event
        per contained file, so the per-file ``delete_by_path`` never fires for
        them. Chroma's ``where`` can't do path-prefix matching, so we get the
        victim paths from the sidecar (which holds every distinct path) and
        delete the matches from both stores in one call each.
        """
        victims = await self._meta.delete_under_dir(dir_path)
        if not victims:
            return
        col = self._require()
        await asyncio.to_thread(col.delete, where={"file_path": {"$in": victims}})

    async def all_file_paths(self) -> set[str]:
        """Every distinct ``file_path`` currently represented in the index.

        Served from the sidecar (one row per file), so this is a cheap indexed
        read rather than a full Chroma scan. Used by the watcher's startup
        reconcile to find indexed files that no longer exist on disk.
        """
        return await self._meta.all_file_paths()

    async def path_mtimes(self) -> dict[str, datetime]:
        """Newest indexed ``modified_at`` per ``file_path``.

        Lets the startup scan skip files that are already indexed *and* unchanged
        on disk, while still re-indexing anything edited while the app was closed.
        Served from the sidecar, so it's a cheap read rather than an O(N) scan of
        every chunk in Chroma.
        """
        return await self._meta.path_mtimes()

    async def query(
        self,
        embedding: list[float],
        top_k: int,
        modified_at_range: tuple[datetime, datetime] | None = None,
    ) -> list[QueryHit]:
        col = self._require()
        where_clauses: list[dict] = [_NOT_FILENAME_ONLY_CLAUSE]
        if modified_at_range is not None:
            start, end = modified_at_range
            # ISO 8601 strings sort lexicographically by date, so $gte/$lte
            # work directly on the stored "modified_at" string metadata.
            where_clauses.append({"modified_at": {"$gte": start.isoformat()}})
            where_clauses.append({"modified_at": {"$lt": end.isoformat()}})
        fetch_n = min(top_k * _QUERY_OVER_FETCH, _QUERY_OVER_FETCH_MAX)
        kwargs: dict = {
            "query_embeddings": [embedding],
            "n_results": fetch_n,
            "where": where_clauses[0] if len(where_clauses) == 1 else {"$and": where_clauses},
        }
        result = await asyncio.to_thread(col.query, **kwargs)
        all_hits = _result_to_hits(result)
        folder = self._settings.watch_folder
        limit = self._settings.max_match_distance
        # Drop neighbours that are merely the *closest* but not actually
        # relevant — without this, every query returns top_k files no matter how
        # unrelated, so a topic with no real match still lists junk "sources".
        scoped = [
            h
            for h in all_hits
            if h.distance <= limit and _under_watch_folder(h.file_path, folder)
        ]
        return scoped[:top_k]

    async def query_filename_only(
        self,
        embedding: list[float],
        n: int,
        modified_at_range: tuple[datetime, datetime] | None = None,
    ) -> list[QueryHit]:
        """Semantic search restricted to filename-only stubs.

        Counterpart to :meth:`query`, which *excludes* those stubs. Lets a
        conceptual question ("office timings") reach a file whose *name* is
        related ("attendance.xlsx") even though the file has no extractable body
        and the query words aren't literal substrings of the name. The
        filename's meaning is what's embedded (see ``_filename_to_text`` in the
        indexer), so semantic similarity does the bridging.
        """
        if n <= 0:
            return []
        col = self._require()
        where_clauses: list[dict] = [{"extraction_source": "filename_only"}]
        if modified_at_range is not None:
            start, end = modified_at_range
            where_clauses.append({"modified_at": {"$gte": start.isoformat()}})
            where_clauses.append({"modified_at": {"$lt": end.isoformat()}})
        fetch_n = min(n * _QUERY_OVER_FETCH, _QUERY_OVER_FETCH_MAX)
        where = where_clauses[0] if len(where_clauses) == 1 else {"$and": where_clauses}
        result = await asyncio.to_thread(
            col.query,
            query_embeddings=[embedding],
            n_results=fetch_n,
            where=where,
        )
        all_hits = _result_to_hits(result)
        folder = self._settings.watch_folder
        limit = self._settings.max_filename_match_distance
        scoped = [
            h
            for h in all_hits
            if h.distance <= limit and _under_watch_folder(h.file_path, folder)
        ]
        return scoped[:n]

    async def most_recent(self, n: int) -> list[QueryHit]:
        """Return up to ``n`` files, newest ``modified_at`` first.

        Used for "what's the latest file I worked on?". Served from the sidecar
        (one row per file, already deduped, ordered by mtime), so it's a cheap
        indexed read rather than the O(N) scan + Python sort it used to be.
        """
        return await self._meta.most_recent(n, self._settings.watch_folder)

    async def query_by_filename_substrings(
        self, substrings: list[str], n: int
    ) -> list[QueryHit]:
        """Return up to ``n`` files whose ``file_name`` has a *word* starting
        with any of the given keywords (case-insensitive). One representative
        chunk per matched file.

        Matching is token-prefix, not raw substring: the filename is split on
        separators and camelCase, and a keyword matches only if some token starts
        with it. So "mcf" still surfaces "mcfcoreinstaller.zip", but "time" no
        longer matches inside "Microsoft...Runtime...appx" — raw-substring
        matching dragged unrelated binaries in as false positives.

        Used for "tell me about <name>" — embeddings can't link a query word
        like "screenshot" to a file whose chunk text doesn't contain it (think
        OCR'd images), but filename matching reliably surfaces the file. Served
        from the sidecar rather than a full Chroma scan.
        """
        return await self._meta.query_by_filename_substrings(
            substrings, n, self._settings.watch_folder
        )

    async def has_path(self, file_path: str) -> bool:
        return await self._meta.has_path(file_path)

    async def count(self) -> int:
        col = self._require()
        return await asyncio.to_thread(col.count)

    async def heartbeat(self) -> bool:
        if self._client is None:
            return False
        try:
            await asyncio.to_thread(self._client.heartbeat)
            return True
        except Exception as ex:
            logger.debug("Chroma heartbeat failed: %s", ex)
            return False


def _result_to_hits(result: dict) -> list[QueryHit]:
    ids_outer = result.get("ids") or []
    if not ids_outer:
        return []
    ids = ids_outer[0] or []
    distances = (result.get("distances") or [[]])[0] or [0.0] * len(ids)
    documents = (result.get("documents") or [[]])[0] or [""] * len(ids)
    metadatas = (result.get("metadatas") or [[]])[0] or [{}] * len(ids)

    hits: list[QueryHit] = []
    for i, chunk_id in enumerate(ids):
        meta = metadatas[i] or {}
        modified_iso = meta.get("modified_at")
        modified_at = None
        if isinstance(modified_iso, str):
            try:
                modified_at = datetime.fromisoformat(modified_iso)
            except ValueError:
                modified_at = None
        hits.append(
            QueryHit(
                chunk_id=chunk_id,
                file_path=str(meta.get("file_path", "")),
                file_name=str(meta.get("file_name", "")),
                chunk_index=int(meta.get("chunk_index", 0) or 0),
                text=documents[i] or "",
                distance=float(distances[i] or 0.0),
                modified_at=modified_at,
                extraction_source=_safe_extraction_source(meta.get("extraction_source")),
            )
        )
    return hits
