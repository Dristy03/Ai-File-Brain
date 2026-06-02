from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime
from typing import Protocol, runtime_checkable

from ai_file_brain.config import AiFileBrainSettings
from ai_file_brain.core.models import FileChunk, QueryHit

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

    async def initialize(self) -> None:
        await asyncio.to_thread(self._init_sync)

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

    async def delete_by_path(self, file_path: str) -> None:
        col = self._require()
        await asyncio.to_thread(col.delete, where={"file_path": file_path})

    async def delete_under_dir(self, dir_path: str) -> None:
        """Delete every chunk whose file lives under ``dir_path`` (recursive).

        Used when a whole directory is removed from, or moved out of, the watch
        tree — watchdog reports that as a single directory event, not one event
        per contained file, so the per-file ``delete_by_path`` never fires for
        them. Chroma's ``where`` can't do path-prefix matching, so we enumerate
        distinct paths and delete the matches in one call.
        """
        paths = await self.all_file_paths()
        norm = os.path.normcase(os.path.normpath(dir_path))
        prefix = norm if norm.endswith(os.sep) else norm + os.sep
        victims = [
            p
            for p in paths
            if os.path.normcase(os.path.normpath(p)).startswith(prefix)
        ]
        if not victims:
            return
        col = self._require()
        await asyncio.to_thread(col.delete, where={"file_path": {"$in": victims}})

    async def all_file_paths(self) -> set[str]:
        """Every distinct ``file_path`` currently represented in the index.

        Scans all chunk metadata (O(N) in collection size), so callers should
        treat it as a maintenance operation, not a hot-path query. Used by the
        watcher's startup reconcile to find indexed files that no longer exist
        on disk.
        """
        col = self._require()
        result = await asyncio.to_thread(col.get, include=["metadatas"])
        metas = result.get("metadatas") or []
        paths: set[str] = set()
        for meta in metas:
            if not meta:
                continue
            p = meta.get("file_path")
            if isinstance(p, str) and p:
                paths.add(p)
        return paths

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
        """Return up to ``n`` chunks, sorted by ``modified_at`` desc, deduped by file_path.

        Used for "what's the latest file I worked on?" — pulls metadata + docs
        and sorts in Python because Chroma can't ``order by`` metadata. Scans
        every chunk, so cost is O(N) where N is the collection size. Fine
        until N gets into the hundreds of thousands.
        """
        if n <= 0:
            return []
        col = self._require()
        result = await asyncio.to_thread(
            col.get,
            include=["metadatas", "documents"],
        )
        return _sort_dedupe_most_recent(result, n, self._settings.watch_folder)

    async def query_by_filename_substrings(
        self, substrings: list[str], n: int
    ) -> list[QueryHit]:
        """Return up to ``n`` chunks whose ``file_name`` has a *word* starting
        with any of the given keywords (case-insensitive). One chunk per matched
        file, the lowest-index chunk first.

        Matching is token-prefix, not raw substring: the filename is split on
        separators and camelCase, and a keyword matches only if some token starts
        with it. So "mcf" still surfaces "mcfcoreinstaller.zip", but "time" no
        longer matches inside "Microsoft...Runtime...appx" — raw-substring
        matching dragged unrelated binaries in as false positives.

        Used for "tell me about <name>" — embeddings can't link a query word
        like "screenshot" to a file whose chunk text doesn't contain it (think
        OCR'd images), but filename matching reliably surfaces the file.
        """
        if not substrings or n <= 0:
            return []
        needles = [s.lower() for s in substrings if s]
        if not needles:
            return []
        col = self._require()
        result = await asyncio.to_thread(
            col.get,
            include=["metadatas", "documents"],
        )
        return _filter_by_filename(result, needles, n, self._settings.watch_folder)

    async def has_path(self, file_path: str) -> bool:
        col = self._require()
        result = await asyncio.to_thread(
            col.get,
            where={"file_path": file_path},
            limit=1,
            include=[],
        )
        ids = result.get("ids") or []
        return bool(ids)

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


_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _filename_tokens(file_name: str) -> list[str]:
    """Split a filename into lowercase word tokens on separators and camelCase.

    'Microsoft.NET.Native.Runtime' -> ['microsoft','net','native','runtime'];
    'mcfcoreinstaller.zip' -> ['mcfcoreinstaller','zip']. Powers token-prefix
    filename matching: 'time' won't match inside 'runtime', but 'mcf' still
    matches the start of 'mcfcoreinstaller'.
    """
    spaced = _CAMEL_BOUNDARY_RE.sub(" ", file_name).lower()
    return [t for t in _NON_ALNUM_RE.split(spaced) if t]


def _filter_by_filename(
    result: dict, needles: list[str], n: int, watch_folder: str
) -> list[QueryHit]:
    ids = result.get("ids") or []
    documents = result.get("documents") or [""] * len(ids)
    metadatas = result.get("metadatas") or [{}] * len(ids)

    # Pick the lowest-chunk-index match per file so we get the start of the file,
    # not an arbitrary chunk somewhere inside.
    best_per_path: dict[str, tuple[int, QueryHit]] = {}
    for i, chunk_id in enumerate(ids):
        meta = metadatas[i] or {}
        file_name = str(meta.get("file_name", ""))
        file_path = str(meta.get("file_path", ""))
        if not file_name or not file_path:
            continue
        if not _under_watch_folder(file_path, watch_folder):
            continue
        tokens = _filename_tokens(file_name)
        if not any(tok.startswith(needle) for tok in tokens for needle in needles):
            continue
        chunk_index = int(meta.get("chunk_index", 0) or 0)
        existing = best_per_path.get(file_path)
        if existing is not None and existing[0] <= chunk_index:
            continue
        modified_iso = meta.get("modified_at")
        modified_at = None
        if isinstance(modified_iso, str):
            try:
                modified_at = datetime.fromisoformat(modified_iso)
            except ValueError:
                modified_at = None
        hit = QueryHit(
            chunk_id=chunk_id,
            file_path=file_path,
            file_name=file_name,
            chunk_index=chunk_index,
            text=documents[i] or "",
            distance=0.0,
            modified_at=modified_at,
            extraction_source=_safe_extraction_source(meta.get("extraction_source")),
        )
        best_per_path[file_path] = (chunk_index, hit)

    return [hit for _index, hit in best_per_path.values()][:n]


def _sort_dedupe_most_recent(
    result: dict, n: int, watch_folder: str
) -> list[QueryHit]:
    ids = result.get("ids") or []
    documents = result.get("documents") or [""] * len(ids)
    metadatas = result.get("metadatas") or [{}] * len(ids)

    items: list[tuple[datetime, str, QueryHit]] = []
    for i, chunk_id in enumerate(ids):
        meta = metadatas[i] or {}
        modified_iso = meta.get("modified_at")
        if not isinstance(modified_iso, str):
            continue
        try:
            modified_at = datetime.fromisoformat(modified_iso)
        except ValueError:
            continue
        file_path = str(meta.get("file_path", ""))
        if not _under_watch_folder(file_path, watch_folder):
            continue
        hit = QueryHit(
            chunk_id=chunk_id,
            file_path=file_path,
            file_name=str(meta.get("file_name", "")),
            chunk_index=int(meta.get("chunk_index", 0) or 0),
            text=documents[i] or "",
            distance=0.0,
            modified_at=modified_at,
            extraction_source=_safe_extraction_source(meta.get("extraction_source")),
        )
        items.append((modified_at, file_path, hit))

    items.sort(key=lambda x: x[0], reverse=True)
    seen: set[str] = set()
    hits: list[QueryHit] = []
    for _modified, file_path, hit in items:
        if file_path in seen:
            continue
        seen.add(file_path)
        hits.append(hit)
        if len(hits) >= n:
            break
    return hits


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


def _safe_extraction_source(raw):
    """Coerce a metadata value to a valid ExtractionSource, defaulting to native
    for old chunks that predate the field or for any unrecognized value."""
    if raw in ("native", "ocr", "mixed", "filename_only"):
        return raw
    return "native"
