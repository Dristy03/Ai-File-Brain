from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ai_file_brain.config import AiFileBrainSettings
from ai_file_brain.core.models import FileChunk
from ai_file_brain.core.storage import ChromaVectorRepository


class _StubCollection:
    def __init__(self) -> None:
        self.last_metadatas: list[dict] | None = None
        self.last_query_where: dict | None = None

    def upsert(self, ids, embeddings, documents, metadatas):
        self.last_metadatas = list(metadatas)

    def query(self, query_embeddings, n_results, where=None):
        self.last_query_where = where
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


def _chunk(file_path: str, file_name: str, modified: datetime, *, text: str = "body",
           chunk_index: int = 0) -> FileChunk:
    return FileChunk(
        id=f"{file_path}::{chunk_index}",
        file_path=file_path,
        file_name=file_name,
        chunk_index=chunk_index,
        text=text,
        created_at=modified,
        modified_at=modified,
    )


async def _repo_with_sidecar(tmp_path) -> ChromaVectorRepository:
    """A repo whose sidecar is initialized on a temp DB and whose Chroma
    collection is a no-op stub that just absorbs upserts/deletes."""
    settings = AiFileBrainSettings()
    settings.watch_folder = str(tmp_path / "watch")
    settings.chroma_path = str(tmp_path / "db")
    repo = ChromaVectorRepository(settings)
    repo._collection = _SidecarStubCollection()
    await repo._meta.initialize()
    return repo


class _SidecarStubCollection:
    """Absorbs the Chroma side of upsert/delete so sidecar-backed repo methods
    can be exercised without a real Chroma collection. Counts delete calls."""

    def __init__(self) -> None:
        self.delete_calls = 0

    def upsert(self, ids, embeddings, documents, metadatas):
        pass

    def delete(self, where=None):
        self.delete_calls += 1


@pytest.mark.asyncio
async def test_extraction_source_round_trips_through_metadata(tmp_path):
    settings = AiFileBrainSettings()
    settings.chroma_path = str(tmp_path / "db")
    repo = ChromaVectorRepository(settings)
    stub = _StubCollection()
    repo._collection = stub  # bypass real chromadb init
    await repo._meta.initialize()  # upsert_batch now mirrors into the sidecar

    now = datetime.now(UTC)
    chunk_native = FileChunk(
        id="a",
        file_path="/p/native.txt",
        file_name="native.txt",
        chunk_index=0,
        text="native body",
        created_at=now,
        modified_at=now,
        extraction_source="native",
    )
    chunk_ocr = FileChunk(
        id="b",
        file_path="/p/scan.png",
        file_name="scan.png",
        chunk_index=0,
        text="ocr body",
        created_at=now,
        modified_at=now,
        extraction_source="ocr",
    )

    await repo.upsert_batch([chunk_native, chunk_ocr], [[0.1, 0.2], [0.3, 0.4]])

    assert stub.last_metadatas is not None
    assert stub.last_metadatas[0]["extraction_source"] == "native"
    assert stub.last_metadatas[1]["extraction_source"] == "ocr"


@pytest.mark.asyncio
async def test_query_excludes_filename_only_chunks():
    """Filename-only stubs (.zip, .exe, …) must not pollute semantic results."""
    repo = ChromaVectorRepository(AiFileBrainSettings())
    stub = _StubCollection()
    repo._collection = stub

    await repo.query([0.1, 0.2, 0.3], top_k=5)

    assert stub.last_query_where == {"extraction_source": {"$ne": "filename_only"}}


@pytest.mark.asyncio
async def test_query_filename_only_restricts_to_stubs():
    """The filename-only semantic pass is the inverse of query(): it searches
    *only* filename-only stubs so a conceptual question can reach a file by its
    name."""
    repo = ChromaVectorRepository(AiFileBrainSettings())
    stub = _StubCollection()
    repo._collection = stub

    await repo.query_filename_only([0.1, 0.2, 0.3], n=5)

    assert stub.last_query_where == {"extraction_source": "filename_only"}


@pytest.mark.asyncio
async def test_query_filename_only_with_time_window_ands_clauses():
    repo = ChromaVectorRepository(AiFileBrainSettings())
    stub = _StubCollection()
    repo._collection = stub

    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 2, 1, tzinfo=UTC)
    await repo.query_filename_only([0.1, 0.2, 0.3], n=5, modified_at_range=(start, end))

    where = stub.last_query_where
    assert where is not None and "$and" in where
    assert {"extraction_source": "filename_only"} in where["$and"]


@pytest.mark.asyncio
async def test_upsert_writes_numeric_modified_at_ts():
    """The time-window filter needs a numeric field; Chroma rejects the ISO
    string for $gte/$lt. Upsert must mirror modified_at into modified_at_ts."""
    repo = ChromaVectorRepository(AiFileBrainSettings())
    stub = _StubCollection()
    repo._collection = stub
    await repo._meta.initialize()

    modified = datetime(2026, 6, 4, 9, 30, tzinfo=UTC)
    await repo.upsert_batch([_chunk("/p/x.txt", "x.txt", modified)], [[0.1]])

    assert stub.last_metadatas is not None
    ts = stub.last_metadatas[0]["modified_at_ts"]
    assert isinstance(ts, float)
    assert ts == modified.timestamp()


@pytest.mark.asyncio
async def test_query_time_window_filters_on_numeric_ts():
    """The range clauses must target modified_at_ts with numeric bounds, not the
    ISO string (which Chroma's $gte/$lt reject)."""
    repo = ChromaVectorRepository(AiFileBrainSettings())
    stub = _StubCollection()
    repo._collection = stub

    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 2, 1, tzinfo=UTC)
    await repo.query([0.1, 0.2, 0.3], top_k=5, modified_at_range=(start, end))

    clauses = stub.last_query_where["$and"]
    assert {"modified_at_ts": {"$gte": start.timestamp()}} in clauses
    assert {"modified_at_ts": {"$lt": end.timestamp()}} in clauses
    # The old string filter must be gone entirely.
    assert not any("modified_at" in c and "modified_at_ts" not in c for c in clauses)


@pytest.mark.asyncio
async def test_query_with_time_window_still_excludes_filename_only():
    repo = ChromaVectorRepository(AiFileBrainSettings())
    stub = _StubCollection()
    repo._collection = stub

    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 2, 1, tzinfo=UTC)
    await repo.query([0.1, 0.2, 0.3], top_k=5, modified_at_range=(start, end))

    where = stub.last_query_where
    assert where is not None and "$and" in where
    clauses = where["$and"]
    assert {"extraction_source": {"$ne": "filename_only"}} in clauses


@pytest.mark.asyncio
async def test_filechunk_default_extraction_source_is_native():
    now = datetime.now(UTC)
    chunk = FileChunk(
        id="a",
        file_path="/p/x.txt",
        file_name="x.txt",
        chunk_index=0,
        text="x",
        created_at=now,
        modified_at=now,
    )
    assert chunk.extraction_source == "native"


class _BackfillStubCollection:
    """In-memory Chroma stand-in for the modified_at_ts backfill: holds records
    as {id: metadata}, and supports count/get(where,include)/update."""

    def __init__(self, records: dict[str, dict]) -> None:
        self.records = records
        self.update_calls = 0

    def count(self) -> int:
        return len(self.records)

    def get(self, include=None, where=None, limit=None):
        ids = list(self.records)
        if where is not None and "modified_at_ts" in where:
            # Mimic Chroma: only records that actually carry the numeric field
            # match a numeric $gte filter.
            ids = [i for i in ids if self.records[i].get("modified_at_ts") is not None]
        out = {"ids": ids}
        if include and "metadatas" in include:
            out["metadatas"] = [dict(self.records[i]) for i in ids]
        return out

    def update(self, ids, metadatas):
        self.update_calls += 1
        for chunk_id, meta in zip(ids, metadatas):
            self.records[chunk_id] = dict(meta)


@pytest.mark.asyncio
async def test_backfill_populates_missing_modified_at_ts(tmp_path):
    """Chunks indexed before the numeric field existed get it seeded from their
    ISO modified_at, without a re-index."""
    settings = AiFileBrainSettings()
    settings.chroma_path = str(tmp_path / "db")
    repo = ChromaVectorRepository(settings)
    iso = "2026-06-04T09:30:00+00:00"
    stub = _BackfillStubCollection(
        {
            "a": {"file_path": "/p/a.txt", "modified_at": iso},
            "b": {"file_path": "/p/b.txt", "modified_at": iso, "modified_at_ts": 1.0},
            "c": {"file_path": "/p/c.txt", "modified_at": "not-a-date"},
        }
    )
    repo._collection = stub

    await repo._backfill_modified_ts_if_needed()

    # 'a' gets the parsed timestamp; existing 'b' is left untouched; 'c' is
    # skipped (unparseable) but keeps its other metadata.
    assert stub.records["a"]["modified_at_ts"] == datetime.fromisoformat(iso).timestamp()
    assert stub.records["a"]["file_path"] == "/p/a.txt"  # other keys preserved
    assert stub.records["b"]["modified_at_ts"] == 1.0
    assert "modified_at_ts" not in stub.records["c"]


@pytest.mark.asyncio
async def test_backfill_noops_when_all_present(tmp_path):
    """Steady state: every chunk already has the field, so no update is issued."""
    settings = AiFileBrainSettings()
    settings.chroma_path = str(tmp_path / "db")
    repo = ChromaVectorRepository(settings)
    stub = _BackfillStubCollection(
        {"a": {"file_path": "/p/a.txt", "modified_at": "x", "modified_at_ts": 1.0}}
    )
    repo._collection = stub

    await repo._backfill_modified_ts_if_needed()

    assert stub.update_calls == 0


class _ScopingStubCollection:
    """Returns a fixed mix of in-folder and out-of-folder chunks regardless of query."""

    def __init__(self, watch_folder: str) -> None:
        from pathlib import Path as _Path

        now = datetime.now(UTC).isoformat()
        self._inside = {
            "id": "inside-1",
            "file_path": str(_Path(watch_folder) / "keep.txt"),
            "file_name": "keep.txt",
            "modified_at": now,
        }
        self._outside_d = {
            "id": "old-d-1",
            "file_path": r"D:\old\stale.txt",
            "file_name": "stale.txt",
            "modified_at": now,
        }

    def _meta(self, entry):
        return {
            "file_path": entry["file_path"],
            "file_name": entry["file_name"],
            "chunk_index": 0,
            "modified_at": entry["modified_at"],
            "created_at": entry["modified_at"],
            "extraction_source": "native",
        }

    def query(self, query_embeddings, n_results, where=None):
        entries = [self._outside_d, self._inside] * (n_results // 2 + 1)
        entries = entries[:n_results]
        return {
            "ids": [[e["id"] for e in entries]],
            "documents": [["body" for _ in entries]],
            "metadatas": [[self._meta(e) for e in entries]],
            "distances": [[0.1 for _ in entries]],
        }

    def get(self, include=None, where=None, limit=None):
        entries = [self._outside_d, self._inside]
        return {
            "ids": [e["id"] for e in entries],
            "documents": ["body" for _ in entries],
            "metadatas": [self._meta(e) for e in entries],
        }


@pytest.mark.asyncio
async def test_query_scopes_results_to_current_watch_folder(tmp_path):
    """Old chunks from a previous watch folder must not pollute results."""
    settings = AiFileBrainSettings()
    settings.watch_folder = str(tmp_path)
    repo = ChromaVectorRepository(settings)
    repo._collection = _ScopingStubCollection(settings.watch_folder)

    hits = await repo.query([0.1, 0.2, 0.3], top_k=3)

    assert hits, "expected at least one in-folder hit"
    assert all(str(tmp_path) in h.file_path for h in hits)
    assert not any(h.file_path.startswith(r"D:\old") for h in hits)


@pytest.mark.asyncio
async def test_most_recent_scopes_to_current_watch_folder(tmp_path):
    """most_recent now reads the sidecar; out-of-folder files must not leak."""
    repo = await _repo_with_sidecar(tmp_path)
    watch = repo._settings.watch_folder
    now = datetime.now(UTC)
    await repo.upsert_batch(
        [
            _chunk(str(Path(watch) / "keep.txt"), "keep.txt", now),
            _chunk(r"D:\old\stale.txt", "stale.txt", now),
        ],
        [[0.1], [0.2]],
    )

    hits = await repo.most_recent(10)

    assert hits, "expected at least one in-folder hit"
    assert all(watch in h.file_path for h in hits)
    assert not any(h.file_path.startswith(r"D:\old") for h in hits)


@pytest.mark.asyncio
async def test_most_recent_orders_newest_first(tmp_path):
    repo = await _repo_with_sidecar(tmp_path)
    watch = repo._settings.watch_folder
    older = datetime(2026, 1, 1, tzinfo=UTC)
    newer = datetime(2026, 6, 1, tzinfo=UTC)
    await repo.upsert_batch(
        [
            _chunk(str(Path(watch) / "old.txt"), "old.txt", older),
            _chunk(str(Path(watch) / "new.txt"), "new.txt", newer),
        ],
        [[0.1], [0.2]],
    )

    hits = await repo.most_recent(10)

    assert [h.file_name for h in hits] == ["new.txt", "old.txt"]


@pytest.mark.asyncio
async def test_filename_substring_scopes_to_current_watch_folder(tmp_path):
    repo = await _repo_with_sidecar(tmp_path)
    watch = repo._settings.watch_folder
    now = datetime.now(UTC)
    # Both files have a "txt" token; only the in-folder one should win.
    await repo.upsert_batch(
        [
            _chunk(str(Path(watch) / "keep.txt"), "keep.txt", now),
            _chunk(r"D:\old\stale.txt", "stale.txt", now),
        ],
        [[0.1], [0.2]],
    )

    hits = await repo.query_by_filename_substrings(["txt"], n=10)

    assert hits, "expected at least one in-folder hit"
    assert all(watch in h.file_path for h in hits)


@pytest.mark.asyncio
async def test_path_mtimes_returns_mtime_per_file(tmp_path):
    repo = await _repo_with_sidecar(tmp_path)
    watch = repo._settings.watch_folder
    older = datetime(2026, 1, 1, tzinfo=UTC)
    newer = datetime(2026, 6, 1, tzinfo=UTC)
    a = str(Path(watch) / "a.txt")
    b = str(Path(watch) / "b.txt")
    await repo.upsert_batch(
        [_chunk(a, "a.txt", older), _chunk(b, "b.txt", newer)],
        [[0.1], [0.2]],
    )

    mtimes = await repo.path_mtimes()

    assert mtimes == {a: older, b: newer}


@pytest.mark.asyncio
async def test_delete_by_path_removes_from_sidecar(tmp_path):
    repo = await _repo_with_sidecar(tmp_path)
    watch = repo._settings.watch_folder
    now = datetime.now(UTC)
    p = str(Path(watch) / "doomed.txt")
    await repo.upsert_batch([_chunk(p, "doomed.txt", now)], [[0.1]])
    assert await repo.has_path(p)

    await repo.delete_by_path(p)

    assert not await repo.has_path(p)
    assert await repo.all_file_paths() == set()


@pytest.mark.asyncio
async def test_delete_by_path_skips_chroma_for_unindexed_path(tmp_path):
    """A delete for a path the sidecar doesn't know must NOT hit Chroma — that's
    the wasted per-file cost during a fresh scan of all-new files."""
    repo = await _repo_with_sidecar(tmp_path)
    stub = repo._collection

    await repo.delete_by_path(str(Path(repo._settings.watch_folder) / "never_indexed.txt"))

    assert stub.delete_calls == 0


@pytest.mark.asyncio
async def test_delete_by_path_hits_chroma_for_indexed_path(tmp_path):
    repo = await _repo_with_sidecar(tmp_path)
    stub = repo._collection
    now = datetime.now(UTC)
    p = str(Path(repo._settings.watch_folder) / "real.txt")
    await repo.upsert_batch([_chunk(p, "real.txt", now)], [[0.1]])

    await repo.delete_by_path(p)

    assert stub.delete_calls == 1
