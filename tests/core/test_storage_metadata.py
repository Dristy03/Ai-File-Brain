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
    can be exercised without a real Chroma collection."""

    def upsert(self, ids, embeddings, documents, metadatas):
        pass

    def delete(self, where=None):
        pass


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
