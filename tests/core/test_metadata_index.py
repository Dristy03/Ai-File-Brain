from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai_file_brain.config import AiFileBrainSettings
from ai_file_brain.core.metadata_index import (
    FileMetadataIndex,
    chunks_from_chroma_get,
)
from ai_file_brain.core.models import FileChunk
from ai_file_brain.core.storage import ChromaVectorRepository


def _chunk(file_path, file_name, modified, *, text="body", chunk_index=0) -> FileChunk:
    return FileChunk(
        id=f"{file_path}::{chunk_index}",
        file_path=file_path,
        file_name=file_name,
        chunk_index=chunk_index,
        text=text,
        created_at=modified,
        modified_at=modified,
    )


async def _index(tmp_path) -> FileMetadataIndex:
    idx = FileMetadataIndex(tmp_path)
    await idx.initialize()
    return idx


@pytest.mark.asyncio
async def test_upsert_keeps_lowest_chunk_index_as_representative(tmp_path):
    idx = await _index(tmp_path)
    now = datetime.now(UTC)
    await idx.upsert_files(
        [
            _chunk("/p/a.txt", "a.txt", now, text="second", chunk_index=1),
            _chunk("/p/a.txt", "a.txt", now, text="first", chunk_index=0),
            _chunk("/p/a.txt", "a.txt", now, text="third", chunk_index=2),
        ]
    )

    hits = await idx.most_recent(10, watch_folder="")
    assert len(hits) == 1
    assert hits[0].text == "first"
    assert hits[0].chunk_index == 0


@pytest.mark.asyncio
async def test_reupsert_replaces_file_row(tmp_path):
    idx = await _index(tmp_path)
    now = datetime.now(UTC)
    await idx.upsert_files([_chunk("/p/a.txt", "a.txt", now, text="old")])
    await idx.upsert_files([_chunk("/p/a.txt", "a.txt", now, text="new")])

    hits = await idx.most_recent(10, watch_folder="")
    assert len(hits) == 1
    assert hits[0].text == "new"


@pytest.mark.asyncio
async def test_delete_under_dir_returns_victims_and_prunes(tmp_path):
    idx = await _index(tmp_path)
    now = datetime.now(UTC)
    keep = str(tmp_path / "other" / "keep.txt")
    v1 = str(tmp_path / "sub" / "a.txt")
    v2 = str(tmp_path / "sub" / "deep" / "b.txt")
    await idx.upsert_files(
        [
            _chunk(keep, "keep.txt", now),
            _chunk(v1, "a.txt", now),
            _chunk(v2, "b.txt", now),
        ]
    )

    victims = await idx.delete_under_dir(str(tmp_path / "sub"))

    assert set(victims) == {v1, v2}
    assert await idx.all_file_paths() == {keep}


@pytest.mark.asyncio
async def test_path_mtimes_and_has_path(tmp_path):
    idx = await _index(tmp_path)
    older = datetime(2026, 1, 1, tzinfo=UTC)
    await idx.upsert_files([_chunk("/p/a.txt", "a.txt", older)])

    assert await idx.has_path("/p/a.txt")
    assert not await idx.has_path("/p/missing.txt")
    assert await idx.path_mtimes() == {"/p/a.txt": older}


@pytest.mark.asyncio
async def test_filename_substring_token_prefix_matching(tmp_path):
    idx = await _index(tmp_path)
    now = datetime.now(UTC)
    await idx.upsert_files(
        [
            _chunk("/p/mcfcoreinstaller.zip", "mcfcoreinstaller.zip", now),
            _chunk("/p/Microsoft.NET.Native.Runtime.appx", "Microsoft.NET.Native.Runtime.appx", now),
        ]
    )

    # "mcf" matches the start of a token; "time" must NOT match inside "Runtime".
    mcf = await idx.query_by_filename_substrings(["mcf"], n=10, watch_folder="")
    time = await idx.query_by_filename_substrings(["time"], n=10, watch_folder="")

    assert [h.file_name for h in mcf] == ["mcfcoreinstaller.zip"]
    assert time == []


@pytest.mark.asyncio
async def test_is_empty(tmp_path):
    idx = await _index(tmp_path)
    assert await idx.is_empty()
    await idx.upsert_files([_chunk("/p/a.txt", "a.txt", datetime.now(UTC))])
    assert not await idx.is_empty()


def test_chunks_from_chroma_get_skips_pathless_and_defaults_bad_mtime():
    result = {
        "ids": ["1", "2", "3"],
        "documents": ["doc1", "doc2", "doc3"],
        "metadatas": [
            {"file_path": "/p/a.txt", "file_name": "a.txt", "chunk_index": 0,
             "modified_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat()},
            # Missing file_path -> dropped.
            {"file_name": "ghost.txt", "modified_at": "2026-01-01T00:00:00+00:00"},
            # Unparseable mtime -> epoch fallback, still kept.
            {"file_path": "/p/c.txt", "file_name": "c.txt", "modified_at": "not-a-date"},
        ],
    }

    chunks = chunks_from_chroma_get(result)

    paths = {c.file_path for c in chunks}
    assert paths == {"/p/a.txt", "/p/c.txt"}
    c = next(c for c in chunks if c.file_path == "/p/c.txt")
    assert c.modified_at == datetime.fromtimestamp(0, tz=UTC)


class _BackfillStubCollection:
    """A Chroma stub that already holds chunks (count > 0) and serves them via
    get, so the repo's one-time sidecar backfill has something to seed from."""

    def __init__(self, metadatas, documents):
        self._metadatas = metadatas
        self._documents = documents

    def count(self):
        return len(self._metadatas)

    def get(self, include=None, where=None, limit=None):
        return {
            "ids": [str(i) for i in range(len(self._metadatas))],
            "documents": list(self._documents),
            "metadatas": list(self._metadatas),
        }


@pytest.mark.asyncio
async def test_backfill_seeds_empty_sidecar_from_chroma(tmp_path):
    settings = AiFileBrainSettings()
    settings.watch_folder = str(tmp_path)
    settings.chroma_path = str(tmp_path / "db")
    repo = ChromaVectorRepository(settings)
    now = datetime(2026, 6, 1, tzinfo=UTC)
    repo._collection = _BackfillStubCollection(
        metadatas=[
            {"file_path": str(tmp_path / "a.txt"), "file_name": "a.txt",
             "chunk_index": 0, "modified_at": now.isoformat()},
            # Two chunks of one file: only the representative row should land.
            {"file_path": str(tmp_path / "b.txt"), "file_name": "b.txt",
             "chunk_index": 1, "modified_at": now.isoformat()},
            {"file_path": str(tmp_path / "b.txt"), "file_name": "b.txt",
             "chunk_index": 0, "modified_at": now.isoformat()},
        ],
        documents=["a body", "b second", "b first"],
    )

    await repo._meta.initialize()
    await repo._backfill_sidecar_if_needed()

    paths = await repo.all_file_paths()
    assert paths == {str(tmp_path / "a.txt"), str(tmp_path / "b.txt")}
    hits = await repo.query_by_filename_substrings(["b"], n=10)
    assert len(hits) == 1
    assert hits[0].text == "b first"  # lowest chunk_index won


@pytest.mark.asyncio
async def test_backfill_noop_when_sidecar_already_populated(tmp_path):
    settings = AiFileBrainSettings()
    settings.watch_folder = str(tmp_path)
    settings.chroma_path = str(tmp_path / "db")
    repo = ChromaVectorRepository(settings)
    now = datetime.now(UTC)
    repo._collection = _BackfillStubCollection(
        metadatas=[{"file_path": "/should/not/import.txt", "file_name": "import.txt",
                    "chunk_index": 0, "modified_at": now.isoformat()}],
        documents=["x"],
    )
    await repo._meta.initialize()
    # Pre-seed the sidecar so backfill must skip the Chroma scan entirely.
    await repo._meta.upsert_files([_chunk(str(tmp_path / "real.txt"), "real.txt", now)])

    await repo._backfill_sidecar_if_needed()

    assert await repo.all_file_paths() == {str(tmp_path / "real.txt")}
