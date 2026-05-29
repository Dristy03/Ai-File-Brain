from __future__ import annotations

from pathlib import Path

import pytest

from ai_file_brain.config import AiFileBrainSettings
from ai_file_brain.core.chunking import ChunkingService
from ai_file_brain.core.models import FileChunk
from ai_file_brain.core.watcher import IndexingPipeline, _filename_to_text


@pytest.mark.parametrize(
    "file_name, expected",
    [
        ("attendance.xlsx", "attendance"),
        ("Office_Attendance_May.xlsx", "Office Attendance May"),
        ("quarterly-report.final.pdf", "quarterly report final"),
        ("GalaxyConnectWinUI.csproj", "Galaxy Connect Win UI"),
        ("README", "README"),
    ],
)
def test_filename_to_text_splits_into_words(file_name, expected):
    assert _filename_to_text(file_name) == expected


class _FakeEmbedder:
    def __init__(self) -> None:
        self.embed_batch_inputs: list[list[str]] = []

    async def embed(self, text: str) -> list[float]:
        return [0.1] * 8

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.embed_batch_inputs.append(list(texts))
        return [[0.1] * 8 for _ in texts]


class _FakeRepo:
    def __init__(self) -> None:
        self.upserted: list[FileChunk] = []
        self.upserted_embeddings: list[list[float]] = []
        self.deletions: list[str] = []

    async def initialize(self) -> None: ...
    async def upsert(self, *a, **kw) -> None: ...
    async def upsert_batch(self, chunks, embeddings) -> None:
        assert len(chunks) == len(embeddings)
        self.upserted.extend(chunks)
        self.upserted_embeddings.extend(embeddings)
    async def delete_by_path(self, path: str) -> None:
        self.deletions.append(path)
    async def has_path(self, path: str) -> bool:
        return False
    async def query(self, *a, **kw):
        return []
    async def count(self) -> int:
        return 0
    async def heartbeat(self) -> bool:
        return True


def _build_pipeline(settings: AiFileBrainSettings) -> tuple[IndexingPipeline, _FakeRepo]:
    repo = _FakeRepo()
    pipeline = IndexingPipeline(
        chunker=ChunkingService(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap),
        embedder=_FakeEmbedder(),
        repo=repo,
        settings=settings,
    )
    return pipeline, repo


@pytest.mark.asyncio
async def test_pipeline_indexes_small_text_file(tmp_path: Path):
    settings = AiFileBrainSettings()
    pipeline, repo = _build_pipeline(settings)

    f = tmp_path / "note.txt"
    f.write_text("a short note about ai file brain", encoding="utf-8")

    count = await pipeline.index_file(str(f))
    assert count == 1
    assert len(repo.upserted) == 1
    assert repo.upserted[0].extraction_source == "native"


@pytest.mark.asyncio
async def test_pipeline_batch_embeds_multi_chunk_file(tmp_path: Path):
    """A file that splits into several chunks is embedded as a batch, and every
    chunk is paired 1:1 with a vector before upsert."""
    settings = AiFileBrainSettings()
    pipeline, repo = _build_pipeline(settings)

    f = tmp_path / "long.txt"
    f.write_text("word " * 3000, encoding="utf-8")  # ~15k chars -> many chunks

    expected = len(
        ChunkingService(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        ).chunk(f.read_text(encoding="utf-8"))
    )
    assert expected > 1  # sanity: we actually exercised the multi-chunk path

    count = await pipeline.index_file(str(f))
    assert count == expected
    assert len(repo.upserted) == expected
    assert len(repo.upserted_embeddings) == expected


@pytest.mark.asyncio
async def test_pipeline_folds_filename_meaning_into_content_embedding(tmp_path: Path):
    """A content file's chunks are embedded with the filename's *meaning*
    prepended, so a conceptual query ('office timings') can reach the file by
    its name ('Office_Attendance_May') even when the body never says it. The
    *stored* chunk text stays pure content."""
    settings = AiFileBrainSettings()
    embedder = _FakeEmbedder()
    repo = _FakeRepo()
    pipeline = IndexingPipeline(
        chunker=ChunkingService(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        ),
        embedder=embedder,
        repo=repo,
        settings=settings,
    )

    f = tmp_path / "Office_Attendance_May.txt"
    f.write_text("roster of staff present", encoding="utf-8")

    await pipeline.index_file(str(f))

    assert embedder.embed_batch_inputs, "expected one batched embed call"
    embedded = embedder.embed_batch_inputs[0][0]
    assert "Office Attendance May" in embedded  # filename meaning folded in
    assert "roster of staff present" in embedded
    # Stored/displayed text is the raw content only — no filename pollution.
    assert repo.upserted[0].text == "roster of staff present"


@pytest.mark.asyncio
async def test_pipeline_skips_files_over_size_cap(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AFB_MAX_FILE_SIZE_BYTES", "100")
    settings = AiFileBrainSettings()
    assert settings.max_file_size_bytes == 100

    pipeline, repo = _build_pipeline(settings)

    big = tmp_path / "huge.txt"
    big.write_bytes(b"x" * 500)  # 500 bytes > 100-byte cap

    count = await pipeline.index_file(str(big))
    assert count == 0
    assert repo.upserted == []
    # Stale chunks for that path are cleared.
    assert str(big) in repo.deletions


@pytest.mark.asyncio
async def test_pipeline_indexes_unsupported_extension_as_filename_only(tmp_path: Path):
    """Files we can't extract text from still get a tiny filename-only chunk so
    "do I have files about X" can still find them. The chunk is excluded from
    semantic search via the extraction_source metadata filter in the repo."""
    settings = AiFileBrainSettings()
    pipeline, repo = _build_pipeline(settings)

    f = tmp_path / "mcfcoreinstaller.zip"
    f.write_bytes(b"PK\x03\x04")  # would-be ZIP header

    count = await pipeline.index_file(str(f))
    assert count == 1
    assert len(repo.upserted) == 1
    chunk = repo.upserted[0]
    assert chunk.extraction_source == "filename_only"
    assert chunk.file_name == "mcfcoreinstaller.zip"
    # The chunk text is the filename itself so it has *something* to embed and
    # so the chunk-level document is non-empty.
    assert chunk.text == "mcfcoreinstaller.zip"
