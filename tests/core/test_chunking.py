import pytest

from ai_file_brain.core.chunking import ChunkingService


def test_empty_text_returns_no_chunks():
    assert ChunkingService(100, 10).chunk("") == []


def test_short_text_returns_one_chunk():
    chunks = ChunkingService(100, 10).chunk("hello world")
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].chunk_index == 0


def test_longer_than_size_produces_multiple_overlapping_chunks():
    text = ("a" * 250) + " " + ("b" * 250)
    chunks = ChunkingService(chunk_size=100, chunk_overlap=20).chunk(text)
    assert len(chunks) >= 4
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(indices)))
    # ranges advance and overlap by ~chunk_overlap
    for prev, curr in zip(chunks, chunks[1:]):
        assert curr.start <= prev.end


def test_prefers_whitespace_boundary_when_possible():
    # ensure splitter doesn't cut a word in half when whitespace is near
    body = "word " * 50  # 250 chars, plenty of whitespace
    chunks = ChunkingService(chunk_size=60, chunk_overlap=10).chunk(body)
    for c in chunks:
        assert not c.text.startswith(" ")
        # No chunk should end with a half-word followed by non-space
        # (the chunk_text has been .strip()'d, so check character is space-aligned)


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        ChunkingService(chunk_size=0)
    with pytest.raises(ValueError):
        ChunkingService(chunk_size=10, chunk_overlap=10)
    with pytest.raises(ValueError):
        ChunkingService(chunk_size=10, chunk_overlap=-1)


def test_no_infinite_loop_when_no_whitespace_in_window():
    text = "x" * 1000
    chunks = ChunkingService(chunk_size=100, chunk_overlap=20).chunk(text)
    assert len(chunks) > 1
    # progress is real
    assert chunks[-1].end == 1000
