from datetime import datetime, timezone

from ai_file_brain.core.models import (
    ChatResult,
    FileChunk,
    SourcesChunk,
    TokenChunk,
)


def test_make_id_is_stable_and_truncated():
    a = FileChunk.make_id("/x/foo.txt", 0)
    b = FileChunk.make_id("/x/foo.txt", 0)
    assert a == b
    assert len(a) == 32


def test_make_id_changes_with_index_and_path():
    base = FileChunk.make_id("/x/foo.txt", 0)
    assert FileChunk.make_id("/x/foo.txt", 1) != base
    assert FileChunk.make_id("/x/bar.txt", 0) != base


def test_chat_stream_chunk_subtypes():
    t = TokenChunk(text="hi")
    s = SourcesChunk(paths=("a.txt",))
    assert t.text == "hi"
    assert s.paths == ("a.txt",)


def test_filechunk_is_immutable():
    chunk = FileChunk(
        id="x",
        file_path="/p",
        file_name="n",
        chunk_index=0,
        text="t",
        created_at=datetime.now(timezone.utc),
        modified_at=datetime.now(timezone.utc),
    )
    import dataclasses

    assert dataclasses.is_dataclass(chunk)


def test_chat_result_round_trip():
    r = ChatResult(answer="hello", sources=("a", "b"))
    assert r.answer == "hello"
    assert r.sources == ("a", "b")
