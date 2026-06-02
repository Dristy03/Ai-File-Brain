from datetime import UTC, datetime
from typing import Any

import pytest

from ai_file_brain.config import AiFileBrainSettings
from ai_file_brain.core.chat import ChatService, _filename_keywords
from ai_file_brain.core.models import (
    QueryHit,
    SourcesChunk,
    StatusChunk,
    TokenChunk,
)


class FakeEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeRepo:
    def __init__(
        self,
        hits: list[QueryHit],
        recent_hits: list[QueryHit] | None = None,
    ) -> None:
        self.hits = hits
        self.recent_hits = recent_hits if recent_hits is not None else hits
        self.filename_only_hits: list[QueryHit] = []
        self.last_modified_at_range: tuple | None = None
        self.most_recent_calls: list[int] = []
        self.query_calls: int = 0

    async def initialize(self): ...
    async def upsert(self, *a, **kw): ...
    async def upsert_batch(self, *a, **kw): ...
    async def delete_by_path(self, *a, **kw): ...
    async def has_path(self, *a, **kw):
        return False
    async def count(self):
        return 0
    async def heartbeat(self):
        return True

    async def query(self, embedding, top_k, modified_at_range=None):
        self.query_calls += 1
        self.last_modified_at_range = modified_at_range
        return self.hits[:top_k]

    async def query_filename_only(self, embedding, n, modified_at_range=None):
        # Default fake: no filename-only semantic matches. Tests that exercise
        # the name->concept bridge set ``filename_only_hits`` on the instance.
        return self.filename_only_hits[:n]

    async def most_recent(self, n: int) -> list[QueryHit]:
        self.most_recent_calls.append(n)
        return self.recent_hits[:n]

    async def query_by_filename_substrings(
        self, substrings: list[str], n: int
    ) -> list[QueryHit]:
        # Default fake: no filename matches. Individual tests can monkey-patch
        # this on the instance when they want to verify the hybrid retrieval.
        return []


class FakeOllama:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.last_kwargs: dict[str, Any] | None = None

    async def chat(self, **kwargs: Any):
        self.last_kwargs = kwargs

        async def gen():
            for tok in self.tokens:
                yield {"message": {"content": tok}}

        return gen()


def _settings() -> AiFileBrainSettings:
    return AiFileBrainSettings(
        watch_folder=".",
        ollama_url="http://x",
        chroma_path="./tmp",
        embedding_model="m",
        chat_model="m",
        chunk_size=100,
        chunk_overlap=10,
        top_k=3,
    )


@pytest.mark.asyncio
async def test_ask_stream_yields_tokens_then_sources():
    hits = [
        QueryHit(
            chunk_id="1",
            file_path="/a.txt",
            file_name="a.txt",
            chunk_index=0,
            text="content from a",
            distance=0.1,
            modified_at=datetime.now(UTC),
        ),
        QueryHit(
            chunk_id="2",
            file_path="/b.txt",
            file_name="b.txt",
            chunk_index=0,
            text="content from b",
            distance=0.2,
            modified_at=None,
        ),
        QueryHit(
            chunk_id="3",
            file_path="/a.txt",  # duplicate path -> dedup
            file_name="a.txt",
            chunk_index=1,
            text="more from a",
            distance=0.3,
            modified_at=None,
        ),
    ]
    chat = ChatService(_settings(), FakeEmbedder(), FakeRepo(hits), FakeOllama(["Hel", "lo"]))

    chunks = []
    async for c in chat.ask_stream("what?"):
        chunks.append(c)

    tokens = [c.text for c in chunks if isinstance(c, TokenChunk)]
    sources_chunks = [c for c in chunks if isinstance(c, SourcesChunk)]
    assert tokens == ["Hel", "lo"]
    assert len(sources_chunks) == 1
    assert sources_chunks[0].paths == ("/a.txt", "/b.txt")


@pytest.mark.asyncio
async def test_ask_stream_emits_status_and_sources_before_first_token():
    """Status pings + sources should arrive before any answer tokens so the UI
    can show 'Embedding…', 'Searching…', 'Reading: a.txt, b.txt…', 'Thinking…'
    while the LLM is still warming up."""
    hits = [
        QueryHit(
            chunk_id="1",
            file_path="/a.txt",
            file_name="a.txt",
            chunk_index=0,
            text="content from a",
            distance=0.1,
            modified_at=datetime.now(UTC),
        ),
        QueryHit(
            chunk_id="2",
            file_path="/b.txt",
            file_name="b.txt",
            chunk_index=0,
            text="content from b",
            distance=0.2,
            modified_at=None,
        ),
    ]
    chat = ChatService(_settings(), FakeEmbedder(), FakeRepo(hits), FakeOllama(["Hi"]))

    chunks = []
    async for c in chat.ask_stream("what?"):
        chunks.append(c)

    # At least one status message and one source chunk arrive before any token.
    first_token_idx = next(i for i, c in enumerate(chunks) if isinstance(c, TokenChunk))
    before_token = chunks[:first_token_idx]
    assert any(isinstance(c, StatusChunk) for c in before_token), \
        "expected StatusChunk before first TokenChunk"
    assert any(isinstance(c, SourcesChunk) for c in before_token), \
        "expected SourcesChunk before first TokenChunk so files appear while LLM warms up"

    # The status messages should cover the user-visible phases.
    status_msgs = [c.message.lower() for c in chunks if isinstance(c, StatusChunk)]
    assert any("embed" in m for m in status_msgs)
    assert any("search" in m for m in status_msgs)
    assert any("read" in m for m in status_msgs)
    assert any("think" in m for m in status_msgs)


@pytest.mark.asyncio
async def test_ask_stream_handles_no_hits():
    chat = ChatService(_settings(), FakeEmbedder(), FakeRepo([]), FakeOllama([]))
    chunks = [c async for c in chat.ask_stream("hi")]
    tokens = [c for c in chunks if isinstance(c, TokenChunk)]
    sources = [c for c in chunks if isinstance(c, SourcesChunk)]
    assert any("couldn't find" in t.text for t in tokens)
    assert sources and sources[0].paths == ()


@pytest.mark.asyncio
async def test_ask_aggregates_into_chat_result():
    hits = [
        QueryHit("1", "/a.txt", "a.txt", 0, "stuff", 0.1, None),
    ]
    chat = ChatService(_settings(), FakeEmbedder(), FakeRepo(hits), FakeOllama(["abc", "def"]))
    result = await chat.ask("?")
    assert result.answer == "abcdef"
    assert result.sources == ("/a.txt",)


@pytest.mark.asyncio
async def test_temporal_question_passes_time_window_to_repo():
    hits = [QueryHit("1", "/a.txt", "a.txt", 0, "stuff", 0.1, None)]
    repo = FakeRepo(hits)
    chat = ChatService(_settings(), FakeEmbedder(), repo, FakeOllama(["ok"]))
    await chat.ask("what was I working on yesterday?")
    assert repo.last_modified_at_range is not None
    start, end = repo.last_modified_at_range
    assert (end - start).total_seconds() == 86400  # exactly one day


@pytest.mark.asyncio
async def test_non_temporal_question_passes_no_window():
    hits = [QueryHit("1", "/a.txt", "a.txt", 0, "stuff", 0.1, None)]
    repo = FakeRepo(hits)
    chat = ChatService(_settings(), FakeEmbedder(), repo, FakeOllama(["ok"]))
    await chat.ask("explain how ranking works")
    assert repo.last_modified_at_range is None


@pytest.mark.asyncio
async def test_temporal_no_hits_uses_window_specific_message():
    repo = FakeRepo([])
    chat = ChatService(_settings(), FakeEmbedder(), repo, FakeOllama([]))
    chunks = [c async for c in chat.ask_stream("what changed last week?")]
    tokens = [c for c in chunks if isinstance(c, TokenChunk)]
    assert any("last week" in t.text for t in tokens)


@pytest.mark.asyncio
async def test_recency_question_calls_most_recent_not_query():
    recent = [
        QueryHit("1", "/new.txt", "new.txt", 0, "fresh", 0.0, datetime(2026, 5, 10, tzinfo=UTC)),
        QueryHit("2", "/old.txt", "old.txt", 0, "stale", 0.0, datetime(2026, 4, 1, tzinfo=UTC)),
    ]
    repo = FakeRepo(hits=[], recent_hits=recent)
    chat = ChatService(_settings(), FakeEmbedder(), repo, FakeOllama(["new.txt"]))

    result = await chat.ask("what is the latest file I worked on?")

    assert repo.most_recent_calls == [3]   # top_k=3 from _settings()
    assert repo.query_calls == 0           # embedding-similarity path not taken
    assert result.sources == ("/new.txt", "/old.txt")


@pytest.mark.asyncio
async def test_recency_no_hits_returns_helpful_message():
    repo = FakeRepo(hits=[], recent_hits=[])
    chat = ChatService(_settings(), FakeEmbedder(), repo, FakeOllama([]))
    chunks = [c async for c in chat.ask_stream("show me the most recent files")]
    tokens = [c for c in chunks if isinstance(c, TokenChunk)]
    assert any("haven't indexed" in t.text for t in tokens)


@pytest.mark.asyncio
async def test_recency_branch_uses_relaxed_system_prompt():
    recent = [QueryHit("1", "/x.txt", "x.txt", 0, "x", 0.0, datetime(2026, 5, 10, tzinfo=UTC))]
    repo = FakeRepo(hits=[], recent_hits=recent)
    ollama = FakeOllama(["ok"])
    chat = ChatService(_settings(), FakeEmbedder(), repo, ollama)
    await chat.ask("what is the latest file I worked on?")
    assert ollama.last_kwargs is not None
    system_msg = ollama.last_kwargs["messages"][0]
    assert system_msg["role"] == "system"
    # Relaxed prompt that lets the model answer from filename + date metadata.
    assert "modified" in system_msg["content"].lower()
    assert "newest first" in system_msg["content"].lower()


@pytest.mark.asyncio
async def test_history_passes_prior_turns_into_followup():
    """Second turn must include the first turn's user+assistant in messages."""
    hits = [QueryHit("1", "/a.txt", "a.txt", 0, "stuff about widgets", 0.1, None)]
    repo = FakeRepo(hits=hits)
    ollama = FakeOllama(["first answer"])
    chat = ChatService(_settings(), FakeEmbedder(), repo, ollama)

    # Turn 1
    await chat.ask("what is in a.txt?")
    first_kwargs = ollama.last_kwargs
    assert first_kwargs is not None
    assert len(first_kwargs["messages"]) == 2  # system + user

    # Turn 2 — same ChatService instance
    ollama.tokens = ["second answer"]
    await chat.ask("tell me more about it")
    second_kwargs = ollama.last_kwargs
    assert second_kwargs is not None
    roles = [m["role"] for m in second_kwargs["messages"]]
    # system, user(turn1), assistant(turn1), user(turn2)
    assert roles == ["system", "user", "assistant", "user"]
    # Assistant message in history is the full text of the prior answer.
    assert second_kwargs["messages"][2]["content"] == "first answer"


@pytest.mark.asyncio
async def test_failed_turn_not_recorded_in_history():
    """A turn that errored mid-stream shouldn't poison the history."""
    hits = [QueryHit("1", "/a.txt", "a.txt", 0, "x", 0.1, None)]
    repo = FakeRepo(hits=hits)

    class BoomOllama:
        last_kwargs = None

        async def chat(self, **kwargs):
            BoomOllama.last_kwargs = kwargs
            raise RuntimeError("boom")

    chat = ChatService(_settings(), FakeEmbedder(), repo, BoomOllama())
    await chat.ask("first?")  # errors mid-stream

    # Replace with a normal client and try a second turn
    good = FakeOllama(["ok"])
    chat._ollama = good  # type: ignore[attr-defined]
    await chat.ask("second?")
    roles = [m["role"] for m in good.last_kwargs["messages"]]
    # Failed first turn should NOT appear in history: just system + new user.
    assert roles == ["system", "user"]


@pytest.mark.asyncio
async def test_clear_history_resets_state():
    hits = [QueryHit("1", "/a.txt", "a.txt", 0, "x", 0.1, None)]
    repo = FakeRepo(hits=hits)
    ollama = FakeOllama(["one"])
    chat = ChatService(_settings(), FakeEmbedder(), repo, ollama)

    await chat.ask("turn one")
    chat.clear_history()

    ollama.tokens = ["two"]
    await chat.ask("turn two")
    roles = [m["role"] for m in ollama.last_kwargs["messages"]]
    assert roles == ["system", "user"]


@pytest.mark.asyncio
async def test_history_capped_to_max_turns():
    """Bounded history prevents context-window blowup."""
    from ai_file_brain.core import chat as chat_module

    hits = [QueryHit("1", "/a.txt", "a.txt", 0, "x", 0.1, None)]
    repo = FakeRepo(hits=hits)
    ollama = FakeOllama(["a"])
    chat = ChatService(_settings(), FakeEmbedder(), repo, ollama)

    # Run MAX + 3 turns
    n_turns = chat_module.MAX_HISTORY_TURNS + 3
    for i in range(n_turns):
        ollama.tokens = [f"answer-{i}"]
        await chat.ask(f"question {i}")

    # On the last turn, history should have at most MAX_HISTORY_TURNS prior turns
    # = MAX_HISTORY_TURNS * 2 messages, PLUS system + new user = +2.
    final_messages = ollama.last_kwargs["messages"]
    assert len(final_messages) <= chat_module.MAX_HISTORY_TURNS * 2 + 2


def test_filename_keywords_keeps_three_char_acronyms_and_codenames():
    """3-char tokens like project codenames and acronyms must surface so users
    can ask 'do I have files on mcf core' and find 'mcfcoreinstaller.zip'."""
    kws = _filename_keywords("do I have files on mcf core?")
    assert "mcf" in kws
    assert "core" in kws


def test_filename_keywords_filters_common_three_char_fillers():
    """Generic 3-char English words shouldn't substring-match random filenames."""
    kws = _filename_keywords("show me the new project files I can see")
    assert "new" not in kws
    assert "see" not in kws
    assert "use" not in kws  # not in this prompt but verify the rule
    # Real content words still drive the match.
    assert "project" in kws


def test_filename_keywords_filters_generic_container_nouns():
    """Generic 'container' nouns ('file', 'document', 'folder') are pure noise as
    filename substrings — 'file' alone matches makefile, logfile, FileList.txt…
    They must be dropped so real content words drive the match."""
    kws = _filename_keywords("is there any file of office timings?")
    assert "file" not in kws
    assert "there" not in kws
    assert "office" in kws
    assert "timings" in kws
    for generic in ("document", "documents", "folder", "files", "doc"):
        assert generic not in _filename_keywords(f"show the {generic} about budget")


@pytest.mark.asyncio
async def test_filename_only_hit_rendered_without_body_in_prompt():
    """Filename-only hits must not have a content body in the LLM prompt — they
    have no real text to include, only the filename + modified date."""
    hits = [
        QueryHit(
            chunk_id="1",
            file_path="/p/mcfcoreinstaller.zip",
            file_name="mcfcoreinstaller.zip",
            chunk_index=0,
            text="mcfcoreinstaller.zip",  # placeholder = filename
            distance=0.0,
            modified_at=datetime(2026, 5, 1, tzinfo=UTC),
            extraction_source="filename_only",
        )
    ]
    ollama = FakeOllama(["ok"])
    chat = ChatService(_settings(), FakeEmbedder(), FakeRepo(hits), ollama)
    await chat.ask("do I have files on mcf core?")

    msgs = ollama.last_kwargs["messages"]
    user_msg = msgs[-1]["content"]
    assert "filename only" in user_msg.lower()
    assert "mcfcoreinstaller.zip" in user_msg
    # System prompt should warn the model about filename-only entries.
    system_msg = msgs[0]["content"]
    assert "filename only" in system_msg.lower()


@pytest.mark.asyncio
async def test_conceptual_query_surfaces_filename_only_semantic_match():
    """The reported bug: asking about 'office timings' must surface a file whose
    *name* is conceptually related ('attendance.xlsx') even though it's an
    unsupported type (filename-only, no body) and the query words aren't literal
    substrings of the name. The filename-only semantic pass bridges the gap."""
    # Content search + substring search both return unrelated junk, like the
    # real failure where the generic word "file" matched FileListAbsolute.txt.
    junk = QueryHit("j", "/dl/FileListAbsolute.txt", "FileListAbsolute.txt", 0,
                    "build artifact paths", 0.4, datetime(2026, 5, 1, tzinfo=UTC))
    attendance = QueryHit(
        chunk_id="att",
        file_path="/dl/attendance.xlsx",
        file_name="attendance.xlsx",
        chunk_index=0,
        text="attendance.xlsx",
        distance=0.2,
        modified_at=datetime(2026, 5, 1, tzinfo=UTC),
        extraction_source="filename_only",
    )
    repo = FakeRepo(hits=[junk])
    repo.filename_only_hits = [attendance]
    ollama = FakeOllama(["ok"])
    chat = ChatService(_settings(), FakeEmbedder(), repo, ollama)

    result = await chat.ask("is there any file of office timings?")

    assert "/dl/attendance.xlsx" in result.sources
    # And it reaches the LLM prompt as a filename-only entry (no invented body).
    user_msg = ollama.last_kwargs["messages"][-1]["content"]
    assert "attendance.xlsx" in user_msg
    assert "filename only" in user_msg.lower()


@pytest.mark.asyncio
async def test_empty_retrieval_with_history_still_calls_llm():
    """A follow-up whose own retrieval finds nothing must still reach the LLM so
    it can answer from the earlier conversation — e.g. 'what's the second-half
    start time?' after the attendance doc was already surfaced. Without this it
    dead-ends on 'couldn't find' before the model ever sees the history."""
    repo = FakeRepo(hits=[])  # current retrieval returns nothing
    ollama = FakeOllama(["2:00 PM"])
    chat = ChatService(_settings(), FakeEmbedder(), repo, ollama)
    # Seed a prior turn as if the attendance doc was surfaced earlier.
    chat._history = [
        {"role": "user", "content": "Attendence.docx — second half starts at 2:00 PM"},
        {"role": "assistant", "content": "The attendance policy covers office hours."},
    ]

    result = await chat.ask("what is the second half starting time of office?")

    assert ollama.last_kwargs is not None, "LLM should be called so it can use history"
    assert result.answer == "2:00 PM"
    assert result.sources == ()


@pytest.mark.asyncio
async def test_empty_retrieval_without_history_says_not_found():
    """With no fresh matches AND no prior conversation, still say 'couldn't find'
    rather than calling the LLM with nothing to ground on."""
    repo = FakeRepo(hits=[])
    ollama = FakeOllama(["should not be used"])
    chat = ChatService(_settings(), FakeEmbedder(), repo, ollama)

    result = await chat.ask("is there any file on quantum widgets?")

    assert ollama.last_kwargs is None, "LLM must NOT be called when nothing to ground on"
    assert "couldn't find" in result.answer


@pytest.mark.asyncio
async def test_filename_only_junk_dropped_when_content_matches():
    """Filename-only stubs (installers, runtimes, archives) must not pad the
    sources when real content matched. They're kept only if at least as close as
    the best content hit, so unrelated binaries sitting just under the loose
    filename ceiling stop showing up as 'sources'."""
    content = QueryHit(
        chunk_id="c",
        file_path="/dl/Attendence.docx",
        file_name="Attendence.docx",
        chunk_index=0,
        text="office hours and attendance policy",
        distance=0.36,
        modified_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    junk = QueryHit(
        chunk_id="j",
        file_path="/dl/Git-2.54.0-64-bit.exe",
        file_name="Git-2.54.0-64-bit.exe",
        chunk_index=0,
        text="Git-2.54.0-64-bit.exe",
        distance=0.53,  # closer than the loose filename ceiling, but worse than content
        modified_at=datetime(2026, 5, 1, tzinfo=UTC),
        extraction_source="filename_only",
    )
    repo = FakeRepo(hits=[content])
    repo.filename_only_hits = [junk]
    chat = ChatService(_settings(), FakeEmbedder(), repo, FakeOllama(["ok"]))

    result = await chat.ask("is there any files on office timings?")

    assert "/dl/Attendence.docx" in result.sources
    assert "/dl/Git-2.54.0-64-bit.exe" not in result.sources


@pytest.mark.asyncio
async def test_recency_intent_beats_time_window():
    # "latest" should pick recency, not interpret "last 7 days" or similar.
    recent = [QueryHit("1", "/x.txt", "x.txt", 0, "x", 0.0, datetime(2026, 5, 10, tzinfo=UTC))]
    repo = FakeRepo(hits=[], recent_hits=recent)
    chat = ChatService(_settings(), FakeEmbedder(), repo, FakeOllama(["ok"]))
    await chat.ask("what is the latest thing I worked on today?")
    assert repo.most_recent_calls == [3]
    assert repo.last_modified_at_range is None  # not routed through window filter
