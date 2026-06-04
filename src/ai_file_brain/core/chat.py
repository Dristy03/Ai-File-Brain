from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import AsyncIterator

from ollama import AsyncClient

from ai_file_brain.config import AiFileBrainSettings
from ai_file_brain.core.embedding import EmbeddingService
from ai_file_brain.core.models import (
    ChatResult,
    ChatStreamChunk,
    QueryHit,
    SourcesChunk,
    StatusChunk,
    TokenChunk,
)
from ai_file_brain.core.storage import VectorRepository, _under_watch_folder
from ai_file_brain.core.time_intent import (
    RecencyIntent,
    TimeWindow,
    parse_recency_intent,
    parse_time_intent,
)

# Lower temperature => more deterministic answers, less likely to hallucinate
# imaginary filenames or invented conversation context.
CHAT_TEMPERATURE = 0.2

# Stopwords stripped from a question before filename matching.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being",
    "i", "you", "we", "they", "he", "she", "it", "me", "us", "them",
    "my", "your", "our", "their", "his", "her", "its",
    "this", "that", "these", "those",
    "what", "which", "who", "whom", "where", "when", "why", "how",
    "tell", "show", "give", "find", "list", "explain", "summarise", "summarize",
    "about", "any", "some", "all", "more", "most", "less", "least",
    "do", "did", "does", "have", "has", "had",
    "from", "into", "by", "at", "as", "but", "if", "then", "than",
    "can", "could", "should", "would", "will", "shall",
    "please", "thanks", "thank",
    "there", "here",
    # Generic "container" nouns people use to mean "anything" — as filename
    # substrings they're pure noise ("file" matches FileListAbsolute.txt,
    # makefile, logfile…). Real content words in the question still drive the
    # match; the semantic passes handle a file whose meaning lives in its name.
    "file", "files", "folder", "folders",
    "document", "documents", "doc", "docs", "attachment", "attachments",
    # Common 3-char English fillers — kept here (not added at len>=3 cutoff
    # below) so we don't substring-match them against random filenames.
    "use", "new", "old", "let", "got", "say", "see", "way", "yet", "via",
    "per", "lot", "etc", "now", "one", "two", "ten",
})


def _filename_keywords(question: str) -> list[str]:
    """Extract content words from a question for filename-substring matching.

    Returns lowercase tokens of length >= 3 that aren't stopwords. The 3-char
    floor lets short acronyms / project codenames ("api", "csv", "mcf", "rfc")
    surface; common 3-char English fillers are screened by ``_STOPWORDS``.
    """
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]+", question.lower())
    return [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS]

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about the user's local files.\n"
    "Ground every answer in the file excerpts in this message AND any file excerpts "
    "or facts already established earlier in this conversation. The user may refer "
    "to things from prior turns (e.g. 'the third one', 'that file') — resolve such "
    "references using the conversation so far. If the answer is in neither the new "
    "excerpts nor the prior conversation, say so.\n"
    "Some entries are marked 'filename only' — those files exist on the user's "
    "machine but their contents are not indexed (typically archives, executables, "
    "media, or other binaries). For those, acknowledge that the file exists and "
    "give the filename, but do not invent any contents or summary for them."
)

# Recency questions are answered from the file *metadata* (filename + modified
# date), not the chunk text. Demand a strict format so small models (llama3.2)
# don't hallucinate framing like "multiple timestamps".
RECENCY_SYSTEM_PROMPT = (
    "You are a helpful assistant. The user is asking which files are the most "
    "recently modified. The list below is already sorted newest first; the first "
    "entry is the most recent. Output ONLY a numbered list, one line per file, in "
    "the exact format: '<N>. <file name> — <modified-at>'. Do not add commentary, "
    "qualifiers, or sentences before or after the list. Use the list verbatim — do "
    "not invent extra timestamps or merge entries."
)


# Cap on prior turns kept for the LLM (each turn is two messages: user + assistant).
# Without a cap, long sessions blow past llama3.2's context window.
MAX_HISTORY_TURNS = 6


class ChatService:
    def __init__(
        self,
        settings: AiFileBrainSettings,
        embedder: EmbeddingService,
        vector_repo: VectorRepository,
        ollama: AsyncClient,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._vector_repo = vector_repo
        self._ollama = ollama
        # Multi-turn history. Each entry is an Ollama message dict.
        # Replaying the full prior user message (with file chunks) lets follow-up
        # questions like "tell me about <file from previous answer>" work.
        self._history: list[dict[str, str]] = []
        # Background fire-and-forget tasks that purge stale index entries spotted
        # at query time. Kept referenced so they aren't garbage-collected mid-run.
        self._purge_tasks: set[asyncio.Task] = set()

    def clear_history(self) -> None:
        self._history.clear()

    def _filter_existing(self, hits: list[QueryHit]) -> list[QueryHit]:
        """Drop hits whose backing file no longer exists under the watch folder.

        A file deleted from the watch folder while the app wasn't running (or via
        a filesystem event the watcher missed) can still have chunks in the
        index. We never want those in an answer, so this is the last line of
        defense at retrieval time — and we opportunistically purge the stale
        chunks so future queries don't pay for them.

        Files *outside* the current watch folder are left untouched: they belong
        to a previously-watched folder that's intentionally retained (see storage
        scoping), so their on-disk presence isn't ours to judge.
        """
        folder = self._settings.watch_folder
        kept: list[QueryHit] = []
        for hit in hits:
            path = hit.file_path
            if path and _under_watch_folder(path, folder) and not os.path.exists(path):
                self._purge_stale(path)
                continue
            kept.append(hit)
        return kept

    def _purge_stale(self, file_path: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._safe_delete(file_path))
        self._purge_tasks.add(task)
        task.add_done_callback(self._purge_tasks.discard)

    async def _safe_delete(self, file_path: str) -> None:
        try:
            await self._vector_repo.delete_by_path(file_path)
        except Exception as ex:
            logger.debug("Opportunistic purge failed for %s: %s", file_path, ex)

    async def ask(self, question: str) -> ChatResult:
        answer_parts: list[str] = []
        sources: tuple[str, ...] = ()
        async for chunk in self.ask_stream(question):
            if isinstance(chunk, TokenChunk):
                answer_parts.append(chunk.text)
            elif isinstance(chunk, SourcesChunk):
                sources = chunk.paths
        return ChatResult(answer="".join(answer_parts), sources=sources)

    async def ask_stream(self, question: str) -> AsyncIterator[ChatStreamChunk]:
        question = (question or "").strip()
        if not question:
            yield SourcesChunk(paths=())
            return

        recency = parse_recency_intent(question)
        window = None if recency else parse_time_intent(question)

        if recency is not None:
            yield StatusChunk(message="Finding your most recent files…")
            hits = await self._vector_repo.most_recent(self._settings.top_k)
        else:
            yield StatusChunk(message="Embedding your question…")
            embedding = await self._embedder.embed(question)
            yield StatusChunk(message="Searching your files…")
            keywords = _filename_keywords(question)
            modified_range = (window.start, window.end) if window else None

            # The three retrieval passes are independent once we have the
            # embedding, so fire them concurrently — total latency becomes the
            # slowest single call instead of the sum. Substring matching scans
            # the whole collection, so overlapping it with the two vector
            # queries is the biggest win.
            async def _name_pass() -> list[QueryHit]:
                if not keywords:
                    return []
                return await self._vector_repo.query_by_filename_substrings(
                    keywords, n=self._settings.top_k
                )

            sem_hits, name_hits, fname_sem_hits = await asyncio.gather(
                self._vector_repo.query(
                    embedding,
                    self._settings.top_k,
                    modified_at_range=modified_range,
                ),
                _name_pass(),
                # Third signal: semantic match over filename-only stubs
                # (unsupported files whose contents aren't indexed). Bridges a
                # conceptual question to a file whose *name* is related —
                # e.g. "office timings" -> "attendance.xlsx" — which neither
                # content search (excludes stubs) nor literal substring
                # matching can reach.
                self._vector_repo.query_filename_only(
                    embedding,
                    self._settings.top_k,
                    modified_at_range=modified_range,
                ),
            )
            # Filename-only stubs (unsupported binaries: .exe, .zip, .appx …)
            # carry a weaker signal than real content — their vector is just the
            # filename's words — so on a content-rich query they crowd in as junk
            # "sources" (installers, runtimes) at distances just under the loose
            # filename ceiling. Only let a filename-only semantic match through
            # when it's at least as close as the best real-content match. With no
            # content match (best_content is None) they pass freely, so the
            # concept->filename bridge ("office timings" -> "attendance.xlsx" when
            # that file has no indexed body) still works.
            best_content = min((h.distance for h in sem_hits), default=None)
            if best_content is not None:
                fname_sem_hits = [
                    h for h in fname_sem_hits if h.distance <= best_content
                ]
            # Priority order: literal name match (high precision), then content
            # semantic, then filename semantic. Dedupe by path, cap at top_k so
            # the LLM context stays tight.
            hits = _merge_unique(
                [name_hits, sem_hits, fname_sem_hits], self._settings.top_k
            )

        # Final guard: never surface a file that's been deleted from the watch
        # folder, even if a filesystem event was missed or its index chunks
        # weren't purged. Also opportunistically cleans those chunks up.
        hits = self._filter_existing(hits)

        if not hits:
            if recency is not None:
                yield TokenChunk(text="I haven't indexed any files yet.")
                yield SourcesChunk(paths=())
                return
            if window is not None:
                yield TokenChunk(
                    text=f"I couldn't find any files modified during {window.label}."
                )
                yield SourcesChunk(paths=())
                return
            if not self._history:
                # Nothing retrieved and no prior conversation to lean on.
                yield TokenChunk(
                    text="I couldn't find any relevant content in your files."
                )
                yield SourcesChunk(paths=())
                return
            # No fresh matches, but this may be a follow-up about a file we
            # surfaced earlier — e.g. "what's the second-half start time?" after
            # the attendance doc was shown. The follow-up's own wording can embed
            # too far from that file to re-retrieve it, so rather than dead-end on
            # "couldn't find", fall through and let the LLM answer from the
            # conversation history (which still holds the earlier excerpts).

        # Emit sources EARLY so the UI can show "considering these files"
        # while the LLM is still loading and prefilling.
        seen_paths: list[str] = []
        for hit in hits:
            if hit.file_path and hit.file_path not in seen_paths:
                seen_paths.append(hit.file_path)
        yield SourcesChunk(paths=tuple(seen_paths))

        unique_names: list[str] = []
        for hit in hits:
            if hit.file_name and hit.file_name not in unique_names:
                unique_names.append(hit.file_name)
        if unique_names:
            preview = ", ".join(unique_names[:3])
            if len(unique_names) > 3:
                preview += f" (+{len(unique_names) - 3} more)"
            yield StatusChunk(
                message=f"Reading {len(unique_names)} file(s): {preview}"
            )

        user_message = _build_user_message(question, hits, window, recency)
        system_prompt = RECENCY_SYSTEM_PROMPT if recency is not None else SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            *self._history,
            {"role": "user", "content": user_message},
        ]

        yield StatusChunk(message="Thinking…")

        answer_parts: list[str] = []
        completed = False
        try:
            stream = await self._ollama.chat(
                model=self._settings.chat_model,
                messages=messages,
                stream=True,
                options={"temperature": CHAT_TEMPERATURE},
            )
            async for part in stream:
                token = (part.get("message") or {}).get("content", "")
                if token:
                    answer_parts.append(token)
                    yield TokenChunk(text=token)
            completed = True
        except Exception as ex:
            logger.exception("Ollama chat stream failed")
            yield TokenChunk(text=f"\n[error: {ex}]")

        if completed:
            self._history.append({"role": "user", "content": user_message})
            self._history.append({"role": "assistant", "content": "".join(answer_parts)})
            # Trim oldest turns so history stays bounded.
            max_messages = MAX_HISTORY_TURNS * 2
            if len(self._history) > max_messages:
                self._history = self._history[-max_messages:]


def _merge_unique(groups: list[list[QueryHit]], limit: int) -> list[QueryHit]:
    """Concatenate ``groups`` in order, drop duplicates by file_path, cap at
    ``limit``. Earlier groups win placement, so pass them highest-priority first.
    """
    seen: set[str] = set()
    out: list[QueryHit] = []
    for group in groups:
        for hit in group:
            if hit.file_path in seen:
                continue
            seen.add(hit.file_path)
            out.append(hit)
            if len(out) >= limit:
                return out
    return out


def _build_user_message(
    question: str,
    hits: list[QueryHit],
    window: TimeWindow | None = None,
    recency: RecencyIntent | None = None,
) -> str:
    blocks: list[str] = []
    if recency is not None:
        blocks.append(
            "[The following files are the most recently modified ones in the index, "
            "sorted newest first. The first entry is the most recent.]"
        )
    elif window is not None:
        blocks.append(
            f"[Filtered to files modified during {window.label} "
            f"({window.start.isoformat()} to {window.end.isoformat()})]"
        )
    for hit in hits:
        modified = hit.modified_at.isoformat() if hit.modified_at else "unknown"
        if hit.extraction_source == "filename_only":
            blocks.append(
                f"--- File: {hit.file_name} "
                f"(filename only — contents not indexed, modified: {modified}) ---"
            )
        else:
            blocks.append(
                f"--- File: {hit.file_name} (modified: {modified}) ---\n{hit.text}"
            )
    blocks.append("")
    blocks.append(f"Question: {question}")
    return "\n\n".join(blocks)
