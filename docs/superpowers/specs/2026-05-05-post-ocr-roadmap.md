# Post-OCR Roadmap

**Date:** 2026-05-05
**Status:** Draft for review
**Context:** The original product concept (`srbd-poc/AI_File_Brain.docx`, April 2026) named four differentiators: (1) passive whole-machine indexing, (2) cross-file-type reasoning, (3) temporal awareness, (4) fully local. The Python port covers (1) and (4) end to end. (2) is partial — text + PDF + (as of `637dac1`) images. (3) is unimplemented in chat.

This document lists the five concrete gaps left from the original plan, ordered by recommended priority. Each gap is shippable as a single PR with its own spec.

---

## Recommended order

| # | Item | Bucket | Why this slot |
|---|---|---|---|
| 1 | DOCX extraction | S — Small | Word docs are the single biggest missing format for the named target users (researchers, freelancers). Pattern is identical to the existing `pdf.py` extractor. Small, high signal. |
| 2 | Code-file extraction | S — Small | Same pattern, even simpler than DOCX (it's effectively `PlainTextExtractor` with a wider extension set). Bundle into the same wave as DOCX or ship right after. |
| 3 | Smart exclusions | M — Medium | Privacy and performance both improve. Required before users feel safe pointing the watcher at `C:\Users\ASUS` (which is what `user-settings.toml` already does). |
| 4 | Temporal awareness in chat | L — Large | It is listed as a headline differentiator in the original concept doc. The data is already on every chunk; the work is in the retrieval/prompt layer. |
| 5 | Live tray indexing status | S — Small | UX polish. Could ship anytime once #3 lands; deferred only because the others move the product forward more. |

---

## 1. DOCX extraction  — *Small*

**Goal:** index `.docx` files so a user's Word docs flow through chunking and embedding the same way `.pdf` does.

**Approach:**
- Add `python-docx` to `pyproject.toml` (pure wheel).
- New module `src/ai_file_brain/core/extraction/docx.py` exposing `DocxExtractor`.
- Walk paragraphs, then tables (cell-by-cell), then headers/footers; concatenate with newlines. No OCR for inline images in this pass — embedded images stay out of scope. Return `ExtractionResult(source="native")`.
- Register `.docx` in `extraction/__init__.py`.

**Edge cases:** password-protected docs raise on open → log + return empty. `.doc` (legacy binary) is *not* supported by `python-docx`; skip with a one-line log.

**Tests:** synthesise a `.docx` via `python-docx` itself in a fixture, assert text round-trips. Mirror the test naming used in `test_extraction.py`.

**Out of scope:** track-changes/comments, embedded images via OCR, `.doc` legacy format.

---

## 2. Code-file extraction  — *Small*

**Goal:** index source-code files so developer queries (`"where is the OCR engine initialised?"`) work.

**Approach:**
- Reuse `PlainTextExtractor` directly — code is just UTF-8 text. No new module.
- Extend `_EXTRACTORS` in `extraction/__init__.py` to register a vetted extension set (e.g. `.py .js .ts .tsx .jsx .java .cs .go .rs .rb .php .c .cpp .h .hpp .sh .yml .yaml .toml .json .md .ini .cfg .sql`). Bind each to the existing `PlainTextExtractor()` instance.
- Optional: cap individual file size at, say, 2 MB to avoid embedding generated lockfiles or minified bundles. Log + skip when over.

**Why this is dead simple:** the extractor already returns `ExtractionResult(source="native")`. Only the registry list needs to grow.

**Risk:** the user's currently-configured watch folder (`C:\Users\ASUS\Documents`) is unlikely to contain code, but a future user pointing at `~/projects` would suddenly index `node_modules/`. Item #3 (smart exclusions) is the proper guard. Until #3 lands, document the size cap as the only safety net.

**Tests:** add the extensions to `test_is_supported`; one round-trip test per representative extension is unnecessary — pick `.py` and `.json`.

**Out of scope:** language-aware chunking (chunking on AST/function boundaries instead of fixed character windows). The current `ChunkingService` works fine for code; AST-aware chunking is a future investigation.

---

## 3. Smart exclusions  — *Medium*

**Goal:** let the user pre-empt indexing of system directories, secret stores, build artefacts, and oversized files.

**Approach:**
- Add to `AiFileBrainSettings`:
  - `excluded_globs: list[str]` — glob patterns relative to the watch folder root *and* absolute, e.g. `["**/node_modules/**", "**/.git/**", "**/AppData/**", "**/__pycache__/**"]`.
  - `excluded_extensions: list[str]` — e.g. `[".lock", ".pyc"]` (orthogonal to the registered extractor extensions; this is a *negative* filter applied first).
  - `max_file_size_bytes: int` — default 10 MB.
- Wire the filter in two places:
  - `FileWatcherService._initial_scan` — skip excluded files during the recursive walk.
  - `FileWatcherService._handle_event` — skip excluded paths before debouncing.
- Defaults shipped in `settings.toml`: ship a sensible Windows default list (`AppData`, `node_modules`, `.git`, `__pycache__`, `*.lock`, `*.pyc`, `Library/Caches/...` placeholder for cross-platform later).
- Tray-menu UI for managing exclusions is *not* in this scope. Editing `user-settings.toml` is the v1 escape hatch; a UI can come later.

**Tests:** unit tests on a path-matcher helper (preferred over coupling tests to the watcher). Integration test: watch a tmp dir, drop one allowed and one excluded file, assert the pipeline only sees the allowed one.

**Out of scope:** content-based exclusion (e.g. detect-and-skip `.env` files containing secrets), per-folder UI.

---

## 4. Temporal awareness in chat  — *Large*

**Goal:** make the chat answer questions like *"what was I working on yesterday?"* or *"summarise the contracts I edited last week."*

**Why it's larger than it looks:** the chunk-level timestamps already exist (`created_at`, `modified_at` on `FileChunk`, persisted to Chroma metadata). The work is in three layers above storage.

**Approach:**

1. **Time-aware retrieval.** Extend `ChromaVectorRepository.query` (or add a new method) to accept an optional `where` filter for `modified_at` ranges. Chroma supports metadata filters natively — this is roughly:
   ```
   col.query(query_embeddings=..., where={"modified_at": {"$gte": iso, "$lte": iso}})
   ```
2. **Time-intent extractor.** A small classifier-by-prompt step in `ChatService` that asks the LLM (or a deterministic regex first pass) to extract a time window from the user's question (`"yesterday"`, `"last week"`, `"in March"`, `null`). Resolve relative phrases against the system clock. If `null`, behave exactly as today.
3. **Prompt augmentation.** When a time window is detected, include it in the system prompt's grounding so the LLM doesn't hallucinate timestamps from chunk content. Surface the window in the source attribution if helpful.

**Risk:** the LLM-based time-intent extractor is the soft part. A regex pass for the cheap, common cases (`yesterday`, `last week`, `last N days`, `in <month>`) covers ~80% with no extra LLM call; LLM fallback for the rest.

**Tests:** unit tests on the time-intent parser (dozens of phrase examples → expected `(start, end)` tuples). Integration test on retrieval: seed two chunks with different `modified_at` values, query with a time filter, assert only the in-window chunk comes back.

**Out of scope:** "yesterday I was in NYC" geographic reasoning, time-zone configuration UI, time-aware *generation* (e.g. *"draft a follow-up to the email I sent on Monday"*).

---

## 5. Live tray indexing status  — *Small*

**Goal:** the tray icon's tooltip / menu shows what the watcher is currently doing — *"Indexing scan.pdf…"*, *"Idle (1,247 chunks)"*, *"OCR queued: 3 files"*.

**Approach:**
- The `FileWatcherService` already calls a `ProgressCallback` with `IndexingProgress(file_path, state, detail)`. This data is already plumbed to `StatusBarViewModel` for the chat-window strip.
- Wire the same signal to a `TrayIconController` that updates the tray tooltip on each event. Throttle to ~1 update/sec to avoid Win32 tooltip churn.
- Add an "Idle" state computed from "no progress events for >2s".

**Tests:** unit test on the throttling logic; visual verification of the tray tooltip is manual.

**Out of scope:** rich tray menu sub-items per recently-indexed file, animated tray icons.

---

## What this roadmap is *not*

- It is not a sprint plan or a deadline commitment.
- It does not propose any architectural refactors. The current shape (extractor protocol, indexing pipeline, vector repo) handles items 1-3 with no structural change. Item 4 needs a single new method on the repo. Item 5 needs no core changes at all.
- It does not introduce new external dependencies beyond `python-docx` (item 1). RapidOCR / PyMuPDF / Pillow already landed with the OCR work.

## Suggested first move

Bundle items #1 and #2 into one PR titled "expand format coverage: docx + code files". Total touch surface: ~80 LoC + tests. Ships in one short session and immediately raises the value of every existing feature (chat, OCR, watcher) by widening their input.
