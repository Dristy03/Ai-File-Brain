# AI File Brain — Python Port Design

**Date:** 2026-04-28
**Scope:** v1, full feature parity with the previous WPF design
**Stack:** Python 3.12 + PySide6 + ChromaDB PersistentClient + Ollama
**Builds on:** the WPF v1 spec from the prior repo (tray icon + chat window + RAG pipeline)

## Why Python

Single self-contained `.exe` is the goal. With ChromaDB the only practical paths from .NET are (a) bundle a Python sidecar or (b) drop ChromaDB. Going fully Python solves it natively: `chromadb.PersistentClient` runs in-process — no server, no HTTP, no sidecar — and PyInstaller bundles the whole stack into a single launchable folder.

## Tech stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.12 | Broad compat with PyInstaller and Chroma; LTS-feeling. |
| UI | PySide6 (Qt 6, LGPL) | Native widgets, `QSystemTrayIcon`, signals/slots = MVVM analogue. |
| Vector store | ChromaDB `PersistentClient` | In-process, persists to a folder; we supply embeddings so `onnxruntime` is not pulled in at runtime. |
| LLM / embeddings | Ollama via `ollama` Python client | Same models as .NET version: `nomic-embed-text` + `llama3.2`. |
| PDF | `pypdf` | Pure Python, no native deps. |
| File watching | `watchdog` | Cross-platform, the de facto choice. |
| Async on Qt | `qasync` | Lets `async def` coroutines run on the Qt event loop. |
| Tests | `pytest` + `pytest-qt` + `pytest-asyncio` | TDD discipline. |
| Lint/format | `ruff` | Single tool. |
| Env / deps | `uv` | Fast venvs + lockfile. |
| Packaging | PyInstaller `--onedir` | Folder + top-level `.exe`. Fast cold start. |

## Project structure

```
ai-file-brain/
├── pyproject.toml
├── settings.toml
├── pyinstaller.spec
├── docs/superpowers/specs/
├── src/ai_file_brain/
│   ├── __main__.py
│   ├── config.py
│   ├── core/
│   │   ├── models.py
│   │   ├── extraction/{plain_text,pdf}.py + factory
│   │   ├── chunking.py
│   │   ├── embedding.py
│   │   ├── storage.py
│   │   ├── chat.py
│   │   └── watcher.py
│   └── app/
│       ├── main.py
│       ├── di.py
│       ├── view_models/{main_window_vm,status_bar_vm}.py
│       ├── services/{tray_icon_service,health_check_service}.py
│       ├── views/main_window.py
│       └── models/chat_turn.py
└── tests/
    ├── core/
    └── app/
```

`core` and `app` mirror the previous `AiFileBrain.Core` / `AiFileBrain.App` boundaries.

## Behavioural parity

The user-visible feature set is identical to the WPF v1 spec:

- Tray icon with left-click toggle, right-click menu (Show / Pause indexing [disabled v1] / Quit).
- Chat window: streaming transcript, sources list per turn, multi-line input, Send/Stop, status strip showing watch folder + chunk count + Ollama and Chroma health.
- File watcher: `.txt` and `.pdf`, 500 ms per-file debounce, on-rename delete-then-reindex, on-startup full scan.
- Indexing pipeline: extract → 2000-char chunks with 400-char overlap → embed via Ollama → upsert into ChromaDB collection `ai-file-brain`.
- RAG: top-5 retrieval, system prompt restricts answers to retrieved excerpts, response streamed token-by-token.
- Health check: 10 s polling, status flags flip on failure, no popups.
- Window close hides; tray-menu Quit shuts down the app.

## Behavioural shifts from the .NET version

- `ChromaDbUrl` → `chroma_path` (a folder on disk).
- ChromaDB Python client is sync; storage repo wraps calls in `asyncio.to_thread` so the async surface is preserved.
- DI is plain factory functions in `app/di.py` — no container framework.
- Logging via `logging.getLogger(__name__)` instead of `ILogger<T>`.
- File-access retry is a small `retry_with_backoff` decorator instead of pulling in `tenacity`.

## Error handling

| Scenario | Behaviour |
|---|---|
| Ollama unreachable | `OllamaHealthy = False`; status strip shows `Ollama ✗`; send raises → rendered as error on the current `ChatTurn`. |
| Chroma init failure | Fail fast at startup with actionable message. |
| Locked / unreadable file | 3 retries with 1/2/4 s backoff; then warn-and-skip. |
| Unhandled exception on UI thread | Logged + `QMessageBox.warning`, app stays alive. |
| Unhandled exception on background task | `loop.set_exception_handler` logs; does not crash. |

## Testing strategy

- **Core**: pure-function tests for chunking, extractors, ID hashing; mocked-collaborator tests for embedding, storage, chat, watcher.
- **App**: ViewModel tests with a fake `ChatService` whose `ask_stream` yields scripted chunks; assert `ChatTurn.answer` grows correctly and `sources` populate at end. Tests for `HealthCheckService` flag transitions. PySide6 widgets are smoke-tested via `pytest-qt` in CI; full UI automation is out of scope for v1.

## Packaging

`pyinstaller.spec` configures:

- Entry point: `src/ai_file_brain/app/main.py`.
- Onedir mode.
- Hidden imports for `chromadb` (collects telemetry submodules) and `PySide6` (Qt plugins).
- Excludes `onnxruntime` (we don't need Chroma's default embedder).
- Bundles `settings.toml` and `assets/tray_icon.ico` next to the .exe.

Output: `dist/ai-file-brain/ai-file-brain.exe` plus a `_internal/` folder. Ship the parent folder.

## Out of scope (v2+)

- Launch on Windows startup.
- Persistent chat history across restarts.
- Click-citation-to-open-file.
- In-app settings UI.
- Pause/resume indexing (menu item present but disabled).
- Dark mode.
- MSIX / auto-update.
- Batch embedding.
- DOCX / OCR.
