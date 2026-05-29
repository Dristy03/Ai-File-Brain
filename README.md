# AI File Brain

A local-first desktop app that watches a folder, indexes your files into a local vector store, and answers natural-language questions about them — all without leaving your machine.

## Features

**Indexing & extraction**

- Real-time file watcher (initial recursive scan + debounced create/modify/delete/move events)
- Plain text · `.txt .md .rst`
- PDF · `pypdf` fast path with **per-page OCR fallback** for scanned and mixed PDFs
- DOCX · paragraphs, tables, and headers/footers
- Source code · `.py .js .ts .tsx .jsx .java .cs .go .rs .rb .php .c .cpp .cc .h .hpp .sh .bash .ps1 .sql`
- Config / data · `.yml .yaml .toml .json .ini .cfg .env`
- Images via OCR · `.png .jpg .jpeg .tif .tiff .bmp .webp` (multi-page TIFFs OCR every page; animated GIFs first frame only)
- Per-chunk `extraction_source` metadata (`native | ocr | mixed`) carried through the vector store

**Retrieval & chat**

- Local embeddings via Ollama (`nomic-embed-text`)
- Persistent ChromaDB vector store (cosine space, in-process, no server)
- Local LLM chat via Ollama (`llama3.2`), grounded on top-k retrieved chunks with file-name and modified-time citations
- **Temporal queries** — natural-language time scoping translated to a Chroma `modified_at` range filter:
  *yesterday · today · this/last week · this/last month · last N days/weeks/months · "in March" · "in January 2024"*

**Desktop UX**

- PySide6 chat window
- System tray with show/hide, change-watch-folder, quit
- **Live tray indexing status** — current activity surfaced in the tooltip, throttled to one update per second, falls back to a quiet baseline when idle

**Safety / config knobs**

- **Smart exclusions** — directory-name and extension skip lists, with sensible Windows defaults (`AppData`, `node_modules`, `.git`, `__pycache__`, `.venv`, `dist`, `build`, `.lock`, `.pyc`…)
- `max_file_size_bytes` cap (10 MiB default) defangs runaway lockfiles and generated bundles
- `ocr_enabled` kill switch reverts behaviour to text-only exactly
- `pdf_ocr_min_native_chars`, `pdf_ocr_per_page_min_chars`, `pdf_ocr_render_dpi` for the OCR fallback
- All settings overridable via `AFB_*` environment variables

## Stack

- Python 3.12
- PySide6 (Qt 6) + qasync for the desktop UI and async event loop
- ChromaDB `PersistentClient` for the vector store
- Ollama for embeddings (`nomic-embed-text`) and chat (`llama3.2`)
- `watchdog` for file events
- `pypdf` + `python-docx` for native text extraction
- `rapidocr-onnxruntime` + `PyMuPDF` + `Pillow` for OCR (no external binaries)
- PyInstaller for the single-folder build

## Prerequisites

1. **Python 3.12** on PATH.
2. **[Ollama](https://ollama.com)** running locally. Pull the models once:
   ```
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```
3. **[uv](https://docs.astral.sh/uv/)** (recommended) or pip.

## Set up

```bash
uv venv
uv pip install -e ".[dev]"
```

## Configure

Edit `settings.toml` for defaults, or `user-settings.toml` for personal overrides (the latter is gitignored). Any setting can also be overridden by setting `AFB_<UPPERCASE_FIELD>` in the environment.

Notable settings:

| Setting | Default | What it does |
|---|---|---|
| `watch_folder` | `C:/Users/ASUS/Documents/AIFileBrainTest` | Folder watched recursively |
| `chunk_size` / `chunk_overlap` | `2000` / `400` | Character-window chunking |
| `top_k` | `5` | Number of retrieved chunks fed to the LLM |
| `ocr_enabled` | `true` | OCR for images and scanned PDFs |
| `pdf_ocr_min_native_chars` | `50` | Below this total native text, OCR fallback engages |
| `pdf_ocr_per_page_min_chars` | `10` | Below this on a page (in fallback mode), the page is OCR'd |
| `pdf_ocr_render_dpi` | `220` | DPI for PDF page rendering before OCR |
| `max_file_size_bytes` | `10485760` (10 MiB) | Files larger than this are skipped |
| `excluded_dir_names` | see `settings.toml` | Path components that mark a file as ignored |
| `excluded_extensions` | `.lock .pyc .pyo .log .tmp .bak` | Extensions never indexed |

## Run

```bash
uv run ai-file-brain
# or
uv run python -m ai_file_brain.app.main
```

The app opens a chat window and parks itself in the system tray. Closing the window hides it; quit from the tray menu. The tray tooltip shows what the indexer is currently doing and the running chunk count.

## Try it out

Drop a screenshot, scanned receipt, Word doc, source file, or PDF into your watch folder. Within seconds it will appear in the tray's "Indexing …" line; once indexed, ask the chat things like:

- "Summarise the receipts I added this week"
- "What was I working on yesterday?"
- "Where do I initialise the OCR engine?"
- "Pull the action items out of last month's meeting notes"

Set `ocr_enabled = false` in `user-settings.toml` if you want to revert to text-only behaviour.

## Test

```bash
uv run pytest
```

OCR-touching tests are marked `slow`; run only the fast suite with:

```bash
uv run pytest -m "not slow"
```

## Build a standalone folder (single .exe + deps)

```bash
uv pip install -e ".[build]"
uv run pyinstaller pyinstaller.spec
```

Output lands in `dist/ai-file-brain/` — ship the whole folder.
