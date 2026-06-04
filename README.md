# AI File Brain

A local-first desktop app that watches a folder, indexes your files into a local vector store, and answers natural-language questions about them — all without leaving your machine.

## Features

**Indexing & extraction**

- Real-time file watcher (initial recursive scan + debounced create/modify/delete/move events)
- **Large watch roots are safe** — the initial scan runs off the UI thread, prunes excluded directories *before descending* (so a tree like `C:\` never enumerates `Windows`/`node_modules`/`AppData`), and feeds a bounded worker pool so indexing can't flood Ollama or exhaust memory
- **Two indexing tiers, fully config-driven** (anything in neither list is ignored):
  - **Full content** (`content_extensions`) — opened, text extracted, chunked, embedded:
    - Plain text · `.txt .md .rst`
    - PDF · `pypdf` fast path with **per-page OCR fallback** for scanned and mixed PDFs
    - Word · `.docx` (paragraphs, tables, headers/footers) and legacy `.doc` (FIB + piece-table parsing)
    - PowerPoint · `.pptx` (slide text, tables, speaker notes) and legacy `.ppt` (OLE text atoms)
    - Excel · `.xlsx` (`openpyxl`) and legacy `.xls` (`xlrd`) — cell values across every sheet, sheet titles preserved
  - **Filename only** (`name_only_extensions`) — just the file *name* is embedded as a stub so it's findable by name, contents not read:
    - Source code · `.py .js .ts .tsx .jsx .java .cs .go .rs .rb .php .c .cpp .cc .h .hpp .sh .bash .sql` (`.ps1` is in `excluded_extensions`, so PowerShell scripts are skipped)
    - Images · `.png .jpg .jpeg .tif .tiff .bmp .webp .gif` (image OCR is **off** this phase — name only)
    - Media / archives · `.mp4 .zip`
- Per-chunk `extraction_source` metadata (`native | ocr | mixed | filename_only`) carried through the vector store; filename-only stubs are excluded from semantic content search

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

- **Two-tier indexing** — `content_extensions` (full text) and `name_only_extensions` (filename stub); anything in neither is skipped entirely
- **Smart exclusions** — directory-name and extension skip lists, with sensible Windows defaults (`AppData`, `node_modules`, `.git`, `__pycache__`, `.venv`, `dist`, `build`, `.lock`, `.pyc`…); exclusions win over both tiers
- `max_concurrent_indexing` (4 default) bounds how many files are extracted/embedded at once so a huge watch root can't overwhelm the machine
- `max_file_size_bytes` cap (200 MiB default; `0` disables) defangs runaway lockfiles and generated bundles
- `ocr_enabled` governs the **scanned-PDF** OCR fallback (image OCR is off this phase — images are name-only)
- `pdf_ocr_min_native_chars`, `pdf_ocr_per_page_min_chars`, `pdf_ocr_render_dpi` for the PDF OCR fallback
- All settings overridable via `AFB_*` environment variables

## Stack

- Python 3.12
- PySide6 (Qt 6) + qasync for the desktop UI and async event loop
- ChromaDB `PersistentClient` for the vector store
- Ollama for embeddings (`nomic-embed-text`) and chat (`llama3.2`)
- `watchdog` for file events
- `pypdf` + `python-docx` + `python-pptx` + `openpyxl` for modern formats; `xlrd` + `olefile` for legacy `.xls` / `.ppt` / `.doc`
- `rapidocr-onnxruntime` + `PyMuPDF` + `Pillow` for scanned-PDF OCR (no external binaries)
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
| `watch_folder` | `~/Downloads` | Folder watched recursively (`~` expands to your home) |
| `chunk_size` / `chunk_overlap` | `2000` / `400` | Character-window chunking |
| `top_k` | `5` | Number of retrieved chunks fed to the LLM |
| `content_extensions` | `.txt .md .rst .pdf .docx .pptx .xlsx .doc .ppt .xls` | Extensions whose **content** is extracted, chunked, and embedded |
| `name_only_extensions` | code, images, `.mp4 .zip` | Extensions indexed by **filename only** (a stub); anything in neither list is ignored |
| `max_concurrent_indexing` | `4` | Max files extracted/embedded at once during scans |
| `ocr_enabled` | `true` | OCR fallback for **scanned PDFs** (image OCR is off this phase) |
| `pdf_ocr_min_native_chars` | `50` | Below this total native text, OCR fallback engages |
| `pdf_ocr_per_page_min_chars` | `10` | Below this on a page (in fallback mode), the page is OCR'd |
| `pdf_ocr_render_dpi` | `220` | DPI for PDF page rendering before OCR |
| `max_file_size_bytes` | `209715200` (200 MiB; `0` = unlimited) | Files larger than this are skipped |
| `max_extracted_chars` | `50000000` (`0` = unlimited) | Caps extracted text chars per file (bounds memory on giant spreadsheets/logs) |
| `excluded_dir_names` | see `settings.toml` | Path components that mark a file as ignored (wins over both tiers) |
| `excluded_extensions` | `.lock .pyc .pyo .log .tmp .bak .ps1` | Extensions never indexed (wins over both tiers) |

## Run

```bash
uv run ai-file-brain
# or
uv run python -m ai_file_brain.app.main
```

The app opens a chat window and parks itself in the system tray. Closing the window hides it; quit from the tray menu. The tray tooltip shows what the indexer is currently doing and the running chunk count.

## Try it out

Drop a Word doc, PDF, Markdown note, or text file in for full-text search — or any code file, image, spreadsheet, or zip to make it findable by name. Within seconds it will appear in the tray's "Indexing …" line; once indexed, ask the chat things like:

- "Summarise the meeting notes I added this week" *(full content)*
- "What was I working on yesterday?" *(temporal)*
- "Do I have anything about the Q3 budget?" *(matches content and filenames)*
- "Pull the action items out of last month's meeting notes" *(full content)*

This phase indexes document **content** (`.txt .md .rst .pdf .docx .pptx .xlsx .doc .ppt .xls`); code, images, and a few extra types are findable by **name only** (their contents aren't read). Adjust `content_extensions` / `name_only_extensions` in `user-settings.toml` to change what's indexed.

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
