# OCR & Image Extraction — Design

**Date:** 2026-05-05
**Status:** Approved (pending implementation plan)
**Scope:** Add OCR-based text extraction so AI File Brain can index image files and scanned/mixed PDFs.

---

## 1. Goal

Today the indexing pipeline supports `.txt` (via `aiofiles`) and `.pdf` (via `pypdf`). Files containing text only as pixels — screenshots, photos of receipts, scanned reports, image-only PDFs, mixed PDFs with a scanned cover page — silently produce zero indexable text. This work adds OCR so those files contribute to the vector store, while preserving the project's "local-first, ship a folder, no extra binaries" identity.

## 2. Decisions (settled during brainstorming)

| Decision | Choice |
|---|---|
| Coverage | Standalone image files **plus** per-page OCR fallback for sparse PDFs |
| OCR engine | RapidOCR (`rapidocr-onnxruntime`) — pure wheel, ONNX models bundled, no external binary |
| PDF rendering for OCR | PyMuPDF — pure wheel, no Poppler dependency |
| PDF fallback trigger | Document-level sparseness check first; if sparse, per-page OCR for pages whose native text is empty |
| Default languages | English only, configurable |
| Provenance metadata | Each chunk carries `extraction_source ∈ {"native", "ocr"}`; document-level granularity propagated by the extractor |
| Scheduling | Inline in the existing indexing pipeline; no separate queue |

## 3. Non-Goals

- HEIC / AVIF / animated image-frame OCR.
- Per-chunk OCR provenance (document-level is enough).
- GPU / CUDA RapidOCR builds.
- Surfacing "this answer came from OCR" in the chat UI. Metadata is captured so a future change can do this; the UI itself is out of scope.
- Replacing `pypdf` with PyMuPDF for native text extraction. The fast path stays unchanged.

## 4. Architecture

### 4.1 New / changed files

```
src/ai_file_brain/core/extraction/
├── __init__.py          (UPDATED — register image extractor and new exts)
├── plain_text.py        (UPDATED — return ExtractionResult)
├── pdf.py               (UPDATED — OCR fallback, return ExtractionResult)
├── image.py             (NEW — standalone image OCR)
└── ocr.py               (NEW — RapidOCR singleton + helpers)

src/ai_file_brain/core/
├── models.py            (UPDATED — ExtractionResult, FileChunk.extraction_source)
├── storage.py           (UPDATED — round-trip extraction_source through Chroma metadata)
└── watcher.py           (UPDATED — propagate extraction source into FileChunk)

src/ai_file_brain/
└── config.py            (UPDATED — new OCR settings)
```

### 4.2 Component responsibilities

**`ocr.py`** — owns the lone `RapidOCR` instance.

- Lazy singleton: model files load on the first call, then are reused for the lifetime of the process.
- One async entry point: `ocr_image_bytes(data: bytes) -> str`. Internally uses `asyncio.to_thread` because RapidOCR is synchronous.
- Accepts a `numpy.ndarray` or raw bytes; the helper handles both. Returns recognised text joined with `\n` between text blocks.
- Reads `ocr_enabled` and `ocr_languages` from settings. If `ocr_enabled` is `False`, returns `""` immediately (and logs once at module-load time).

**`image.py`** — `ImageExtractor` registered for `.png .jpg .jpeg .tiff .tif .bmp .webp`.

- Reads the file with `aiofiles`, opens with Pillow, normalises mode (RGB), iterates pages for multi-page TIFFs, calls `ocr_image_bytes` per page, joins results.
- Animated images: only the first frame is OCR'd (Pillow exposes this directly via `Image.open` without seeking).
- Unreadable / corrupt images: log a warning, return empty `ExtractionResult(text="", source="ocr")`.
- Returns `ExtractionResult(text=..., source="ocr")` always (even for empty results — the *attempt* was OCR-based).

**`pdf.py`** — keeps current pypdf fast path, adds OCR fallback.

- Step 1 (unchanged): try `pypdf` against every page; collect per-page text.
- Step 2 (new): if total non-whitespace text length < `pdf_ocr_min_native_chars` (default `50`), enter fallback:
  - Open the document with PyMuPDF.
  - For each page, if the page's native text length < `pdf_ocr_per_page_min_chars` (default `10`), render the page to an image at `pdf_ocr_render_dpi` (default `220`) and OCR it; otherwise keep the native text.
  - Track which pages were OCR'd vs native.
- Step 3: produce the result.
  - If no fallback ran → `source="native"`.
  - If fallback ran and every page used OCR → `source="ocr"`.
  - If fallback ran and some pages used native, some OCR → `source="mixed"`.
- Errors during PyMuPDF open / page render: log, return whatever native text we already had with `source="native"`. We never raise out of the extractor.

**`models.py`**

```python
@dataclass(frozen=True, slots=True)
class ExtractionResult:
    text: str
    source: Literal["native", "ocr", "mixed"]

@dataclass(frozen=True, slots=True)
class FileChunk:
    # ... existing fields ...
    extraction_source: Literal["native", "ocr", "mixed"] = "native"
```

`FileChunk.extraction_source` defaults to `"native"` so existing call sites and existing Chroma rows that pre-date this change continue to round-trip correctly.

**`storage.py`** — `FileChunk → Chroma metadata` dict adds `"extraction_source": str`. On read, missing metadata defaults to `"native"`.

**`watcher.py` / `IndexingPipeline`** — the only change in `_index_file_once` is that `extractor.extract(...)` now returns `ExtractionResult`. The pipeline reads `result.text` for chunking and stamps every `FileChunk` it produces with `result.source`.

### 4.3 Extractor protocol update

Current:

```python
class TextExtractor(Protocol):
    async def extract(self, file_path: str) -> str: ...
```

New:

```python
class TextExtractor(Protocol):
    async def extract(self, file_path: str) -> ExtractionResult: ...
```

`PlainTextExtractor` is updated to wrap its return in `ExtractionResult(text=..., source="native")`. Trivial diff.

## 5. Settings

Added to `settings.toml` (and the `AiFileBrainSettings` model):

```toml
ocr_enabled = true
ocr_languages = ["en"]
pdf_ocr_min_native_chars = 50      # below this total, switch to per-page fallback
pdf_ocr_per_page_min_chars = 10    # below this on a page, OCR that page
pdf_ocr_render_dpi = 220
```

`ocr_enabled = false` is the escape hatch: image files index to empty, PDFs behave exactly as today (pypdf only). All settings overridable via `AFB_*` env vars consistent with the existing pattern.

## 6. Dependencies

Added to `pyproject.toml` `dependencies`:

```
rapidocr-onnxruntime>=1.4
pymupdf>=1.24
pillow>=10.4
```

All pure wheels on Windows, no external binaries. Estimated PyInstaller bundle impact: +30–50 MB (ONNX models + PyMuPDF native libs). Acceptable for the goal.

## 7. Error handling

| Failure | Behaviour |
|---|---|
| Image file unreadable / corrupt | Log warning, return `ExtractionResult("", "ocr")`. Pipeline treats this like any other empty extraction (clears prior chunks for that path). |
| PyMuPDF cannot open a PDF | Log warning, return whatever pypdf produced with `source="native"`. |
| RapidOCR raises mid-image | Log warning, treat as empty for that image/page; continue with the rest of the document. |
| `ocr_enabled = false` | Image extractor returns `""`; PDF extractor never enters fallback. |
| Unsupported image format (HEIC, animated GIF beyond first frame) | First-frame attempt; if Pillow can't decode, log and return empty. |

The extractors never raise — consistent with the existing `pdf.py` philosophy that an extractor failure should not crash indexing.

## 8. Performance notes

- RapidOCR loads models once (~0.5–1.0 s on first call); subsequent calls reuse the singleton.
- Per-image OCR latency on CPU: roughly 0.2–1.0 s for typical screenshots; scales with image size.
- PDF render at 220 DPI is a deliberate tradeoff between OCR accuracy and CPU/memory cost; configurable.
- Indexing already runs off the UI thread (`qasync` event loop), so a slow OCR job will not freeze the chat window. It will, however, sit in front of the next file in the watcher queue — accepted per the brainstorming decision.

## 9. Tests

Added to `tests/core/test_extraction.py` (plus a small storage round-trip test):

1. `test_image_extractor_reads_text_from_png` — generate a PNG with PIL containing a known string; assert OCR result contains that string (fuzzy substring match — OCR is not exact).
2. `test_image_extractor_returns_empty_for_blank_image` — blank canvas → empty text, no exception, `source="ocr"`.
3. `test_image_extractor_handles_corrupt_file` — random bytes in `.png` → log + empty result, no exception.
4. `test_pdf_extractor_native_fast_path` — text-only PDF → existing assertions plus `source == "native"`.
5. `test_pdf_extractor_image_only_pdf_uses_ocr` — synthesise a one-page PDF whose only content is a rendered image of known text (built with PyMuPDF in the fixture); assert OCR text is returned and `source == "ocr"`.
6. `test_pdf_extractor_mixed_pages_returns_mixed_source` — two-page PDF: page 1 native text, page 2 image-only; assert both are present and `source == "mixed"`.
7. `test_ocr_disabled_short_circuits` — `ocr_enabled = false` → image extractor returns empty, scanned PDF returns empty.
8. `test_extraction_source_round_trips_through_storage` — index a file end-to-end, query Chroma, confirm metadata carries the flag.

OCR-touching tests are marked `@pytest.mark.slow` so they can be excluded with `-m "not slow"` in tight loops; they still run by default.

## 10. Migration / backwards compatibility

- Existing Chroma rows have no `extraction_source` metadata. Reads default to `"native"`.
- Existing PDFs already in the index will not be re-OCR'd unless they are modified or deleted/re-added; the watcher's existing `has_path` check is unchanged.
- A user who wants to retroactively OCR their corpus can delete `chroma-data/` and let the initial scan rebuild it.
- No schema migration script needed.
