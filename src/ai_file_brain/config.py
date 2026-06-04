from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict, TomlConfigSettingsSource

DEFAULTS_TOML = "settings.toml"
USER_OVERRIDES_TOML = "user-settings.toml"


class AiFileBrainSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AFB_",
        extra="ignore",
        case_sensitive=False,
    )

    watch_folder: str = r"C:\Users\ASUS\Documents\AIFileBrainTest"
    ollama_url: str = "http://127.0.0.1:11434"
    chroma_path: str = "./chroma-data"
    embedding_model: str = "nomic-embed-text"
    chat_model: str = "llama3.2"
    chunk_size: int = 2000
    chunk_overlap: int = 400
    top_k: int = 5

    # Cosine-distance ceiling for a chunk to count as relevant (cosine space:
    # 0 = identical, 2 = opposite). Hits farther than this are dropped before
    # they reach the answer, so unrelated files stop appearing as "sources" and
    # a query with no real match returns a clean "not found". Tune lower for
    # stricter matching, higher to admit weaker matches.
    max_match_distance: float = 0.5
    # Filename-only stubs (unsupported types: .exe, .m4a …) embed just the
    # filename's words, so their distances run higher than full-content chunks
    # for the same intent. Give them a looser ceiling so a name-based match
    # ("attendance.xlsx" for "office timings") still survives.
    max_filename_match_distance: float = 0.85

    ocr_enabled: bool = True
    ocr_languages: list[str] = ["en"]
    # Run OCR on the GPU via DirectML when available (any DX12 GPU, incl. Intel
    # Arc). Falls back to CPU automatically if the DirectML runtime isn't
    # installed (pip install onnxruntime-directml), so this is always safe to
    # leave on. Note: shares the GPU with Ollama, so heavy OCR + chat at the same
    # instant time-slice; set False to force CPU OCR if that ever matters.
    ocr_use_gpu: bool = True
    pdf_ocr_min_native_chars: int = 50
    pdf_ocr_per_page_min_chars: int = 10
    pdf_ocr_render_dpi: int = 220
    # Max pages to OCR per PDF (0 = unlimited). Pages are rendered + OCR'd one at a
    # time, so memory is bounded regardless; this only caps worst-case CPU time on
    # a huge scanned document. Raise/zero it if you need every page of giant scans.
    pdf_ocr_max_pages: int = 0

    # Hard upper bound on per-file size before indexing. Mainly a backstop against
    # runaway generated junk (lockfiles, bundles, DB dumps), NOT a memory guard:
    # OCR streams page-by-page and text extraction is bounded, so large real
    # documents are fine. 200 MiB by default; set 0 to disable the cap entirely.
    max_file_size_bytes: int = 200 * 1024 * 1024

    # Upper bound on extracted text characters per file (0 = unlimited). Bounds
    # peak memory + embedding work when a file's *text* dwarfs its on-disk size —
    # chiefly a spreadsheet flattened to text or a giant log/JSON. The default is
    # deliberately huge (~50 MB of text: a 5,000+ page book) so it never touches a
    # real document, only pathological data dumps. The worst offenders (xlsx,
    # plain text) stop reading at the cap; other types are truncated post-extract.
    max_extracted_chars: int = 50_000_000

    # Upper bound on how many files are indexed (extracted + embedded) at once.
    # Bulk scans feed a bounded work queue drained by this many workers, so a
    # huge watch root (e.g. C:\) can't spawn millions of concurrent tasks and
    # flood Ollama / exhaust memory. Raise for faster indexing on beefy boxes;
    # lower to be gentler on the machine while you keep working.
    max_concurrent_indexing: int = 4

    # --- What gets indexed, in two tiers (anything in neither list is ignored) ---
    #
    # Tier 1 — full content: the file is opened, its text extracted, chunked and
    # embedded. Only extensions with a registered extractor work here; an entry
    # without one silently falls back to a filename-only stub.
    content_extensions: list[str] = [
        ".txt",
        ".md",
        ".rst",
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".ppt",
        ".xls",
        ".doc",
    ]
    # Tier 2 — filename only: no content is read; just the file's name is
    # embedded so it's findable by name. Used for files we can't (or choose not
    # to) read this phase — code, images (OCR off), and a few extra doc/media/
    # archive types. Everything not in either list is skipped entirely.
    name_only_extensions: list[str] = [
        # source code — names only for now
        ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".cs", ".go", ".rs",
        ".rb", ".php", ".c", ".cpp", ".cc", ".h", ".hpp", ".sh", ".bash",
        ".ps1", ".sql",
        # images — OCR disabled this phase, so name-only
        ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif",
        # extra media / archives, by name
        ".mp4", ".zip",
    ]

    # Names of directories that, if they appear anywhere in a path, cause the
    # file to be skipped. Case-insensitive. Designed for noisy / private dirs
    # that no user wants indexed.
    excluded_dir_names: list[str] = [
        "AppData",
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".idea",
        ".vscode",
        "dist",
        "build",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "site-packages",
        "Recycle.Bin",
        "$RECYCLE.BIN",
    ]

    # Extensions that should never be indexed even if otherwise routable.
    # Note: registered extractor extensions are the *positive* list; this is
    # for extensions that aren't routed today but might be in the future, or
    # when a user wants to defang an otherwise-routable extension.
    excluded_extensions: list[str] = [
        ".lock",
        ".pyc",
        ".pyo",
        ".log",
        ".tmp",
        ".bak",
    ]

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        # Priority (first wins): init args > env > user overrides > defaults file > secrets
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls, toml_file=USER_OVERRIDES_TOML),
            TomlConfigSettingsSource(settings_cls, toml_file=DEFAULTS_TOML),
            file_secret_settings,
        )

    def chroma_path_resolved(self) -> Path:
        return Path(self.chroma_path).expanduser().resolve()


def save_user_overrides(updates: dict[str, Any], path: str | Path = USER_OVERRIDES_TOML) -> Path:
    """Merge ``updates`` into the user-overrides TOML and write it back.

    Returns the resolved path that was written.
    """
    target = Path(path)
    existing: dict[str, Any] = {}
    if target.exists():
        try:
            existing = tomllib.loads(target.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            existing = {}
    existing.update(updates)
    target.write_text(_dump_flat_toml(existing), encoding="utf-8")
    return target.resolve()


def _dump_flat_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    for key in sorted(data):
        value = data[key]
        if isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            lines.append(f"{key} = {value}")
        else:
            text = str(value).replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{key} = "{text}"')
    return "\n".join(lines) + "\n"
