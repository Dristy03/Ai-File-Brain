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

    ocr_enabled: bool = True
    ocr_languages: list[str] = ["en"]
    pdf_ocr_min_native_chars: int = 50
    pdf_ocr_per_page_min_chars: int = 10
    pdf_ocr_render_dpi: int = 220

    max_file_size_bytes: int = 10 * 1024 * 1024

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
