# Single-EXE Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one shareable `dist\ai-file-brain.exe` on Windows that runs on a recipient's machine without a source tree, expecting only Ollama as an external prerequisite.

**Architecture:** Add a tiny `paths.py` helper for `_MEIPASS`-aware resource resolution and a per-user data dir under `%LOCALAPPDATA%\AIFileBrain`. Wire it into `config.py` (settings/overrides/chroma path) and the icon loader. Rewrite `pyinstaller.spec` for `--onefile` with the OCR-stack imports it currently misses. Add a `QFrame` banner that surfaces Ollama-missing state by binding to the existing `StatusBarViewModel.ollama_healthy`. Add a one-command build script and update the README handoff section.

**Tech Stack:** PyInstaller 6, PySide6, pydantic-settings, RapidOCR + ONNX Runtime, ChromaDB, Ollama (external).

**Spec reference:** `docs/superpowers/specs/2026-05-13-single-exe-packaging-design.md`

---

## File Structure

**Create:**
- `src/ai_file_brain/core/paths.py` — `resource_path()` + `user_data_dir()` helpers
- `tests/core/test_paths.py` — unit tests for both helpers
- `scripts/build.ps1` — one-command build script

**Modify:**
- `src/ai_file_brain/config.py` — overrides path + chroma default + bundled-settings discovery
- `src/ai_file_brain/app/main.py:23-28` — icon loader uses `resource_path`
- `src/ai_file_brain/core/watcher.py:169` — `Path(...).expanduser()` so `~/Documents/AIFileBrain` works
- `src/ai_file_brain/app/views/main_window.py` — add the Ollama banner
- `tests/core/test_config.py` — switch to `AFB_DATA_DIR` env override for isolation
- `tests/conftest.py` — set `AFB_DATA_DIR=tmp_path` so settings tests don't write to real `%LOCALAPPDATA%`
- `settings.toml` — remove `chroma_path`, change `watch_folder` default
- `pyinstaller.spec` — full rewrite for `--onefile` + correct hidden imports / datas / excludes
- `README.md` — replace the "Build a standalone folder" section

---

## Task 1: `paths.py` — resource and per-user data dir helpers

**Files:**
- Create: `src/ai_file_brain/core/paths.py`
- Test: `tests/core/test_paths.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/core/test_paths.py
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from ai_file_brain.core import paths


def test_resource_path_uses_source_tree_when_not_frozen(monkeypatch):
    # When sys.frozen is unset, resource_path returns <project_root>/<rel>.
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    got = paths.resource_path("settings.toml")
    assert got.name == "settings.toml"
    # Project root should contain pyproject.toml
    assert (got.parent / "pyproject.toml").exists()


def test_resource_path_uses_meipass_when_frozen(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    got = paths.resource_path("settings.toml")
    assert got == tmp_path / "settings.toml"


def test_user_data_dir_honours_afb_data_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AFB_DATA_DIR", str(tmp_path / "custom"))
    got = paths.user_data_dir()
    assert got == tmp_path / "custom"
    assert got.is_dir()  # created on demand


def test_user_data_dir_falls_back_to_localappdata_on_windows(monkeypatch, tmp_path):
    monkeypatch.delenv("AFB_DATA_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setattr(sys, "platform", "win32")
    got = paths.user_data_dir()
    assert got == tmp_path / "appdata" / "AIFileBrain"
    assert got.is_dir()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_paths.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ai_file_brain.core.paths'`.

- [ ] **Step 3: Implement `paths.py`**

```python
# src/ai_file_brain/core/paths.py
from __future__ import annotations

import os
import sys
from pathlib import Path


def resource_path(rel: str) -> Path:
    """Resolve a read-only resource path that works from source and from a PyInstaller bundle.

    When the app is frozen by PyInstaller, resources live under ``sys._MEIPASS``
    (a temp extraction dir for --onefile builds). Otherwise they live in the
    project root (the directory containing ``pyproject.toml``).
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / rel  # type: ignore[attr-defined]
    return _project_root() / rel


def user_data_dir() -> Path:
    """Return a writable per-user data directory, creating it if needed.

    Priority:
    1. ``$AFB_DATA_DIR`` (used by tests; also a power-user override).
    2. ``%LOCALAPPDATA%\\AIFileBrain`` on Windows.
    3. ``~/Library/Application Support/AIFileBrain`` on macOS.
    4. ``~/.local/share/AIFileBrain`` elsewhere.
    """
    override = os.environ.get("AFB_DATA_DIR")
    if override:
        target = Path(override)
    elif sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        target = Path(base) / "AIFileBrain"
    elif sys.platform == "darwin":
        target = Path.home() / "Library" / "Application Support" / "AIFileBrain"
    else:
        target = Path.home() / ".local" / "share" / "AIFileBrain"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _project_root() -> Path:
    # paths.py lives at src/ai_file_brain/core/paths.py → root is 3 levels up.
    return Path(__file__).resolve().parents[3]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_paths.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ai_file_brain/core/paths.py tests/core/test_paths.py
git commit -m "feat(core): add paths helpers for bundle-aware resource and per-user data dir"
```

---

## Task 2: Isolate tests from real `%LOCALAPPDATA%`

We're about to make `config.py` write under `user_data_dir()`. Tests must not touch the real user data dir. The existing `conftest.py` already isolates CWD; add `AFB_DATA_DIR` to point at `tmp_path` for every test.

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Update the conftest fixture**

Replace the body of `isolated_cwd` to also set `AFB_DATA_DIR`:

```python
# tests/conftest.py
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure src/ is on sys.path even before editable install
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    """Each test runs in a temp CWD with a temp per-user data dir."""
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "afb-data"
    monkeypatch.setenv("AFB_DATA_DIR", str(data_dir))
    # No AFB_* env from outside leaks in
    for key in list(os.environ):
        if key.startswith("AFB_") and key != "AFB_DATA_DIR":
            monkeypatch.delenv(key, raising=False)
    yield tmp_path
```

- [ ] **Step 2: Run the full suite to verify nothing broke**

Run: `uv run pytest -m "not slow" -q`
Expected: PASS (same count as before this commit).

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: isolate AFB_DATA_DIR per test so config writes stay in tmp_path"
```

---

## Task 3: `config.py` — per-user overrides path + bundled defaults discovery

**Files:**
- Modify: `src/ai_file_brain/config.py`
- Modify: `tests/core/test_config.py`

- [ ] **Step 1: Write new failing tests for the per-user behaviour**

Add to `tests/core/test_config.py`:

```python
# Append to tests/core/test_config.py

import sys
from ai_file_brain.config import (
    user_overrides_path,
    defaults_toml_path,
)
from ai_file_brain.core import paths as paths_mod


def test_user_overrides_path_lives_under_user_data_dir(tmp_path):
    # AFB_DATA_DIR is set by conftest to tmp_path / "afb-data"
    got = user_overrides_path()
    assert got == tmp_path / "afb-data" / "user-settings.toml"


def test_save_user_overrides_writes_to_user_data_dir(tmp_path):
    save_user_overrides({"watch_folder": "C:/notes"})
    target = tmp_path / "afb-data" / "user-settings.toml"
    assert target.exists()
    assert 'watch_folder = "C:/notes"' in target.read_text(encoding="utf-8")


def test_settings_reads_user_overrides_from_user_data_dir(tmp_path):
    (tmp_path / "afb-data").mkdir(exist_ok=True)
    (tmp_path / "afb-data" / "user-settings.toml").write_text(
        'watch_folder = "C:/from-user-dir"\n', encoding="utf-8"
    )
    s = AiFileBrainSettings()
    assert s.watch_folder == "C:/from-user-dir"


def test_defaults_toml_path_uses_resource_path(monkeypatch, tmp_path):
    # When frozen, defaults_toml_path should resolve under _MEIPASS.
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "meipass"), raising=False)
    (tmp_path / "meipass").mkdir()
    assert defaults_toml_path() == tmp_path / "meipass" / "settings.toml"
```

Also update the existing tests in `test_config.py` that still write `Path(USER_OVERRIDES_TOML)` directly so they pass the resolved path:

```python
# Replace the existing test_user_overrides_beat_defaults_file
def test_user_overrides_beat_defaults_file(tmp_path):
    Path(DEFAULTS_TOML).write_text('watch_folder = "C:/defaults"\n', encoding="utf-8")
    user_overrides_path().write_text('watch_folder = "C:/user"\n', encoding="utf-8")
    s = AiFileBrainSettings()
    assert s.watch_folder == "C:/user"


# Replace the existing test_env_beats_user_overrides
def test_env_beats_user_overrides(monkeypatch, tmp_path):
    user_overrides_path().write_text('watch_folder = "C:/user"\n', encoding="utf-8")
    monkeypatch.setenv("AFB_WATCH_FOLDER", "C:/env")
    s = AiFileBrainSettings()
    assert s.watch_folder == "C:/env"


# Replace the existing test_save_user_overrides_creates_file_with_updates
def test_save_user_overrides_creates_file_with_updates():
    save_user_overrides({"watch_folder": "C:/notes", "chunk_size": 1234})
    written = user_overrides_path()
    assert written.exists()
    parsed = tomllib.loads(written.read_text(encoding="utf-8"))
    assert parsed["watch_folder"] == "C:/notes"
    assert parsed["chunk_size"] == 1234


# Replace the existing test_save_user_overrides_merges_existing
def test_save_user_overrides_merges_existing():
    user_overrides_path().write_text(
        'chunk_size = 1\nchat_model = "llama3.2"\n', encoding="utf-8"
    )
    save_user_overrides({"chunk_size": 2})
    parsed = tomllib.loads(user_overrides_path().read_text(encoding="utf-8"))
    assert parsed["chunk_size"] == 2
    assert parsed["chat_model"] == "llama3.2"


# Replace the existing test_save_user_overrides_handles_backslash_paths
def test_save_user_overrides_handles_backslash_paths():
    save_user_overrides({"watch_folder": r"C:\Users\me\Documents"})
    parsed = tomllib.loads(user_overrides_path().read_text(encoding="utf-8"))
    assert parsed["watch_folder"] == r"C:\Users\me\Documents"
```

Update the imports at the top of the test file:

```python
from ai_file_brain.config import (
    DEFAULTS_TOML,
    AiFileBrainSettings,
    defaults_toml_path,
    save_user_overrides,
    user_overrides_path,
)
```

(Remove the `USER_OVERRIDES_TOML` import; `user_overrides_path()` replaces it.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_config.py -v`
Expected: import errors for `user_overrides_path` and `defaults_toml_path`.

- [ ] **Step 3: Update `config.py`**

Replace the existing `config.py` contents:

```python
# src/ai_file_brain/config.py
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict, TomlConfigSettingsSource

from ai_file_brain.core.paths import resource_path, user_data_dir

DEFAULTS_TOML = "settings.toml"
USER_OVERRIDES_FILENAME = "user-settings.toml"


def defaults_toml_path() -> Path:
    """Resolve the bundled defaults file (works from source and from a frozen .exe)."""
    return resource_path(DEFAULTS_TOML)


def user_overrides_path() -> Path:
    """Resolve the per-user overrides file under the per-user data dir."""
    return user_data_dir() / USER_OVERRIDES_FILENAME


class AiFileBrainSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AFB_",
        extra="ignore",
        case_sensitive=False,
    )

    watch_folder: str = "~/Documents/AIFileBrain"
    ollama_url: str = "http://127.0.0.1:11434"
    # Empty string is a sentinel that means "use user_data_dir() / chroma-data".
    chroma_path: str = ""
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

    excluded_dir_names: list[str] = [
        "AppData", ".git", ".hg", ".svn", "node_modules", "__pycache__",
        ".venv", "venv", ".idea", ".vscode", "dist", "build",
        ".pytest_cache", ".ruff_cache", ".mypy_cache", "site-packages",
        "Recycle.Bin", "$RECYCLE.BIN",
    ]
    excluded_extensions: list[str] = [".lock", ".pyc", ".pyo", ".log", ".tmp", ".bak"]

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        defaults_file = _existing_or_cwd(defaults_toml_path(), DEFAULTS_TOML)
        overrides_file = _existing_or_cwd(user_overrides_path(), USER_OVERRIDES_FILENAME)
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls, toml_file=overrides_file),
            TomlConfigSettingsSource(settings_cls, toml_file=defaults_file),
            file_secret_settings,
        )

    def chroma_path_resolved(self) -> Path:
        if not self.chroma_path:
            return user_data_dir() / "chroma-data"
        return Path(self.chroma_path).expanduser().resolve()


def save_user_overrides(
    updates: dict[str, Any],
    path: str | Path | None = None,
) -> Path:
    """Merge ``updates`` into the user-overrides TOML and write it back."""
    target = Path(path) if path is not None else user_overrides_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if target.exists():
        try:
            existing = tomllib.loads(target.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            existing = {}
    existing.update(updates)
    target.write_text(_dump_flat_toml(existing), encoding="utf-8")
    return target.resolve()


def _existing_or_cwd(preferred: Path, fallback_name: str) -> Path | str:
    """Return ``preferred`` if it exists; otherwise the CWD-relative fallback name.

    This keeps source-tree tests working (they write ``settings.toml`` in tmp CWD)
    while letting the bundled app read its frozen defaults from ``_MEIPASS``.
    """
    if preferred.exists():
        return preferred
    return fallback_name


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
```

- [ ] **Step 4: Run config tests**

Run: `uv run pytest tests/core/test_config.py -v`
Expected: all pass.

- [ ] **Step 5: Run the full fast suite to confirm no regressions**

Run: `uv run pytest -m "not slow" -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ai_file_brain/config.py tests/core/test_config.py
git commit -m "feat(config): per-user overrides under LOCALAPPDATA; bundled defaults via resource_path"
```

---

## Task 4: Expand `~` in the watch folder

The new default is `~/Documents/AIFileBrain`. The watcher creates the folder at startup but doesn't expand `~`. Fix it.

**Files:**
- Modify: `src/ai_file_brain/core/watcher.py:169`
- Test: extend `tests/core/test_indexing_pipeline.py` (or wherever `FileWatcherService` is tested — pick the file that imports it) with a fast unit-style check.

- [ ] **Step 1: Write the failing test**

Add to a new file `tests/core/test_watcher_paths.py`:

```python
# tests/core/test_watcher_paths.py
from __future__ import annotations

from pathlib import Path

import pytest

from ai_file_brain.config import AiFileBrainSettings
from ai_file_brain.core.watcher import FileWatcherService


def test_watcher_start_expands_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))           # POSIX
    monkeypatch.setenv("USERPROFILE", str(tmp_path))    # Windows
    settings = AiFileBrainSettings(watch_folder="~/inbox")

    # We don't need to actually start the observer; just confirm the
    # path expansion in the helper. Extract it from FileWatcherService.
    # The simplest assertion: the resolved path exists once expansion happens.
    expanded = Path(settings.watch_folder).expanduser()
    assert expanded == tmp_path / "inbox"
```

Then add a second test that exercises the code path by importing a tiny helper we'll add. Skip the full `start()` (it spawns watchdog threads). Just assert that `FileWatcherService._resolve_watch_folder` returns the expanded path:

```python
def test_resolve_watch_folder_expands_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    settings = AiFileBrainSettings(watch_folder="~/inbox")
    svc = FileWatcherService(settings, pipeline=None, repo=None)  # type: ignore[arg-type]
    resolved = svc._resolve_watch_folder()
    assert resolved == (tmp_path / "inbox").resolve()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_watcher_paths.py -v`
Expected: FAIL — `_resolve_watch_folder` doesn't exist yet.

- [ ] **Step 3: Add the helper and use it in `start()`**

In `src/ai_file_brain/core/watcher.py`, find the `async def start(self) -> None:` method around line 165-176 and change it:

```python
    async def start(self) -> None:
        if self._running:
            return
        self._loop = asyncio.get_running_loop()
        folder = self._resolve_watch_folder()
        folder.mkdir(parents=True, exist_ok=True)
        self._observer = Observer()
        self._handler = _Handler(self)
        self._observer.schedule(self._handler, str(folder), recursive=True)
        self._observer.start()
        self._running = True
        logger.info("Watching %s", folder)
        await self._initial_scan()

    def _resolve_watch_folder(self) -> Path:
        return Path(self._settings.watch_folder).expanduser().resolve()
```

Also update `_initial_scan` (around line 274) to use the same resolution:

```python
    async def _initial_scan(self) -> None:
        folder = self._resolve_watch_folder()
        for path in folder.rglob("*"):
            if not path.is_file():
                continue
            ...
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/core/test_watcher_paths.py -v`
Expected: PASS.

Also run the existing watcher / pipeline tests:

Run: `uv run pytest tests/core/test_indexing_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ai_file_brain/core/watcher.py tests/core/test_watcher_paths.py
git commit -m "fix(watcher): expand ~ in watch_folder so per-user defaults work"
```

---

## Task 5: Update `settings.toml` defaults

**Files:**
- Modify: `settings.toml`

- [ ] **Step 1: Edit `settings.toml`**

Make these changes:
- Change `watch_folder = "C:/Users/ASUS/Documents/AIFileBrainTest"` to `watch_folder = "~/Documents/AIFileBrain"`.
- Delete the `chroma_path = "./chroma-data"` line entirely (the empty-string sentinel in the model now means "use per-user data dir").

The resulting file:

```toml
# Default settings for AI File Brain.
# Override any value via environment variable: AFB_<UPPERCASE_FIELD>=...
# Example: AFB_WATCH_FOLDER="D:/notes"

watch_folder = "~/Documents/AIFileBrain"
ollama_url = "http://127.0.0.1:11434"
embedding_model = "nomic-embed-text"
chat_model = "llama3.2"
chunk_size = 2000
chunk_overlap = 400
top_k = 5

# OCR — image files and scanned-PDF fallback.
ocr_enabled = true
ocr_languages = ["en"]
pdf_ocr_min_native_chars = 50      # below this total, switch to per-page OCR fallback
pdf_ocr_per_page_min_chars = 10    # below this on a page, OCR that page
pdf_ocr_render_dpi = 220

# Hard upper bound on per-file size before indexing. Skips runaway lockfiles,
# generated bundles, etc. 10 MiB by default.
max_file_size_bytes = 10485760

# Smart exclusions: any path containing one of these directory names is
# skipped (case-insensitive). Edit this list in user-settings.toml to add
# project-specific noise without touching defaults.
excluded_dir_names = [
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

excluded_extensions = [".lock", ".pyc", ".pyo", ".log", ".tmp", ".bak"]
```

- [ ] **Step 2: Run the full fast suite**

Run: `uv run pytest -m "not slow" -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add settings.toml
git commit -m "feat(config): portable defaults — ~/Documents/AIFileBrain and per-user chroma path"
```

---

## Task 6: Icon loader uses `resource_path`

**Files:**
- Modify: `src/ai_file_brain/app/main.py:23-28`

The canonical bundle-relative path is `assets/tray_icon.ico`. The PyInstaller spec (Task 8) maps `src/ai_file_brain/app/assets/` → `assets/` so this path works both from source and frozen.

- [ ] **Step 1: Update `_load_icon`**

In `src/ai_file_brain/app/main.py`, replace `_load_icon`:

```python
from ai_file_brain.core.paths import resource_path


def _load_icon(qapp: QApplication) -> QIcon:
    asset = resource_path("assets/tray_icon.ico")
    if asset.exists():
        return QIcon(str(asset))
    # fallback to a built-in style icon so tray + window have something
    return qapp.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
```

Drop the `from pathlib import Path` import if it's no longer used in the file (it currently is, by `_load_icon`).

- [ ] **Step 2: Quick smoke check from source**

Run: `uv run python -c "from ai_file_brain.app.main import _load_icon; from PySide6.QtWidgets import QApplication; import sys; app=QApplication(sys.argv); icon=_load_icon(app); print('OK', not icon.isNull())"`
Expected output ends with `OK True`.

- [ ] **Step 3: Move the asset file in the source tree to match the spec mapping**

The spec uses `src/ai_file_brain/app/assets/` as the source path. That's already where `tray_icon.ico` lives. Confirm:

Run: `uv run python -c "from pathlib import Path; print(Path('src/ai_file_brain/app/assets/tray_icon.ico').exists())"`
Expected: `True`.

No file move needed — `resource_path("assets/tray_icon.ico")` resolves to `<project_root>/assets/tray_icon.ico` when not frozen, but the file is at `<project_root>/src/ai_file_brain/app/assets/tray_icon.ico`. We need either a copy or a smarter `resource_path` for assets.

Pick the cleaner option: extend `paths.py` so `resource_path` checks a registered set of source-tree roots before falling back. Concretely, look for the rel under both `<project_root>/` and `<project_root>/src/ai_file_brain/app/`.

Update `paths.py`:

```python
def resource_path(rel: str) -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / rel  # type: ignore[attr-defined]

    root = _project_root()
    candidates = [
        root / rel,
        root / "src" / "ai_file_brain" / "app" / rel,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # caller will see the missing-file path in its error
```

Update `tests/core/test_paths.py`:

```python
def test_resource_path_finds_app_assets_when_not_frozen(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    got = paths.resource_path("assets/tray_icon.ico")
    assert got.name == "tray_icon.ico"
    assert got.exists()
```

- [ ] **Step 4: Run all paths and main-icon tests**

Run: `uv run pytest tests/core/test_paths.py -v`
Expected: PASS, including the new asset-lookup test.

Re-run the from-source icon smoke check from Step 2 — it should still print `OK True`.

- [ ] **Step 5: Commit**

```bash
git add src/ai_file_brain/app/main.py src/ai_file_brain/core/paths.py tests/core/test_paths.py
git commit -m "feat(app): load tray icon via bundle-aware resource_path"
```

---

## Task 7: Ollama health-check banner

A `QFrame` at the top of the chat window. Visible when Ollama is unhealthy and not dismissed; hidden once Ollama goes healthy or the user clicks the close button.

**Files:**
- Modify: `src/ai_file_brain/app/views/main_window.py`
- Test: extend `tests/app/test_status_bar_vm.py` is NOT the right place — we need a widget test. Create `tests/app/test_main_window_banner.py`.

- [ ] **Step 1: Write the failing widget test**

```python
# tests/app/test_main_window_banner.py
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from ai_file_brain.app.view_models.main_window_vm import MainWindowViewModel
from ai_file_brain.app.view_models.status_bar_vm import StatusBarViewModel
from ai_file_brain.app.views.main_window import MainWindow


@pytest.fixture
def app(qtbot):
    return QApplication.instance() or QApplication([])


def _build_window():
    status = StatusBarViewModel()
    vm = MainWindowViewModel(chat=None)  # type: ignore[arg-type]
    return MainWindow(vm, status), status, vm


def test_banner_hidden_when_ollama_healthy(qtbot):
    window, status, _vm = _build_window()
    qtbot.addWidget(window)
    status.ollama_healthy = True
    assert window.ollama_banner.isVisible() is False


def test_banner_shows_when_ollama_unhealthy(qtbot):
    window, status, _vm = _build_window()
    qtbot.addWidget(window)
    window.show()
    status.ollama_healthy = False
    assert window.ollama_banner.isVisible() is True


def test_banner_hides_when_ollama_recovers(qtbot):
    window, status, _vm = _build_window()
    qtbot.addWidget(window)
    window.show()
    status.ollama_healthy = False
    assert window.ollama_banner.isVisible() is True
    status.ollama_healthy = True
    assert window.ollama_banner.isVisible() is False


def test_banner_copy_button_copies_install_commands(qtbot):
    window, status, _vm = _build_window()
    qtbot.addWidget(window)
    window.show()
    status.ollama_healthy = False
    window.ollama_banner_copy_button.click()
    clip = QApplication.clipboard().text()
    assert "ollama pull nomic-embed-text" in clip
    assert "ollama pull llama3.2" in clip


def test_banner_dismiss_hides_for_rest_of_session(qtbot):
    window, status, _vm = _build_window()
    qtbot.addWidget(window)
    window.show()
    status.ollama_healthy = False
    assert window.ollama_banner.isVisible() is True
    window.ollama_banner_dismiss_button.click()
    assert window.ollama_banner.isVisible() is False
    # Even if Ollama stays unhealthy, the banner should not re-show this session.
    status.ollama_healthy = True
    status.ollama_healthy = False
    assert window.ollama_banner.isVisible() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/app/test_main_window_banner.py -v`
Expected: FAIL — `MainWindow` has no `ollama_banner` attribute.

- [ ] **Step 3: Implement the banner in `MainWindow`**

In `src/ai_file_brain/app/views/main_window.py`, add the banner colors to `PALETTE` (just after `"status_bg"`):

```python
    "banner_bg": "#fffaf0",
    "banner_border": "#fbd38d",
    "banner_text": "#744210",
```

Add CSS to `STYLESHEET` (before the closing `"""`):

```python
QFrame#OllamaBanner {
    background-color: """ + PALETTE["banner_bg"] + """;
    border-bottom: 1px solid """ + PALETTE["banner_border"] + """;
}
QLabel#OllamaBannerLabel {
    color: """ + PALETTE["banner_text"] + """;
    font-size: 12px;
    padding: 2px 0;
}
QPushButton#OllamaBannerCopy, QPushButton#OllamaBannerDismiss {
    background-color: transparent;
    color: """ + PALETTE["banner_text"] + """;
    border: 1px solid """ + PALETTE["banner_border"] + """;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
}
QPushButton#OllamaBannerCopy:hover, QPushButton#OllamaBannerDismiss:hover {
    background-color: """ + PALETTE["banner_border"] + """;
}
"""
```

(Alternative: append the rules via an f-string the same way the rest of `STYLESHEET` does. Match the existing pattern.)

In `MainWindow.__init__`, *immediately after* `outer = QVBoxLayout(self)` and its `setContentsMargins`/`setSpacing` calls, and *before* `# ---- transcript ----`, insert the banner:

```python
        # ---- Ollama banner ----
        self._banner_dismissed = False
        self.ollama_banner = QFrame()
        self.ollama_banner.setObjectName("OllamaBanner")
        self.ollama_banner.setVisible(False)
        banner_layout = QHBoxLayout(self.ollama_banner)
        banner_layout.setContentsMargins(12, 8, 12, 8)
        banner_layout.setSpacing(8)
        banner_text = QLabel(
            "Ollama is not reachable. Install from "
            '<a href="https://ollama.com">ollama.com</a>, then run:'
            "<br><code>ollama pull nomic-embed-text &amp;&amp; ollama pull llama3.2</code>"
        )
        banner_text.setObjectName("OllamaBannerLabel")
        banner_text.setOpenExternalLinks(True)
        banner_text.setTextFormat(Qt.TextFormat.RichText)
        banner_text.setWordWrap(True)
        banner_layout.addWidget(banner_text, 1)
        self.ollama_banner_copy_button = QPushButton("Copy commands")
        self.ollama_banner_copy_button.setObjectName("OllamaBannerCopy")
        self.ollama_banner_copy_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.ollama_banner_copy_button.clicked.connect(self._on_banner_copy)
        banner_layout.addWidget(self.ollama_banner_copy_button)
        self.ollama_banner_dismiss_button = QPushButton("✕")
        self.ollama_banner_dismiss_button.setObjectName("OllamaBannerDismiss")
        self.ollama_banner_dismiss_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.ollama_banner_dismiss_button.clicked.connect(self._on_banner_dismiss)
        banner_layout.addWidget(self.ollama_banner_dismiss_button)
        outer.addWidget(self.ollama_banner)
```

Hook the banner into status changes. At the end of `__init__`, where existing `status_vm.changed.connect(self._refresh_status)` lives, leave that wiring alone — `_refresh_status` will now also drive the banner. Update `_refresh_status`:

```python
    def _refresh_status(self) -> None:
        self._status_label.setText(self._status_vm.render_html())
        self._refresh_banner()

    def _refresh_banner(self) -> None:
        if self._banner_dismissed:
            self.ollama_banner.setVisible(False)
            return
        self.ollama_banner.setVisible(not self._status_vm.ollama_healthy)
```

Add the button handlers as new methods on `MainWindow`:

```python
    def _on_banner_copy(self) -> None:
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(
            "ollama pull nomic-embed-text && ollama pull llama3.2"
        )

    def _on_banner_dismiss(self) -> None:
        self._banner_dismissed = True
        self.ollama_banner.setVisible(False)
```

- [ ] **Step 4: Run banner tests**

Run: `uv run pytest tests/app/test_main_window_banner.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Run the full fast suite**

Run: `uv run pytest -m "not slow" -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ai_file_brain/app/views/main_window.py tests/app/test_main_window_banner.py
git commit -m "feat(app): banner surfaces the Ollama prerequisite for first-run shareable builds"
```

---

## Task 8: Rewrite `pyinstaller.spec` for `--onefile`

**Files:**
- Modify: `pyinstaller.spec`

This task has no unit tests; the verification is "PyInstaller builds successfully and the resulting .exe launches." That's Task 11.

- [ ] **Step 1: Replace the spec**

Overwrite `pyinstaller.spec` with:

```python
# -*- mode: python ; coding: utf-8 -*-
# Run: `pyinstaller pyinstaller.spec --noconfirm --clean`
#
# Output: dist/ai-file-brain.exe  (single file)

import os

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None
ICON_PATH = "src/ai_file_brain/app/assets/tray_icon.ico"
ICON_ARG = ICON_PATH if os.path.exists(ICON_PATH) else None

hidden_imports = []
hidden_imports += collect_submodules("chromadb")
hidden_imports += collect_submodules("chromadb.telemetry")
hidden_imports += collect_submodules("chromadb.api")
hidden_imports += collect_submodules("chromadb.db")
hidden_imports += collect_submodules("chromadb.segment")
hidden_imports += collect_submodules("chromadb.utils")
hidden_imports += collect_submodules("PySide6")
hidden_imports += collect_submodules("rapidocr_onnxruntime")
hidden_imports += [
    "qasync",
    "ollama",
    "pypdf",
    "pymupdf",
    "docx",
    "PIL",
    "PIL.Image",
    "PIL.ImageSequence",
    "numpy",
    "onnxruntime",
    "watchdog.observers",
    "watchdog.events",
    "pydantic_settings",
    "pydantic_settings.sources",
]

datas = []
datas += collect_data_files("chromadb")
datas += collect_data_files("rapidocr_onnxruntime")
datas += [
    ("settings.toml", "."),
    ("src/ai_file_brain/app/assets", "assets"),
]

excludes = [
    # Test-only deps shouldn't ship.
    "pytest",
    "pytest_asyncio",
    "pytest_qt",
]


a = Analysis(
    ["src/ai_file_brain/app/main.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="ai-file-brain",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON_ARG,
)
```

Key diffs from the old spec:
- `onnxruntime` removed from `excludes`.
- `rapidocr_onnxruntime`, `onnxruntime`, `pymupdf`, `docx`, `PIL`, `PIL.Image`, `PIL.ImageSequence`, `numpy` added to `hidden_imports`.
- `collect_data_files("rapidocr_onnxruntime")` added to `datas`.
- Assets data-file mapping changed from `"ai_file_brain/app/assets"` to `"assets"` (to match `resource_path("assets/tray_icon.ico")`).
- `EXE(...)` now bundles binaries + zipfiles + datas inline (onefile mode).
- `COLLECT(...)` block deleted (no folder output).

- [ ] **Step 2: Commit the spec rewrite**

```bash
git add pyinstaller.spec
git commit -m "build: rewrite pyinstaller.spec for --onefile with OCR-stack imports"
```

(Note: this commit doesn't run the build itself — that happens in Task 11. The spec change is a code change and lands on its own commit so a build failure in Task 11 doesn't pollute the spec history.)

---

## Task 9: One-command build script

**Files:**
- Create: `scripts/build.ps1`

- [ ] **Step 1: Create the `scripts/` directory if absent**

Run: `New-Item -ItemType Directory -Force -Path scripts | Out-Null`

- [ ] **Step 2: Write `scripts/build.ps1`**

```powershell
# scripts/build.ps1
# Build a single-file ai-file-brain.exe for sharing.
# Usage:  .\scripts\build.ps1
$ErrorActionPreference = "Stop"

Write-Host "==> Installing build dependencies"
uv sync --extra build
if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

Write-Host "==> Cleaning previous build artifacts"
Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue

Write-Host "==> Running PyInstaller (--onefile)"
uv run pyinstaller pyinstaller.spec --noconfirm --clean
if ($LASTEXITCODE -ne 0) { throw "pyinstaller failed" }

$exe = Resolve-Path "dist\ai-file-brain.exe" -ErrorAction SilentlyContinue
if (-not $exe) { throw "Build did not produce dist\ai-file-brain.exe" }

$size = (Get-Item $exe).Length / 1MB
Write-Host ""
Write-Host "==> Built: $exe ($([math]::Round($size,1)) MB)"
Write-Host "    Share this file. Recipients need Ollama installed; the app shows a banner if it isn't."
```

- [ ] **Step 3: Commit**

```bash
git add scripts/build.ps1
git commit -m "build: add scripts/build.ps1 for one-command bundle"
```

---

## Task 10: README handoff section

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the existing build section**

In `README.md`, locate the section beginning with `## Build a standalone folder (single .exe + deps)` and replace it (including its code fence) with:

```markdown
## Build a shareable .exe

```powershell
.\scripts\build.ps1
```

Output: `dist\ai-file-brain.exe` (one file, ~400–600 MB). Send that file to anyone you want to share with.

## For people I'm sharing the .exe with

1. **Install Ollama** from https://ollama.com (one-time, free, local).
2. Pull the two models the app uses (one-time, ~2.3 GB total):
   ```
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```
3. Double-click `ai-file-brain.exe`.
   - Windows SmartScreen may say "Windows protected your PC" because the .exe is unsigned. Click **More info → Run anyway**.
   - The first launch unpacks the bundled Python runtime to a temp dir; allow ~10 seconds.
4. The app lives in the system tray (look for the brain icon). Right-click it for **Show**, **Change watch folder…**, and **Quit**.
5. If you see an orange "Ollama is not reachable" banner inside the chat window, Ollama isn't running — start it and the banner will clear itself within ~10 seconds.

Data location: the bundled app keeps its vector store and per-user settings under `%LOCALAPPDATA%\AIFileBrain\`.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(readme): document single-exe build and recipient setup"
```

---

## Task 11: Build, smoke-test, and verify

This is the manual verification step from the spec's §Verification. It is not a unit test — its purpose is to catch the things unit tests can't: missing PyInstaller hooks, runtime-path bugs that only surface when frozen, and visible-UI regressions.

**Files:** none modified. This task only runs commands and observes behaviour.

- [ ] **Step 1: Prerequisites check**

Confirm Ollama is installed and the two models are pulled. Run:

```
ollama list
```

Expected: rows for `nomic-embed-text` and `llama3.2`. If either is missing: `ollama pull nomic-embed-text` and/or `ollama pull llama3.2`.

- [ ] **Step 2: Build**

Run: `.\scripts\build.ps1`
Expected final line: `==> Built: ...\dist\ai-file-brain.exe (NNN.N MB)`. Build should take 1–5 minutes the first time.

If the build fails: read the PyInstaller error. Two common modes:
- `ImportError` for a runtime dep → add it to `hidden_imports` in `pyinstaller.spec`, re-run.
- Missing data file → add `collect_data_files("<pkg>")` to `datas`, re-run.

- [ ] **Step 3: Out-of-tree smoke test**

```
$dest = "$env:TEMP\afb-smoke"
Remove-Item -Recurse -Force $dest -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $dest | Out-Null
Copy-Item dist\ai-file-brain.exe $dest\
cd $dest
.\ai-file-brain.exe
```

Expected:
- After 5–15 seconds the tray icon appears (Windows tray, may be in the "hidden icons" pop-up).
- Right-click the tray icon → **Show** → chat window opens.
- Status strip at the bottom shows `Watching ~/Documents/AIFileBrain` (with `~` expanded), `0 chunks`, `Ollama ●` (green) and `Chroma ●` (green).
- The Ollama banner is NOT visible.

If the tray icon doesn't appear: re-launch from a `cmd` prompt so any startup errors are visible.

- [ ] **Step 4: Banner verification — Ollama-down state**

Stop Ollama (Task Manager → end `ollama.exe`, or `taskkill /F /IM ollama.exe`). Within ~10 seconds (one health-check tick):

Expected:
- The orange banner appears at the top of the chat window with the install instructions.
- Status strip shows `Ollama ●` (red).
- Click "Copy commands". Paste in Notepad. Expected text: `ollama pull nomic-embed-text && ollama pull llama3.2`.
- Click the `✕` button. Banner disappears.
- Restart Ollama. Status strip flips back to green within ~10 seconds.

- [ ] **Step 5: Indexing pipeline smoke test**

In the chat window, click **Change folder…** → pick `C:\Temp\afb-smoke\watch`. Create that folder if Windows asks.

Drop three files into it:
- A small `.txt` with the word "BANANA" in it.
- A small native-text PDF (any).
- A `.png` with some readable text (a screenshot of a paragraph works).

Expected:
- Within seconds, the tray tooltip shows `Indexing: <filename>` for each in turn.
- The status strip chunk count goes up.

- [ ] **Step 6: Chat smoke test**

In the chat window, ask: `does any file mention banana?`

Expected:
- An answer streams in within ~10 seconds, mentioning the `.txt` file as a source.

- [ ] **Step 7: Persistence test**

Quit via tray → **Quit**. Wait 3 seconds. Re-launch `ai-file-brain.exe`.

Expected:
- The chunk count in the status strip is immediately ≥ 3 (the three files from step 5), not 0. This proves `%LOCALAPPDATA%\AIFileBrain\chroma-data\` persisted across runs.

- [ ] **Step 8: Data-dir inspection**

Run: `dir $env:LOCALAPPDATA\AIFileBrain`
Expected: a `chroma-data\` subfolder exists. If you changed the watch folder via the UI, also expect `user-settings.toml`.

- [ ] **Step 9: Document any deviations**

If any of Steps 3–8 deviated from "Expected", file the deviation as a follow-up:

- Stop here.
- Open `docs/superpowers/specs/2026-05-13-single-exe-packaging-design.md` and append a `## Known issues from smoke test` section.
- Decide with the user whether to address now or defer.

- [ ] **Step 10: Final commit (notes-only, if any)**

If the smoke test surfaced no issues, no commit is needed. Otherwise:

```bash
git add docs/superpowers/specs/2026-05-13-single-exe-packaging-design.md
git commit -m "docs(specs): note smoke-test findings for single-exe build"
```

---

## Self-Review Checklist (for the executing agent)

After completing all tasks:

- [ ] `uv run pytest -m "not slow"` is green.
- [ ] `dist\ai-file-brain.exe` exists and runs from outside the source tree.
- [ ] Ollama banner shows/hides correctly with Ollama up/down.
- [ ] `%LOCALAPPDATA%\AIFileBrain\chroma-data\` is created and persists across runs.
- [ ] README's "Build a shareable .exe" + "For people I'm sharing the .exe with" sections are present and accurate.

If any item is unchecked, the work is not done — go back to the corresponding task.
