# Single-EXE Packaging — Design

**Date:** 2026-05-13
**Scope:** Produce one shareable `ai-file-brain.exe` for Windows from the current Python codebase.
**Builds on:** the v1 design doc at `docs/superpowers/specs/2026-04-28-ai-file-brain-python-design.md`, which scoped packaging as `--onedir`.

## Why this exists

The current `pyinstaller.spec` is `--onedir` and was last updated before OCR, DOCX, and image extraction shipped. A build run today would either fail to bundle the new dependencies or crash at runtime the first time OCR is triggered (the spec excludes `onnxruntime`, which RapidOCR requires). The deliverable also needs to be one file the developer can email or upload — not a folder.

Ollama is an external prerequisite and cannot be bundled (separate Go binary plus ~2.3 GB of model weights). The plan accepts this and surfaces the requirement to the recipient via README and an in-app banner.

## Non-goals

- Code signing the `.exe`. Recipients will see Windows SmartScreen on first run.
- Mac / Linux builds.
- Auto-installing Ollama or pulling models from inside the app.
- Auto-update / MSIX / installer (.msi, NSIS, Inno Setup).
- Reducing the binary size below what `--onefile` + `upx=True` produces.

## Architecture

Five small, focused changes:

1. Rewrite `pyinstaller.spec` for `--onefile` with correct hidden imports, data files, and excludes.
2. Introduce `src/ai_file_brain/core/paths.py` for bundle-aware resource and per-user data paths.
3. Update three call sites (`config.py`, icon load in app startup, optionally `ocr.py`) to use those paths.
4. Add a first-run banner that surfaces the Ollama prerequisite, driven by the existing `HealthCheckService`.
5. Add a build script and update the README.

Everything else (the indexing pipeline, chat, watcher, UI) is untouched.

## Component changes

### `pyinstaller.spec` (rewrite)

- Switch to `--onefile`: drop `exclude_binaries=True` from `EXE(...)`, remove the `COLLECT(...)` block, pass `a.binaries`, `a.zipfiles`, `a.datas` directly into `EXE(...)`.
- Hidden imports add: `rapidocr_onnxruntime`, `onnxruntime`, `pymupdf`, `docx`, `PIL`, `PIL.Image`, `PIL.ImageSequence`, `numpy`.
- Data files add: `collect_data_files("rapidocr_onnxruntime")` (the `.onnx` weights are non-Python data). Keep `collect_data_files("chromadb")`, the bundled `settings.toml`, and the `assets/` folder.
- Excludes: remove `onnxruntime` (RapidOCR needs it). Keep `pytest*` excludes.
- `upx=True` stays. If AV false-positives become a delivery problem later, flip to `upx=False`.

### `src/ai_file_brain/core/paths.py` (new)

Two functions:

- `resource_path(rel: str) -> Path` — when frozen, returns `Path(sys._MEIPASS) / rel`. Otherwise returns the source-tree path (the project root). Used to locate `settings.toml`, `tray_icon.ico`, and any RapidOCR model files at runtime.
- `user_data_dir() -> Path` — on Windows, returns `Path(os.environ["LOCALAPPDATA"]) / "AIFileBrain"`. Creates it if missing. Mac and Linux fall back to `~/Library/Application Support/AIFileBrain` and `~/.local/share/AIFileBrain` respectively, even though Mac/Linux builds are out of scope today — the helper is OS-portable so we don't paint ourselves into a corner.

### `config.py` (modify)

- Change `chroma_path: str = "./chroma-data"` to `chroma_path: str = ""` (sentinel for "use per-user default").
- `chroma_path_resolved()` returns `user_data_dir() / "chroma-data"` when `chroma_path` is empty, otherwise the explicit value (so an `AFB_CHROMA_PATH` env var still overrides).
- Remove the `chroma_path = "./chroma-data"` line from `settings.toml` so the sentinel default isn't shadowed by the bundled defaults file.
- `USER_OVERRIDES_TOML` constant becomes a function that returns `user_data_dir() / "user-settings.toml"`. Overrides written by the running app are per-user, not next to the .exe (the .exe lives in a temp-extracted dir under `--onefile` and would be wiped between runs).
- `DEFAULTS_TOML` for `TomlConfigSettingsSource` is read from `resource_path("settings.toml")` so the bundled defaults are always findable both from source and when frozen.
- `watch_folder` default in `settings.toml` changes from the hardcoded `C:/Users/ASUS/Documents/AIFileBrainTest` to `~/Documents/AIFileBrain`. The watcher expands `~` and creates the folder at `watcher.py:169`; confirm that line uses `Path(...).expanduser()`.

### Icon loading (modify wherever `tray_icon.ico` is loaded)

Pick one canonical relative path used everywhere: `assets/tray_icon.ico`.

- Update the spec's data-file mapping to `("src/ai_file_brain/app/assets", "assets")` (source path → bundle path). Today's mapping (`"ai_file_brain/app/assets"`) changes.
- `resource_path("assets/tray_icon.ico")` resolves to `<src tree>/src/ai_file_brain/app/assets/tray_icon.ico` from source, and `<_MEIPASS>/assets/tray_icon.ico` when frozen. `resource_path` is the only piece of code aware of the `src/` prefix.
- All callers that load the icon use the same `resource_path("assets/tray_icon.ico")` call. Find the call sites by searching for `tray_icon.ico` in `src/ai_file_brain/app/` and switch them.

### Ollama health-check banner (modify UI)

`HealthCheckService` already exposes Ollama health via `StatusBarViewModel`. Add a one-shot banner inside the chat window:

- A `QFrame` at the top of the chat window, hidden by default.
- Bind visibility to `status_vm.ollama_healthy`: shows when `False`, hides when `True`.
- Banner contents: a label with the install/pull instructions, a "Copy commands" `QPushButton` that copies `ollama pull nomic-embed-text && ollama pull llama3.2` to the clipboard, and a dismiss `X`.
- Dismiss is session-scoped (no persistence). User can re-trigger by relaunching with Ollama still missing.
- No new dependency.

### OCR model path (conditional)

RapidOCR usually finds its bundled `.onnx` weights via `importlib.resources` once the data files are collected by `collect_data_files`. If a smoke-test build shows it can't find them under `--onefile`, pass an explicit `det_model_path` / `rec_model_path` to `RapidOCR()` in `ocr.py` pointing at `resource_path("rapidocr_onnxruntime/models/...")`. This is a contingency, not a planned change.

### `scripts/build.ps1` (new)

```powershell
uv sync --extra build
Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue
uv run pyinstaller pyinstaller.spec --noconfirm --clean
Write-Host "Built: $(Resolve-Path dist\ai-file-brain.exe)"
```

Single command for the developer; produces `dist\ai-file-brain.exe`.

### README (modify)

Replace the existing "Build a standalone folder" section with:

- **Build a single .exe** — point at `scripts/build.ps1`; note the artifact path.
- **For people I'm sharing the .exe with** — install Ollama (link), `ollama pull nomic-embed-text`, `ollama pull llama3.2`, double-click the .exe, accept the SmartScreen "More info → Run anyway" prompt because the .exe is unsigned, app lives in the tray.

## Failure-mode analysis

| Scenario | What happens | Mitigation |
|---|---|---|
| Recipient hasn't installed Ollama | Health check fails, banner appears with copy-paste instructions, chat returns an error if attempted | Banner is the mitigation; chat error already exists. |
| Recipient hasn't pulled models | Ollama is reachable but embedding/chat calls 404 on model. Health check today only checks reachability, not model presence. | Out of scope to detect model presence; README is explicit. Optional future: probe the model in the health check. |
| SmartScreen / AV blocks the unsigned .exe | Windows shows a warning; some AVs may quarantine UPX'd binaries | Document the SmartScreen step in README. If AV becomes a real problem, flip `upx=False` and re-build. |
| `_MEIPASS` extraction collides with a previous instance | Rare; PyInstaller handles it via per-PID temp dirs | None needed. |
| Per-user data dir is read-only (locked-down corp machine) | First write to `chroma-data` fails, app shows the existing `QMessageBox.critical` from `di.py:51-58` | Existing error path already exists; message includes the offending path. |
| Recipient launches the .exe from a network drive | Slower cold-start as the .exe self-extracts over the network | Tell users to copy locally; document in README. |

## Verification

Manual checklist on the developer's machine after building:

1. Build runs cleanly: `scripts\build.ps1` succeeds.
2. `dist\ai-file-brain.exe` exists; size is in the 300–700 MB range.
3. Move the .exe to a folder outside the source tree (e.g. `C:\Temp\afb-smoke\`). Confirms no accidental CWD-relative path dependencies survived.
4. Launch the .exe. Tray icon appears, chat window opens, no missing-file dialog.
5. With Ollama running and models pulled: status strip shows `Ollama ✓` and `Chroma ✓`; the first-run banner does NOT appear.
6. With Ollama stopped: the first-run banner appears; status strip shows `Ollama ✗`; clicking "Copy commands" puts the two pull commands on the clipboard.
7. Drop a `.txt`, a small `.pdf`, and a `.png` into the watch folder. Each appears in the tray tooltip's "Indexing …" line and the chunk count ticks up.
8. Ask a chat question that matches one of the dropped files. Receive a streamed answer with sources.
9. Quit via the tray menu. Relaunch. Confirm the chunk count is non-zero immediately — proves `%LOCALAPPDATA%\AIFileBrain\chroma-data\` persisted across runs.
10. Inspect `%LOCALAPPDATA%\AIFileBrain\` and confirm `chroma-data/` lives there (not next to the .exe, not in `%TEMP%`).

Not verified by this checklist: that the .exe runs on a *different* clean Windows machine without Python installed. PyInstaller bundles the interpreter so it should, but the only honest test is shipping it to a tester.

## Out of scope (future)

- Code signing (an EV cert and signing pipeline).
- Auto-detect Ollama-missing-model state in the health check (currently only reachability is checked).
- Splitting the build for two channels (debug-console vs release-windowed).
- Mac / Linux builds.
- An installer (.msi via WiX, or NSIS / Inno Setup).
