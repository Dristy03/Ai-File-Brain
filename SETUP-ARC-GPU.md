# Setting up AI File Brain with Intel Arc GPU on a new PC

This guide gets the app running with **GPU-accelerated Ollama** on an Intel Arc GPU.

Stock Ollama only uses the **CPU** on Intel Arc machines (no CUDA/ROCm). To use the GPU
you need a separate **IPEX-LLM** build of Ollama. The app itself talks to Ollama at
`http://127.0.0.1:11434`, so once the GPU build is serving on that port, nothing in the
app needs to change.

The setup has **two independent halves**: the app, and the GPU Ollama build.

---

## Half 1 — The app

1. Install **Python 3.12** and **[uv](https://docs.astral.sh/uv/)**, both on PATH.
2. In the project folder:
   ```powershell
   uv venv
   uv pip install -e ".[dev]"
   ```
3. Set your watch folder. The default in `settings.toml` points at a folder that
   won't exist on a new PC. Create `user-settings.toml` (gitignored) with:
   ```toml
   watch_folder = "C:/Users/<you>/Documents/AIFileBrainTest"
   ```
   (Or set the `AFB_WATCH_FOLDER` environment variable.)

---

## Half 2 — The Arc GPU Ollama build

### Step 1 — One-time machine prep
- **Update the Intel graphics driver** (ships the GPU / oneAPI runtime).
- **Turn Windows Developer Mode ON** — Settings → System → For developers.
  Required for the symlink step below (`mklink`).
- **Install Python 3.11** — *not 3.12*. The IPEX install fails on 3.12
  (`sentencepiece~=0.1.98` has no 3.12 Windows wheel).

### Step 2 — Build the GPU Ollama
Pick any folder outside the repo (this guide uses `D:\ipex-ollama`):
```powershell
uv python install 3.11
uv venv D:\ipex-ollama\env --python 3.11
D:\ipex-ollama\env\Scripts\activate
pip install --pre --upgrade "ipex-llm[cpp]"
cd D:\ipex-ollama
init-ollama.bat
```
This installs `ipex-llm` + `bigdl-core-cpp` (Ollama 0.9.3, SYCL backend) and symlinks
the GPU `ollama` binaries into `D:\ipex-ollama`.

### Step 3 — Add the launcher
Copy `start-gpu-ollama.bat` from the old PC's `D:\ipex-ollama` into the new one.
If you don't have it, create `D:\ipex-ollama\start-gpu-ollama.bat` with:

```bat
@echo off
set "IPEX_DIR=D:\ipex-ollama"

REM IPEX dir goes FIRST on PATH so a stock Ollama can't hijack the server/runner,
REM plus the oneAPI/SYCL runtime DLLs.
set "PATH=%IPEX_DIR%;%IPEX_DIR%\env\Library\bin;%PATH%"

REM --- Intel GPU runtime settings ---
set "ONEAPI_DEVICE_SELECTOR=level_zero:0"
set "SYCL_CACHE_PERSISTENT=1"
set "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1"
set "ZES_ENABLE_SYSMAN=1"

REM Push all model layers onto the GPU.
set "OLLAMA_NUM_GPU=999"

REM Keep the model resident so there's no slow cold-reload after idle.
set "OLLAMA_KEEP_ALIVE=60m"

REM Reuse the models you already pulled.
set "OLLAMA_MODELS=%USERPROFILE%\.ollama\models"

cd /d "%IPEX_DIR%"
echo Starting GPU Ollama server on http://127.0.0.1:11434 ...
echo (Leave this window open. Close it to stop the server.)
"%IPEX_DIR%\ollama.exe" serve
```

> ⚠️ **Multiple GPUs:** `ONEAPI_DEVICE_SELECTOR=level_zero:0` assumes the Arc is
> device 0. If the PC has more than one GPU, change `:0` to the right index.

### Step 4 — Stop stock Ollama from stealing port 11434
Both builds serve on port `11434`; whichever starts first wins. If stock Ollama
auto-starts at login, your GPU build can't bind the port and the app silently falls
back to the slow CPU one.

Disable stock Ollama's autostart (only if stock Ollama was ever installed):

1. Press `Win + R`, type `shell:startup`, press Enter.
2. Rename `Ollama.lnk` → `Ollama.lnk.disabled`
   (if extensions are hidden it shows as `Ollama` → rename to `Ollama.disabled`).

Or via PowerShell:
```powershell
Rename-Item "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\Ollama.lnk" "Ollama.lnk.disabled"
```
**To re-enable later:** rename it back to `Ollama.lnk`.

### Step 5 — Pull the models
```powershell
ollama pull nomic-embed-text
ollama pull llama3.2
```

---

## Running it

1. Double-click **`start-gpu-ollama.bat`** and leave the window open.
2. Start the app:
   ```powershell
   uv run ai-file-brain
   ```

✅ **It's working** when the Ollama window shows model layers loading onto
`SYCL0 (Intel Arc …)`. The first answer is slow (one-time GPU/SYCL warm-up),
then it's fast (~34 tok/s warm on an Arc 140V).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| App is slow / no GPU activity | A stock CPU Ollama is on port 11434. Close it, confirm Step 4, restart `start-gpu-ollama.bat`. |
| `mklink` error in `init-ollama.bat` | Windows Developer Mode is OFF — turn it on (Step 1). |
| `pip install` fails on `sentencepiece` | You're on Python 3.12. Rebuild the venv with 3.11. |
| No layers on GPU / wrong device | Wrong `ONEAPI_DEVICE_SELECTOR` index — try `:1`, etc. |
| First answer very slow | Normal — one-time model load + SYCL kernel JIT. Cached afterward. |
