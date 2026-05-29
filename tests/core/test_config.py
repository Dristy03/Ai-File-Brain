from pathlib import Path

import tomllib

from ai_file_brain.config import (
    DEFAULTS_TOML,
    USER_OVERRIDES_TOML,
    AiFileBrainSettings,
    save_user_overrides,
)


def test_defaults_used_when_no_files_present():
    s = AiFileBrainSettings()
    assert s.embedding_model == "nomic-embed-text"
    assert s.chat_model == "llama3.2"
    assert s.chunk_size == 2000


def test_settings_toml_overrides_defaults(tmp_path: Path, monkeypatch):
    # conftest already chdir'd into tmp_path
    Path(DEFAULTS_TOML).write_text(
        'watch_folder = "C:/from-defaults"\nchunk_size = 999\n', encoding="utf-8"
    )
    s = AiFileBrainSettings()
    assert s.watch_folder == "C:/from-defaults"
    assert s.chunk_size == 999


def test_user_overrides_beat_defaults_file():
    Path(DEFAULTS_TOML).write_text('watch_folder = "C:/defaults"\n', encoding="utf-8")
    Path(USER_OVERRIDES_TOML).write_text('watch_folder = "C:/user"\n', encoding="utf-8")
    s = AiFileBrainSettings()
    assert s.watch_folder == "C:/user"


def test_env_beats_user_overrides(monkeypatch):
    Path(USER_OVERRIDES_TOML).write_text('watch_folder = "C:/user"\n', encoding="utf-8")
    monkeypatch.setenv("AFB_WATCH_FOLDER", "C:/env")
    s = AiFileBrainSettings()
    assert s.watch_folder == "C:/env"


def test_save_user_overrides_creates_file_with_updates():
    save_user_overrides({"watch_folder": "C:/notes", "chunk_size": 1234})
    written = Path(USER_OVERRIDES_TOML)
    assert written.exists()
    parsed = tomllib.loads(written.read_text(encoding="utf-8"))
    assert parsed["watch_folder"] == "C:/notes"
    assert parsed["chunk_size"] == 1234


def test_save_user_overrides_merges_existing():
    Path(USER_OVERRIDES_TOML).write_text('chunk_size = 1\nchat_model = "llama3.2"\n', encoding="utf-8")
    save_user_overrides({"chunk_size": 2})
    parsed = tomllib.loads(Path(USER_OVERRIDES_TOML).read_text(encoding="utf-8"))
    assert parsed["chunk_size"] == 2
    assert parsed["chat_model"] == "llama3.2"


def test_save_user_overrides_handles_backslash_paths():
    save_user_overrides({"watch_folder": r"C:\Users\me\Documents"})
    parsed = tomllib.loads(Path(USER_OVERRIDES_TOML).read_text(encoding="utf-8"))
    assert parsed["watch_folder"] == r"C:\Users\me\Documents"
