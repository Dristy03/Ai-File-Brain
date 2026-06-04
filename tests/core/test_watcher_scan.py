from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

from ai_file_brain.config import AiFileBrainSettings
from ai_file_brain.core.watcher import FileWatcherService


def _service() -> FileWatcherService:
    # _needs_index only touches settings + the filesystem, so pipeline/repo can
    # be omitted for this unit.
    return FileWatcherService(AiFileBrainSettings(), pipeline=None, repo=None)


def _disk_mtime(path: str) -> datetime:
    return datetime.fromtimestamp(os.path.getmtime(path), tz=UTC)


def test_new_file_is_indexed(tmp_path):
    f = tmp_path / "new.txt"
    f.write_text("hi")
    svc = _service()
    # No prior index entry -> must index.
    assert svc._needs_index(str(f), None) is True


def test_unchanged_file_is_skipped(tmp_path):
    f = tmp_path / "same.txt"
    f.write_text("hi")
    svc = _service()
    # Indexed at exactly the on-disk mtime -> unchanged -> skip.
    assert svc._needs_index(str(f), _disk_mtime(str(f))) is False


def test_file_edited_while_closed_is_reindexed(tmp_path):
    f = tmp_path / "edited.txt"
    f.write_text("hi")
    svc = _service()
    # We last indexed it well before its current on-disk mtime -> re-index.
    stale = _disk_mtime(str(f)) - timedelta(minutes=5)
    assert svc._needs_index(str(f), stale) is True


def test_missing_file_is_let_through(tmp_path):
    svc = _service()
    # Can't stat it -> let index_file deal with it rather than silently dropping.
    assert svc._needs_index(str(tmp_path / "gone.txt"), datetime.now(UTC)) is True
