from __future__ import annotations

import pytest

from ai_file_brain.app.services.indexing_activity_service import (
    IndexingActivityService,
    format_activity_label,
)
from ai_file_brain.app.view_models.status_bar_vm import StatusBarViewModel
from ai_file_brain.core.watcher import IndexingProgress


def test_format_indexing_label_uses_basename():
    p = IndexingProgress("C:/dir/report.pdf", "indexing")
    assert format_activity_label(p) == "Indexing report.pdf…"


def test_format_indexed_includes_chunk_detail():
    p = IndexingProgress("/notes/a.txt", "indexed", "5 chunks")
    assert format_activity_label(p) == "Indexed a.txt (5 chunks)"


def test_format_indexed_without_detail():
    p = IndexingProgress("/notes/a.txt", "indexed", "")
    assert format_activity_label(p) == "Indexed a.txt"


def test_format_deleted_label():
    p = IndexingProgress("/notes/old.docx", "deleted")
    assert format_activity_label(p) == "Removed old.docx"


def test_format_error_label():
    p = IndexingProgress("/notes/broken.pdf", "error", "boom")
    assert format_activity_label(p) == "Error indexing broken.pdf"


def test_format_handles_missing_path():
    p = IndexingProgress("", "indexing")
    assert format_activity_label(p) == "Indexing ?…"


@pytest.fixture
def status_vm(qapp):
    return StatusBarViewModel()


def test_first_event_publishes_immediately(status_vm):
    svc = IndexingActivityService(status_vm)
    svc.on_progress(IndexingProgress("/x/foo.pdf", "indexing"))
    assert status_vm.current_activity == "Indexing foo.pdf…"


def test_subsequent_event_during_cooldown_is_buffered(status_vm):
    svc = IndexingActivityService(status_vm)
    svc.on_progress(IndexingProgress("/x/a.pdf", "indexing"))
    svc.on_progress(IndexingProgress("/x/b.pdf", "indexing"))
    # Tooltip still shows the first event because the throttle window has not fired.
    assert status_vm.current_activity == "Indexing a.pdf…"


def test_idle_clears_activity(qtbot, status_vm, monkeypatch):
    monkeypatch.setattr(IndexingActivityService, "IDLE_AFTER_MS", 50)
    svc = IndexingActivityService(status_vm)
    svc.on_progress(IndexingProgress("/x/a.pdf", "indexed", "1 chunks"))
    assert status_vm.current_activity == "Indexed a.pdf (1 chunks)"
    qtbot.waitUntil(lambda: status_vm.current_activity == "", timeout=1500)


def test_buffered_event_emitted_after_cooldown(qtbot, status_vm, monkeypatch):
    monkeypatch.setattr(IndexingActivityService, "UPDATE_INTERVAL_MS", 50)
    monkeypatch.setattr(IndexingActivityService, "IDLE_AFTER_MS", 5000)
    svc = IndexingActivityService(status_vm)
    svc.on_progress(IndexingProgress("/x/first.pdf", "indexing"))
    svc.on_progress(IndexingProgress("/x/second.pdf", "indexing"))
    assert status_vm.current_activity == "Indexing first.pdf…"
    qtbot.waitUntil(
        lambda: status_vm.current_activity == "Indexing second.pdf…", timeout=1500
    )
