from __future__ import annotations

import os

from PySide6.QtCore import QObject, QTimer

from ai_file_brain.app.view_models.status_bar_vm import StatusBarViewModel
from ai_file_brain.core.watcher import IndexingProgress


def format_activity_label(progress: IndexingProgress) -> str:
    """Render an IndexingProgress event as a short, human-friendly label.

    Returns ``""`` if the event should not produce a visible activity line
    (currently never — every state has a label — but kept for future use).
    """
    name = os.path.basename(progress.file_path) if progress.file_path else "?"
    state = progress.state
    if state == "indexing":
        return f"Indexing {name}…"
    if state == "indexed":
        detail = (progress.detail or "").strip()
        return f"Indexed {name}" + (f" ({detail})" if detail else "")
    if state == "deleted":
        return f"Removed {name}"
    if state == "error":
        return f"Error indexing {name}"
    return f"{state} {name}".strip()


class IndexingActivityService(QObject):
    """Throttled relay from IndexingProgress events to the status view-model.

    Two timers shape the UX:

    * **Update throttle** (``UPDATE_INTERVAL_MS``): the first event in a burst
      is published immediately; subsequent events during the cooldown window
      are coalesced and published when the timer fires.
    * **Idle timeout** (``IDLE_AFTER_MS``): when no progress events have
      arrived for this many ms, the activity is cleared so the tray tooltip
      and status bar fall back to a quiet baseline.
    """

    UPDATE_INTERVAL_MS = 1000
    IDLE_AFTER_MS = 2000

    def __init__(
        self,
        status: StatusBarViewModel,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._status = status
        self._latest: IndexingProgress | None = None
        self._last_emitted_id: int = 0

        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._on_update_timer)

        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.timeout.connect(self._on_idle_timer)

    def on_progress(self, progress: IndexingProgress) -> None:
        self._latest = progress
        self._idle_timer.start(self.IDLE_AFTER_MS)
        if not self._update_timer.isActive():
            self._publish_latest()
            self._update_timer.start(self.UPDATE_INTERVAL_MS)

    def _on_update_timer(self) -> None:
        if self._latest is None:
            return
        if id(self._latest) == self._last_emitted_id:
            return  # nothing new during the cooldown window
        self._publish_latest()
        self._update_timer.start(self.UPDATE_INTERVAL_MS)

    def _on_idle_timer(self) -> None:
        self._status.current_activity = ""
        self._latest = None
        self._last_emitted_id = 0

    def _publish_latest(self) -> None:
        if self._latest is None:
            return
        self._status.current_activity = format_activity_label(self._latest)
        self._last_emitted_id = id(self._latest)
