from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class StatusBarViewModel(QObject):
    changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._watch_folder = ""
        self._chunk_count = 0
        self._ollama_healthy = False
        self._chroma_healthy = False
        self._current_activity = ""

    @property
    def watch_folder(self) -> str:
        return self._watch_folder

    @watch_folder.setter
    def watch_folder(self, v: str) -> None:
        if v != self._watch_folder:
            self._watch_folder = v
            self.changed.emit()

    @property
    def chunk_count(self) -> int:
        return self._chunk_count

    @chunk_count.setter
    def chunk_count(self, v: int) -> None:
        if v != self._chunk_count:
            self._chunk_count = v
            self.changed.emit()

    @property
    def ollama_healthy(self) -> bool:
        return self._ollama_healthy

    @ollama_healthy.setter
    def ollama_healthy(self, v: bool) -> None:
        if v != self._ollama_healthy:
            self._ollama_healthy = v
            self.changed.emit()

    @property
    def chroma_healthy(self) -> bool:
        return self._chroma_healthy

    @chroma_healthy.setter
    def chroma_healthy(self, v: bool) -> None:
        if v != self._chroma_healthy:
            self._chroma_healthy = v
            self.changed.emit()

    @property
    def current_activity(self) -> str:
        return self._current_activity

    @current_activity.setter
    def current_activity(self, v: str) -> None:
        v = v or ""
        if v != self._current_activity:
            self._current_activity = v
            self.changed.emit()

    def render(self) -> str:
        ollama = "✓" if self._ollama_healthy else "✗"
        chroma = "✓" if self._chroma_healthy else "✗"
        folder = self._watch_folder or "(none)"
        return (
            f"Watching {folder} · {self._chunk_count} chunks · "
            f"Ollama {ollama} · Chroma {chroma}"
        )

    def render_html(self) -> str:
        folder = self._watch_folder or "(none)"
        folder_display = _elide_middle(folder, 50)
        return (
            f'<span style="color:#4a5568;">Watching </span>'
            f'<span style="color:#1a202c; font-weight:500;">{_escape(folder_display)}</span>'
            f'<span style="color:#a0aec0;"> &nbsp;·&nbsp; </span>'
            f'<span style="color:#1a202c; font-weight:500;">{self._chunk_count}</span>'
            f'<span style="color:#4a5568;"> chunks</span>'
            f'<span style="color:#a0aec0;"> &nbsp;·&nbsp; </span>'
            f'{_dot(self._ollama_healthy)} <span style="color:#4a5568;">Ollama</span>'
            f'<span style="color:#a0aec0;"> &nbsp;·&nbsp; </span>'
            f'{_dot(self._chroma_healthy)} <span style="color:#4a5568;">Chroma</span>'
        )

    def render_tooltip(self) -> str:
        folder = self._watch_folder or "(none)"
        return (
            f"Watching: {folder}\n"
            f"{self._chunk_count} chunks indexed\n"
            f"Ollama: {'connected' if self._ollama_healthy else 'unreachable'}\n"
            f"Chroma: {'connected' if self._chroma_healthy else 'unreachable'}"
        )


def _dot(healthy: bool) -> str:
    color = "#38a169" if healthy else "#e53e3e"
    return f'<span style="color:{color}; font-size:14px;">●</span>'


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _elide_middle(text: str, max_chars: int) -> str:
    """Trim a long string by replacing its middle with '…' so the head and tail
    are still recognisable. Used for long watch-folder paths so the status
    strip doesn't push the window wider than narrow office screens."""
    if len(text) <= max_chars or max_chars < 3:
        return text
    keep = (max_chars - 1) // 2
    return text[:keep] + "…" + text[-keep:]
