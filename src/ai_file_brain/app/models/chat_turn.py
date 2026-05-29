from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class ChatTurn(QObject):
    answer_changed = Signal()
    sources_changed = Signal()
    error_changed = Signal()
    status_changed = Signal()

    def __init__(self, *, question: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._question = question
        self._answer = ""
        self._sources: tuple[str, ...] = ()
        self._error: str | None = None
        self._status: str = ""

    @property
    def question(self) -> str:
        return self._question

    @property
    def answer(self) -> str:
        return self._answer

    @property
    def sources(self) -> tuple[str, ...]:
        return self._sources

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def status(self) -> str:
        return self._status

    def append_answer(self, text: str) -> None:
        if not text:
            return
        self._answer += text
        self.answer_changed.emit()
        if self._status:
            self._status = ""
            self.status_changed.emit()

    def set_sources(self, paths: tuple[str, ...]) -> None:
        self._sources = paths
        self.sources_changed.emit()

    def set_error(self, message: str) -> None:
        self._error = message
        self.error_changed.emit()
        if self._status:
            self._status = ""
            self.status_changed.emit()

    def set_status(self, message: str) -> None:
        if message == self._status:
            return
        self._status = message
        self.status_changed.emit()
