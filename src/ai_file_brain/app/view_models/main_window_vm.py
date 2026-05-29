from __future__ import annotations

import asyncio
import logging

from PySide6.QtCore import QObject, Signal, Slot

from ai_file_brain.app.models.chat_turn import ChatTurn
from ai_file_brain.core.chat import ChatService
from ai_file_brain.core.models import SourcesChunk, StatusChunk, TokenChunk

logger = logging.getLogger(__name__)


class MainWindowViewModel(QObject):
    turn_appended = Signal(object)            # ChatTurn
    is_sending_changed = Signal(bool)
    input_text_changed = Signal(str)

    def __init__(self, chat: ChatService, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._chat = chat
        self._turns: list[ChatTurn] = []
        self._input_text = ""
        self._is_sending = False
        self._send_task: asyncio.Task | None = None

    @property
    def turns(self) -> list[ChatTurn]:
        return self._turns

    @property
    def input_text(self) -> str:
        return self._input_text

    @input_text.setter
    def input_text(self, v: str) -> None:
        if v != self._input_text:
            self._input_text = v
            self.input_text_changed.emit(v)

    @property
    def is_sending(self) -> bool:
        return self._is_sending

    def can_send(self) -> bool:
        return not self._is_sending and bool(self._input_text.strip())

    @Slot()
    def send(self) -> None:
        if not self.can_send():
            return
        question = self._input_text.strip()
        self.input_text = ""
        turn = ChatTurn(question=question)
        self._turns.append(turn)
        self.turn_appended.emit(turn)
        self._send_task = asyncio.ensure_future(self._stream(turn, question))

    @Slot()
    def stop(self) -> None:
        if self._send_task and not self._send_task.done():
            self._send_task.cancel()

    async def _stream(self, turn: ChatTurn, question: str) -> None:
        self._set_sending(True)
        try:
            async for chunk in self._chat.ask_stream(question):
                if isinstance(chunk, TokenChunk):
                    turn.append_answer(chunk.text)
                elif isinstance(chunk, SourcesChunk):
                    turn.set_sources(chunk.paths)
                elif isinstance(chunk, StatusChunk):
                    turn.set_status(chunk.message)
        except asyncio.CancelledError:
            turn.append_answer("\n\n[stopped]")
            raise
        except Exception as ex:
            logger.exception("Chat stream failed")
            turn.set_error(str(ex))
        finally:
            self._set_sending(False)

    def _set_sending(self, value: bool) -> None:
        if self._is_sending != value:
            self._is_sending = value
            self.is_sending_changed.emit(value)
