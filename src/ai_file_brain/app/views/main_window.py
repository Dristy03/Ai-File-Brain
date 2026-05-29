from __future__ import annotations

import os
from collections.abc import Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ai_file_brain.app.models.chat_turn import ChatTurn
from ai_file_brain.app.view_models.main_window_vm import MainWindowViewModel
from ai_file_brain.app.view_models.status_bar_vm import StatusBarViewModel


PALETTE = {
    "bg": "#f7f9fc",
    "surface": "#ffffff",
    "surface_alt": "#edf2f7",
    "border": "#e2e8f0",
    "text": "#1a202c",
    "text_muted": "#4a5568",
    "text_subtle": "#718096",
    "accent": "#3182ce",
    "accent_hover": "#2c5282",
    "accent_text": "#ffffff",
    "danger": "#c53030",
    "question_bg": "#ebf4ff",
    "question_border": "#bee3f8",
    "answer_bg": "#ffffff",
    "answer_border": "#e2e8f0",
    "status_bg": "#edf2f7",
}

STYLESHEET = f"""
QWidget#MainWindowRoot {{
    background-color: {PALETTE['bg']};
}}
QWidget#TranscriptHost {{
    background-color: {PALETTE['bg']};
}}
QScrollArea {{
    background-color: {PALETTE['bg']};
    border: none;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #cbd5e0;
    border-radius: 5px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: #a0aec0;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QFrame#ChatTurnCard {{
    background-color: transparent;
}}
QLabel#QuestionLabel {{
    background-color: {PALETTE['question_bg']};
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['question_border']};
    border-radius: 12px;
    padding: 10px 14px;
    font-size: 13px;
    font-weight: 600;
}}
QLabel#AnswerLabel {{
    background-color: {PALETTE['answer_bg']};
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['answer_border']};
    border-radius: 12px;
    padding: 10px 14px;
    font-size: 13px;
}}
QLabel#StatusPill {{
    background-color: {PALETTE['surface_alt']};
    color: {PALETTE['text_subtle']};
    border: 1px dashed {PALETTE['border']};
    border-radius: 12px;
    padding: 8px 14px;
    font-size: 12px;
    font-style: italic;
}}
QLabel#SourcesLabel {{
    color: {PALETTE['text_subtle']};
    font-size: 11px;
    padding: 2px 14px;
}}
QLabel#ErrorLabel {{
    color: {PALETTE['danger']};
    font-size: 12px;
    padding: 4px 14px;
    font-weight: 500;
}}

QWidget#InputBar {{
    background-color: {PALETTE['surface']};
    border-top: 1px solid {PALETTE['border']};
}}
QPlainTextEdit#ChatInput {{
    background-color: {PALETTE['surface']};
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 13px;
    selection-background-color: {PALETTE['accent']};
}}
QPlainTextEdit#ChatInput:focus {{
    border: 1px solid {PALETTE['accent']};
}}

QPushButton#SendButton {{
    background-color: {PALETTE['accent']};
    color: {PALETTE['accent_text']};
    border: none;
    border-radius: 8px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 600;
    min-width: 64px;
}}
QPushButton#SendButton:hover {{
    background-color: {PALETTE['accent_hover']};
}}
QPushButton#SendButton:disabled {{
    background-color: #a0aec0;
}}
QPushButton#SendButton[mode="stop"] {{
    background-color: {PALETTE['danger']};
}}
QPushButton#SendButton[mode="stop"]:hover {{
    background-color: #9b2c2c;
}}

QWidget#StatusStrip {{
    background-color: {PALETTE['status_bg']};
    border-top: 1px solid {PALETTE['border']};
}}
QLabel#StatusLabel {{
    color: {PALETTE['text_muted']};
    font-size: 11px;
}}
QToolButton#ChangeFolderButton, QToolButton#CopyConversationButton {{
    background-color: {PALETTE['surface']};
    color: {PALETTE['accent']};
    border: 1px solid {PALETTE['border']};
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 500;
}}
QToolButton#ChangeFolderButton:hover, QToolButton#CopyConversationButton:hover {{
    background-color: {PALETTE['accent']};
    color: {PALETTE['accent_text']};
    border-color: {PALETTE['accent']};
}}
QToolButton#ChangeFolderButton:pressed, QToolButton#CopyConversationButton:pressed {{
    background-color: {PALETTE['accent_hover']};
}}
QToolButton#CopyConversationButton:disabled {{
    background-color: {PALETTE['surface_alt']};
    color: {PALETTE['text_subtle']};
    border-color: {PALETTE['border']};
}}
"""


class _ChatTurnWidget(QFrame):
    def __init__(self, turn: ChatTurn, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ChatTurnCard")
        self._turn = turn

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)

        # Every chat-turn label sets minWidth=0 + Ignored horizontal size policy
        # so a long unbreakable token (URL, path) doesn't push the transcript
        # host wider than the scroll viewport and clip the right edge.
        def _shrinkable(label: QLabel) -> None:
            label.setMinimumWidth(0)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        self._question_label = QLabel(turn.question)
        self._question_label.setObjectName("QuestionLabel")
        self._question_label.setWordWrap(True)
        self._question_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        _shrinkable(self._question_label)
        layout.addWidget(self._question_label)

        self._status_label = QLabel("")
        self._status_label.setObjectName("StatusPill")
        self._status_label.setWordWrap(True)
        self._status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._status_label.setVisible(False)
        _shrinkable(self._status_label)
        layout.addWidget(self._status_label)

        self._sources_label = QLabel("")
        self._sources_label.setObjectName("SourcesLabel")
        self._sources_label.setWordWrap(True)
        self._sources_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._sources_label.setVisible(False)
        _shrinkable(self._sources_label)
        layout.addWidget(self._sources_label)

        self._answer_label = QLabel("")
        self._answer_label.setObjectName("AnswerLabel")
        self._answer_label.setWordWrap(True)
        self._answer_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._answer_label.setVisible(False)
        _shrinkable(self._answer_label)
        layout.addWidget(self._answer_label)

        self._error_label = QLabel("")
        self._error_label.setObjectName("ErrorLabel")
        self._error_label.setWordWrap(True)
        self._error_label.setVisible(False)
        _shrinkable(self._error_label)
        layout.addWidget(self._error_label)

        turn.answer_changed.connect(self._refresh_answer)
        turn.sources_changed.connect(self._refresh_sources)
        turn.error_changed.connect(self._refresh_error)
        turn.status_changed.connect(self._refresh_status)
        # Render any state the turn already has (e.g. status that arrived
        # synchronously between construction and signal connection).
        self._refresh_status()
        self._refresh_sources()
        self._refresh_answer()
        self._refresh_error()

    def _refresh_answer(self) -> None:
        self._answer_label.setText(self._turn.answer)
        self._answer_label.setVisible(bool(self._turn.answer))

    def _refresh_sources(self) -> None:
        if not self._turn.sources:
            self._sources_label.setVisible(False)
            self._sources_label.setToolTip("")
            return
        # Show basenames in the label so long paths can't push the card wider
        # than the viewport. Full paths live in the tooltip on hover.
        basenames = [os.path.basename(s) or s for s in self._turn.sources]
        rendered = "  ·  ".join(basenames)
        self._sources_label.setText(f"Sources: {rendered}")
        self._sources_label.setToolTip("\n".join(self._turn.sources))
        self._sources_label.setVisible(True)

    def _refresh_error(self) -> None:
        msg = self._turn.error
        if not msg:
            self._error_label.setVisible(False)
            return
        self._error_label.setText(f"Error: {msg}")
        self._error_label.setVisible(True)

    def _refresh_status(self) -> None:
        msg = self._turn.status
        if not msg:
            self._status_label.setVisible(False)
            return
        self._status_label.setText(msg)
        self._status_label.setVisible(True)


class MainWindow(QWidget):
    def __init__(
        self,
        vm: MainWindowViewModel,
        status_vm: StatusBarViewModel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._vm = vm
        self._status_vm = status_vm
        self._is_quitting = False
        self._change_folder_handler: Callable[[], None] | None = None

        self.setObjectName("MainWindowRoot")
        self.setWindowTitle("AI File Brain")
        self.setMinimumSize(680, 560)
        self.setStyleSheet(STYLESHEET)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- transcript ----
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._transcript_host = QWidget()
        self._transcript_host.setObjectName("TranscriptHost")
        self._transcript_layout = QVBoxLayout(self._transcript_host)
        self._transcript_layout.setContentsMargins(8, 8, 8, 8)
        self._transcript_layout.setSpacing(4)
        self._empty_label = QLabel(
            "Drop .txt or .pdf files into your watch folder, then ask a question."
        )
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet(
            f"color: {PALETTE['text_subtle']}; font-size: 12px; padding: 32px;"
        )
        self._empty_label.setWordWrap(True)
        self._transcript_layout.addWidget(self._empty_label)
        self._transcript_layout.addStretch(1)
        self._scroll.setWidget(self._transcript_host)
        outer.addWidget(self._scroll, 1)

        # ---- input bar ----
        input_bar = QWidget()
        input_bar.setObjectName("InputBar")
        input_row = QHBoxLayout(input_bar)
        input_row.setContentsMargins(12, 10, 12, 10)
        input_row.setSpacing(8)

        self._input = _EnterToSendTextEdit(self._on_enter_pressed)
        self._input.setObjectName("ChatInput")
        self._input.setPlaceholderText("Ask a question about your files…  (Shift+Enter for newline)")
        self._input.setMaximumHeight(96)
        input_row.addWidget(self._input, 1)

        self._send_button = QPushButton("Send")
        self._send_button.setObjectName("SendButton")
        self._send_button.setProperty("mode", "send")
        self._send_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._send_button.clicked.connect(self._on_send_clicked)
        input_row.addWidget(self._send_button)

        outer.addWidget(input_bar)

        # ---- status strip ----
        status_strip = QWidget()
        status_strip.setObjectName("StatusStrip")
        status_layout = QHBoxLayout(status_strip)
        status_layout.setContentsMargins(12, 6, 12, 6)
        status_layout.setSpacing(10)

        self._status_label = QLabel(self._status_vm.render_html())
        self._status_label.setObjectName("StatusLabel")
        self._status_label.setTextFormat(Qt.TextFormat.RichText)
        self._status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        # Allow the label to shrink below its rich-text natural width so the
        # Copy / Change-folder buttons stay on-screen on narrow displays.
        self._status_label.setMinimumWidth(0)
        self._status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._status_label.setToolTip(self._status_vm.render_tooltip())
        status_layout.addWidget(self._status_label, 1)

        self._copy_button = QToolButton()
        self._copy_button.setObjectName("CopyConversationButton")
        self._copy_button.setText("Copy")
        self._copy_button.setToolTip("Copy the whole conversation to the clipboard")
        self._copy_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._copy_button.clicked.connect(self._on_copy_clicked)
        status_layout.addWidget(self._copy_button)

        self._change_folder_button = QToolButton()
        self._change_folder_button.setObjectName("ChangeFolderButton")
        self._change_folder_button.setText("Change folder…")
        self._change_folder_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._change_folder_button.clicked.connect(self._on_change_folder_clicked)
        status_layout.addWidget(self._change_folder_button)

        outer.addWidget(status_strip)

        # ---- bindings ----
        vm.turn_appended.connect(self._on_turn_appended)
        vm.is_sending_changed.connect(self._on_sending_changed)
        vm.input_text_changed.connect(self._on_input_text_changed)
        status_vm.changed.connect(self._refresh_status)
        self._input.textChanged.connect(self._sync_input_to_vm)

    # ---- public API ----

    def show_and_raise(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def toggle_visibility(self) -> None:
        if self.isVisible():
            self.hide()
        else:
            self.show_and_raise()

    def mark_quitting(self) -> None:
        self._is_quitting = True

    def set_change_folder_handler(self, handler: Callable[[], None]) -> None:
        self._change_folder_handler = handler

    # ---- event handlers ----

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._is_quitting:
            event.accept()
            return
        event.ignore()
        self.hide()

    def _on_enter_pressed(self) -> None:
        self._on_send_clicked()

    def _on_send_clicked(self) -> None:
        if self._vm.is_sending:
            self._vm.stop()
        else:
            self._vm.send()

    def _on_change_folder_clicked(self) -> None:
        if self._change_folder_handler is not None:
            self._change_folder_handler()

    def _on_copy_clicked(self) -> None:
        transcript = self._build_transcript()
        if not transcript:
            return
        QApplication.clipboard().setText(transcript)
        self._copy_button.setText("Copied ✓")
        self._copy_button.setEnabled(False)
        QTimer.singleShot(1200, self._restore_copy_button)

    def _restore_copy_button(self) -> None:
        self._copy_button.setText("Copy")
        self._copy_button.setEnabled(True)

    def _build_transcript(self) -> str:
        lines: list[str] = []
        for turn in self._vm.turns:
            lines.append(f"You: {turn.question}")
            if turn.answer:
                lines.append(f"AI: {turn.answer}")
            if turn.sources:
                lines.append("Sources: " + " · ".join(turn.sources))
            if turn.error:
                lines.append(f"Error: {turn.error}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def _on_turn_appended(self, turn: ChatTurn) -> None:
        if self._empty_label is not None:
            self._empty_label.setVisible(False)
        widget = _ChatTurnWidget(turn)
        # insert before the trailing stretch
        self._transcript_layout.insertWidget(self._transcript_layout.count() - 1, widget)
        self._scroll.verticalScrollBar().rangeChanged.connect(self._scroll_to_bottom)

    def _scroll_to_bottom(self) -> None:
        bar = self._scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _on_sending_changed(self, sending: bool) -> None:
        self._send_button.setText("Stop" if sending else "Send")
        self._send_button.setProperty("mode", "stop" if sending else "send")
        # Re-polish so the property selector takes effect.
        self._send_button.style().unpolish(self._send_button)
        self._send_button.style().polish(self._send_button)

    def _on_input_text_changed(self, text: str) -> None:
        if self._input.toPlainText() != text:
            self._input.blockSignals(True)
            self._input.setPlainText(text)
            self._input.blockSignals(False)

    def _sync_input_to_vm(self) -> None:
        self._vm.input_text = self._input.toPlainText()

    def _refresh_status(self) -> None:
        self._status_label.setText(self._status_vm.render_html())
        self._status_label.setToolTip(self._status_vm.render_tooltip())


class _EnterToSendTextEdit(QPlainTextEdit):
    def __init__(self, on_submit) -> None:
        super().__init__()
        self._on_submit = on_submit

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event)
                return
            self._on_submit()
            return
        super().keyPressEvent(event)
