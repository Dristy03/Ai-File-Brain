from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QObject
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from ai_file_brain.app.view_models.status_bar_vm import StatusBarViewModel


class TrayIconService(QObject):
    def __init__(
        self,
        icon: QIcon,
        toggle_window: Callable[[], None],
        show_window: Callable[[], None],
        change_folder: Callable[[], None],
        quit_app: Callable[[], None],
        status: StatusBarViewModel,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._toggle = toggle_window
        self._quit = quit_app
        self._status = status
        self.is_quitting = False

        self._tray = QSystemTrayIcon(icon, self)
        self._tray.setToolTip("AI File Brain")

        menu = QMenu()
        show_act = QAction("Show chat", menu)
        show_act.triggered.connect(show_window)
        menu.addAction(show_act)

        change_act = QAction("Change watch folder…", menu)
        change_act.triggered.connect(change_folder)
        menu.addAction(change_act)

        pause_act = QAction("Pause indexing", menu)
        pause_act.setEnabled(False)
        menu.addAction(pause_act)

        menu.addSeparator()
        quit_act = QAction("Quit", menu)
        quit_act.triggered.connect(self._on_quit)
        menu.addAction(quit_act)

        self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._on_activated)

        status.changed.connect(self._refresh_tooltip)
        self._refresh_tooltip()

    def attach(self) -> None:
        self._tray.show()

    def detach(self) -> None:
        self._tray.hide()

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._toggle()

    def _on_quit(self) -> None:
        self.is_quitting = True
        self._quit()

    def _refresh_tooltip(self) -> None:
        base = f"AI File Brain — {self._status.chunk_count} chunks indexed"
        activity = self._status.current_activity
        self._tray.setToolTip(f"{base}\n{activity}" if activity else base)
