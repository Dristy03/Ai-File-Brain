from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import qasync
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox, QStyle

from ai_file_brain.app.di import build_container
from ai_file_brain.config import AiFileBrainSettings


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def _load_icon(qapp: QApplication) -> QIcon:
    asset = Path(__file__).parent / "assets" / "tray_icon.ico"
    if asset.exists():
        return QIcon(str(asset))
    # fallback to a built-in style icon so tray + window have something
    return qapp.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)


def main() -> int:
    _configure_logging()

    qapp = QApplication.instance() or QApplication(sys.argv)
    qapp.setApplicationName("AI File Brain")
    qapp.setQuitOnLastWindowClosed(False)

    icon = _load_icon(qapp)
    qapp.setWindowIcon(icon)

    try:
        settings = AiFileBrainSettings()
    except Exception as ex:
        QMessageBox.critical(None, "AI File Brain", f"Failed to load settings: {ex}")
        return 1

    loop = qasync.QEventLoop(qapp)
    asyncio.set_event_loop(loop)

    container = build_container(settings, qapp, icon)

    def _on_unhandled(loop_, context):
        msg = context.get("message", "Unhandled error")
        exc = context.get("exception")
        logging.getLogger("ai_file_brain").error("Unhandled task error: %s (%s)", msg, exc)

    loop.set_exception_handler(_on_unhandled)

    with loop:
        loop.create_task(container.startup())
        loop.run_forever()

    return 0


if __name__ == "__main__":
    sys.exit(main())
