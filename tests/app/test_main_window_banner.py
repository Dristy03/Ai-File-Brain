from collections.abc import AsyncIterator

from ai_file_brain.app.view_models.main_window_vm import MainWindowViewModel
from ai_file_brain.app.view_models.status_bar_vm import StatusBarViewModel
from ai_file_brain.app.views.main_window import MainWindow
from ai_file_brain.core.models import ChatStreamChunk


class _NoChat:
    async def ask_stream(self, question: str) -> AsyncIterator[ChatStreamChunk]:
        return
        yield  # pragma: no cover — makes this an async generator


def _window(qtbot):
    status_vm = StatusBarViewModel()
    window = MainWindow(MainWindowViewModel(_NoChat()), status_vm)
    qtbot.addWidget(window)
    return window, status_vm


def test_banner_hidden_before_first_probe(qtbot):
    window, _status_vm = _window(qtbot)
    # ollama_checked is False until the first probe — no banner even though
    # ollama_healthy defaults to False.
    assert window._banner.isHidden()


def test_banner_shows_when_ollama_down(qtbot):
    window, status_vm = _window(qtbot)
    status_vm.ollama_checked = True
    status_vm.ollama_healthy = False
    assert not window._banner.isHidden()
    assert "Ollama" in window._banner.text()


def test_banner_hides_when_ollama_recovers(qtbot):
    window, status_vm = _window(qtbot)
    status_vm.ollama_checked = True
    status_vm.ollama_healthy = False
    assert not window._banner.isHidden()

    status_vm.ollama_healthy = True
    assert window._banner.isHidden()


def test_banner_shows_missing_model_with_pull_command(qtbot):
    window, status_vm = _window(qtbot)
    status_vm.ollama_checked = True
    status_vm.ollama_healthy = True
    status_vm.missing_models = ("llama3.2",)
    assert not window._banner.isHidden()
    text = window._banner.text()
    assert "llama3.2" in text
    assert "ollama pull llama3.2" in text


def test_banner_clears_when_model_installed(qtbot):
    window, status_vm = _window(qtbot)
    status_vm.ollama_checked = True
    status_vm.ollama_healthy = True
    status_vm.missing_models = ("llama3.2",)
    assert not window._banner.isHidden()

    status_vm.missing_models = ()
    assert window._banner.isHidden()
