from pathlib import Path

import pytest

from ai_file_brain.app.services.watch_folder_service import WatchFolderService
from ai_file_brain.app.view_models.status_bar_vm import StatusBarViewModel
from ai_file_brain.config import USER_OVERRIDES_TOML, AiFileBrainSettings


class FakeWatcher:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def start(self) -> None:
        self.calls.append("start")

    async def stop(self) -> None:
        self.calls.append("stop")


@pytest.mark.asyncio
async def test_change_to_hot_swaps_and_persists(tmp_path: Path, qtbot):
    new_folder = tmp_path / "notes"
    new_folder.mkdir()

    settings = AiFileBrainSettings(watch_folder=str(tmp_path / "old"))
    (tmp_path / "old").mkdir()
    watcher = FakeWatcher()
    status = StatusBarViewModel()
    svc = WatchFolderService(settings, watcher, status)

    result = await svc.change_to(str(new_folder))

    assert result == str(new_folder.resolve())
    assert settings.watch_folder == str(new_folder.resolve())
    assert status.watch_folder == str(new_folder.resolve())
    assert watcher.calls == ["stop", "start"]
    assert Path(USER_OVERRIDES_TOML).exists()


@pytest.mark.asyncio
async def test_change_to_same_path_is_noop(tmp_path: Path, qtbot):
    folder = tmp_path / "same"
    folder.mkdir()

    settings = AiFileBrainSettings(watch_folder=str(folder))
    watcher = FakeWatcher()
    svc = WatchFolderService(settings, watcher, StatusBarViewModel())

    await svc.change_to(str(folder))
    assert watcher.calls == []
    assert not Path(USER_OVERRIDES_TOML).exists()


@pytest.mark.asyncio
async def test_change_to_rejects_missing_path(tmp_path: Path, qtbot):
    settings = AiFileBrainSettings(watch_folder=str(tmp_path))
    watcher = FakeWatcher()
    svc = WatchFolderService(settings, watcher, StatusBarViewModel())

    with pytest.raises(FileNotFoundError):
        await svc.change_to(str(tmp_path / "does-not-exist"))
    assert watcher.calls == []


@pytest.mark.asyncio
async def test_change_to_rejects_file_path(tmp_path: Path, qtbot):
    f = tmp_path / "thing.txt"
    f.write_text("x")

    settings = AiFileBrainSettings(watch_folder=str(tmp_path))
    watcher = FakeWatcher()
    svc = WatchFolderService(settings, watcher, StatusBarViewModel())

    with pytest.raises(NotADirectoryError):
        await svc.change_to(str(f))
    assert watcher.calls == []
