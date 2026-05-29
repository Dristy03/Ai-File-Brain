from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure src/ is on sys.path even before editable install
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    """Run each test in a temp working directory so settings.toml lookups stay clean."""
    monkeypatch.chdir(tmp_path)
    # Make sure no AFB_* env leaks across processes
    for key in list(os.environ):
        if key.startswith("AFB_"):
            monkeypatch.delenv(key, raising=False)
    yield tmp_path
