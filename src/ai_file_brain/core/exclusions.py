from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import PurePath


def is_excluded(
    file_path: str,
    excluded_dir_names: Iterable[str],
    excluded_extensions: Iterable[str],
) -> bool:
    """Return True if ``file_path`` should be skipped for indexing.

    Two cheap checks:

    * **Directory-name match.** Any path component (case-insensitive) appearing
      in ``excluded_dir_names`` excludes the file. This catches things like
      ``AppData``, ``node_modules``, ``__pycache__`` regardless of where they
      sit in the tree.
    * **Extension match.** The file's extension (case-insensitive, leading dot
      included) appearing in ``excluded_extensions`` excludes the file.

    Designed to be O(parts) and allocation-light because it runs on every
    create/modify/move event.
    """
    ext = os.path.splitext(file_path)[1].lower()
    excluded_exts = {e.lower() for e in excluded_extensions}
    if ext and ext in excluded_exts:
        return True

    excluded_dirs = {d.lower() for d in excluded_dir_names}
    if not excluded_dirs:
        return False
    parts = PurePath(file_path).parts
    # Skip the last part (the file name itself) — only directory components matter.
    return any(part.lower() in excluded_dirs for part in parts[:-1])
