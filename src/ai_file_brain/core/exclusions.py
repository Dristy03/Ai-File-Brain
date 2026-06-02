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


# Indexing tiers, returned by ``classify_path``.
TIER_CONTENT = "content"  # extract + embed the file's text
TIER_NAME_ONLY = "name_only"  # embed just the filename as a stub


def classify_path(
    file_path: str,
    content_exts: frozenset[str],
    name_only_exts: frozenset[str],
    excluded_dir_names: Iterable[str],
    excluded_extensions: Iterable[str],
) -> str | None:
    """Decide how a path should be indexed, or ``None`` to ignore it.

    ``content_exts`` / ``name_only_exts`` must already be lowercase sets (the
    caller precomputes them once). Exclusion rules win over both tiers, so a
    file in an excluded dir or with an excluded extension is always ignored.
    """
    if is_excluded(file_path, excluded_dir_names, excluded_extensions):
        return None
    ext = os.path.splitext(file_path)[1].lower()
    if ext in content_exts:
        return TIER_CONTENT
    if ext in name_only_exts:
        return TIER_NAME_ONLY
    return None


def prune_excluded_dirs(dir_names: list[str], excluded_dir_names: Iterable[str]) -> list[str]:
    """Filter a directory listing down to the ones worth descending into.

    Used to prune an ``os.walk`` *in place* (``dirnames[:] = prune_excluded_dirs(...)``)
    so the walk never even enters excluded subtrees like ``node_modules`` or
    ``AppData`` — as opposed to entering them and discarding every file after the
    fact, which is what makes scanning a huge root (e.g. ``C:\\``) crawl.
    """
    excluded = {d.lower() for d in excluded_dir_names}
    if not excluded:
        return dir_names
    return [d for d in dir_names if d.lower() not in excluded]
