from __future__ import annotations

from ai_file_brain.core.exclusions import is_excluded

DIRS = ["AppData", "node_modules", ".git", "__pycache__"]
EXTS = [".lock", ".pyc"]


def test_path_with_excluded_dir_component_is_excluded():
    assert is_excluded(r"C:\Users\me\AppData\Local\foo.txt", DIRS, EXTS)
    assert is_excluded(r"/home/me/proj/node_modules/lib/index.js", DIRS, EXTS)
    assert is_excluded(r"/home/me/proj/.git/config", DIRS, EXTS)


def test_path_outside_excluded_dirs_is_allowed():
    assert not is_excluded(r"C:\Users\me\Documents\notes.txt", DIRS, EXTS)
    assert not is_excluded(r"/home/me/proj/src/main.py", DIRS, EXTS)


def test_dir_name_match_is_case_insensitive():
    assert is_excluded(r"C:\Users\me\appdata\Local\x.txt", DIRS, EXTS)
    assert is_excluded(r"/home/me/Node_Modules/x.js", DIRS, EXTS)


def test_extension_match_is_case_insensitive():
    assert is_excluded(r"/home/me/proj/uv.lock", DIRS, EXTS)
    assert is_excluded(r"/home/me/proj/x.PYC", DIRS, EXTS)


def test_filename_alone_matching_excluded_dir_does_not_count():
    # "AppData" as a filename component is not an excluded dir.
    assert not is_excluded(r"/home/me/AppData", DIRS, EXTS)
    # ...but if it sits inside another excluded dir, the *containing* dir wins.
    assert is_excluded(r"/home/me/.git/AppData", DIRS, EXTS)


def test_no_extension_no_excluded_dir_means_allowed():
    assert not is_excluded(r"/home/me/Makefile", DIRS, EXTS)


def test_empty_lists_means_nothing_excluded():
    assert not is_excluded(r"C:\Users\me\AppData\foo.lock", [], [])
