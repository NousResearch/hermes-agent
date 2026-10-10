"""Tests: local media paths must resolve inside ssh/WSL sandboxes on a Windows host.

Regression: with Hermes running on Windows and the terminal backend on WSL over ssh,
_resolve_container_fallback passed the raw Windows path (C:\\Users\\... or a
backslash-mangled POSIX path) to the sandbox's `head -c` command. The sandbox is a
Linux shell, so the file was never found and base64.b64decode failed with
"Only base64 data is allowed" — breaking vision_analyze for every local image.
"""

import pytest

from tools.image_source import _sandbox_exec_path


def test_windows_drive_path_maps_to_drvfs():
    assert (
        _sandbox_exec_path(r"C:\Users\steve\Pictures\cat.png")
        == "/mnt/c/Users/steve/Pictures/cat.png"
    )


def test_backslash_mangled_posix_path_gets_forward_slashes():
    assert (
        _sandbox_exec_path(r"\mnt\c\Users\steve\Pictures\cat.png")
        == "/mnt/c/Users/steve/Pictures/cat.png"
    )


def test_posix_path_untouched():
    assert _sandbox_exec_path("/home/steve/cat.png") == "/home/steve/cat.png"


def test_relative_path_untouched():
    assert _sandbox_exec_path("cat.png") == "cat.png"


def test_lowercase_drive_is_normalized():
    assert (
        _sandbox_exec_path(r"c:\Users\steve\cat.png")
        == "/mnt/c/Users/steve/cat.png"
    )
