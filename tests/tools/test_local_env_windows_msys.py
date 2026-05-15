"""Tests for the Windows / Git Bash MSYS-path normalization in
``LocalEnvironment``.

Background
----------
On Windows, ``pwd -P`` inside Git Bash emits paths like
``/c/Users/NVIDIA``. ``subprocess.Popen(..., cwd=...)`` only accepts
native Windows paths (``C:\\Users\\NVIDIA``), and the validation done
by ``_resolve_safe_cwd`` was also checking the MSYS form against
``os.path.isdir``, which returns ``False`` on Windows. The combined
effect was a warning logged on every single terminal call:

    LocalEnvironment cwd '/c/Users/NVIDIA' is missing on disk;
    falling back to '/' so terminal commands keep working.

These tests fake the Windows env on Linux CI by patching ``_IS_WINDOWS``
and ``os.path.isdir`` so the MSYS path tests as "missing" exactly like
on the real OS.
"""

import os
from unittest.mock import patch

import pytest

from tools.environments import local as local_mod
from tools.environments.local import (
    LocalEnvironment,
    _msys_to_windows_path,
    _resolve_safe_cwd,
)


# ---------------------------------------------------------------------------
# _msys_to_windows_path — pure-function unit tests
# ---------------------------------------------------------------------------

class TestMsysToWindowsPath:
    def test_noop_on_non_windows(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", False)
        # On a non-Windows host the function must never rewrite the path
        # — POSIX-style paths are real paths there.
        assert _msys_to_windows_path("/c/Users/NVIDIA") == "/c/Users/NVIDIA"
        assert _msys_to_windows_path("/home/teknium") == "/home/teknium"

    def test_translates_drive_path(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        assert _msys_to_windows_path("/c/Users/NVIDIA") == r"C:\Users\NVIDIA"
        assert _msys_to_windows_path("/d/Projects/foo bar") == r"D:\Projects\foo bar"

    def test_translates_bare_drive_root(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        # Bare "/c" alone should resolve to the drive root.
        assert _msys_to_windows_path("/c") == "C:\\"
        # Trailing slash on the drive letter is also a root.
        assert _msys_to_windows_path("/c/") == "C:\\"

    def test_idempotent_on_already_windows_path(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        assert _msys_to_windows_path(r"C:\Users\NVIDIA") == r"C:\Users\NVIDIA"

    def test_does_not_translate_multi_char_first_segment(self, monkeypatch):
        """``/tmp/foo`` and ``/home/x`` must NOT be misread as drive paths
        just because they start with ``/`` and a single letter — the regex
        only matches when the first segment is exactly one character."""
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        assert _msys_to_windows_path("/tmp/foo") == "/tmp/foo"
        assert _msys_to_windows_path("/home/x") == "/home/x"

    def test_empty_string(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        assert _msys_to_windows_path("") == ""


# ---------------------------------------------------------------------------
# _resolve_safe_cwd — Windows fast path
# ---------------------------------------------------------------------------

class TestResolveSafeCwdWindows:
    def test_msys_path_resolves_to_native_when_native_exists(
        self, monkeypatch, tmp_path,
    ):
        """The whole point of this fix: a Git Bash ``/c/Users/x`` value
        should resolve to its native equivalent if that native dir exists,
        WITHOUT falling back to the temp dir."""
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)

        # tmp_path is a real native dir on the test host. Build a fake
        # MSYS form pointing at it and prove the resolver finds it.
        native = str(tmp_path)
        # Construct a synthetic MSYS form for whatever tmp_path is.
        # On Linux CI tmp_path is /tmp/... ; the resolver shouldn't even
        # try to translate that (regex won't match), so emulate the
        # mapping by pointing the translator at the real native dir.
        with patch.object(
            local_mod, "_msys_to_windows_path", return_value=native
        ):
            assert _resolve_safe_cwd("/c/whatever") == native


# ---------------------------------------------------------------------------
# End-to-end: _update_cwd via marker file (Windows simulation)
# ---------------------------------------------------------------------------

class TestUpdateCwdWindowsMsys:
    def test_marker_file_msys_path_stored_in_native_form(
        self, monkeypatch, tmp_path,
    ):
        """When Git Bash writes ``/c/Users/x`` to the cwd marker file on
        Windows, ``_update_cwd`` must translate to native form before
        validating and storing — otherwise ``os.path.isdir`` rejects a
        perfectly real directory."""
        original = tmp_path / "starting"
        original.mkdir()

        # Fake Windows for the test
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)

        with patch.object(
            LocalEnvironment, "init_session", autospec=True, return_value=None
        ):
            env = LocalEnvironment(cwd=str(original), timeout=10)

        # Pretend Git Bash wrote an MSYS path that maps to tmp_path/"next"
        new_dir = tmp_path / "next"
        new_dir.mkdir()

        with open(env._cwd_file, "w") as f:
            f.write("/c/whatever/from/bash")

        # Translate the synthetic MSYS string to the real native dir.
        def fake_translate(p):
            if p == "/c/whatever/from/bash":
                return str(new_dir)
            return p

        with patch.object(local_mod, "_msys_to_windows_path", side_effect=fake_translate):
            env._update_cwd({"output": "", "returncode": 0})

        assert env.cwd == str(new_dir)


# ---------------------------------------------------------------------------
# End-to-end: _extract_cwd_from_output rollback when marker is invalid
# ---------------------------------------------------------------------------

class TestExtractCwdFromOutputWindowsMsys:
    def test_stale_msys_marker_does_not_clobber_cwd(self, monkeypatch, tmp_path):
        """When the cwd marker in stdout points at a non-existent path,
        ``LocalEnvironment._extract_cwd_from_output`` must roll back to
        the previous cwd instead of propagating a bad value."""
        original = tmp_path / "starting"
        original.mkdir()

        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)

        with patch.object(
            LocalEnvironment, "init_session", autospec=True, return_value=None
        ):
            env = LocalEnvironment(cwd=str(original), timeout=10)

        marker = env._cwd_marker
        result = {
            "output": f"some command output\n{marker}/c/no/such/path{marker}\n",
            "returncode": 0,
        }

        # Translation produces a path that doesn't exist on disk → rollback.
        with patch.object(
            local_mod,
            "_msys_to_windows_path",
            return_value=str(tmp_path / "definitely-does-not-exist"),
        ):
            env._extract_cwd_from_output(result)

        assert env.cwd == str(original)

    def test_valid_msys_marker_normalized_to_native(self, monkeypatch, tmp_path):
        original = tmp_path / "starting"
        original.mkdir()
        new_dir = tmp_path / "next"
        new_dir.mkdir()

        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)

        with patch.object(
            LocalEnvironment, "init_session", autospec=True, return_value=None
        ):
            env = LocalEnvironment(cwd=str(original), timeout=10)

        marker = env._cwd_marker
        result = {
            "output": f"x\n{marker}/c/whatever{marker}\n",
            "returncode": 0,
        }

        with patch.object(local_mod, "_msys_to_windows_path", return_value=str(new_dir)):
            env._extract_cwd_from_output(result)

        assert env.cwd == str(new_dir)
