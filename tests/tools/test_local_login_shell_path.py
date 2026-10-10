"""Login-shell PATH capture for zsh/fish users (#129103).

The terminal snapshot is built by ``bash -l``, so a zsh login shell's
~/.zprofile (Homebrew, pyenv, conda on macOS) is never read. When $SHELL names
zsh/fish, the login PATH is captured once (login-only, bounded) and merged in
front of the inherited PATH before the snapshot is taken.
"""

import os
import stat
from unittest.mock import patch

import pytest

from tools.environments import local as local_mod
from tools.environments.local import (
    _apply_login_shell_path,
    _capture_user_login_shell_path,
    _extract_login_shell_path,
    _merge_login_shell_path,
    _reset_login_shell_path_cache,
    _user_login_shell_for_path_capture,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    _reset_login_shell_path_cache()
    yield
    _reset_login_shell_path_cache()


def _make_executable(path):
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class TestUserLoginShellSelection:
    def test_returns_zsh_when_executable(self, tmp_path, monkeypatch):
        fake = tmp_path / "zsh"
        fake.write_text("#!/bin/sh\n")
        _make_executable(fake)
        monkeypatch.setenv("SHELL", str(fake))
        assert _user_login_shell_for_path_capture() == str(fake)

    def test_returns_fish_when_executable(self, tmp_path, monkeypatch):
        fake = tmp_path / "fish"
        fake.write_text("#!/bin/sh\n")
        _make_executable(fake)
        monkeypatch.setenv("SHELL", str(fake))
        assert _user_login_shell_for_path_capture() == str(fake)

    def test_ignores_bash_family(self, tmp_path, monkeypatch):
        for name in ("bash", "sh", "dash", "ksh"):
            fake = tmp_path / name
            fake.write_text("#!/bin/sh\n")
            _make_executable(fake)
            monkeypatch.setenv("SHELL", str(fake))
            assert _user_login_shell_for_path_capture() is None, name

    def test_ignores_missing_or_nonexecutable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SHELL", str(tmp_path / "no-such-zsh"))
        assert _user_login_shell_for_path_capture() is None
        fake = tmp_path / "zsh"
        fake.write_text("#!/bin/sh\n")
        fake.chmod(0o644)
        monkeypatch.setenv("SHELL", str(fake))
        assert _user_login_shell_for_path_capture() is None

    def test_ignores_empty_shell(self, monkeypatch):
        monkeypatch.setenv("SHELL", "")
        assert _user_login_shell_for_path_capture() is None


class TestExtractLoginShellPath:
    def test_extracts_between_sentinels(self):
        out = (
            "motd banner\n"
            "__HERMES_LOGIN_PATH_START__/opt/homebrew/bin:/usr/bin:/bin\n"
            "__HERMES_LOGIN_PATH_END__\n"
        )
        assert _extract_login_shell_path(out) == "/opt/homebrew/bin:/usr/bin:/bin"

    def test_last_start_wins(self):
        out = (
            "__HERMES_LOGIN_PATH_START__/stale__HERMES_LOGIN_PATH_END__\n"
            "__HERMES_LOGIN_PATH_START__/fresh:/usr/bin__HERMES_LOGIN_PATH_END__"
        )
        assert _extract_login_shell_path(out) == "/fresh:/usr/bin"

    def test_missing_markers_returns_none(self):
        assert _extract_login_shell_path("/opt/homebrew/bin:/usr/bin") is None
        assert _extract_login_shell_path("") is None
        assert _extract_login_shell_path("__HERMES_LOGIN_PATH_START__/a:/b") is None


class TestMergeLoginShellPath:
    def test_login_first_deduped(self):
        merged = _merge_login_shell_path(
            "/usr/bin:/bin", "/opt/homebrew/bin:/usr/bin:/bin")
        entries = merged.split(":")
        assert entries[0] == "/opt/homebrew/bin"
        assert entries.count("/usr/bin") == 1
        assert entries.count("/bin") == 1

    def test_existing_only_entries_kept(self):
        merged = _merge_login_shell_path("/custom/bin:/usr/bin", "/opt/homebrew/bin:/usr/bin")
        assert merged.split(":") == ["/opt/homebrew/bin", "/usr/bin", "/custom/bin"]

    def test_empties_dropped(self):
        assert _merge_login_shell_path("/usr/bin:", ":/opt/homebrew/bin:") == \
            "/opt/homebrew/bin:/usr/bin"


class TestCapture:
    def test_probe_uses_ladder_and_printenv(self, tmp_path, monkeypatch):
        fake = tmp_path / "zsh"
        fake.write_text("#!/bin/sh\n")
        _make_executable(fake)
        monkeypatch.setenv("SHELL", str(fake))
        seen = []

        class _FakeProc:
            def communicate(self, timeout=None):
                return ("__HERMES_LOGIN_PATH_START__/a:/b__HERMES_LOGIN_PATH_END__", "")

        def _fake_popen(args, **kwargs):
            seen.append(args)
            assert "printenv PATH" in args[-1]
            assert "${PATH}" not in args[-1]
            assert args[1] == "-ilc"
            return _FakeProc()

        with patch.object(local_mod.subprocess, "Popen", _fake_popen):
            assert _capture_user_login_shell_path() == "/a:/b"
        assert len(seen) == 1
        assert seen[0][0] == str(fake)

    def test_probe_ladder_falls_back_to_lc_when_ilc_fails(self, tmp_path, monkeypatch):
        fake = tmp_path / "zsh"
        fake.write_text("#!/bin/sh\n")
        _make_executable(fake)
        monkeypatch.setenv("SHELL", str(fake))
        seen_flags = []

        class _SuccessProc:
            def communicate(self, timeout=None):
                return ("__HERMES_LOGIN_PATH_START__/from_lc__HERMES_LOGIN_PATH_END__", "")

        class _FailProc:
            def communicate(self, timeout=None):
                return ("no markers here", "")

        def _fake_popen(args, **kwargs):
            flag = args[1]
            seen_flags.append(flag)
            if flag == "-ilc":
                return _FailProc()
            return _SuccessProc()

        with patch.object(local_mod.subprocess, "Popen", _fake_popen):
            assert _capture_user_login_shell_path() == "/from_lc"
        assert seen_flags == ["-ilc", "-lc"]

    def test_nonzero_exit_still_trusts_sentinel(self, tmp_path, monkeypatch):
        fake = tmp_path / "zsh"
        fake.write_text("#!/bin/sh\n")
        _make_executable(fake)
        monkeypatch.setenv("SHELL", str(fake))

        class _FakeProc:
            def communicate(self, timeout=None):
                return ("banner\n__HERMES_LOGIN_PATH_START__/x:/y__HERMES_LOGIN_PATH_END__", "")

        with patch.object(local_mod.subprocess, "Popen", lambda *a, **k: _FakeProc()):
            assert _capture_user_login_shell_path() == "/x:/y"

    def test_failure_does_not_cache_and_allows_reprobe(self, tmp_path, monkeypatch):
        fake = tmp_path / "zsh"
        fake.write_text("#!/bin/sh\n")
        _make_executable(fake)
        monkeypatch.setenv("SHELL", str(fake))
        attempt = 0

        class _SuccessProc:
            def communicate(self, timeout=None):
                return ("__HERMES_LOGIN_PATH_START__/recovered__HERMES_LOGIN_PATH_END__", "")

        def _flaky_popen(args, **kwargs):
            nonlocal attempt
            attempt += 1
            if attempt <= 2:  # first capture tries -ilc then -lc and fails
                raise OSError("slow shell timeout")
            return _SuccessProc()

        with patch.object(local_mod.subprocess, "Popen", _flaky_popen):
            # First call fails on both ladder rungs, returns None, does not cache None
            assert _capture_user_login_shell_path() is None
            assert attempt == 2
            # Second call re-probes and recovers
            assert _capture_user_login_shell_path() == "/recovered"
            # Third call uses success cache without re-probing
            assert _capture_user_login_shell_path() == "/recovered"
            assert attempt == 3

    def test_timeout_kills_group_and_returns_none(self, tmp_path, monkeypatch):
        fake = tmp_path / "zsh"
        fake.write_text("#!/bin/sh\n")
        _make_executable(fake)
        monkeypatch.setenv("SHELL", str(fake))
        import subprocess as sp

        killed = {}

        class _HangProc:
            pid = 12345

            def communicate(self, timeout=None):
                if timeout == 2:
                    return ("", "")
                raise sp.TimeoutExpired(cmd="zsh", timeout=timeout)

            def kill(self):
                killed["kill"] = True

        with patch.object(local_mod.subprocess, "Popen", lambda *a, **k: _HangProc()), \
                patch.object(local_mod.os, "killpg", lambda *a, **k: killed.setdefault("killpg", True)):
            assert _capture_user_login_shell_path(timeout=0.01) is None
        assert killed.get("killpg") or killed.get("kill")


class TestApplyLoginShellPath:
    def test_merges_captured_path_in_front(self, monkeypatch):
        monkeypatch.setattr(
            local_mod, "_capture_user_login_shell_path", lambda: "/opt/homebrew/bin:/usr/bin")
        assert _apply_login_shell_path("/usr/bin:/bin") == \
            "/opt/homebrew/bin:/usr/bin:/bin"

    def test_no_capture_leaves_path_untouched(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_capture_user_login_shell_path", lambda: None)
        assert _apply_login_shell_path("/usr/bin:/bin") == "/usr/bin:/bin"

    def test_make_run_env_merges_login_path(self, tmp_path, monkeypatch):
        """End-to-end through _make_run_env: login entries win over /usr/bin."""
        fake = tmp_path / "zsh"
        fake.write_text("#!/bin/sh\n")
        _make_executable(fake)
        monkeypatch.setenv("SHELL", str(fake))
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        monkeypatch.setattr(local_mod, "_managed_runtime_path_entries", lambda: [])
        monkeypatch.setattr(local_mod, "_resolve_hermes_bin_dir", lambda: None)
        monkeypatch.setattr(
            local_mod, "_capture_user_login_shell_path",
            lambda: "/opt/homebrew/bin:/usr/bin:/bin")
        env = local_mod._make_run_env({})
        entries = env["PATH"].split(os.pathsep)
        assert entries[0] == "/opt/homebrew/bin"
        assert entries.count("/opt/homebrew/bin") == 1
