"""Tests for LocalEnvironment.wrap_argv hook.

A plugin (Hermes Tether) needs to prepend ``sandbox-exec`` to every host
process LocalEnvironment starts.  The hook must be called for all three
spawn purposes (command, login, long_lived), must leave output/cwd/env
unchanged when it is the default identity, and must not break process-group
kill (the exec-in-place replacement keeps the PID).
"""

import contextlib
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import textwrap
import time
from unittest.mock import patch

import pytest

from tools.environments.local import LocalEnvironment


@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "logs").mkdir(exist_ok=True)


def _make_marker_script(tmp_path):
    """Return a shell script that records its invocation and execs the rest.

    The script writes a sentinel line to *log* and then replaces itself
    with ``$@`` so the PID stays the same — exactly what ``sandbox-exec``
    does.
    """
    marker = tmp_path / "marker.sh"
    log = tmp_path / "marker.log"
    marker.write_text(textwrap.dedent(f"""\
        #!/bin/sh
        echo "MARKER_WAS_HERE" \u003e\u003e {shlex.quote(str(log))}
        exec "$@"
    """))
    marker.chmod(0o755)
    return str(marker), str(log)


def _wait_for_file(path, timeout=5.0):
    p = Path(path)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if p.exists() and p.stat().st_size:
            return True
        time.sleep(0.05)
    return p.exists() and p.stat().st_size


def _read_log_lines(log_path):
    if not os.path.exists(log_path):
        return []
    with open(log_path) as f:
        return [line.strip() for line in f if line.strip()]


class _PrefixEnv(LocalEnvironment):
    """Subclass that prepends a marker program via wrap_argv."""

    def __init__(self, marker, *args, **kwargs):
        self._marker = marker
        super().__init__(*args, **kwargs)

    def wrap_argv(self, argv, env, *, purpose):
        return [self._marker, *argv]


class TestWrapArgvHook:
    """Behaviour-contract tests for the wrap_argv seam."""

    @pytest.mark.platforms("posix")
    def test_login_shell_path_goes_through_hook(self, tmp_path):
        """init_session() calls _run_bash with login=True; the hook must fire."""
        marker, log = _make_marker_script(tmp_path)
        env = _PrefixEnv(marker, cwd=str(tmp_path), timeout=10)
        try:
            assert _wait_for_file(log), "marker never wrote a log entry"
            lines = _read_log_lines(log)
            assert len(lines) >= 1, f"expected at least one log line, got {lines!r}"
            # The first spawn is the login-shell snapshot.
            assert lines[0] == "MARKER_WAS_HERE"
        finally:
            env.cleanup()

    @pytest.mark.platforms("posix")
    def test_command_path_goes_through_hook(self, tmp_path):
        """execute() calls _run_bash with login=False; the hook must fire."""
        marker, log = _make_marker_script(tmp_path)
        with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = _PrefixEnv(marker, cwd=str(tmp_path), timeout=10)
        try:
            result = env.execute("echo hello-wrap")
            assert result.get("output", "").strip() == "hello-wrap"
            lines = _read_log_lines(log)
            assert len(lines) >= 1, f"expected at least one log line, got {lines!r}"
            assert lines[-1] == "MARKER_WAS_HERE"
        finally:
            env.cleanup()

    @pytest.mark.platforms("posix")
    def test_long_lived_path_goes_through_hook(self, tmp_path):
        """start_process() calls _spawn_long_lived; the hook must fire."""
        marker, log = _make_marker_script(tmp_path)
        with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = _PrefixEnv(marker, cwd=str(tmp_path), timeout=10)
        try:
            stream = env.start_process("test_ll", [sys.executable, "-c", "print('ll-ok')"])
            # Drain stdout until the process exits.
            stdout_bytes = b""
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                chunk = stream.stdout.read(1024)
                if chunk:
                    stdout_bytes += chunk
                if stream.poll() is not None:
                    break
                time.sleep(0.05)
            assert b"ll-ok" in stdout_bytes, f"got {stdout_bytes!r}"
            lines = _read_log_lines(log)
            assert len(lines) >= 1, f"expected at least one log line, got {lines!r}"
            assert lines[-1] == "MARKER_WAS_HERE"
        finally:
            env.cleanup()

    
    def test_default_wrap_argv_is_identity(self, tmp_path):
        """Without an override, wrap_argv must return argv unchanged."""
        env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
        argv = ["bash", "-c", "echo hi"]
        run_env = {"PATH": "/usr/bin"}
        assert env.wrap_argv(argv, run_env, purpose="command") is argv
        env.cleanup()

    @pytest.mark.platforms("posix")
    def test_wrap_argv_preserves_cwd_and_env(self, tmp_path):
        """A prefix that execs in place must not alter cwd or the child env."""
        marker, _log = _make_marker_script(tmp_path)
        with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = _PrefixEnv(marker, cwd=str(tmp_path), timeout=10)
        # Inject a sentinel env var through the backend's env dict.
        env.env["WRAP_TEST_SENTINEL"] = "sentinel-value"
        try:
            result = env.execute(
                'echo "cwd=$PWD sentinel=$WRAP_TEST_SENTINEL"')
            output = result.get("output", "").strip()
            assert f"cwd={tmp_path}" in output, f"cwd lost: {output!r}"
            assert "sentinel=sentinel-value" in output, f"env lost: {output!r}"
        finally:
            env.cleanup()

    @pytest.mark.platforms("posix")
    def test_process_group_kill_survives_exec_prefix(self, tmp_path):
        """When wrap_argv prepends an exec-in-place marker, the timeout kill
        must still reach the whole process group — the PID after exec is the
        same as before."""
        marker, log = _make_marker_script(tmp_path)
        with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = _PrefixEnv(marker, cwd=str(tmp_path), timeout=10)
        try:
            result = env.execute("sleep 300", timeout=2)
            # The command must have been killed (timed out or killed).
            assert result.get("returncode") not in (0, None), (
                f"expected non-zero exit, got {result!r}")
            # Verify the marker was invoked.
            lines = _read_log_lines(log)
            assert len(lines) >= 1
            assert lines[-1] == "MARKER_WAS_HERE"
        finally:
            env.cleanup()

    
    def test_purpose_argument_reaches_wrap_argv(self, tmp_path):
        """The purpose kwarg must be passed and have the documented values."""
        purposes = []

        class _RecordingEnv(LocalEnvironment):
            def wrap_argv(self, argv, env, *, purpose):
                purposes.append(purpose)
                return argv

        env = _RecordingEnv(cwd=str(tmp_path), timeout=10)
        try:
            # init_session fires first with login.
            assert "login" in purposes
            result = env.execute("true")
            assert result.get("returncode") == 0
            assert "command" in purposes
        finally:
            env.cleanup()


class TestConfinedLocalEnvironment:
    """A LocalEnvironment with a command prefix must not keep the host fast path."""

    def test_unconfined_local_environment_keeps_host_filesystem(self, tmp_path):
        from tools.environments.base import env_uses_host_filesystem

        env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
        try:
            assert env_uses_host_filesystem(env) is True
        finally:
            env.cleanup()

    @pytest.mark.platforms("posix")
    def test_argv_prefix_turns_host_filesystem_off(self, tmp_path):
        from tools.environments.base import env_uses_host_filesystem

        marker, _log = _make_marker_script(tmp_path)
        env = LocalEnvironment(cwd=str(tmp_path), timeout=10, argv_prefix=[marker])
        try:
            assert env_uses_host_filesystem(env) is False
        finally:
            env.cleanup()

    @pytest.mark.platforms("posix")
    def test_wrap_argv_override_turns_host_filesystem_off(self, tmp_path):
        from tools.environments.base import env_uses_host_filesystem

        marker, _log = _make_marker_script(tmp_path)
        env = _PrefixEnv(marker, cwd=str(tmp_path), timeout=10)
        try:
            assert env_uses_host_filesystem(env) is False
        finally:
            env.cleanup()

    @pytest.mark.platforms("posix")
    def test_search_files_runs_inside_the_prefix(self, tmp_path):
        from tools.file_operations import ShellFileOperations

        marker, _log = _make_marker_script(tmp_path)
        tree = tmp_path / "tree"
        tree.mkdir()
        (tree / "a.txt").write_text("needle-in-haystack\n")
        env = _PrefixEnv(marker, cwd=str(tmp_path), timeout=20)
        spawned = []
        real_popen = subprocess.Popen

        def spy(args, *a, **kw):
            spawned.append(list(args) if not isinstance(args, str) else [args])
            return real_popen(args, *a, **kw)

        try:
            ops = ShellFileOperations(env, cwd=str(tmp_path))
            with patch.object(subprocess, "Popen", spy):
                result = ops.search("needle-in-haystack", path=str(tree))
            assert result.total_count == 1
            escaped = [a for a in spawned if a and a[0] != marker]
            assert escaped == [], f"spawned outside the prefix: {escaped!r}"
        finally:
            env.cleanup()

    @pytest.mark.platforms("posix")
    def test_read_file_goes_through_execute(self, tmp_path):
        from tools.file_operations import ShellFileOperations

        marker, log = _make_marker_script(tmp_path)
        target = tmp_path / "f.txt"
        target.write_text("hello\n")
        env = _PrefixEnv(marker, cwd=str(tmp_path), timeout=20)
        try:
            before = len(_read_log_lines(log))
            result = ShellFileOperations(env, cwd=str(tmp_path)).read_file(str(target))
            assert "hello" in result.content
            assert len(_read_log_lines(log)) > before, "read did not use the prefix"
        finally:
            env.cleanup()


class TestArgvPrefixValidation:
    def test_str_prefix_is_rejected(self, tmp_path):
        with pytest.raises(TypeError):
            LocalEnvironment(cwd=str(tmp_path), timeout=10, argv_prefix="sandbox-exec")

    def test_bytes_and_non_str_items_are_rejected(self, tmp_path):
        with pytest.raises(TypeError):
            LocalEnvironment(cwd=str(tmp_path), timeout=10, argv_prefix=b"x")
        with pytest.raises(TypeError):
            LocalEnvironment(cwd=str(tmp_path), timeout=10, argv_prefix=["ok", 3])

    def test_list_prefix_is_accepted(self, tmp_path):
        with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = LocalEnvironment(cwd=str(tmp_path), timeout=10, argv_prefix=["env", "-i"])
        try:
            assert env._argv_prefix == ("env", "-i")
        finally:
            env.cleanup()


class TestWindowsIdentity:
    def test_wrap_argv_is_identity_on_windows_and_warns_once(self, tmp_path, monkeypatch, caplog):
        import logging

        import tools.environments.local as local_mod

        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        monkeypatch.setattr(local_mod, "_WARNED_WINDOWS_PREFIX", False, raising=False)
        with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            with caplog.at_level(logging.WARNING, logger=local_mod.logger.name):
                env = LocalEnvironment(cwd=str(tmp_path), timeout=10, argv_prefix=["sandbox-exec"])
                argv = ["bash", "-c", "x"]
                assert env.wrap_argv(argv, {}, purpose="command") is argv
                assert env.wrap_argv(argv, {}, purpose="login") is argv
        warnings = [r for r in caplog.records if "argv_prefix" in r.getMessage()]
        assert len(warnings) == 1
