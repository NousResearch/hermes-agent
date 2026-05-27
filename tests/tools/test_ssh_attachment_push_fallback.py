"""Regression test: read_file auto-pushes host-local files to SSH backends.

When a WebUI file upload targets a session that uses an SSH terminal backend,
the uploaded file lands in the host's attachment inbox but is absent on the
remote machine.  Previously read_file returned "File not found" immediately;
after the fix it calls env.push_file() and retries the stat so the agent can
read the content without manual intervention.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, call, patch

import pytest

from tools.file_operations import ShellFileOperations, ReadResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(*, stat_exit_codes, cat_output="content", file_size=7):
    """Return a mock environment whose execute() replays the given stat codes.

    stat_exit_codes: list of exit codes returned on successive stat calls.
    """
    call_count = {"n": 0}

    def _exec(cmd):
        result = MagicMock()
        if cmd.startswith("wc -c"):
            idx = call_count["n"]
            call_count["n"] += 1
            code = stat_exit_codes[idx] if idx < len(stat_exit_codes) else 0
            result.exit_code = code
            result.stdout = str(file_size) if code == 0 else ""
        elif cmd.startswith("head -c"):
            result.exit_code = 0
            result.stdout = cat_output[:1000]
        elif cmd.startswith("cat "):
            result.exit_code = 0
            result.stdout = cat_output
        else:
            result.exit_code = 0
            result.stdout = ""
        return result

    env = MagicMock()
    env.execute = _exec
    env.get_cwd.return_value = "/remote/cwd"
    return env


# ---------------------------------------------------------------------------
# _push_local_file_if_needed
# ---------------------------------------------------------------------------

class TestPushLocalFileIfNeeded:
    def test_returns_false_when_file_absent_locally(self, tmp_path):
        env = MagicMock()
        env.push_file = MagicMock()
        ops = ShellFileOperations(env)
        assert ops._push_local_file_if_needed(str(tmp_path / "ghost.txt")) is False
        env.push_file.assert_not_called()

    def test_returns_false_when_env_has_no_push_file(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('{"k": 1}')
        env = MagicMock(spec=["execute", "get_cwd"])  # no push_file attribute
        ops = ShellFileOperations(env)
        assert ops._push_local_file_if_needed(str(f)) is False

    def test_returns_true_and_calls_push_on_success(self, tmp_path):
        f = tmp_path / "cookies.json"
        f.write_text('{"session": "abc"}')
        env = MagicMock()
        env.push_file = MagicMock()
        ops = ShellFileOperations(env)
        assert ops._push_local_file_if_needed(str(f)) is True
        env.push_file.assert_called_once_with(str(f), str(f))

    def test_returns_false_when_push_raises(self, tmp_path):
        f = tmp_path / "upload.txt"
        f.write_text("hello")
        env = MagicMock()
        env.push_file = MagicMock(side_effect=RuntimeError("scp failed"))
        ops = ShellFileOperations(env)
        assert ops._push_local_file_if_needed(str(f)) is False


# ---------------------------------------------------------------------------
# read_file with SSH push fallback
# ---------------------------------------------------------------------------

class TestReadFilePushFallback:
    def test_read_file_pushes_and_reads_when_remote_missing(self, tmp_path):
        """Simulates: file in local attachment inbox, SSH backend returns not-found
        on first stat, succeeds after push_file transfers it."""
        attachment = tmp_path / "nb.json"
        attachment.write_text('{"token": "secret"}')

        push_calls = []

        def _push(local_path, remote_path):
            push_calls.append((local_path, remote_path))

        env = MagicMock()
        env.push_file = _push
        env.get_cwd.return_value = "/home/carry"

        call_n = {"n": 0}

        def _exec(cmd):
            r = MagicMock()
            if "wc -c" in cmd:
                # First stat fails (remote missing), second succeeds (after push)
                r.exit_code = 1 if call_n["n"] == 0 else 0
                r.stdout = "" if call_n["n"] == 0 else str(attachment.stat().st_size)
                call_n["n"] += 1
            elif "head -c" in cmd:
                r.exit_code = 0
                r.stdout = '{"token": "secret"}'
            elif cmd.startswith("cat "):
                r.exit_code = 0
                r.stdout = '{"token": "secret"}'
            else:
                r.exit_code = 0
                r.stdout = ""
            return r

        env.execute = _exec

        ops = ShellFileOperations(env)
        # Patch _exec to use our mock
        ops._exec = _exec

        result = ops.read_file(str(attachment))

        assert len(push_calls) == 1
        assert push_calls[0] == (str(attachment), str(attachment))
        assert result.error is None
        assert result.content is not None

    def test_read_file_falls_through_to_suggest_when_push_fails(self, tmp_path):
        """If push_file raises, read_file still returns the suggest-similar fallback."""
        attachment = tmp_path / "broken.json"
        attachment.write_text("{}")

        env = MagicMock()
        env.push_file = MagicMock(side_effect=RuntimeError("scp: connection timeout"))
        env.get_cwd.return_value = "/remote"

        def _exec(cmd):
            r = MagicMock()
            r.exit_code = 1
            r.stdout = ""
            return r

        ops = ShellFileOperations(env)
        ops._exec = _exec
        ops._suggest_similar_files = MagicMock(
            return_value=ReadResult(error=f"File not found: {attachment}")
        )

        result = ops.read_file(str(attachment))

        assert result.error is not None
        ops._suggest_similar_files.assert_called_once()

    def test_read_file_skips_push_when_env_lacks_push_file(self, tmp_path):
        """Local environments don't expose push_file; fallback must not crash."""
        f = tmp_path / "local.txt"
        f.write_text("hello local")

        env = MagicMock(spec=["execute", "get_cwd"])
        env.get_cwd.return_value = str(tmp_path)

        def _exec(cmd):
            r = MagicMock()
            r.exit_code = 1
            r.stdout = ""
            return r

        ops = ShellFileOperations(env)
        ops._exec = _exec
        ops._suggest_similar_files = MagicMock(return_value=ReadResult(error="not found"))

        result = ops.read_file(str(f))
        assert result.error is not None


# ---------------------------------------------------------------------------
# SSHEnvironment.push_file delegates to _scp_upload
# ---------------------------------------------------------------------------

class TestSSHEnvironmentPushFile:
    def test_push_file_calls_scp_upload(self):
        from tools.environments.ssh import SSHEnvironment
        env = object.__new__(SSHEnvironment)
        env._scp_upload = MagicMock()
        env.push_file("/local/nb.json", "/remote/home/nb.json")
        env._scp_upload.assert_called_once_with("/local/nb.json", "/remote/home/nb.json")
