"""Tests for login-shell PATH preservation in _run_bash.

On Debian-based systems, /etc/profile unconditionally resets PATH for
non-root shells, discarding venv entries set by the Dockerfile's ENV PATH.
_run_bash must save and restore PATH around the login shell so the snapshot
(export -p) captures the correct venv entries.

See: https://github.com/NousResearch/hermes-agent/issues/56634
"""

import os
from unittest.mock import patch

import pytest

from tools.environments.local import _make_run_env, _path_env_key


class TestLoginShellPathPreservation:
    """Verify that _run_bash wraps login=True invocations with PATH
    save/restore so Debian's /etc/profile reset doesn't discard venv
    entries from the captured snapshot."""

    def test_path_restore_prelude_appended_for_login(self, tmp_path, monkeypatch):
        """When login=True, the cmd_string should include PATH save/restore."""
        from tools.environments.local import LocalEnvironment

        monkeypatch.setenv("HOME", str(tmp_path))
        env = LocalEnvironment(cwd=str(tmp_path), env={})

        # Capture the args passed to Popen
        captured = {}

        original_popen = __import__("subprocess").Popen

        def mock_popen(args, **kwargs):
            captured["args"] = args
            captured["env"] = kwargs.get("env")
            # Return a dummy process that exits immediately
            raise SystemExit(0)

        with patch("subprocess.Popen", side_effect=mock_popen):
            try:
                env._run_bash("echo hello", login=True)
            except SystemExit:
                pass

        cmd = captured["args"][-1]  # bash -l -c <cmd>
        assert "__hermes_prelogin_path" in cmd
        assert "unset __hermes_prelogin_path" in cmd

    def test_no_path_restore_for_non_login(self, tmp_path, monkeypatch):
        """When login=False, no PATH save/restore should be injected."""
        from tools.environments.local import LocalEnvironment

        monkeypatch.setenv("HOME", str(tmp_path))
        env = LocalEnvironment(cwd=str(tmp_path), env={})

        captured = {}

        def mock_popen(args, **kwargs):
            captured["args"] = args
            raise SystemExit(0)

        with patch("subprocess.Popen", side_effect=mock_popen):
            try:
                env._run_bash("echo hello", login=False)
            except SystemExit:
                pass

        cmd = captured["args"][2]
        assert "__hermes_prelogin_path" not in cmd

    def test_make_run_env_preserves_venv_path(self, tmp_path, monkeypatch):
        """_make_run_env should keep venv entries on PATH."""
        venv_bin = "/opt/hermes/.venv/bin"
        original_path = f"{venv_bin}:/usr/local/bin:/usr/bin:/bin"
        monkeypatch.setenv("PATH", original_path)
        monkeypatch.setenv("HOME", str(tmp_path))

        run_env = _make_run_env({})
        path_key = _path_env_key(run_env)
        assert path_key is not None
        assert venv_bin in run_env[path_key]
