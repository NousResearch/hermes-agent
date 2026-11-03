"""Tests that init_session() respects the configured cwd.

The bug: when terminal.cwd is set in config.yaml, the configured path was
displayed in the TUI banner but actual terminal commands ran in os.getcwd()
(the directory where ``hermes chat`` was started).

Root cause: init_session() captures the login shell environment by running
``pwd -P`` inside a ``bash -l -c`` bootstrap.  Profile scripts (.bashrc,
.bash_profile, etc.) can change the working directory before ``pwd -P``
runs, so _update_cwd() overwrites self.cwd with the wrong directory.

Fix: the bootstrap now includes an explicit ``cd`` back to self.cwd before
running ``pwd -P``, so the configured cwd is always what gets recorded.
"""

from tempfile import TemporaryFile
from unittest.mock import MagicMock

from tools.environments.base import BaseEnvironment


class _TestableEnv(BaseEnvironment):
    """Concrete subclass for testing base class methods."""

    def __init__(self, cwd="/tmp", timeout=10):
        super().__init__(cwd=cwd, timeout=timeout)

    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        raise NotImplementedError("Use mock")

    def cleanup(self):
        pass


class TestInitSessionCwdRespect:
    """init_session() must preserve the configured cwd."""

    def test_bootstrap_contains_cd_to_configured_cwd(self):
        """The bootstrap script must cd to self.cwd before running pwd."""
        env = _TestableEnv(cwd="/my/project")

        # Capture the bootstrap script that init_session would pass to _run_bash
        captured = {}

        def mock_run_bash(cmd_string, *, login=False, timeout=120, stdin_data=None):
            captured["cmd"] = cmd_string
            mock = MagicMock()
            mock.poll.return_value = 0
            mock.returncode = 0
            stdout = TemporaryFile(mode="w+b")
            stdout.seek(0)
            mock.stdout = stdout
            return mock

        env._run_bash = mock_run_bash
        env.init_session()

        assert "cmd" in captured, "init_session did not call _run_bash"
        bootstrap = captured["cmd"]

        # The cd must appear before pwd -P so the configured cwd is recorded
        cd_pos = bootstrap.find("builtin cd")
        pwd_pos = bootstrap.find("pwd -P")
        assert cd_pos != -1, "bootstrap must contain 'builtin cd'"
        assert pwd_pos != -1, "bootstrap must contain 'pwd -P'"
        assert cd_pos < pwd_pos, (
            "builtin cd must appear before pwd -P in the bootstrap so "
            "the configured cwd is what gets recorded"
        )

        # The cd target must be the configured path (shlex.quote only adds
        # quotes when the path contains shell-special characters)
        assert "/my/project" in bootstrap, (
            "bootstrap cd must target the configured cwd (/my/project)"
        )


    def test_bootstrap_cd_uses_shlex_quote(self):
        """Paths with spaces must be properly quoted in the bootstrap cd."""
        env = _TestableEnv(cwd="/my project/with spaces")

        captured = {}

        def mock_run_bash(cmd_string, *, login=False, timeout=120, stdin_data=None):
            captured["cmd"] = cmd_string
            mock = MagicMock()
            mock.poll.return_value = 0
            mock.returncode = 0
            stdout = TemporaryFile(mode="w+b")
            stdout.seek(0)
            mock.stdout = stdout
            return mock

        env._run_bash = mock_run_bash
        env.init_session()

        bootstrap = captured["cmd"]
        # shlex.quote wraps paths with spaces in single quotes
        assert "'/my project/with spaces'" in bootstrap, (
            "bootstrap cd must properly quote paths with spaces"
        )
