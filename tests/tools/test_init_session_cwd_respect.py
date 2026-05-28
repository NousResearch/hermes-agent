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

from tools.environments.base import (
    BaseEnvironment,
    _SNAPSHOT_SECRET_ENV_RE,
    _snapshot_export_command,
)


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

    def test_configured_cwd_survives_init_session(self):
        """self.cwd must be the configured path after init_session completes."""
        configured_cwd = "/my/project"
        env = _TestableEnv(cwd=configured_cwd)

        marker = env._cwd_marker

        def mock_run_bash(cmd_string, *, login=False, timeout=120, stdin_data=None):
            mock = MagicMock()
            mock.poll.return_value = 0
            mock.returncode = 0
            # Simulate output where pwd reports the configured cwd
            output = f"snapshot output\n{marker}{configured_cwd}{marker}\n"
            stdout = TemporaryFile(mode="w+b")
            stdout.write(output.encode("utf-8"))
            stdout.seek(0)
            mock.stdout = stdout
            return mock

        env._run_bash = mock_run_bash
        env.init_session()

        assert env.cwd == configured_cwd, (
            f"Expected cwd={configured_cwd!r} after init_session, got {env.cwd!r}"
        )

    def test_default_cwd_still_works(self):
        """When no custom cwd is configured, default /tmp behavior is preserved."""
        env = _TestableEnv()  # default cwd="/tmp"

        marker = env._cwd_marker

        def mock_run_bash(cmd_string, *, login=False, timeout=120, stdin_data=None):
            mock = MagicMock()
            mock.poll.return_value = 0
            mock.returncode = 0
            output = f"snapshot output\n{marker}/tmp{marker}\n"
            stdout = TemporaryFile(mode="w+b")
            stdout.write(output.encode("utf-8"))
            stdout.seek(0)
            mock.stdout = stdout
            return mock

        env._run_bash = mock_run_bash
        env.init_session()

        assert env.cwd == "/tmp"

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

    def test_snapshot_capture_filters_secret_env_names(self):
        env = _TestableEnv(cwd="/tmp")
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
        assert "export -p | grep -Eiv" in bootstrap
        assert "TOKEN" in bootstrap
        assert "PASSWORD" in bootstrap
        assert "KEY" in bootstrap
        assert "CREDENTIALS?" in bootstrap
        assert "AUTHORIZATION" in bootstrap
        assert "BEARER" in bootstrap

    def test_snapshot_refresh_filters_secret_env_names(self):
        env = _TestableEnv(cwd="/tmp")
        env._snapshot_ready = True

        wrapped = env._wrap_command("true", "/tmp")

        assert "export -p | grep -Eiv" in wrapped
        assert "TOKEN" in wrapped
        assert "PASSWORD" in wrapped
        assert "KEY" in wrapped
        assert "CREDENTIALS?" in wrapped
        assert "AUTHORIZATION" in wrapped

    def test_snapshot_filter_does_not_drop_auth_socket_names(self):
        cmd = _snapshot_export_command("/tmp/snap.sh")
        assert "AUTHORIZATION" in cmd
        assert "BEARER" in cmd
        assert "AUTH($|_)" not in cmd

    def test_snapshot_filter_scrubs_common_secret_names_but_keeps_socket_names(self):
        lines = "\n".join(
            [
                'declare -x OPENAI_API_KEY="secret"',
                'declare -x api_key="secret"',
                'declare -x OPENAI_KEY="secret"',
                'declare -x GOOGLE_APPLICATION_CREDENTIALS="secret"',
                'declare -x AWS_ACCESS_KEY_ID="secret"',
                'declare -x NPM_CONFIG__AUTH_TOKEN="secret"',
                'declare -x SSH_AUTH_SOCK="/tmp/ssh.sock"',
                'declare -x XAUTHORITY="/tmp/auth"',
                'declare -x TOKENIZER_PARALLELISM="false"',
            ]
        )
        import subprocess

        proc = subprocess.run(
            ["grep", "-Eiv", _SNAPSHOT_SECRET_ENV_RE],
            input=lines,
            text=True,
            capture_output=True,
            check=True,
        )
        snapshot = proc.stdout
        assert "OPENAI_API_KEY" not in snapshot
        assert "api_key" not in snapshot
        assert "OPENAI_KEY" not in snapshot
        assert "GOOGLE_APPLICATION_CREDENTIALS" not in snapshot
        assert "AWS_ACCESS_KEY_ID" not in snapshot
        assert "NPM_CONFIG__AUTH_TOKEN" not in snapshot
        assert "SSH_AUTH_SOCK" in snapshot
        assert "XAUTHORITY" in snapshot
        assert "TOKENIZER_PARALLELISM" in snapshot
