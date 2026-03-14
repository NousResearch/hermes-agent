"""Tests for the local persistent shell backend."""

import glob as glob_mod

import pytest

from tools.environments.local import LocalEnvironment
from tools.environments.persistent_shell import PersistentShellMixin


class TestLocalConfig:
    def test_local_persistent_default_false(self, monkeypatch):
        monkeypatch.delenv("TERMINAL_LOCAL_PERSISTENT", raising=False)
        from tools.terminal_tool import _get_env_config
        assert _get_env_config()["local_persistent"] is False

    def test_local_persistent_true(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_LOCAL_PERSISTENT", "true")
        from tools.terminal_tool import _get_env_config
        assert _get_env_config()["local_persistent"] is True

    def test_local_persistent_yes(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_LOCAL_PERSISTENT", "yes")
        from tools.terminal_tool import _get_env_config
        assert _get_env_config()["local_persistent"] is True


class TestMergeOutput:
    def test_stdout_only(self):
        assert PersistentShellMixin._merge_output("out", "") == "out"

    def test_stderr_only(self):
        assert PersistentShellMixin._merge_output("", "err") == "err"

    def test_both(self):
        assert PersistentShellMixin._merge_output("out", "err") == "out\nerr"

    def test_empty(self):
        assert PersistentShellMixin._merge_output("", "") == ""

    def test_strips_trailing_newlines(self):
        assert PersistentShellMixin._merge_output("out\n\n", "err\n") == "out\nerr"


class TestLocalOneShotRegression:
    def test_echo(self):
        env = LocalEnvironment(persistent=False)
        r = env.execute("echo hello")
        assert r["returncode"] == 0
        assert "hello" in r["output"]
        env.cleanup()

    def test_exit_code(self):
        env = LocalEnvironment(persistent=False)
        r = env.execute("exit 42")
        assert r["returncode"] == 42
        env.cleanup()

    def test_state_does_not_persist(self):
        env = LocalEnvironment(persistent=False)
        env.execute("export HERMES_ONESHOT_LOCAL=yes")
        r = env.execute("echo $HERMES_ONESHOT_LOCAL")
        assert r["output"].strip() == ""
        env.cleanup()


class TestLocalPersistent:
    @pytest.fixture
    def env(self):
        e = LocalEnvironment(persistent=True)
        yield e
        e.cleanup()

    def test_echo(self, env):
        r = env.execute("echo hello-persistent")
        assert r["returncode"] == 0
        assert "hello-persistent" in r["output"]

    def test_env_var_persists(self, env):
        env.execute("export HERMES_LOCAL_PERSIST_TEST=works")
        r = env.execute("echo $HERMES_LOCAL_PERSIST_TEST")
        assert r["output"].strip() == "works"

    def test_cwd_persists(self, env):
        env.execute("cd /tmp")
        r = env.execute("pwd")
        assert r["output"].strip() == "/tmp"

    def test_exit_code(self, env):
        r = env.execute("(exit 42)")
        assert r["returncode"] == 42

    def test_stderr(self, env):
        r = env.execute("echo oops >&2")
        assert r["returncode"] == 0
        assert "oops" in r["output"]

    def test_multiline_output(self, env):
        r = env.execute("echo a; echo b; echo c")
        lines = r["output"].strip().splitlines()
        assert lines == ["a", "b", "c"]

    def test_timeout_then_recovery(self, env):
        r = env.execute("sleep 999", timeout=2)
        assert r["returncode"] in (124, 130)
        r = env.execute("echo alive")
        assert r["returncode"] == 0
        assert "alive" in r["output"]

    def test_large_output(self, env):
        r = env.execute("seq 1 1000")
        assert r["returncode"] == 0
        lines = r["output"].strip().splitlines()
        assert len(lines) == 1000
        assert lines[0] == "1"
        assert lines[-1] == "1000"

    def test_shell_variable_persists(self, env):
        env.execute("MY_LOCAL_VAR=hello123")
        r = env.execute("echo $MY_LOCAL_VAR")
        assert r["output"].strip() == "hello123"

    def test_cleanup_removes_temp_files(self, env):
        env.execute("echo warmup")
        prefix = env._temp_prefix
        assert len(glob_mod.glob(f"{prefix}-*")) > 0
        env.cleanup()
        remaining = glob_mod.glob(f"{prefix}-*")
        assert remaining == []

    def test_state_does_not_leak_between_instances(self):
        env1 = LocalEnvironment(persistent=True)
        env2 = LocalEnvironment(persistent=True)
        try:
            env1.execute("export LEAK_TEST=from_env1")
            r = env2.execute("echo $LEAK_TEST")
            assert r["output"].strip() == ""
        finally:
            env1.cleanup()
            env2.cleanup()

    def test_special_characters_in_command(self, env):
        r = env.execute("echo 'hello world'")
        assert r["output"].strip() == "hello world"

    def test_pipe_command(self, env):
        r = env.execute("echo hello | tr 'h' 'H'")
        assert r["output"].strip() == "Hello"

    def test_multiple_commands_semicolon(self, env):
        r = env.execute("X=42; echo $X")
        assert r["output"].strip() == "42"
