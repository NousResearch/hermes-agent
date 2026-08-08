"""Tests for the H2 argv-form support in terminal_tool.

The argv-list form skips bash entirely, eliminating the bash-quoting bug
class on apostrophes / unbalanced quotes inside JSON bodies. These tests
verify the dispatch path without spawning real processes.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from tools.terminal_tool import TERMINAL_SCHEMA, terminal_tool


def _mock_env(stdout: str = "ok", returncode: int = 0):
    """Build a mock env that records execute / execute_argv calls."""
    env = MagicMock()
    env.execute.return_value = {"output": stdout, "returncode": returncode}
    env.execute_argv.return_value = {"output": stdout, "returncode": returncode}
    env.env = {}
    return env


class TestArgvValidation:
    """Pure validation paths — no env needed; the guards run before env lookup."""

    def test_command_and_argv_together_returns_error(self):
        result = json.loads(terminal_tool(command="echo hi", argv=["echo", "hi"]))
        assert "error" in result
        assert "either" in result["error"].lower() or "both" in result["error"].lower()

    def test_empty_argv_falls_through_as_no_command(self):
        # argv=[] is treated as "argv not provided"; with empty command too,
        # we get the empty-command guard error.
        result = json.loads(terminal_tool(command="", argv=[]))
        assert "error" in result
        assert "empty" in result["error"].lower()

    def test_argv_with_non_string_returns_error(self):
        result = json.loads(terminal_tool(command="", argv=["curl", 42, "http://x"]))  # type: ignore[list-item]
        assert "error" in result
        assert "argv" in result["error"].lower()

    def test_argv_rejects_background(self):
        result = json.loads(
            terminal_tool(command="", argv=["sleep", "60"], background=True)
        )
        assert "error" in result
        assert "background" in result["error"].lower()

    def test_argv_rejects_pty(self):
        result = json.loads(terminal_tool(command="", argv=["python"], pty=True))
        assert "error" in result
        assert "pty" in result["error"].lower()


class TestArgvDispatchPath:
    """End-to-end dispatch — inject env via _active_environments dict."""

    def _install_env(self, monkeypatch_env):
        """Stash a mock env in _active_environments under a unique task_id.

        Register an override entry so _resolve_container_task_id keeps our
        custom task_id intact (otherwise it collapses to "default" and we'd
        pollute the global slot).
        """
        from tools import terminal_tool as tt
        task_id = "test-argv-dispatch"
        tt._task_env_overrides[task_id] = {}  # marker to keep task_id distinct
        tt._active_environments[task_id] = monkeypatch_env
        # Update activity timestamp so the env isn't reaped mid-test.
        tt._last_activity[task_id] = 1e15
        return task_id

    def _cleanup(self, task_id):
        from tools import terminal_tool as tt
        tt._active_environments.pop(task_id, None)
        tt._task_env_overrides.pop(task_id, None)
        tt._last_activity.pop(task_id, None)

    def test_argv_routes_to_execute_argv_not_execute(self):
        env = _mock_env(stdout="hello")
        task_id = self._install_env(env)
        try:
            argv_payload = ["curl", "-X", "POST", "-H", "Content-Type: application/json",
                            "-d", '{"body":"It\'s done"}', "http://localhost:3100/api/issues/AOS-8"]
            result_json = terminal_tool(command="", argv=argv_payload, task_id=task_id)
            result = json.loads(result_json)

            # Dispatch landed on execute_argv with the literal argv list — never
            # touched bash, so the apostrophe in "It's done" is safe.
            env.execute_argv.assert_called_once()
            env.execute.assert_not_called()
            called_argv = env.execute_argv.call_args.args[0]
            assert called_argv == argv_payload
            assert called_argv[6] == '{"body":"It\'s done"}', (
                "Apostrophe in JSON body must reach the env verbatim — "
                "the whole point of argv form is no shell parsing"
            )
            assert result["exit_code"] == 0
            assert result["output"] == "hello"
        finally:
            self._cleanup(task_id)

    def test_string_command_still_routes_to_execute(self):
        env = _mock_env()
        task_id = self._install_env(env)
        try:
            terminal_tool(command="echo hello", task_id=task_id)
            env.execute.assert_called_once()
            env.execute_argv.assert_not_called()
        finally:
            self._cleanup(task_id)


class TestSchemaShape:
    def test_schema_advertises_argv_property(self):
        props = TERMINAL_SCHEMA["parameters"]["properties"]
        assert "argv" in props
        assert props["argv"]["type"] == "array"
        assert props["argv"]["items"] == {"type": "string"}

    def test_schema_command_description_mentions_argv_alternative(self):
        # The model should learn from the schema that argv exists as an alt.
        props = TERMINAL_SCHEMA["parameters"]["properties"]
        assert "argv" in props["command"]["description"]

    def test_schema_argv_description_warns_about_shell_features(self):
        props = TERMINAL_SCHEMA["parameters"]["properties"]
        desc = props["argv"]["description"].lower()
        # Model must understand that argv form skips pipes/redirection.
        assert "pipe" in desc or "redirect" in desc or "shell-level" in desc.lower()
        # And must understand the mutual exclusion.
        assert "mutually exclusive" in desc or "not both" in desc


class TestLocalRunArgv:
    """Smoke-test that LocalEnvironment._run_argv actually skips bash."""

    def test_run_argv_uses_subprocess_popen_with_shell_false(self):
        from tools.environments.local import LocalEnvironment

        # Construct a minimal LocalEnvironment without going through init_session
        # (which would actually fork bash). We just need the instance methods.
        env = LocalEnvironment.__new__(LocalEnvironment)
        env.cwd = "/tmp"
        env.env = {}

        with patch("tools.environments.local.subprocess.Popen") as mock_popen, \
                patch("tools.environments.local._make_run_env", return_value={}), \
                patch("tools.environments.local._resolve_safe_cwd", return_value="/tmp"):
            mock_popen.return_value = MagicMock()
            env._run_argv(["echo", "hello"])

            mock_popen.assert_called_once()
            args, kwargs = mock_popen.call_args
            # The first positional arg is the argv list — passed through verbatim.
            assert args[0] == ["echo", "hello"]
            # CRITICAL: shell=False so bash is never invoked.
            assert kwargs["shell"] is False


class TestExecuteArgvBaseInterface:
    """execute_argv on the base class must validate input and skip cwd updates."""

    def test_execute_argv_rejects_non_list(self):
        from tools.environments.local import LocalEnvironment

        env = LocalEnvironment.__new__(LocalEnvironment)
        env.cwd = "/tmp"
        env.env = {}
        env.timeout = 60
        env._snapshot_ready = False

        try:
            env.execute_argv("not a list")  # type: ignore[arg-type]
        except TypeError as e:
            assert "list of strings" in str(e)
        else:
            raise AssertionError("expected TypeError")

    def test_execute_argv_rejects_empty_list(self):
        from tools.environments.local import LocalEnvironment

        env = LocalEnvironment.__new__(LocalEnvironment)
        env.cwd = "/tmp"
        env.env = {}
        env.timeout = 60
        env._snapshot_ready = False

        try:
            env.execute_argv([])
        except TypeError as e:
            assert "non-empty" in str(e).lower()
        else:
            raise AssertionError("expected TypeError")
