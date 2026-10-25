"""The idle reaper must not tear down an environment while a foreground call is running in it.

``_cleanup_inactive_envs`` reaps any key whose ``_last_activity`` is older than ``lifetime_seconds``
(default 300 s). Only background processes refresh that stamp mid-run, while a foreground command may
run for up to 600 s: Docker workspace containers were stopped (exit 137), Modal sandboxes terminated
and SSH lost its control master mid-command.
"""

from __future__ import annotations

import json
import time

import pytest

from tools import code_execution_tool, terminal_tool as tt
from tools.terminal_tool_lifecycle import _cleanup_inactive_envs


class _ReapingEnv:
    """An env whose command outlives ``lifetime_seconds``: one reaper sweep fires from inside it."""

    def __init__(self):
        self.cleaned = False
        self.cleaned_mid_call = None

    def cleanup(self):
        self.cleaned = True

    def execute(self, command, cwd=None, timeout=None, **kwargs):
        _sweep_after_long_call()
        self.cleaned_mid_call = self.cleaned
        return {"output": "hi", "returncode": 0}


def _sweep_after_long_call():
    for key in list(tt._last_activity):
        tt._last_activity[key] = time.time() - 600
    _cleanup_inactive_envs(lifetime_seconds=300)


@pytest.fixture()
def env(monkeypatch):
    """A shared env cached under the collapsed key ``default`` (the raw task id ``session-42`` has
    no session key or isolation override), with only the sweeps a test fires."""
    monkeypatch.setattr(tt, "_active_environments", {})
    monkeypatch.setattr(tt, "_last_activity", {})
    monkeypatch.setattr(tt, "_creation_locks", {})
    monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
    reaping = _ReapingEnv()
    tt._active_environments["default"] = reaping
    tt._last_activity["default"] = time.time()
    return reaping


def test_a_foreground_terminal_call_keeps_its_environment_past_the_lifetime(env):
    result = json.loads(tt.terminal_tool(command="echo hi", task_id="session-42"))

    assert env.cleaned_mid_call is False
    assert tt._active_environments["default"] is env
    assert result["exit_code"] == 0
    # Idle again once the call returns: the countdown restarts from the call's end.
    _sweep_after_long_call()
    assert env.cleaned


def test_a_remote_execute_code_call_keeps_its_environment_past_the_lifetime(env):
    code_execution_tool._execute_remote("print(1)", "session-42", None)

    assert env.cleaned_mid_call is False
    assert tt._active_environments["default"] is env


def test_a_call_that_raises_releases_its_environment_to_the_reaper(env, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(tt, "_run_approval_guards", boom)
    tt.terminal_tool(command="echo hi", task_id="session-42")

    _sweep_after_long_call()
    assert env.cleaned
