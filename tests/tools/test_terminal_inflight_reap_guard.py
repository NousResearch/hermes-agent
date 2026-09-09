"""Regression coverage: the idle reaper must not tear down an environment mid-call.

``_cleanup_inactive_envs`` reaps any task whose ``_last_activity`` is older than
``lifetime_seconds`` (``TERMINAL_LIFETIME_SECONDS`` / ``terminal.lifetime_seconds``, default 300).
``_last_activity`` is only stamped at call boundaries, and only BACKGROUND processes get refreshed
mid-run (through ``process_registry.has_active_processes``). So a foreground ``terminal`` or remote
``execute_code`` call that outlives the threshold was reaped while still using its own environment:

* container/remote backends (docker, modal, daytona, ssh, ...): ``env.cleanup()`` stops the sandbox
  the in-flight call is executing in;
* local: ``LocalEnvironment.cleanup()`` unlinks the session snapshot and cwd file, so the running
  command returns but the session's exported shell environment is gone — breaking the terminal
  tool's own contract that "exported environment variables persist between calls".

The window is wider than the command: the environment is acquired (and stamped) BEFORE the approval
guards, so a flagged command waiting on approval is idle by this measure too.
"""

from __future__ import annotations

import json
import time

import pytest

from tools import code_execution_tool, terminal_tool as tt
from tools.terminal_tool_lifecycle import _cleanup_inactive_envs


class _FakeEnv:
    """Cleanup-only stub; ``_ReapingEnv`` below adds the execute side."""

    def __init__(self):
        self.cleaned = False

    def cleanup(self):
        self.cleaned = True


@pytest.fixture()
def env_registry(monkeypatch):
    """Isolate the module-level lifecycle state; yields a register(task_id, idle_for) helper."""
    monkeypatch.setattr(tt, "_active_environments", {})
    monkeypatch.setattr(tt, "_last_activity", {})
    monkeypatch.setattr(tt, "_calls_in_flight", {})
    monkeypatch.setattr(tt, "_creation_locks", {})

    def register(task_id: str, idle_for: float) -> _FakeEnv:
        env = _FakeEnv()
        tt._active_environments[task_id] = env
        tt._last_activity[task_id] = time.time() - idle_for
        return env

    return register


def test_idle_environment_is_still_reaped(env_registry):
    """Baseline: the guard must not disable the reaper for genuinely idle tasks."""
    env = env_registry("t1", idle_for=600)

    _cleanup_inactive_envs(lifetime_seconds=300)

    assert env.cleaned
    assert "t1" not in tt._active_environments


def test_call_in_flight_survives_the_reaper(env_registry):
    """The whole point: a call longer than lifetime_seconds keeps its environment."""
    env = env_registry("t1", idle_for=600)

    with tt._call_in_flight("t1"):
        _cleanup_inactive_envs(lifetime_seconds=300)

        assert not env.cleaned
        assert tt._active_environments["t1"] is env


def test_environment_is_reapable_again_once_the_call_returns(env_registry):
    """The mark is a hold, not an exemption: the env must not leak past the call."""
    env = env_registry("t1", idle_for=600)
    with tt._call_in_flight("t1"):
        pass

    # Exiting restamps activity, so the countdown starts at call end, not call start.
    assert time.time() - tt._last_activity["t1"] < 5
    _cleanup_inactive_envs(lifetime_seconds=300)
    assert not env.cleaned

    tt._last_activity["t1"] = time.time() - 600
    _cleanup_inactive_envs(lifetime_seconds=300)
    assert env.cleaned


def test_parallel_calls_refcount_the_mark(env_registry):
    """Parallel tool calls in one session share a task_id; the first one out must not release it."""
    env = env_registry("t1", idle_for=600)

    with tt._call_in_flight("t1"):
        with tt._call_in_flight("t1"):
            pass
        tt._last_activity["t1"] = time.time() - 600
        _cleanup_inactive_envs(lifetime_seconds=300)

    assert not env.cleaned
    assert tt._calls_in_flight == {}


def test_both_the_collapsed_and_the_raw_task_id_are_held(env_registry):
    """``_lookup_active_env`` stamps whichever of the two keys owns the env, so both are marked:
    a per-session surface with a CWD-only override collapses to "default" while an env may already
    be cached under the originating task_id."""
    collapsed = env_registry("default", idle_for=600)
    raw = env_registry("session-42", idle_for=600)

    with tt._call_in_flight("default", "session-42"):
        _cleanup_inactive_envs(lifetime_seconds=300)

    assert not collapsed.cleaned and not raw.cleaned


def test_unrelated_tasks_are_unaffected(env_registry):
    """Holding one task must not keep every other idle sandbox alive."""
    held = env_registry("t1", idle_for=600)
    other = env_registry("t2", idle_for=600)

    with tt._call_in_flight("t1"):
        _cleanup_inactive_envs(lifetime_seconds=300)

    assert not held.cleaned
    assert other.cleaned


def test_none_and_duplicate_ids_are_dropped(env_registry):
    """``task_id`` is Optional and often equals the collapsed id; neither may corrupt the refcount."""
    env_registry("t1", idle_for=600)

    with tt._call_in_flight(None, "t1", "t1", ""):
        assert tt._calls_in_flight == {"t1": 1}

    assert tt._calls_in_flight == {}


# ---------------------------------------------------------------------------
# Through the real entry points, with the reaper firing from inside the call.
# ---------------------------------------------------------------------------


def _sweep_mid_call():
    """One reaper sweep from inside a call that has been running longer than ``lifetime_seconds``.

    Acquisition stamps ``_last_activity`` when the call STARTS and never again, so age every stamp
    past the threshold first: that is what the clock does while a long build, a slow script or an
    approval wait runs. Only the in-flight mark can save the environment now.
    """
    for key in list(tt._last_activity):
        tt._last_activity[key] = time.time() - 600
    _cleanup_inactive_envs(lifetime_seconds=300)


@pytest.fixture()
def held_env(env_registry, monkeypatch):
    """A shared environment cached under the COLLAPSED key ``default``, plus a no-op cleanup thread
    so only the sweeps a test fires can reap it.

    ``session-42`` is the raw tool-call ``task_id``; ``_resolve_container_task_id`` collapses it to
    ``default`` (no isolation override, no session key), which is where both acquisition paths cache
    and stamp. A guard that holds the raw id only leaves this environment exposed.
    """
    monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
    env_registry("default", idle_for=0)


class _ReapingEnv(_FakeEnv):
    """Sweeps from inside ``execute`` — a command that outlives ``lifetime_seconds``."""

    def __init__(self, log: list):
        super().__init__()
        self.log = log
        self.calls: list[str] = []

    def execute(self, command, cwd=None, timeout=None, **kwargs):
        self.calls.append(command)
        _sweep_mid_call()
        self.log.append(("execute", self.cleaned))
        return {"output": "hi", "exit_code": 0}


def test_terminal_tool_holds_its_environment_through_approval_and_execution(held_env, monkeypatch):
    """A foreground command that outlives ``lifetime_seconds`` must keep the environment it is
    running in. The reaper is fired twice from inside the call — once from the approval guards
    (a flagged command can wait there for minutes) and once from ``env.execute`` — because the
    mark has to span both, not just execution."""
    log: list = []
    env = _ReapingEnv(log)
    tt._active_environments["default"] = env
    real_guards = tt._run_approval_guards

    def guards(*args, **kwargs):
        _sweep_mid_call()
        log.append(("approval", env.cleaned))
        return real_guards(*args, **kwargs)

    monkeypatch.setattr(tt, "_run_approval_guards", guards)

    result = json.loads(tt.terminal_tool(command="echo hi", task_id="session-42"))

    assert [label for label, _ in log] == ["approval", "execute"]
    assert not any(cleaned for _, cleaned in log)
    assert not env.cleaned and tt._active_environments["default"] is env
    assert result["exit_code"] == 0


def test_execute_code_remote_holds_the_collapsed_environment_key(held_env):
    """``_execute_remote``'s own ``effective_task_id`` is just ``task_id or "default"``, while
    ``_get_or_create_env`` caches and stamps under the collapsed key. Holding the raw id alone let
    the reaper stop the sandbox the script was running in."""
    env = _ReapingEnv([])
    tt._active_environments["default"] = env

    code_execution_tool._execute_remote("print(1)", "session-42", None)

    assert env.calls, "the remote path never reached the environment"
    assert not env.cleaned
    assert tt._active_environments["default"] is env


def test_mark_is_released_when_the_call_raises(env_registry):
    """A tool call that dies must not pin its sandbox for the life of the process."""
    env = env_registry("t1", idle_for=600)

    with pytest.raises(RuntimeError):
        with tt._call_in_flight("t1"):
            raise RuntimeError("boom")

    assert tt._calls_in_flight == {}
    tt._last_activity["t1"] = time.time() - 600
    _cleanup_inactive_envs(lifetime_seconds=300)
    assert env.cleaned
