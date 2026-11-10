"""The idle reaper must keep an environment that hosts a running background process, and any
teardown of such an environment must settle the process instead of leaving its poller behind.

Production keys a session's sandbox ``session:<id>`` (``_active_environments`` / ``_last_activity``,
and ``ProcessSession.task_id``), while ``ProcessSession.owner_task_id`` holds the raw ``<id>``. A
reaper check on the owner id never matched the env key, so a sandbox with a running process was
closed after ``lifetime_seconds`` and its poller kept polling the closed environment.
"""

from __future__ import annotations

import time

import pytest

import tools.process_registry as process_registry_module
from tools import terminal_tool as tt
from tools.process_registry import ProcessRegistry, ProcessSession
from tools.terminal_tool_lifecycle import _cleanup_inactive_envs, _evict_environment_for_task, cleanup_vm

RAW_ID = "20261014_014945_b75b1f"
ENV_KEY = f"session:{RAW_ID}"


class _PolledForever(BaseException):
    """Escapes the poller's ``except Exception`` so a never-ending poll fails the test."""


class _SandboxEnv:
    """Remote env double. Running: answers the poller's reads with a live process. Closed: every
    command returns empty output instead of raising, which the poller reads as "no news" and
    polls again, as it did against a backend whose client was closed."""

    def __init__(self):
        self.closed = False
        self.calls_after_close = 0
        self.on_check = None

    def cleanup(self):
        self.closed = True

    def execute(self, command, **_kwargs):
        if self.closed:
            self.calls_after_close += 1
            if self.calls_after_close > 20:
                raise _PolledForever
            return {"output": "", "returncode": 1}
        if command.startswith("kill -0"):
            if self.on_check is not None:
                hook, self.on_check = self.on_check, None
                hook()
            return {"output": "0\n"}
        return {"output": "0 0\n"}


@pytest.fixture()
def registry(monkeypatch):
    reg = ProcessRegistry()
    monkeypatch.setattr(reg, "_write_checkpoint", lambda *_a, **_k: None)
    monkeypatch.setattr(process_registry_module, "process_registry", reg)
    monkeypatch.setattr(process_registry_module.time, "sleep", lambda _s: None)
    return reg


@pytest.fixture()
def env(monkeypatch):
    monkeypatch.setattr(tt, "_active_environments", {})
    monkeypatch.setattr(tt, "_last_activity", {})
    monkeypatch.setattr(tt, "_creation_locks", {})
    monkeypatch.setattr(tt, "_calls_in_flight", {})
    sandbox = _SandboxEnv()
    tt._active_environments[ENV_KEY] = sandbox
    tt._last_activity[ENV_KEY] = time.time() - 600
    return sandbox


def _background(registry, env, *, task_id=ENV_KEY, owner=RAW_ID, sid="proc_bg"):
    session = ProcessSession(
        id=sid, command="sleep 900", task_id=task_id, owner_task_id=owner, started_at=time.time(),
        pid=4242, pid_scope="sandbox", env_ref=env, notify_on_complete=True)
    registry._running[session.id] = session
    return session


def test_reaper_keeps_a_session_sandbox_with_a_running_process(registry, env):
    _background(registry, env)

    _cleanup_inactive_envs(lifetime_seconds=300)

    assert tt._active_environments.get(ENV_KEY) is env
    assert not env.closed


def test_reaper_keeps_the_sandbox_of_a_process_handed_off_to_another_key(registry, env):
    """A handoff re-keys the session to the new owner's env, but the process still runs in this one."""
    _background(registry, env, task_id="session:parent", owner="parent")

    _cleanup_inactive_envs(lifetime_seconds=300)

    assert tt._active_environments.get(ENV_KEY) is env


def test_reaper_reaps_the_new_owners_idle_sandbox_after_handoff(registry, env):
    """A handoff updates task_id but the running child remains in its original sandbox."""
    new_owner_key = "session:parent"
    unrelated_sandbox = _SandboxEnv()
    tt._active_environments[new_owner_key] = unrelated_sandbox
    tt._last_activity[new_owner_key] = time.time() - 600
    session = _background(registry, env, task_id=new_owner_key, owner="parent")

    _cleanup_inactive_envs(lifetime_seconds=300)

    assert tt._active_environments.get(ENV_KEY) is env
    assert not env.closed
    assert new_owner_key not in tt._active_environments
    assert unrelated_sandbox.closed
    assert session.id in registry._running and not session.exited


def test_reaper_closes_the_sandbox_once_its_process_has_exited(registry, env):
    session = _background(registry, env)
    registry._finish_exited(session, 0)

    _cleanup_inactive_envs(lifetime_seconds=300)

    assert ENV_KEY not in tt._active_environments
    assert env.closed


def test_reaper_closes_an_idle_sandbox_whose_only_process_lives_elsewhere(registry, env):
    _background(registry, object(), task_id="session:other", owner=RAW_ID)

    _cleanup_inactive_envs(lifetime_seconds=300)

    assert env.closed


@pytest.mark.parametrize("teardown", [cleanup_vm, _evict_environment_for_task])
def test_teardown_settles_a_running_process_and_stops_its_poller(registry, env, teardown):
    session = _background(registry, env)
    env.on_check = lambda: teardown(ENV_KEY)

    registry._env_poller_loop(session, env, "/tmp/bg.log", "/tmp/bg.pid", "/tmp/bg.exit")

    assert env.closed
    assert env.calls_after_close <= 2
    assert (session.exited, session.exit_code, session.completion_reason, session.termination_source) == (
        True, -1, "lost", "env_closed")
    assert session.id in registry._finished
    assert session._completion_event.is_set()


def test_reaper_keeps_a_sandbox_whose_process_matches_only_by_key(registry, env):
    """A process recovered from a checkpoint has no env object, only the env key it ran under."""
    _background(registry, None)

    _cleanup_inactive_envs(lifetime_seconds=300)

    assert tt._active_environments.get(ENV_KEY) is env
    assert not env.closed


def _completions(registry):
    events = []
    while not registry.completion_queue.empty():
        events.append(registry.completion_queue.get_nowait())
    return [(e["exit_code"], e["completion_reason"], e["termination_source"]) for e in events
            if e.get("type") == "completion"]


def _receipt(session):
    import json

    from hermes_constants import get_hermes_home

    record = json.loads((get_hermes_home() / "logs" / "process-results" / f"{session.id}.json").read_text())
    return record["exit_code"], record["completion_reason"], record["termination_source"]


def test_poller_seeing_exit_after_teardown_settled_the_process_keeps_the_settled_result(registry, env):
    """Teardown lands while the poller's liveness check is in flight; the check then reports the
    exit. The settled result was already published, so the poller must not change it."""
    session = _background(registry, env)
    real_execute = env.execute
    closed = []

    def execute(command, **kwargs):
        if command.startswith("kill -0") and not closed:
            cleanup_vm(ENV_KEY)  # teardown settles the session mid-RPC
            closed.append(True)
            return {"output": "1\n"}  # ...and the RPC that was already running reports the exit
        if command.startswith("cat ") and closed:
            return {"output": "0\n"}
        return real_execute(command, **kwargs)

    env.execute = execute
    registry._env_poller_loop(session, env, "/tmp/bg.log", "/tmp/bg.pid", "/tmp/bg.exit")

    lost = (-1, "lost", "env_closed")
    assert (session.exit_code, session.completion_reason, session.termination_source) == lost
    assert _completions(registry) == [lost]
    assert _receipt(session) == lost


class _ReleaseHook:
    """Wraps a lock; runs ``hook`` once, right after the next release."""

    def __init__(self, lock, hook):
        self._inner, self._hook = lock, hook

    def __enter__(self):
        return self._inner.__enter__()

    def __exit__(self, *exc):
        result = self._inner.__exit__(*exc)
        hook, self._hook = self._hook, None
        if hook is not None:
            hook()
        return result


def test_settlement_keeps_a_success_published_after_its_snapshot(registry, env):
    """The process exits on its own between settlement's snapshot of running sessions and its
    finish call. The success is published and durable; settlement must not turn it into lost."""
    session = _background(registry, env)
    lock = registry._lock
    natural = []

    def natural_exit():
        registry._lock = lock
        registry._finish_exited(session, 0)
        natural.append(session.id in registry._finished)

    from tools.process_registry_env_scope import settle_env_closed

    registry._lock = _ReleaseHook(lock, natural_exit)
    settled = settle_env_closed(registry, env)

    success = (0, "exited", "")
    assert natural == [True]  # the success was published before settlement's finish call
    assert (session.exit_code, session.completion_reason, session.termination_source) == success
    assert _completions(registry) == [success]
    assert _receipt(session) == success
    assert settled == 0
