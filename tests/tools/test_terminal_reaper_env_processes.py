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

import hermes_agent.tools.process_registry as process_registry_module
from hermes_agent.tools import terminal_tool as tt
from hermes_agent.tools.process_registry import ProcessRegistry, ProcessSession
from hermes_agent.tools.terminal_tool_lifecycle import _cleanup_inactive_envs, _evict_environment_for_task, cleanup_vm

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
def env_cache(monkeypatch):
    for name in ("_active_environments", "_last_activity", "_creation_locks", "_calls_in_flight"):
        monkeypatch.setattr(tt, name, {})


@pytest.fixture()
def env(env_cache):
    sandbox = _SandboxEnv()
    tt._active_environments[ENV_KEY] = sandbox
    tt._last_activity[ENV_KEY] = time.time() - 600
    return sandbox


def _background(registry, env, *, task_id=ENV_KEY, owner=RAW_ID, sid="proc_bg"):
    session = ProcessSession(
        id=sid, command="sleep 900", task_id=task_id, owner_task_id=owner, started_at=time.time(),
        pid=4242, pid_scope="sandbox" if env is not None else "host", env_ref=env, notify_on_complete=True)
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


class _ReconnectingEnv(_SandboxEnv):
    """SSH-like: cleanup() drops the connection, the job keeps running, the next execute() reconnects."""

    cleanup_stops_processes = False
    cleanups = 0

    def cleanup(self):
        self.cleanups += 1

    def execute(self, command, **kwargs):
        if command.startswith("kill -0") and self.cleanups:
            return {"output": "1\n"}  # the job finished after the teardown
        if command.startswith("cat ") and self.cleanups:
            return {"output": "7\n"}
        return super().execute(command, **kwargs)


def test_ssh_backend_declares_that_cleanup_leaves_its_processes_running():
    from hermes_agent.tools.environments.base import BaseEnvironment
    from hermes_agent.tools.environments.ssh import SSHEnvironment

    assert BaseEnvironment.cleanup_stops_processes is True
    assert SSHEnvironment.cleanup_stops_processes is False


@pytest.mark.parametrize("teardown", [cleanup_vm, _evict_environment_for_task])
def test_teardown_of_a_backend_whose_processes_survive_cleanup_keeps_polling(registry, env_cache, teardown):
    """SSH cleanup only closes ControlMaster sockets: the remote job runs on and its poller must
    still collect the real exit code rather than a settled ``lost``."""
    ssh_like = _ReconnectingEnv()
    tt._active_environments[ENV_KEY] = ssh_like
    session = _background(registry, ssh_like)

    teardown(ENV_KEY)
    assert not session.exited
    registry._env_poller_loop(session, ssh_like, "/tmp/bg.log", "/tmp/bg.pid", "/tmp/bg.exit")

    assert ssh_like.cleanups == 1
    assert (session.exit_code, session.completion_reason, session.termination_source) == (7, "exited", "")


class _CleanupFails(_SandboxEnv):
    def cleanup(self):
        raise RuntimeError("sync-back failed before terminate")


@pytest.mark.parametrize("teardown", [cleanup_vm, _evict_environment_for_task])
def test_failed_teardown_does_not_settle_processes_of_a_sandbox_still_running(registry, env_cache, teardown):
    """A cleanup that raised (Modal's sync-back before terminate) may leave the sandbox up."""
    failing = _CleanupFails()
    tt._active_environments[ENV_KEY] = failing
    session = _background(registry, failing)

    teardown(ENV_KEY)

    assert not session.exited
    assert session.id in registry._running


def test_reaper_keeps_a_sandbox_whose_process_matches_only_by_key(registry, env):
    """A host process (local, or recovered from a checkpoint) has no env object, only the env key
    it ran under. Checkpoint recovery skips sandbox processes, so only host ones reach here."""
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

    from hermes_agent.hermes_constants import get_hermes_home

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


def test_settlement_keeps_a_success_published_after_its_snapshot(registry, env, monkeypatch):
    """The process exits on its own between settlement's snapshot of running sessions and its
    finish call. The success is published and durable; settlement must not turn it into lost."""
    session = _background(registry, env)
    real_finish = registry._finish_exited
    natural = []

    def finish_after_natural_exit(s, *args, **kwargs):
        if not natural:  # the process exits on its own right before settlement's finish call
            real_finish(s, 0)
            natural.append(s.id in registry._finished)
        return real_finish(s, *args, **kwargs)

    monkeypatch.setattr(registry, "_finish_exited", finish_after_natural_exit)
    settled = registry.settle_env_closed(env)

    success = (0, "exited", "")
    assert natural == [True]  # the success was published before settlement's finish call
    assert (session.exit_code, session.completion_reason, session.termination_source) == success
    assert _completions(registry) == [success]
    assert _receipt(session) == success
    assert settled == 0
