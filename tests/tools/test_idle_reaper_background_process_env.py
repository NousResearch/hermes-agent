"""The idle reaper keeps an environment while a background process still runs inside it.

``terminal(background=true)`` on a remote backend (SSH, Modal, Daytona, Vercel, persistent
Docker) spawns into the environment cached under the collapsed container key (``default``
for a session-less CLI call) while the process is owned by the raw spawning task id. The
env poller talks to the sandbox through ``env.execute`` and never refreshes
``_last_activity``, so the reaper alone decides whether the sandbox survives an idle
``terminal.lifetime_seconds``: tearing it down kills the process the user started.
"""

from __future__ import annotations

import json
import time

import pytest

from tools import process_registry as process_registry_module
from tools import terminal_tool as tt
from tools.process_registry import ProcessRegistry
from tools.terminal_tool_lifecycle import _cleanup_inactive_envs


class _SandboxEnv:
    """Remote-backend stand-in: answers the launch wrapper with a PID and the poller's
    liveness probe with "alive" until the process is killed."""

    env_type = "ssh"

    def __init__(self):
        self.cwd = "/root"
        self.alive = True
        self.cleaned = False

    def get_temp_dir(self):
        return "/tmp"

    def cleanup(self):
        self.cleaned = True

    def execute(self, command, cwd="", **kwargs):
        if "echo $! >" in command:
            return {"output": "4242\n", "returncode": 0}
        if command.startswith("kill 4242"):
            self.alive = False
            return {"output": "", "returncode": 0}
        if "kill -0" in command:
            return {"output": "0\n" if self.alive else "1\n", "returncode": 0}
        if "hermes_bg_" in command and command.startswith("cat "):
            return {"output": "0\n", "returncode": 0}
        return {"output": "0 0\n", "returncode": 0}


@pytest.fixture()
def sandbox(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    monkeypatch.setattr(tt, "_active_environments", {})
    monkeypatch.setattr(tt, "_last_activity", {})
    monkeypatch.setattr(tt, "_creation_locks", {})
    monkeypatch.setattr(tt, "_calls_in_flight", {})
    monkeypatch.setattr(tt, "_session_cwd", {})
    monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
    registry = ProcessRegistry()
    monkeypatch.setattr(process_registry_module, "process_registry", registry)
    env = _SandboxEnv()
    key = tt._resolve_container_task_id("turn-1")
    tt._active_environments[key] = env
    tt._last_activity[key] = time.time()
    yield env, key, registry
    registry.kill_all()


def test_idle_reaper_keeps_a_remote_sandbox_that_hosts_a_running_background_process(sandbox):
    env, key, registry = sandbox
    result = json.loads(tt.terminal_tool(
        command="python3 -m http.server 8000", task_id="turn-1", background=True, force=True))
    assert result["error"] is None and result["session_id"]
    assert not registry.get(result["session_id"]).exited

    tt._last_activity[key] = time.time() - 600
    _cleanup_inactive_envs(lifetime_seconds=300)

    assert tt._active_environments.get(key) is env, (
        f"sandbox under {key!r} was reaped while background process {result['session_id']} runs in it")
    assert env.cleaned is False
