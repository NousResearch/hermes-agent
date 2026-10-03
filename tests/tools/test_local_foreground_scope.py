"""Foreground local commands get the same transient systemd scope as background ones.

#70716 closed the failure domain for `terminal(background=true)` executors: a worker that
breaches its cgroup limit is killed alone instead of taking the gateway (and the messaging
control plane) with it. A *foreground* command still ran inside the gateway's own cgroup, so
the same heavy build/test could still kill the control plane.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
from typing import cast

import pytest

from tools import process_registry
from tools.environments import local as local_env

# systemd transient scopes only exist on Linux (the code is gated on that host fact).
pytestmark = pytest.mark.platforms("linux")


class _FakeProc:
    """Minimal Popen stand-in: bookkeeping attributes only, no reader thread."""

    _hermes_scope_unit: str | None = None

    def __init__(self, pid: int = 4242):
        self.pid = pid
        self.stdout = self.stderr = self.stdin = None
        self.returncode = None

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0

    def kill(self):
        return None


@pytest.mark.parametrize("case", ["scoped", "private_bus", "not_the_gateway", "not_systemd", "no_scope", "no_wrapper",
                                  "bus_gone"])
def test_gateway_command_is_wrapped_recorded_and_given_the_bus_env(monkeypatch, caplog, case):
    seen: dict = {}
    monkeypatch.setattr(local_env.subprocess, "Popen",
                        lambda args, **kw: (seen.update(argv=list(args), kwargs=kw), _FakeProc())[1])
    monkeypatch.setattr(local_env, "_find_bash", lambda: "/bin/bash")
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: case != "not_the_gateway")
    monkeypatch.setattr(process_registry, "_systemd_run_user_scope_available", lambda: case != "no_scope")
    # "not_systemd" = an s6/Docker supervised gateway: no systemd unit, nothing to scope into.
    if case == "not_systemd":
        monkeypatch.delenv("INVOCATION_ID", raising=False)
    else:
        monkeypatch.setenv("INVOCATION_ID", "x")
    # "bus_gone" = the probe verdict is cached but the user bus vanished since; an address
    # inherited from the manager must not pass for a live bus.
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1/bus")
    bus = {} if case == "bus_gone" else {"DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1/bus"}
    monkeypatch.setattr(process_registry, "systemd_user_bus_env", lambda base: {**base, **bus})
    # Probe said yes; "no_wrapper" = systemd-run is gone from PATH by spawn time.
    real_which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: (
        None if case == "no_wrapper" else "/usr/bin/systemd-run") if name == "systemd-run" else real_which(name, *a, **k))
    monkeypatch.setattr(local_env, "_foreground_degraded_warned", False)
    # "private_bus" = this profile's Bot Desktop published its own Xfce session bus (#125830).
    from tools.bot_desktop import runtime as bot_desktop_runtime
    published = {"DBUS_SESSION_BUS_ADDRESS": "unix:path=/tmp/xfce-bus"} if case == "private_bus" else {}
    monkeypatch.setattr(bot_desktop_runtime, "published_env", lambda: published)
    # No real snapshot bootstrap (it would wait out its timeouts against the fake Popen) and
    # never a real `systemctl --user stop` from the kill path.
    monkeypatch.setattr(local_env.LocalEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(process_registry, "_stop_systemd_unit", lambda unit: True)
    env = local_env.LocalEnvironment()
    caplog.set_level("WARNING", logger=local_env.logger.name)
    monkeypatch.setattr(local_env, "_foreground_scope_issued", False)

    proc = env._run_bash("true")

    argv, kwargs = seen["argv"], seen["kwargs"]
    if case not in ("scoped", "private_bus"):
        assert argv == ["/bin/bash", "-c", "true"]
        assert getattr(proc, "_hermes_scope_unit", None) is None
        assert kwargs["env"] == local_env._make_run_env(env.env)
        assert local_env._foreground_scope_issued is False  # nothing for the host-exit sweep
        # Every fallback after the gateway check is a degraded failure domain, reported once.
        assert ("share the gateway cgroup" in caplog.text) is (case in ("no_scope", "no_wrapper", "bus_gone"))
        return
    assert argv[0].endswith("systemd-run")
    # systemd-run reaches the user manager, but the command keeps the bus it was given.
    own_bus = ["env", "DBUS_SESSION_BUS_ADDRESS=unix:path=/tmp/xfce-bus"] if case == "private_bus" else []
    assert argv[argv.index("--") + 1:] == [*own_bus, "/bin/bash", "-c", "true"]
    properties = [argv[i + 1] for i, token in enumerate(argv) if token == "--property"]
    assert "MemoryAccounting=yes" in properties
    # Own cgroup, but no background-worker cap: a big foreground build must not be OOM-killed.
    assert not any(p.startswith("MemoryMax=") for p in properties)
    # `--unit` takes the bare name; the recorded unit is what a kill path stops.
    assert proc._hermes_scope_unit == f"{argv[argv.index('--unit') + 1]}.scope"
    # The availability probe derives the user-bus variables, so the spawn must carry them
    # too or a system-level unit would fail where the probe succeeded.
    assert kwargs["env"]["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/run/user/1/bus"
    assert "share the gateway cgroup" not in caplog.text
    assert local_env._foreground_scope_issued is True  # arms the host-exit sweep
    # A scope left over from an earlier gateway with the same PID must not collide.
    assert env._run_bash("true")._hermes_scope_unit != proc._hermes_scope_unit


def test_scope_is_stopped_even_when_the_group_kill_raises_and_survives_adoption(monkeypatch, tmp_path):
    """The cgroup is the authoritative cleanup: an unexpected group-kill failure must not leak
    the transient unit, and a command yielded to the background keeps the unit to stop later."""
    stopped: list[str] = []

    def boom(proc):
        raise RuntimeError("group kill blew up")

    monkeypatch.setattr(local_env, "_kill_process_group_posix", boom)
    monkeypatch.setattr(process_registry, "_stop_systemd_unit",
                        lambda unit: stopped.append(unit) or True)
    monkeypatch.setattr(local_env.LocalEnvironment, "init_session", lambda self: None)
    env = local_env.LocalEnvironment()
    proc = _FakeProc(pid=4244)
    proc._hermes_scope_unit = "hermes-fg-4244-1.scope"
    proc.kill = lambda: stopped.append("parent")  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        env._kill_process(proc)
    # The parent is signalled before the cgroup is torn down (tools/AGENTS.md order).
    assert stopped == ["parent", "hermes-fg-4244-1.scope"]

    # Hard exit (no wait allowed): the scope is SIGKILLed without blocking on systemctl.
    spawned: list[list[str]] = []
    monkeypatch.setattr(local_env.os, "killpg", lambda *a: None)
    monkeypatch.setattr(local_env.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(local_env.shutil, "which", lambda name, *a, **k: f"/usr/bin/{name}")
    monkeypatch.setattr(local_env.subprocess, "Popen", lambda args, **kw: spawned.append(list(args)))
    env._force_kill_process(proc)
    assert spawned == [["/usr/bin/systemctl", "--user", "--no-block", "kill", "--signal=SIGKILL",
                        "hermes-fg-4244-1.scope"]]

    monkeypatch.setattr(process_registry, "CHECKPOINT_PATH", tmp_path / "processes.json")
    monkeypatch.setattr(process_registry.ProcessRegistry, "_track_started", lambda self, *a, **k: None)
    monkeypatch.setattr(process_registry.ProcessRegistry, "_safe_host_start_time", lambda self, pid: 0.0)
    session = process_registry.ProcessRegistry().adopt_local(
        cast("subprocess.Popen", proc), command="build", cwd=str(tmp_path))
    assert session.systemd_unit == "hermes-fg-4244-1.scope"

    # A command that exited normally can leave a daemonized descendant holding its scope,
    # outside the gateway cgroup: the host-exit funnel must stop this process's scopes.
    from tools import terminal_tool_lifecycle
    monkeypatch.setattr(terminal_tool_lifecycle, "_scratch_paths", lambda: [])
    monkeypatch.setattr(local_env, "_foreground_scope_issued", True)
    stopped.clear()
    terminal_tool_lifecycle.cleanup_all_environments()
    assert len(stopped) == 1
    assert fnmatch.fnmatchcase(f"hermes-fg-{os.getpid()}-0123abcd.scope", stopped[0])
    assert not fnmatch.fnmatchcase("hermes-fg-1-0123abcd.scope", stopped[0])  # another gateway's
