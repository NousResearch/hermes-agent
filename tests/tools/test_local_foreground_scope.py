"""Foreground local commands get the same transient systemd scope as background ones.

#70716 closed the failure domain for `terminal(background=true)` executors: a worker that
breaches its cgroup limit is killed alone instead of taking the gateway (and the messaging
control plane) with it. A *foreground* command still ran inside the gateway's own cgroup, so
the same heavy build/test could still kill the control plane.

Two layers here: cheap contracts for the argv/env the wrapper produces, and real end-to-end
runs through ``execute`` (the production path: base -> ``_run_bash``) that assert the command
actually lands in its own cgroup, that the environment reaches the child, and that the kill
path really stops the scope. Only the gateway-identity probe is injected where needed — the
availability probe stays real, so a host without a user bus skips instead of asserting a
fake.
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import cast

import pytest

from tools import process_registry
from tools.environments import base as base_env
from tools.environments import local as local_env

# systemd transient scopes only exist on Linux (the code is gated on that host fact).
pytestmark = pytest.mark.platforms("linux")


class _FakeProc:
    """Minimal Popen stand-in: bookkeeping attributes only, no reader thread."""

    # Set by the wrapper when the command really is scoped.
    _hermes_scope_unit: str | None = None

    def __init__(self, pid: int = 4242):
        self.pid = pid
        self.stdout = None
        self.stderr = None
        self.stdin = None
        self.returncode = None

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0

    def kill(self):
        return None


@pytest.fixture
def gateway_identity(monkeypatch):
    """This process is the supervised gateway. Injected because it is a host fact about who
    we are; every other probe and the wrapper itself stay real."""
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)


@pytest.fixture
def scoped_gateway(gateway_identity, monkeypatch):
    """Gateway identity plus a scope-capable host, for the argv-level contracts."""
    monkeypatch.setattr(process_registry, "_systemd_run_user_scope_available", lambda: True)


@pytest.fixture
def env():
    return local_env.LocalEnvironment()


@pytest.fixture
def spawns(monkeypatch):
    """Capture exactly what `_run_bash` hands to Popen."""
    seen: dict = {}

    def fake_popen(args, **kwargs):
        seen["argv"] = list(args)
        seen["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr(local_env.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(local_env, "_find_bash", lambda: "/bin/bash")
    return seen


def _payload(argv):
    """The wrapped command: everything after systemd-run's ``--``."""
    return argv[argv.index("--") + 1:]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _user_manager_reachable() -> bool:
    completed = subprocess.run(["systemctl", "--user", "show", "-p", "Version", "--value"],
                               capture_output=True, text=True, timeout=30)
    return completed.returncode == 0 and bool(completed.stdout.strip())


def _unit_active_state(unit: str) -> str | None:
    """ActiveState of *unit*, or None when the user manager could not be asked. An
    unreachable bus must never read as "the unit is gone" (that is a vacuous pass)."""
    if not _user_manager_reachable():
        return None
    completed = subprocess.run(["systemctl", "--user", "show", unit, "-p", "ActiveState", "--value"],
                               capture_output=True, text=True, timeout=30)
    return completed.stdout.strip() if completed.returncode == 0 else None


def test_gateway_command_is_wrapped_recorded_and_given_the_bus_env(scoped_gateway, env, spawns, monkeypatch):
    monkeypatch.setattr(
        process_registry, "systemd_user_bus_env",
        lambda base: {**base, "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1/bus"})

    proc = env._run_bash("true")

    argv, kwargs = spawns["argv"], spawns["kwargs"]
    assert argv[0].endswith("systemd-run")
    assert _payload(argv) == ["/bin/bash", "-c", "true"]
    properties = [argv[i + 1] for i, token in enumerate(argv) if token == "--property"]
    assert "MemoryAccounting=yes" in properties
    assert any(p.startswith("MemoryMax=") for p in properties)
    # `--unit` takes the bare name; the recorded unit is what a kill path stops.
    assert proc._hermes_scope_unit == f"{argv[argv.index('--unit') + 1]}.scope"
    # The availability probe derives the user-bus variables, so the spawn must carry them
    # too or a system-level unit would fail where the probe succeeded.
    assert kwargs["env"]["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/run/user/1/bus"
    # A scope does not create a session, so the wrapper keeps its own process group.
    assert kwargs["start_new_session"] is True


def test_stdin_and_output_contract_is_unchanged(scoped_gateway, env, spawns):
    env._run_bash("cat", stdin_data="hello")

    assert spawns["kwargs"]["stdin"] is subprocess.PIPE
    assert spawns["kwargs"]["stdout"] is subprocess.PIPE
    assert spawns["kwargs"]["stderr"] is subprocess.STDOUT


@pytest.mark.parametrize("unscoped_because", ["not_the_gateway", "no_scope", "no_wrapper"])
def test_falls_back_to_the_plain_command_and_reports_a_degraded_gateway(
        env, spawns, monkeypatch, unscoped_because):
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process",
                        lambda: unscoped_because != "not_the_gateway")
    monkeypatch.setattr(process_registry, "_systemd_run_user_scope_available",
                        lambda: unscoped_because != "no_scope")
    if unscoped_because == "no_wrapper":  # probe said yes, but systemd-run is gone from PATH
        monkeypatch.setattr(process_registry, "_build_systemd_scope_argv",
                            lambda argv, unit_suffix, *, prefix: list(argv))
    degraded: list[str] = []
    monkeypatch.setattr(local_env, "_warn_foreground_scope_degraded", degraded.append)
    plain_env = local_env._make_run_env(env.env)

    proc = env._run_bash("true")

    assert spawns["argv"] == ["/bin/bash", "-c", "true"]
    assert getattr(proc, "_hermes_scope_unit", None) is None
    assert spawns["kwargs"]["env"] == plain_env
    # Every fallback *after* the gateway check is a degraded failure domain: the command is
    # meant to be isolated, so the host problem is reported once (never once per command).
    assert bool(degraded) is (unscoped_because != "not_the_gateway")


def test_scope_is_stopped_even_when_the_group_kill_raises(monkeypatch):
    """The cgroup is the authoritative cleanup, so an unexpected group-kill failure must
    not leak a transient unit."""
    stopped: list[str] = []

    def boom(proc):
        raise RuntimeError("group kill blew up")

    monkeypatch.setattr(local_env, "_kill_process_group_posix", boom)
    monkeypatch.setattr(process_registry, "_stop_systemd_unit",
                        lambda unit: stopped.append(unit) or True)
    proc = _FakeProc(pid=4244)
    proc._hermes_scope_unit = "hermes-fg-4244-1.scope"

    with pytest.raises(RuntimeError):
        local_env.LocalEnvironment()._kill_process(proc)

    assert stopped == ["hermes-fg-4244-1.scope"]


def test_yielded_command_keeps_its_scope_unit(monkeypatch, tmp_path):
    """Adopting a running foreground command as a background session must carry the unit:
    otherwise a later `process kill` (and checkpoint recovery) has nothing to stop."""
    monkeypatch.setattr(process_registry, "CHECKPOINT_PATH", tmp_path / "processes.json")
    monkeypatch.setattr(process_registry.ProcessRegistry, "_track_started",
                        lambda self, *args, **kwargs: None)
    monkeypatch.setattr(process_registry.ProcessRegistry, "_safe_host_start_time",
                        lambda self, pid: 0.0)
    proc = _FakeProc(pid=4245)
    proc._hermes_scope_unit = "hermes-fg-4245-7.scope"

    session = process_registry.ProcessRegistry().adopt_local(
        cast("subprocess.Popen", proc), command="build", cwd=str(tmp_path))

    assert session.systemd_unit == "hermes-fg-4245-7.scope"


def test_adopting_an_unscoped_command_leaves_no_unit(monkeypatch, tmp_path):
    monkeypatch.setattr(process_registry, "CHECKPOINT_PATH", tmp_path / "processes.json")
    monkeypatch.setattr(process_registry.ProcessRegistry, "_track_started",
                        lambda self, *args, **kwargs: None)
    monkeypatch.setattr(process_registry.ProcessRegistry, "_safe_host_start_time",
                        lambda self, pid: 0.0)

    session = process_registry.ProcessRegistry().adopt_local(
        cast("subprocess.Popen", _FakeProc(pid=4246)), command="build", cwd=str(tmp_path))

    assert session.systemd_unit == ""


@pytest.fixture
def real_scope_probe(monkeypatch):
    """Force a fresh availability probe. Other tests in the suite exercise the
    unavailable path and latch the cached verdict, which would otherwise turn these
    end-to-end runs into silent skips depending on test order."""
    monkeypatch.setattr(process_registry, "_SYSTEMD_SCOPE_AVAILABLE", None)
    monkeypatch.setattr(process_registry, "_SYSTEMD_SCOPE_PROBED_AT", 0.0)
    if not process_registry._systemd_run_user_scope_available():
        pytest.skip("no reachable user systemd bus on this host")


def test_real_gateway_command_lands_in_its_own_scope_end_to_end(gateway_identity, real_scope_probe, tmp_path):
    """Through the production path (`execute` -> `_run_bash` -> Popen): the command runs in
    its own transient scope, its exit code survives, and its cwd is preserved."""
    env = local_env.LocalEnvironment(cwd=str(tmp_path), timeout=60)

    result = env.execute("cat /proc/self/cgroup; printf 'cwd=%s\\n' \"$PWD\"; exit 7", timeout=30)

    assert "hermes-fg-" in result["output"] and ".scope" in result["output"]
    assert f"cwd={tmp_path}" in result["output"]
    assert result["returncode"] == 7


@pytest.fixture
def no_wall_clock_backstop(monkeypatch):
    """``BaseEnvironment._kill_spawned_tree`` (the hard wait-bound backstop) reaps the whole
    tree on its own, which would hide whether the foreground kill path did anything. Stub it
    so this test measures the group kill + scope stop only."""
    monkeypatch.setattr(base_env.BaseEnvironment, "_kill_spawned_tree", lambda self, spawned: None)


def test_timeout_kill_reaps_a_detached_descendant_through_the_scope(
        gateway_identity, real_scope_probe, no_wall_clock_backstop, tmp_path):
    """A foreground command that outlives its timeout leaves nothing behind, and the
    transient unit is gone afterwards.

    Boundary, measured rather than assumed (mutation test: with the scope stop disabled
    this still passes): the group kill plus ``BaseEnvironment._kill_spawned_tree`` already
    cover a ``setsid`` descendant, so this asserts the user-visible guarantee — nothing
    survives — and NOT the scope stop's isolated contribution. That contribution is pinned
    by ``test_scope_is_stopped_even_when_the_group_kill_raises`` (which does fail under the
    same mutation) and by ``test_real_gateway_command_lands_in_its_own_scope_end_to_end``
    for the placement claim.
    """
    pid_file = tmp_path / "detached.pid"
    killed: list[str | None] = []
    original = local_env.LocalEnvironment._kill_process

    def spy(self, proc):
        killed.append(getattr(proc, "_hermes_scope_unit", None))
        return original(self, proc)

    local_env.LocalEnvironment._kill_process = spy
    try:
        result = local_env.LocalEnvironment(timeout=30).execute(
            f"setsid sleep 60 & echo $! > {pid_file}; sleep 60", timeout=2)
    finally:
        local_env.LocalEnvironment._kill_process = original
        # Never leave a transient unit behind if the assertions below fail.
        if killed and killed[0]:
            process_registry._stop_systemd_unit(killed[0])

    assert result["returncode"] == 124, result
    assert len(killed) == 1, killed
    unit = killed[0]
    assert unit and unit.startswith("hermes-fg-")
    detached_pid = int(pid_file.read_text().strip())

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and _pid_alive(detached_pid):
        time.sleep(0.5)
    assert not _pid_alive(detached_pid), f"detached descendant {detached_pid} outlived the scope stop"

    state = _unit_active_state(unit)
    if state is None:
        pytest.skip("user manager unreachable, cannot confirm the unit is gone")
    assert state not in ("active", "activating"), f"{unit} survived the timeout kill ({state})"
