"""Regression: a hello-validation failure must terminate and detach the
rejected child — never leak and adopt it."""

import json
import os
import sys
from pathlib import Path

import pytest

from tui_gateway.host_supervisor import HostSupervisor

_MISMATCH_CHILD = (
    "import json,time\n"
    "print(json.dumps({'type':'hello','hermes_home':'/tmp/homeB','build_sha':'abc'}),flush=True)\n"
    "time.sleep(2)\n"
)


def _make_supervisor(tmp_path):
    return HostSupervisor(
        registry_path=tmp_path / "reg.json",
        argv=[sys.executable, "-c", _MISMATCH_CHILD],
        expected_hermes_home="/tmp/homeA",
        expected_build_sha="abc",
        autostart=False,
    )


def test_hello_mismatch_terminates_and_detaches_child(tmp_path, monkeypatch):
    """Regression: _validate_hello raised while _proc stayed set, so
    is_running() returned True, the child kept running ownerless
    (start_new_session=True), and the next start() silently adopted the
    very host the guard had rejected.

    The termination path is asserted via a spy on _terminate_process (the
    real helper still runs): capturing the pid after start() raises is
    useless because the fix detaches _proc first."""
    import time as _time

    supervisor = _make_supervisor(tmp_path)
    terminated = []
    real_terminate = HostSupervisor._terminate_process

    def _spy(self, proc):
        terminated.append(proc.pid)
        return real_terminate(self, proc)

    monkeypatch.setattr(HostSupervisor, "_terminate_process", _spy)

    with pytest.raises(RuntimeError, match="HERMES_HOME mismatch"):
        supervisor.start()

    assert supervisor.is_running() is False
    assert supervisor._proc is None
    assert len(terminated) == 1, "rejection must terminate the child, not just detach"
    _time.sleep(3)  # child exits on its own ~2s after hello
    assert not Path(f"/proc/{terminated[0]}").exists(), (
        f"rejected child {terminated[0]} still running"
    )



def test_start_retries_fresh_after_mismatch(tmp_path):
    """After a rejected start, the supervisor must spawn a NEW process on
    the next start() (failing again on the same mismatch) — not early-return
    against the adopted child."""
    supervisor = _make_supervisor(tmp_path)
    with pytest.raises(RuntimeError):
        supervisor.start()
    with pytest.raises(RuntimeError):
        supervisor.start()  # respawned and rejected again — proves no adoption
    assert supervisor.is_running() is False


def test_multiplex_home_divergence_triggers_rejection_via_default_constructor(tmp_path):
    """Production-shaped divergence: the supervisor defaults
    expected_hermes_home from get_hermes_home() (ContextVar-sensitive,
    per tui_gateway/server.py:1449 which passes no explicit value), while
    the child reports os.environ HERMES_HOME. A profile override at
    construction time therefore rejects — and must terminate/detach."""
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    # Child reports the *launch* env home; supervisor captures the override.
    child = (
        "import json,time,os\n"
        "print(json.dumps({'type':'hello','hermes_home':os.environ['HERMES_HOME'],'build_sha':'abc'}),flush=True)\n"
        "time.sleep(2)\n"
    )
    token = set_hermes_home_override("/tmp/profileB-home")
    try:
        supervisor = HostSupervisor(
            registry_path=tmp_path / "reg.json",
            argv=[sys.executable, "-c", child],
            expected_build_sha="abc",
            autostart=False,
        )
    finally:
        reset_hermes_home_override(token)

    assert supervisor.expected_hermes_home == "/tmp/profileB-home"
    with pytest.raises(RuntimeError, match="HERMES_HOME mismatch"):
        supervisor.start()
    assert supervisor.is_running() is False
    assert supervisor._proc is None


_MATCHING_CHILD = (
    "import json,time,os\n"
    "print(json.dumps({'type':'hello','hermes_home':os.environ['HERMES_HOME'],'build_sha':'abc','boot_id':'b1'}),flush=True)\n"
    "time.sleep(2)\n"
)

_BUILD_MISMATCH_CHILD = (
    "import json,time,os\n"
    "print(json.dumps({'type':'hello','hermes_home':os.environ['HERMES_HOME'],'build_sha':'WRONGSHA'}),flush=True)\n"
    "time.sleep(2)\n"
)


def test_matching_hello_starts_and_persists_registry(tmp_path):
    """Happy-path control: a matching hello must start cleanly (the new
    try/except must not break normal startup) and persist the registry."""
    registry = tmp_path / "reg.json"
    supervisor = HostSupervisor(
        registry_path=registry,
        argv=[sys.executable, "-c", _MATCHING_CHILD],
        expected_build_sha="abc",
        autostart=False,
    )
    supervisor.start()
    assert supervisor.is_running() is True
    assert registry.exists()
    assert json.loads(registry.read_text())["host_pid"] == supervisor.pid
    supervisor.shutdown()
    assert supervisor.is_running() is False


def test_build_sha_mismatch_terminates_and_detaches(tmp_path):
    """The build-sha validation branch carries the same leak risk as the
    HERMES_HOME branch — rejected child must be terminated and detached."""
    supervisor = HostSupervisor(
        registry_path=tmp_path / "reg.json",
        argv=[sys.executable, "-c", _BUILD_MISMATCH_CHILD],
        expected_build_sha="abc",
        autostart=False,
    )
    with pytest.raises(RuntimeError, match="build mismatch"):
        supervisor.start()
    assert supervisor.is_running() is False
    assert supervisor._proc is None


def test_rejection_writes_no_registry(tmp_path):
    """A rejected child must not leave registry metadata behind — a stale
    registry would let a later boot reconcile onto a dead/mismatched host."""
    registry = tmp_path / "reg.json"
    supervisor = HostSupervisor(
        registry_path=registry,
        argv=[sys.executable, "-c", _MISMATCH_CHILD],
        expected_hermes_home="/tmp/homeA",
        expected_build_sha="abc",
        autostart=False,
    )
    with pytest.raises(RuntimeError):
        supervisor.start()
    assert not registry.exists()
