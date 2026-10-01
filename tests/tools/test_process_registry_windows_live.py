"""LIVE Windows E2E for background-executor spawn parity (#70716 / PR salvage).

Runs ONLY on a real Windows host (the on-demand ``windows-venv-e2e.yml``
lane). The systemd cgroup-isolation feature for local background executors
must be a strict no-op on Windows: jobs spawn exactly as before, output is
captured, exit codes are correct, and no systemd code path is ever reached
— even when the process claims gateway identity.

These tests drive the REAL ``ProcessRegistry.spawn_local`` pipe path on the
live Windows process table (real Popen, real Git Bash shell, real reader
thread) — no mocked spawn.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="live Windows background-executor E2E"
)


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    import tools.process_registry as pr

    reg = pr.ProcessRegistry()
    yield reg
    for sid in list(reg._running):
        try:
            reg.kill_process(sid)
        except Exception:
            pass


def _wait_exit(reg, sid, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        sess = reg._finished.get(sid) or reg._running.get(sid)
        if sess is not None and sess.exited:
            return sess
        time.sleep(0.2)
    raise AssertionError(f"session {sid} did not exit within {timeout}s")


class TestWindowsSpawnParity:
    def test_background_job_runs_output_and_exit_code_unchanged(self, registry):
        """Plain background job: spawned, output captured, exit code correct."""
        session = registry.spawn_local("echo win-live-parity; exit 7")
        done = _wait_exit(registry, session.id)

        assert done.exit_code == 7
        assert "win-live-parity" in done.output_buffer
        # The systemd scope identity must never be recorded on Windows.
        assert done.systemd_unit == ""

    def test_gateway_identity_never_reaches_systemd_path_on_windows(
        self, registry, monkeypatch
    ):
        """Even with full (faked) gateway identity, the Windows spawn takes
        the legacy path: no scope argv is built, no probe runs, and the job
        behaves exactly as without the identity."""
        import tools.process_registry as pr

        monkeypatch.setenv("_HERMES_GATEWAY", "1")
        monkeypatch.setattr(
            "gateway.status.get_running_pid",
            lambda *, cleanup_stale=False: os.getpid(),
        )
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process", lambda: True
        )
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", None)

        scope_builds = []
        monkeypatch.setattr(
            pr,
            "_build_systemd_scope_argv",
            lambda *a, **k: scope_builds.append(a) or a[0],
        )

        session = registry.spawn_local("echo win-live-gateway; exit 3")
        done = _wait_exit(registry, session.id)

        assert done.exit_code == 3
        assert "win-live-gateway" in done.output_buffer
        assert done.systemd_unit == ""
        assert scope_builds == [], "Windows must never build a systemd scope argv"
        # The availability probe must not have flipped to True on Windows.
        assert pr._SYSTEMD_SCOPE_AVAILABLE is not True

    def test_kill_process_windows_plain_path(self, registry):
        """kill_process on Windows works without any systemd unit cleanup."""
        session = registry.spawn_local("sleep 60")
        time.sleep(1.0)
        result = registry.kill_process(session.id)
        assert result.get("status") in {"killed", "already_exited"}
        assert session.systemd_unit == ""
