from __future__ import annotations

from pathlib import Path
import sys
import time

from workstation.supervisor import RecoveryPlane, RuntimeSupervisor, SupervisorState


class _FakeRuntime:
    def __init__(self, healthy: bool = True) -> None:
        self.healthy = healthy
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def test_supervisor_starts_checks_and_stops_runtime(tmp_path):
    runtimes: list[_FakeRuntime] = []

    def factory() -> _FakeRuntime:
        runtime = _FakeRuntime()
        runtimes.append(runtime)
        return runtime

    supervisor = RuntimeSupervisor(
        runtime_factory=factory,
        health_check=lambda runtime: runtime.healthy,
        state_path=tmp_path / "supervisor.json",
    )
    assert supervisor.start() is True
    assert supervisor.state == SupervisorState.RUNNING
    assert supervisor.health().healthy is True
    supervisor.stop()
    assert runtimes[0].stopped is True
    assert supervisor.state == SupervisorState.STOPPED


def test_supervisor_restores_last_good_artifact(tmp_path):
    artifact = tmp_path / "profile.json"
    artifact.write_text('{"version": 1}', encoding="utf-8")
    supervisor = RuntimeSupervisor(
        runtime_factory=lambda: _FakeRuntime(),
        health_check=lambda runtime: runtime.healthy,
        state_path=tmp_path / "supervisor.json",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    checkpoint = supervisor.checkpoint_artifact(artifact, label="known-good")
    artifact.write_text('{"version": 2}', encoding="utf-8")

    assert supervisor.restore_last_good(artifact) is True
    assert artifact.read_text(encoding="utf-8") == '{"version": 1}'
    assert Path(checkpoint.artifact_path).exists()


def test_recovery_plane_quarantines_optional_component(tmp_path):
    plane = RecoveryPlane(tmp_path / "recovery.json")
    assert plane.quarantine("plugin-x", "import failed") is True
    assert plane.is_quarantined("plugin-x") is True
    report = plane.diagnose({"plugin-x": lambda: True, "gateway": lambda: "ok"})
    assert report["plugin-x"]["status"] == "quarantined"
    assert report["gateway"]["status"] == "healthy"
    assert plane.restore("plugin-x") is True
    assert plane.is_quarantined("plugin-x") is False


def test_independent_subprocess_supervisor_recovers_a_stopped_runtime(tmp_path):
    supervisor = RuntimeSupervisor.from_command(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        state_path=tmp_path / "supervisor.json",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    assert supervisor.start()
    assert supervisor.health().healthy
    runtime = supervisor.runtime
    assert runtime is not None and runtime.process is not None
    runtime.process.terminate()
    runtime.process.wait(timeout=5)
    time.sleep(0.02)
    recovered = supervisor.watchdog_step()
    assert recovered.healthy
    assert supervisor.state == SupervisorState.RUNNING
    supervisor.stop()


def test_slow_backend_startup_keeps_same_process_until_ready(tmp_path):
    from workstation.supervisor import SubprocessRuntime
    marker = tmp_path / "ready"
    runtimes = []
    def factory():
        runtime = SubprocessRuntime([sys.executable, "-c",
            "import pathlib,sys,time; time.sleep(.25); pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(30)", str(marker)]).start()
        runtimes.append(runtime)
        return runtime
    supervisor = RuntimeSupervisor(runtime_factory=factory,
        health_check=lambda runtime: runtime.is_alive() and marker.exists(),
        state_path=tmp_path / "supervisor.json", startup_timeout_seconds=5)
    try:
        assert not supervisor.start()
        assert supervisor.state == SupervisorState.STARTING
        original_pid = supervisor.runtime.process.pid
        assert not supervisor.watchdog_step().healthy
        assert len(runtimes) == 1
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert supervisor.watchdog_step().healthy
        assert supervisor.runtime.process.pid == original_pid
        assert supervisor.state == SupervisorState.RUNNING
    finally:
        supervisor.stop()


def test_startup_deadline_fails_closed_without_unbounded_restart(tmp_path):
    supervisor = RuntimeSupervisor.from_command([sys.executable, "-c", "import time; time.sleep(30)"],
        state_path=tmp_path / "supervisor.json", startup_timeout_seconds=0)
    supervisor.health_check = lambda runtime: False
    try:
        assert not supervisor.start()
        assert supervisor.state == SupervisorState.FAILED
        assert supervisor.last_error == "runtime readiness deadline exceeded"
    finally:
        supervisor.stop()
