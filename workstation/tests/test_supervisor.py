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
