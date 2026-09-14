from __future__ import annotations

import pytest

from workstation.isolation import (
    BoundedMCPExecutor,
    ComponentBootManager,
    ControlPlaneMonitor,
    ExtensionQualification,
    QualificationStage,
    ReleaseQualificationGate,
    TrustLabel,
)
from workstation.runtime import CancellationToken


def test_control_plane_suspends_new_external_surface():
    monitor = ControlPlaneMonitor(allowed_destinations={"localhost"})
    decision = monitor.observe_network("api.example.com", task_id="task-1")
    assert decision.decision == "suspend"
    assert decision.reason
    assert monitor.incident_trace[-1]["task_id"] == "task-1"


def test_control_plane_observes_action_spend_and_permission_expansion():
    monitor = ControlPlaneMonitor(allowed_destinations={"localhost"})
    assert monitor.observe_action("read_page", task_id="task-2", allowed_actions=("read_page",)).decision == "allow"
    assert monitor.observe_spend(0.11, task_id="task-2", budget_usd=0.10).decision == "suspend"
    assert monitor.observe_permission("write_file", task_id="task-2", declared_permissions=("read_file",)).decision == "suspend"
    assert len(monitor.incident_trace) == 3


def test_degraded_boot_quarantines_broken_optional_component():
    manager = ComponentBootManager()
    report = manager.boot({"core": lambda: True, "bad-plugin": lambda: 1 / 0})
    assert report["core"].status == "ready"
    assert report["bad-plugin"].status == "quarantined"
    assert manager.is_available("core") is True
    assert manager.is_available("bad-plugin") is False


def test_recovery_plane_quarantine_is_honored_by_degraded_boot(tmp_path):
    from workstation.supervisor import RecoveryPlane

    plane = RecoveryPlane(tmp_path / "recovery.json")
    plane.quarantine("bad-plugin", "known failure")
    report = ComponentBootManager(plane).boot({"bad-plugin": lambda: True})
    assert report["bad-plugin"].status == "quarantined"


def test_mcp_executor_has_page_tool_response_and_time_bounds():
    executor = BoundedMCPExecutor(max_pages=2, max_tools=2, max_response_bytes=20)
    with pytest.raises(ValueError, match="page limit"):
        executor.discover(lambda page: ["tool-a"], pages=3)
    with pytest.raises(ValueError, match="response limit"):
        executor.execute(lambda: "x" * 21)
    assert executor.discover(lambda page: ["tool-a"], pages=2) == ["tool-a", "tool-a"]
    with pytest.raises(ValueError, match="negative"):
        executor.discover(lambda page: [], pages=-1)
    token = CancellationToken()
    token.cancel("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        executor.discover(lambda page: ["tool-a"], pages=1, cancellation=token)


def test_extension_qualification_requires_all_safety_stages():
    qualification = ExtensionQualification()
    result = qualification.run(
        source="https://example.com/plugin.zip",
        requested_capabilities=("read_page",),
        checks={stage: (lambda: True) for stage in QualificationStage},
        trust=TrustLabel.UNTRUSTED,
    )
    assert result.accepted is True
    assert result.stages == list(QualificationStage)


def test_release_qualification_is_fail_closed_until_every_gate_passes():
    gate = ReleaseQualificationGate()
    checks = {stage: (lambda: True) for stage in gate.STAGES}
    assert gate.run(checks)["accepted"] is True
    checks["native_smoke"] = lambda: False
    result = gate.run(checks)
    assert result["accepted"] is False
    assert result["failed_stage"] == "native_smoke"


def test_release_qualification_runner_preserves_stage_evidence(tmp_path):
    from workstation.release_qualification import ReleaseQualificationRunner

    runner = ReleaseQualificationRunner(tmp_path)
    checks = {
        stage: (lambda stage=stage: True)
        for stage in ReleaseQualificationGate.STAGES
    }
    report = runner.run(checks=checks)

    assert report.accepted is True
    assert [stage.name for stage in report.stages] == list(ReleaseQualificationGate.STAGES)
    assert all(stage.passed for stage in report.stages)
    assert report.to_dict()["stages"][-1]["name"] == "migration"


def test_release_qualification_runner_stops_after_first_failed_stage(tmp_path):
    from workstation.release_qualification import ReleaseQualificationRunner

    calls: list[str] = []

    def check(stage: str, passed: bool):
        def run():
            calls.append(stage)
            return passed

        return run

    checks = {
        stage: check(stage, stage != "clean_install")
        for stage in ReleaseQualificationGate.STAGES
    }
    report = ReleaseQualificationRunner(tmp_path).run(checks=checks)

    assert report.accepted is False
    assert report.failed_stage == "clean_install"
    assert calls == ["ownership", "assets", "clean_install"]
    assert [stage.name for stage in report.stages] == calls


def test_clean_install_evidence_must_match_candidate_revision(tmp_path):
    from workstation.release_qualification import ReleaseQualificationRunner

    evidence = tmp_path / "clean-install.json"
    evidence.write_text(
        '{"accepted": true, "candidate_revision": "different", "workstation_home": "C:/clean", "stages": ["clean_install"]}',
        encoding="utf-8",
    )
    runner = ReleaseQualificationRunner(tmp_path)
    result = runner._check_clean_install(evidence, "candidate")

    assert result.passed is False
    assert "revision" in result.details


def test_clean_install_evidence_requires_workstation_home(tmp_path):
    from workstation.release_qualification import ReleaseQualificationRunner

    evidence = tmp_path / "clean-install.json"
    evidence.write_text(
        '{"accepted": true, "candidate_revision": "candidate", "stages": ["clean_install"]}',
        encoding="utf-8",
    )
    runner = ReleaseQualificationRunner(tmp_path)
    result = runner._check_clean_install(evidence, "candidate")

    assert result.passed is False
    assert "workstation_home" in result.details


def test_clean_install_evidence_accepts_matching_revision_and_home(tmp_path):
    from workstation.release_qualification import ReleaseQualificationRunner

    evidence = tmp_path / "clean-install.json"
    evidence.write_text(
        '{"accepted": true, "candidate_revision": "candidate", "workstation_home": "C:/clean", "stages": ["clean_install"]}',
        encoding="utf-8",
    )
    runner = ReleaseQualificationRunner(tmp_path)
    result = runner._check_clean_install(evidence, "candidate")

    assert result.passed is True
    assert "candidate" in result.details


@pytest.mark.parametrize(
    ("workstation_home", "expected"),
    [(".", "absolute"), ("candidate-home", "absolute")],
)
def test_clean_install_evidence_rejects_relative_home(tmp_path, workstation_home, expected):
    from workstation.release_qualification import ReleaseQualificationRunner

    evidence = tmp_path / "clean-install.json"
    evidence.write_text(
        '{"accepted": true, "candidate_revision": "candidate", '
        f'"workstation_home": "{workstation_home}", "stages": ["clean_install"]}}',
        encoding="utf-8",
    )
    result = ReleaseQualificationRunner(tmp_path)._check_clean_install(evidence, "candidate")

    assert result.passed is False
    assert expected in result.details


def test_clean_install_evidence_rejects_home_inside_candidate(tmp_path):
    from workstation.release_qualification import ReleaseQualificationRunner

    evidence = tmp_path / "clean-install.json"
    inside = tmp_path / "candidate-home"
    evidence.write_text(
        '{"accepted": true, "candidate_revision": "candidate", '
        f'"workstation_home": "{inside.as_posix()}", "stages": ["clean_install"]}}',
        encoding="utf-8",
    )
    result = ReleaseQualificationRunner(tmp_path)._check_clean_install(evidence, "candidate")

    assert result.passed is False
    assert "outside the candidate checkout" in result.details
