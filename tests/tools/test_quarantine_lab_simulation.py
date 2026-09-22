from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

from tools.external_skill_admission import _candidate_hash
from tools.quarantine_approval_models import (
    APPROVAL_BINDING_SCHEMA,
    ApprovalBinding,
    evidence_bundle_digest,
)
from tools.quarantine_dev_context import DevHumanApprovalContext
from tools.quarantine_dev_human_approval import derive_confirmation_code
from tools.quarantine_lab_simulation import (
    ALLOW_TO_LAB_SIMULATION,
    LabQuarantineSimulationSink,
    run_lab_quarantine_simulation,
)

FIXED_NOW = datetime(2026, 9, 19, 4, 0, tzinfo=timezone.utc)
ATTEMPT = "1" * 32


class FakeConsole:
    def __init__(self, answer, on_read=None):
        self.answer = answer
        self.on_read = on_read
        self.shown = []

    def is_interactive(self):
        return True

    def show_review(self, text):
        self.shown.append(text)

    def read_confirmation(self, prompt):
        if self.on_read:
            self.on_read()
        return self.answer


def safe_context():
    return DevHumanApprovalContext(
        environment="LAB",
        simulation_only=True,
        background=False,
        production_enforce=False,
        provider_execution_enabled=False,
        browser_execution_enabled=False,
        financial_execution_enabled=False,
        destructive_host_actions_enabled=False,
        active_production_config=False,
        active_skill_store_target=False,
    )


def fixture(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "SKILL.md").write_text("safe fixture\n", encoding="utf-8")
    sha = _candidate_hash(candidate)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "decision.json").write_text('{"decision":"QUARANTINE"}\n', encoding="utf-8")
    digest = evidence_bundle_digest(evidence)
    binding = ApprovalBinding(
        schema_version=APPROVAL_BINDING_SCHEMA,
        candidate_sha256=sha,
        source_id="owner/repo/demo",
        source_version="v1",
        gate_release_sha256="b" * 64,
        policy_version="policy-v1",
        hermes_guard_version="skills-guard-v5",
        cisco_version="cisco-v1",
        nvidia_version="nvidia-v1",
        scanner_completeness=(("hermes", True), ("cisco", True), ("nvidia", True)),
        scanner_reason_codes=(("hermes", ()), ("cisco", ("review",)), ("nvidia", ())),
        evidence_digest=digest,
    )
    external = SimpleNamespace(
        mode="enforce",
        decision="QUARANTINE",
        approval_binding=binding,
        candidate_sha256=sha,
        evidence_dir=str(evidence),
    )
    return candidate, binding, external


def states(root: Path):
    return [
        json.loads(p.read_text(encoding="utf-8"))["state"]
        for p in sorted(root.glob("*.json"))
    ]


def test_successful_flow_returns_only_lab_simulation_and_does_not_modify_candidate(tmp_path):
    candidate, binding, external = fixture(tmp_path)
    before = {p.relative_to(candidate).as_posix(): p.read_bytes() for p in candidate.rglob("*") if p.is_file()}
    code = derive_confirmation_code(binding, ATTEMPT)
    final_calls = []
    root = tmp_path / "audit"

    result = run_lab_quarantine_simulation(
        external,
        candidate,
        safe_context(),
        FakeConsole(f"APPROVE LAB {code}"),
        final_confirm=lambda summary: final_calls.append(summary) or True,
        source_id=binding.source_id,
        source_version=binding.source_version,
        install_attempt_id=ATTEMPT,
        audit_root=root,
        now=FIXED_NOW,
        monotonic=iter([1.0, 2.0]).__next__,
    )

    after = {p.relative_to(candidate).as_posix(): p.read_bytes() for p in candidate.rglob("*") if p.is_file()}
    assert result.allowed is True
    assert result.outcome == ALLOW_TO_LAB_SIMULATION
    assert result.reason == ALLOW_TO_LAB_SIMULATION
    assert len(final_calls) == 1
    assert "simulation only" in final_calls[0]
    assert before == after
    assert states(root) == ["requested", "approved", "simulation_completed"]


def test_final_confirmation_cancel_discards_dev_approval(tmp_path):
    candidate, binding, external = fixture(tmp_path)
    code = derive_confirmation_code(binding, ATTEMPT)
    root = tmp_path / "audit"
    result = run_lab_quarantine_simulation(
        external,
        candidate,
        safe_context(),
        FakeConsole(f"APPROVE LAB {code}"),
        final_confirm=lambda summary: False,
        install_attempt_id=ATTEMPT,
        audit_root=root,
        now=FIXED_NOW,
        monotonic=iter([1.0, 2.0]).__next__,
    )
    assert result.allowed is False
    assert result.reason == "DEV_FINAL_CONFIRMATION_CANCELLED"
    assert states(root) == ["requested", "approved", "final_confirmation_cancelled"]


def test_mutation_during_final_confirmation_is_rejected_before_simulation(tmp_path):
    candidate, binding, external = fixture(tmp_path)
    code = derive_confirmation_code(binding, ATTEMPT)
    root = tmp_path / "audit"

    def final_confirm(_summary):
        (candidate / "SKILL.md").write_text("mutated after first approval\n", encoding="utf-8")
        return True

    result = run_lab_quarantine_simulation(
        external,
        candidate,
        safe_context(),
        FakeConsole(f"APPROVE LAB {code}"),
        final_confirm=final_confirm,
        install_attempt_id=ATTEMPT,
        audit_root=root,
        now=FIXED_NOW,
        monotonic=iter([1.0, 2.0]).__next__,
    )
    assert result.allowed is False
    assert result.reason == "DEV_APPROVAL_BINDING_DRIFT_REJECTED"
    assert states(root) == ["requested", "approved", "binding_drift_rejected"]


def test_simulation_sink_rejects_production_context():
    bad = DevHumanApprovalContext(
        environment="PROD",
        simulation_only=False,
        background=False,
        production_enforce=True,
        provider_execution_enabled=False,
        browser_execution_enabled=False,
        financial_execution_enabled=False,
        destructive_host_actions_enabled=False,
        active_production_config=True,
        active_skill_store_target=True,
    )
    result = LabQuarantineSimulationSink().simulate(SimpleNamespace(approval_binding=None), bad)
    assert result.simulated is False
    assert result.outcome == "dev_context_environment_invalid"


def test_block_never_reaches_final_confirmation(tmp_path):
    candidate, binding, external = fixture(tmp_path)
    external.decision = "BLOCK"
    final_calls = []
    result = run_lab_quarantine_simulation(
        external,
        candidate,
        safe_context(),
        FakeConsole("APPROVE LAB " + derive_confirmation_code(binding, ATTEMPT)),
        final_confirm=lambda summary: final_calls.append(summary) or True,
        install_attempt_id=ATTEMPT,
    )
    assert result.allowed is False
    assert result.reason == "dev_approval_not_quarantine"
    assert final_calls == []


def test_candidate_python_payload_is_never_executed(tmp_path):
    from dataclasses import replace

    candidate, binding, external = fixture(tmp_path)
    sentinel = tmp_path / "EXECUTED.txt"
    (candidate / "payload.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(sentinel)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )
    new_sha = _candidate_hash(candidate)
    binding = replace(binding, candidate_sha256=new_sha)
    external.approval_binding = binding
    external.candidate_sha256 = new_sha
    code = derive_confirmation_code(binding, ATTEMPT)

    result = run_lab_quarantine_simulation(
        external,
        candidate,
        safe_context(),
        FakeConsole(f"APPROVE LAB {code}"),
        final_confirm=lambda summary: True,
        install_attempt_id=ATTEMPT,
        now=FIXED_NOW,
        monotonic=iter([1.0, 2.0]).__next__,
    )

    assert result.allowed is True
    assert result.outcome == ALLOW_TO_LAB_SIMULATION
    assert sentinel.exists() is False
