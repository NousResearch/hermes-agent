from dataclasses import replace
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
from tools.quarantine_lab_integration import DevLabQuarantineIntegration
from tools.quarantine_lab_simulation import ALLOW_TO_LAB_SIMULATION

ATTEMPT = "3" * 32


class FakeConsole:
    def __init__(self, answer):
        self.answer = answer
        self.shown = []

    def is_interactive(self):
        return True

    def show_review(self, text):
        self.shown.append(text)

    def read_confirmation(self, prompt):
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


def fixture(tmp_path: Path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "SKILL.md").write_text("safe fixture\n", encoding="utf-8")
    sha = _candidate_hash(candidate)

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "decision.json").write_text(
        '{"decision":"QUARANTINE"}\n', encoding="utf-8"
    )
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
        scanner_completeness=(
            ("hermes", True),
            ("cisco", True),
            ("nvidia", True),
        ),
        scanner_reason_codes=(
            ("hermes", ()),
            ("cisco", ("review",)),
            ("nvidia", ()),
        ),
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


def test_success_returns_terminal_lab_simulation_without_candidate_execution(tmp_path):
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

    before = {
        p.relative_to(candidate).as_posix(): p.read_bytes()
        for p in candidate.rglob("*")
        if p.is_file()
    }
    code = derive_confirmation_code(binding, ATTEMPT)
    integration = DevLabQuarantineIntegration(
        context=safe_context(),
        review_console=FakeConsole(f"APPROVE LAB {code}"),
        final_confirm=lambda _summary: True,
        install_attempt_id=ATTEMPT,
    )

    result = integration.simulate(
        external,
        candidate,
        source_id=binding.source_id,
        source_version=binding.source_version,
    )

    after = {
        p.relative_to(candidate).as_posix(): p.read_bytes()
        for p in candidate.rglob("*")
        if p.is_file()
    }
    assert result.handled is True
    assert result.allowed is True
    assert result.outcome == ALLOW_TO_LAB_SIMULATION
    assert result.reason == ALLOW_TO_LAB_SIMULATION
    assert result.candidate_sha256 == new_sha
    assert result.evidence_digest == binding.evidence_digest
    assert before == after
    assert sentinel.exists() is False


def test_wrong_phrase_rejects(tmp_path):
    candidate, binding, external = fixture(tmp_path)
    integration = DevLabQuarantineIntegration(
        context=safe_context(),
        review_console=FakeConsole("APPROVE LAB 000000000000"),
        final_confirm=lambda _summary: True,
        install_attempt_id=ATTEMPT,
    )
    result = integration.simulate(
        external,
        candidate,
        source_id=binding.source_id,
        source_version=binding.source_version,
    )
    assert result.allowed is False
    assert result.reason == "DEV_APPROVAL_REJECTED"


def test_final_confirmation_cancel_rejects(tmp_path):
    candidate, binding, external = fixture(tmp_path)
    code = derive_confirmation_code(binding, ATTEMPT)
    integration = DevLabQuarantineIntegration(
        context=safe_context(),
        review_console=FakeConsole(f"APPROVE LAB {code}"),
        final_confirm=lambda _summary: False,
        install_attempt_id=ATTEMPT,
    )
    result = integration.simulate(
        external,
        candidate,
        source_id=binding.source_id,
        source_version=binding.source_version,
    )
    assert result.allowed is False
    assert result.reason == "DEV_FINAL_CONFIRMATION_CANCELLED"


def test_non_quarantine_never_reads_console(tmp_path):
    candidate, _binding, external = fixture(tmp_path)
    external.decision = "BLOCK"

    class MustNotPrompt(FakeConsole):
        def read_confirmation(self, prompt):
            raise AssertionError("non-QUARANTINE must not prompt")

    integration = DevLabQuarantineIntegration(
        context=safe_context(),
        review_console=MustNotPrompt(""),
        final_confirm=lambda _summary: (_ for _ in ()).throw(
            AssertionError("non-QUARANTINE must not final-confirm")
        ),
    )
    result = integration.simulate(external, candidate)
    assert result.allowed is False
    assert result.reason == "dev_approval_not_quarantine"


def test_invalid_production_context_rejects_before_prompt(tmp_path):
    candidate, _binding, external = fixture(tmp_path)
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

    class MustNotPrompt(FakeConsole):
        def read_confirmation(self, prompt):
            raise AssertionError("invalid context must not prompt")

    integration = DevLabQuarantineIntegration(
        context=bad,
        review_console=MustNotPrompt(""),
        final_confirm=lambda _summary: False,
    )
    result = integration.simulate(external, candidate)
    assert result.allowed is False
    assert result.reason == "dev_context_environment_invalid"


def test_timeout_above_frozen_maximum_rejects_before_prompt(tmp_path):
    candidate, _binding, external = fixture(tmp_path)

    class MustNotPrompt(FakeConsole):
        def read_confirmation(self, prompt):
            raise AssertionError("invalid timeout must not prompt")

    integration = DevLabQuarantineIntegration(
        context=safe_context(),
        review_console=MustNotPrompt(""),
        final_confirm=lambda _summary: False,
        timeout_seconds=120.001,
    )
    result = integration.simulate(external, candidate)
    assert result.allowed is False
    assert result.reason == "dev_lab_timeout_invalid"
