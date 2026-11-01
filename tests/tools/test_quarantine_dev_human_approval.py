from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.external_skill_admission import _candidate_hash
from tools.quarantine_approval_models import (
    APPROVAL_BINDING_SCHEMA,
    ApprovalBinding,
    canonical_json_bytes,
    evidence_bundle_digest,
)
from tools.quarantine_dev_context import DevHumanApprovalContext
from tools.quarantine_dev_human_approval import (
    derive_confirmation_code,
    request_dev_human_approval,
)

FIXED_NOW = datetime(2026, 9, 19, 4, 0, tzinfo=timezone.utc)
ATTEMPT = "1" * 32


class FakeConsole:
    def __init__(self, answer="", *, interactive=True, on_read=None):
        self.answer = answer
        self.interactive = interactive
        self.on_read = on_read
        self.shown = []
        self.prompts = []

    def is_interactive(self):
        return self.interactive

    def show_review(self, text):
        self.shown.append(text)

    def read_confirmation(self, prompt):
        self.prompts.append(prompt)
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


def make_fixture(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "SKILL.md").write_text("safe fixture\n", encoding="utf-8")
    candidate_sha = _candidate_hash(candidate)

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "decision.json").write_text(
        '{"decision":"QUARANTINE","reason":"reference_unresolved"}\n',
        encoding="utf-8",
    )
    evidence_digest = evidence_bundle_digest(evidence)
    binding = ApprovalBinding(
        schema_version=APPROVAL_BINDING_SCHEMA,
        candidate_sha256=candidate_sha,
        source_id="owner/repo/demo",
        source_version="v1",
        gate_release_sha256="b" * 64,
        policy_version="skill-admission-gate-v0.1",
        hermes_guard_version="skills-guard-v5",
        cisco_version="skill-scanner 2.1.0",
        nvidia_version="2.11.2",
        scanner_completeness=(
            ("hermes", True),
            ("cisco", False),
            ("nvidia", True),
        ),
        scanner_reason_codes=(
            ("hermes", ()),
            ("cisco", ("reference_unresolved",)),
            ("nvidia", ()),
        ),
        evidence_digest=evidence_digest,
        decision="QUARANTINE",
    )
    external = SimpleNamespace(
        mode="enforce",
        decision="QUARANTINE",
        approval_binding=binding,
        candidate_sha256=candidate_sha,
        evidence_dir=str(evidence),
    )
    return candidate, evidence, binding, external


def audit_states(root: Path):
    if not root.exists():
        return []
    return [
        json.loads(path.read_text(encoding="utf-8"))["state"]
        for path in sorted(root.glob("*.json"))
    ]


def test_confirmation_code_is_exact_binding_and_attempt_sensitive(tmp_path):
    _, _, binding, _ = make_fixture(tmp_path)
    base = derive_confirmation_code(binding, ATTEMPT)
    assert len(base) == 12
    assert base == base.upper()
    int(base, 16)

    mutations = [
        replace(binding, candidate_sha256="a" * 64),
        replace(binding, evidence_digest="c" * 64),
        replace(binding, source_id="other/source"),
        replace(binding, source_version="v2"),
        replace(binding, policy_version="policy-v2"),
        replace(binding, cisco_version="scanner-v3"),
    ]
    for changed in mutations:
        assert derive_confirmation_code(changed, ATTEMPT) != base
    assert derive_confirmation_code(binding, "2" * 32) != base


def test_exact_foreground_phrase_approves_and_audits(tmp_path):
    candidate, _, binding, external = make_fixture(tmp_path)
    code = derive_confirmation_code(binding, ATTEMPT)
    console = FakeConsole(f"APPROVE LAB {code}")
    root = tmp_path / "audit"

    result = request_dev_human_approval(
        external,
        candidate,
        safe_context(),
        console,
        source_id=binding.source_id,
        source_version=binding.source_version,
        install_attempt_id=ATTEMPT,
        audit_root=root,
        now=FIXED_NOW,
        monotonic=iter([10.0, 11.0]).__next__,
    )

    assert result.approved is True
    assert result.reason == "DEV_APPROVAL_APPROVED"
    assert result.install_attempt_id == ATTEMPT
    assert result.confirmation_code == code
    assert audit_states(root) == ["requested", "approved"]
    assert "DEVELOPMENT LAB CONFIRMATION" in console.shown[0]
    assert binding.candidate_sha256 in console.shown[0]
    assert binding.evidence_digest in console.shown[0]


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "approve lab 000000000000",
        "APPROVE  LAB 000000000000",
        "APPROVE LAB 000000000000 extra",
        "YES",
    ],
)
def test_wrong_or_fuzzy_confirmation_is_rejected(tmp_path, answer):
    candidate, _, binding, external = make_fixture(tmp_path)
    code = derive_confirmation_code(binding, ATTEMPT)
    actual = answer.replace("000000000000", code)
    console = FakeConsole(actual)

    result = request_dev_human_approval(
        external,
        candidate,
        safe_context(),
        console,
        install_attempt_id=ATTEMPT,
        now=FIXED_NOW,
        monotonic=iter([1.0, 2.0]).__next__,
    )
    assert result.approved is False
    assert result.reason == "DEV_APPROVAL_REJECTED"


def test_noninteractive_console_fails_before_prompt(tmp_path):
    candidate, _, _, external = make_fixture(tmp_path)
    console = FakeConsole("anything", interactive=False)
    result = request_dev_human_approval(
        external,
        candidate,
        safe_context(),
        console,
        install_attempt_id=ATTEMPT,
    )
    assert result.approved is False
    assert result.reason == "DEV_APPROVAL_NONINTERACTIVE_REJECTED"
    assert console.shown == []
    assert console.prompts == []


@pytest.mark.parametrize("decision", ["BLOCK", "ERROR", "ALLOW_TO_LAB"])
def test_non_quarantine_never_reaches_prompt(tmp_path, decision):
    candidate, _, _, external = make_fixture(tmp_path)
    external.decision = decision
    console = FakeConsole("anything")
    result = request_dev_human_approval(
        external,
        candidate,
        safe_context(),
        console,
        install_attempt_id=ATTEMPT,
    )
    assert result.approved is False
    assert result.reason == "dev_approval_not_quarantine"
    assert console.shown == []
    assert console.prompts == []


def test_prompt_expiry_rejects_late_correct_response(tmp_path):
    candidate, _, binding, external = make_fixture(tmp_path)
    code = derive_confirmation_code(binding, ATTEMPT)
    root = tmp_path / "audit"
    result = request_dev_human_approval(
        external,
        candidate,
        safe_context(),
        FakeConsole(f"APPROVE LAB {code}"),
        install_attempt_id=ATTEMPT,
        audit_root=root,
        now=FIXED_NOW,
        monotonic=iter([10.0, 131.0]).__next__,
        timeout_seconds=120.0,
    )
    assert result.approved is False
    assert result.reason == "DEV_APPROVAL_EXPIRED"
    assert audit_states(root) == ["requested", "expired"]


def test_candidate_mutation_after_human_input_is_rejected(tmp_path):
    candidate, _, binding, external = make_fixture(tmp_path)
    code = derive_confirmation_code(binding, ATTEMPT)

    def mutate():
        (candidate / "SKILL.md").write_text("mutated\n", encoding="utf-8")

    console = FakeConsole(f"APPROVE LAB {code}", on_read=mutate)
    result = request_dev_human_approval(
        external,
        candidate,
        safe_context(),
        console,
        install_attempt_id=ATTEMPT,
        now=FIXED_NOW,
        monotonic=iter([1.0, 2.0]).__next__,
    )
    assert result.approved is False
    assert result.reason == "DEV_APPROVAL_BINDING_DRIFT_REJECTED"


def test_evidence_mutation_before_prompt_fails_without_prompt(tmp_path):
    candidate, evidence, _, external = make_fixture(tmp_path)
    (evidence / "decision.json").write_text('{"decision":"BLOCK"}\n', encoding="utf-8")
    console = FakeConsole("anything")
    result = request_dev_human_approval(
        external,
        candidate,
        safe_context(),
        console,
        install_attempt_id=ATTEMPT,
    )
    assert result.approved is False
    assert result.reason == "dev_approval_evidence_digest_drift"
    assert console.shown == []


def test_source_binding_mismatch_fails_without_prompt(tmp_path):
    candidate, _, _, external = make_fixture(tmp_path)
    console = FakeConsole("anything")
    result = request_dev_human_approval(
        external,
        candidate,
        safe_context(),
        console,
        source_id="wrong/source",
        install_attempt_id=ATTEMPT,
    )
    assert result.approved is False
    assert result.reason == "dev_approval_source_mismatch"
    assert console.shown == []


def test_development_audit_is_labeled_and_contains_no_production_authentication_claim(tmp_path):
    candidate, _, binding, external = make_fixture(tmp_path)
    code = derive_confirmation_code(binding, ATTEMPT)
    root = tmp_path / "audit"
    result = request_dev_human_approval(
        external,
        candidate,
        safe_context(),
        FakeConsole(f"APPROVE LAB {code}"),
        install_attempt_id=ATTEMPT,
        audit_root=root,
        now=FIXED_NOW,
        monotonic=iter([1.0, 2.0]).__next__,
    )
    assert result.approved
    raw = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.json")).lower()
    assert "development_manual_confirmation_not_production_authentication" in raw
    for forbidden in ("signature_b64", "signer_key_id", "private_key", "nonce", "credential_id"):
        assert forbidden not in raw
