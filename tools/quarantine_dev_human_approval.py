"""Foreground-only manual confirmation for LAB quarantine simulation.

This is deliberately not a production signer, signature verifier, approval
artifact, or authentication mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import secrets
import sys
import time
from typing import Any, Protocol

from tools.external_skill_admission import candidate_hash_matches
from tools.quarantine_approval_models import (
    APPROVAL_BINDING_SCHEMA,
    ApprovalBinding,
    approval_scope_digest,
    canonical_json_bytes,
    evidence_bundle_digest,
)
from tools.quarantine_dev_audit import (
    DevApprovalAuditEvent,
    append_dev_approval_audit_event,
)
from tools.quarantine_dev_context import (
    DevHumanApprovalContext,
    validate_dev_approval_context,
)

DEV_CONFIRM_LABEL = b"HERMES_DEV_LAB_CONFIRM_V0"
DEFAULT_PROMPT_TIMEOUT_SECONDS = 120.0
_CONFIRM_RE = re.compile(r"^APPROVE LAB ([0-9A-F]{12})$")
_ATTEMPT_RE = re.compile(r"^[0-9a-f]{32}$")


class ForegroundReviewConsole(Protocol):
    def is_interactive(self) -> bool:
        raise NotImplementedError

    def show_review(self, text: str) -> None:
        raise NotImplementedError

    def read_confirmation(self, prompt: str) -> str:
        raise NotImplementedError


class StdForegroundReviewConsole:
    def is_interactive(self) -> bool:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())

    def show_review(self, text: str) -> None:
        print(text, file=sys.stdout, flush=True)

    def read_confirmation(self, prompt: str) -> str:
        return input(prompt)


@dataclass(frozen=True)
class DevHumanApprovalResult:
    approved: bool
    reason: str
    install_attempt_id: str = ""
    confirmation_code: str = ""
    approval_scope_sha256: str = ""
    audit_root: str = ""


@dataclass(frozen=True)
class BindingRecheck:
    valid: bool
    reason: str


def _utc_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _reason_digest(binding: ApprovalBinding) -> str:
    return hashlib.sha256(
        canonical_json_bytes(binding.scanner_reason_codes)
    ).hexdigest()


def derive_confirmation_code(binding: ApprovalBinding, install_attempt_id: str) -> str:
    if binding.schema_version != APPROVAL_BINDING_SCHEMA:
        raise ValueError("unsupported approval binding schema")
    if binding.decision != "QUARANTINE":
        raise ValueError("development confirmation requires QUARANTINE binding")
    if not _ATTEMPT_RE.fullmatch(install_attempt_id):
        raise ValueError("invalid development install attempt id")
    scope_hex = approval_scope_digest(binding, install_attempt_id)
    payload = (
        DEV_CONFIRM_LABEL
        + b"\x00"
        + bytes.fromhex(scope_hex)
        + b"\x00"
        + install_attempt_id.encode("ascii")
    )
    return hashlib.sha256(payload).hexdigest()[:12].upper()


def format_dev_review_summary(
    binding: ApprovalBinding,
    install_attempt_id: str,
    confirmation_code: str,
) -> str:
    completeness = ", ".join(
        f"{name}={'complete' if complete else 'partial'}"
        for name, complete in binding.scanner_completeness
    ) or "none"
    reasons = ", ".join(
        f"{name}:{code}"
        for name, codes in binding.scanner_reason_codes
        for code in codes
    ) or "none"
    return "\n".join([
        "DEVELOPMENT LAB CONFIRMATION — NOT PRODUCTION AUTHENTICATION",
        f"Source: {binding.source_id}",
        f"Source version: {binding.source_version}",
        f"Candidate SHA256: {binding.candidate_sha256}",
        f"Evidence digest: {binding.evidence_digest}",
        f"Gate release SHA256: {binding.gate_release_sha256}",
        f"Policy version: {binding.policy_version}",
        f"Hermes guard: {binding.hermes_guard_version}",
        f"Cisco scanner: {binding.cisco_version}",
        f"NVIDIA scanner: {binding.nvidia_version}",
        f"Scanner completeness: {completeness}",
        f"Quarantine reasons: {reasons}",
        f"Install attempt: {install_attempt_id}",
        f"Confirmation code: {confirmation_code}",
        f"Type exactly: APPROVE LAB {confirmation_code}",
    ])


def recheck_dev_binding(
    external_result: Any,
    candidate_path: Path,
    context: DevHumanApprovalContext,
    *,
    source_id: str = "",
    source_version: str = "",
) -> BindingRecheck:
    context_result = validate_dev_approval_context(context)
    if not context_result.valid:
        return BindingRecheck(False, context_result.reason)
    if getattr(external_result, "decision", "") != "QUARANTINE":
        return BindingRecheck(False, "dev_approval_not_quarantine")
    binding = getattr(external_result, "approval_binding", None)
    if not isinstance(binding, ApprovalBinding):
        return BindingRecheck(False, "dev_approval_binding_missing")
    if binding.schema_version != APPROVAL_BINDING_SCHEMA:
        return BindingRecheck(False, "dev_approval_binding_schema_invalid")
    if binding.decision != "QUARANTINE":
        return BindingRecheck(False, "dev_approval_binding_decision_invalid")
    if source_id and binding.source_id != source_id:
        return BindingRecheck(False, "dev_approval_source_mismatch")
    if source_version and binding.source_version != source_version:
        return BindingRecheck(False, "dev_approval_source_version_mismatch")
    result_sha = str(getattr(external_result, "candidate_sha256", "") or "").lower()
    if result_sha and result_sha != binding.candidate_sha256.lower():
        return BindingRecheck(False, "dev_approval_result_hash_mismatch")
    if not candidate_hash_matches(Path(candidate_path), binding.candidate_sha256):
        return BindingRecheck(False, "dev_approval_candidate_hash_drift")
    evidence_dir = Path(str(getattr(external_result, "evidence_dir", "") or ""))
    try:
        current_digest = evidence_bundle_digest(evidence_dir)
    except (OSError, ValueError):
        return BindingRecheck(False, "dev_approval_evidence_invalid")
    if current_digest.lower() != binding.evidence_digest.lower():
        return BindingRecheck(False, "dev_approval_evidence_digest_drift")
    return BindingRecheck(True, "dev_approval_binding_stable")


def _audit(
    audit_root: Path | None,
    state: str,
    binding: ApprovalBinding,
    attempt: str,
    when: datetime,
    *,
    code: str = "",
    result: str = "",
) -> None:
    if audit_root is None:
        return
    code_digest = hashlib.sha256(code.encode("ascii")).hexdigest() if code else ""
    append_dev_approval_audit_event(
        Path(audit_root),
        DevApprovalAuditEvent(
            state=state,
            candidate_sha256=binding.candidate_sha256.lower(),
            source_id=binding.source_id,
            source_version=binding.source_version,
            evidence_digest=binding.evidence_digest.lower(),
            quarantine_reason_digest=_reason_digest(binding),
            install_attempt_id=attempt,
            timestamp=_utc_z(when),
            result=result,
            confirmation_code_sha256=code_digest,
        ),
    )


def record_dev_approval_state(
    external_result: Any,
    approval_result: DevHumanApprovalResult,
    state: str,
    *,
    result: str = "",
    now: datetime | None = None,
) -> None:
    binding = getattr(external_result, "approval_binding", None)
    if not isinstance(binding, ApprovalBinding) or not approval_result.audit_root:
        return
    _audit(
        Path(approval_result.audit_root),
        state,
        binding,
        approval_result.install_attempt_id,
        now or datetime.now(timezone.utc),
        code=approval_result.confirmation_code,
        result=result,
    )


def request_dev_human_approval(
    external_result: Any,
    candidate_path: Path,
    context: DevHumanApprovalContext,
    console: ForegroundReviewConsole,
    *,
    source_id: str = "",
    source_version: str = "",
    install_attempt_id: str | None = None,
    audit_root: Path | None = None,
    now: datetime | None = None,
    monotonic: Any = time.monotonic,
    timeout_seconds: float = DEFAULT_PROMPT_TIMEOUT_SECONDS,
) -> DevHumanApprovalResult:
    context_result = validate_dev_approval_context(context)
    if not context_result.valid:
        return DevHumanApprovalResult(False, context_result.reason)

    if getattr(external_result, "decision", "") != "QUARANTINE":
        return DevHumanApprovalResult(False, "dev_approval_not_quarantine")
    binding = getattr(external_result, "approval_binding", None)
    if not isinstance(binding, ApprovalBinding):
        return DevHumanApprovalResult(False, "dev_approval_binding_missing")

    precheck = recheck_dev_binding(
        external_result,
        Path(candidate_path),
        context,
        source_id=source_id,
        source_version=source_version,
    )
    if not precheck.valid:
        return DevHumanApprovalResult(False, precheck.reason)

    if not console.is_interactive():
        return DevHumanApprovalResult(False, "DEV_APPROVAL_NONINTERACTIVE_REJECTED")

    attempt = install_attempt_id or secrets.token_hex(16)
    if not _ATTEMPT_RE.fullmatch(attempt):
        return DevHumanApprovalResult(False, "dev_approval_attempt_id_invalid")

    code = derive_confirmation_code(binding, attempt)
    scope_digest = approval_scope_digest(binding, attempt)
    when = now or datetime.now(timezone.utc)
    root_text = str(audit_root or "")
    _audit(audit_root, "requested", binding, attempt, when, code=code)

    console.show_review(format_dev_review_summary(binding, attempt, code))
    started = float(monotonic())
    try:
        raw_response = console.read_confirmation("Development confirmation: ")
    except (EOFError, KeyboardInterrupt):
        raw_response = ""
    finished = float(monotonic())

    base_result = dict(
        install_attempt_id=attempt,
        confirmation_code=code,
        approval_scope_sha256=scope_digest,
        audit_root=root_text,
    )
    if finished - started > timeout_seconds:
        _audit(audit_root, "expired", binding, attempt, when, code=code, result="DEV_APPROVAL_EXPIRED")
        return DevHumanApprovalResult(False, "DEV_APPROVAL_EXPIRED", **base_result)

    response = str(raw_response).strip(" \t\r\n")
    match = _CONFIRM_RE.fullmatch(response)
    if not match or match.group(1) != code:
        _audit(audit_root, "rejected", binding, attempt, when, code=code, result="DEV_APPROVAL_REJECTED")
        return DevHumanApprovalResult(False, "DEV_APPROVAL_REJECTED", **base_result)

    postcheck = recheck_dev_binding(
        external_result,
        Path(candidate_path),
        context,
        source_id=source_id,
        source_version=source_version,
    )
    if not postcheck.valid:
        _audit(
            audit_root,
            "binding_drift_rejected",
            binding,
            attempt,
            when,
            code=code,
            result="DEV_APPROVAL_BINDING_DRIFT_REJECTED",
        )
        return DevHumanApprovalResult(
            False,
            "DEV_APPROVAL_BINDING_DRIFT_REJECTED",
            **base_result,
        )

    _audit(audit_root, "approved", binding, attempt, when, code=code, result="DEV_APPROVAL_APPROVED")
    return DevHumanApprovalResult(True, "DEV_APPROVAL_APPROVED", **base_result)
