"""Append-only audit for development manual confirmation.

This module is intentionally distinct from production approval artifacts,
signatures, and nonce-ledger state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import re
import secrets
import threading
import time

from tools.quarantine_approval_models import canonical_json_bytes

DEV_AUDIT_MODE = "development_manual_confirmation_not_production_authentication"

_ALLOWED_STATES = {
    "requested",
    "rejected",
    "expired",
    "approved",
    "binding_drift_rejected",
    "final_confirmation_cancelled",
    "simulation_completed",
}
_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX12 = re.compile(r"^[0-9A-F]{12}$")
_REASON = re.compile(r"^[A-Za-z0-9_.:-]{0,128}$")

_ORDER_LOCK = threading.Lock()
_LAST_ORDER_NS = 0


@dataclass(frozen=True)
class DevApprovalAuditEvent:
    state: str
    candidate_sha256: str
    source_id: str
    source_version: str
    evidence_digest: str
    quarantine_reason_digest: str
    install_attempt_id: str
    timestamp: str
    result: str = ""
    confirmation_code_sha256: str = ""
    mode: str = DEV_AUDIT_MODE


def _next_order_ns() -> int:
    global _LAST_ORDER_NS
    with _ORDER_LOCK:
        current = time.monotonic_ns()
        if current <= _LAST_ORDER_NS:
            current = _LAST_ORDER_NS + 1
        _LAST_ORDER_NS = current
        return current


def _is_redirect(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _validate(event: DevApprovalAuditEvent) -> None:
    if event.mode != DEV_AUDIT_MODE:
        raise ValueError("invalid development audit mode")
    if event.state not in _ALLOWED_STATES:
        raise ValueError("unsupported development audit state")
    for name, value in (
        ("candidate sha256", event.candidate_sha256),
        ("evidence digest", event.evidence_digest),
        ("quarantine reason digest", event.quarantine_reason_digest),
    ):
        if not _HEX64.fullmatch(value):
            raise ValueError(f"invalid {name}")
    if not _HEX32.fullmatch(event.install_attempt_id):
        raise ValueError("invalid development install attempt id")
    if event.confirmation_code_sha256 and not _HEX64.fullmatch(event.confirmation_code_sha256):
        raise ValueError("invalid confirmation code digest")
    if event.result and not _REASON.fullmatch(event.result):
        raise ValueError("invalid development audit result")


def append_dev_approval_audit_event(root: Path, event: DevApprovalAuditEvent) -> Path:
    _validate(event)
    root = Path(root)
    if root.exists() and _is_redirect(root):
        raise ValueError("development audit root redirected")
    root.mkdir(parents=True, exist_ok=True)
    if _is_redirect(root) or not root.is_dir():
        raise ValueError("development audit root invalid")

    safe_ts = re.sub(r"[^0-9A-Za-z]", "", event.timestamp) or "time"
    name = (
        f"{safe_ts}-{_next_order_ns():020d}-{event.install_attempt_id}-"
        f"{event.state}-{secrets.token_hex(8)}.json"
    )
    path = root / name
    if _is_redirect(path):
        raise ValueError("development audit path redirected")
    payload = canonical_json_bytes(asdict(event))
    with path.open("xb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    return path
