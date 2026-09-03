"""Grant authority provider abstraction + correlated human approval receipts.

Review #100694 blocker 1 (andrexibiza, 2026-09-01): the persisted grant must
be cryptographically unforgeable from model-facing surfaces, and issuance must
consume a proof that generic model-facing terminal/file/plugin paths cannot
mint.

Architecture (Option A+):

    Human Approval Transport
            |
            v
    Correlated Approval Receipt   (produced by THIS module, one-shot)
            |
            v
    Trusted Grant Issuer          (tools.destructive_grants.issue_grant)
            |
            v
    Grant Authority Provider      (authenticates the canonical payload)
            |
            v
    authenticated one-shot grant
            |
            v
    claim / fencing / settlement  (tools.destructive_grants)
            |
            v
    structured QGA formatter

Design invariants:

1. A grant is authenticated with a keyed tag (HMAC-SHA256 by default), never
   an unkeyed SHA-256.  The key lives ONLY in the memory of the trusted
   Hermes process (``ProcessEphemeralAuthority``): it is not persisted, not
   in the environment, not logged, not exported, not serialized, and not
   copied into child-process configuration.

2. ``execute_code`` children are separate OS processes: they import their own
   copy of this module, so they get their own provider with a DIFFERENT key.
   A grant they mint is rejected by the parent's provider (generation and tag
   mismatch).  They cannot read the parent's key (separate address space) and
   cannot resolve the parent's approval queues (separate memory).

3. Issuance consumes a correlated human approval receipt produced by the
   approval layer AFTER an explicit human ``approve once`` decision.  The
   caller never supplies a free-form evidence dict.  A receipt is one-shot:
   it can issue at most one grant, and replaying it is denied.

4. A restart invalidates all live grants: the provider generation changes,
   so ``verify`` fails with an authority-generation mismatch.  Grants are
   process-bound capabilities (short TTL, re-issue after restart).

5. Hardware-backed protection (TPM/vTPM) is an optional strengthening layer:
   the provider interface is the extension point.  No TPM is required for
   this feature to be secure (portable baseline first).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Authority provider abstraction
# ---------------------------------------------------------------------------


class GrantAuthorityProvider(ABC):
    """Authenticate/verify a canonical grant payload.

    A provider owns a secret that model-facing surfaces (terminal,
    execute_code, write_file) cannot obtain.  ``authenticate`` produces the
    tag stored in the grant; ``verify`` recomputes it and compares in
    constant time.  ``generation`` identifies the provider instance: a new
    process (or a new provider) has a new generation, so grants minted by a
    previous incarnation are rejected.
    """

    @abstractmethod
    def authenticate(self, payload: bytes) -> str:
        """Return the authentication tag for *payload*."""

    @abstractmethod
    def verify(self, payload: bytes, tag: str) -> bool:
        """Return True iff *tag* authenticates *payload* under this provider."""

    @property
    @abstractmethod
    def generation(self) -> str:
        """Opaque provider-instance identifier (changes on restart)."""


class ProcessEphemeralAuthority(GrantAuthorityProvider):
    """Baseline provider: HMAC-SHA256 with a process-local random secret.

    The secret is 256 bits, generated once per process, and is NEVER
    persisted, exported, logged, serialized, or placed in the environment.
    It lives only in this object's memory inside the trusted Hermes process.
    """

    _SECRET_BYTES = 32  # 256 bits

    def __init__(self) -> None:
        self._secret = secrets.token_bytes(self._SECRET_BYTES)
        self._generation = uuid.uuid4().hex

    def authenticate(self, payload: bytes) -> str:
        return hmac.new(self._secret, payload, hashlib.sha256).hexdigest()

    def verify(self, payload: bytes, tag: str) -> bool:
        if not isinstance(tag, str) or len(tag) != 64:
            return False
        try:
            expected = hmac.new(self._secret, payload, hashlib.sha256).hexdigest()
        except (TypeError, ValueError):
            return False
        return hmac.compare_digest(expected, tag)

    @property
    def generation(self) -> str:
        return self._generation

    def __repr__(self) -> str:
        # Never leak the secret through repr/errors.
        return f"<ProcessEphemeralAuthority generation={self._generation[:8]}…>"


# Module-level singleton: one provider per trusted process.  A child
# execute_code process imports this module fresh and gets its OWN provider
# with a DIFFERENT secret — that is the isolation property the adversarial
# tests prove.
_AUTHORITY: Optional[GrantAuthorityProvider] = None


def get_authority() -> GrantAuthorityProvider:
    """Return the process-local authority provider (lazy singleton)."""
    global _AUTHORITY
    if _AUTHORITY is None:
        _AUTHORITY = ProcessEphemeralAuthority()
    return _AUTHORITY


def reset_authority_for_tests() -> None:
    """Drop the singleton (tests only): the next get_authority() call mints a
    fresh provider with a new secret and a new generation."""
    global _AUTHORITY
    _AUTHORITY = None


# ---------------------------------------------------------------------------
# Correlated human approval receipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HumanApprovalReceipt:
    """Proof of an explicit human ``approve once`` decision.

    Produced ONLY by :func:`request_destructive_grant_approval` after the
    approval layer returned a real human decision.  The caller never
    constructs one.  A receipt is one-shot: :func:`consume_receipt` removes
    it from the process-local store, so the same receipt cannot issue a
    second grant (replay denied).
    """

    receipt_id: str
    request_id: str
    request_digest: str
    session_id: str
    turn_id: str
    tool_call_id: str
    operation: str
    vm_id: str
    device: str
    fs_type: str
    label: str
    incarnation_product_uuid: str
    incarnation_boot_id: str
    incarnation_hostname: str
    issued_at: float
    expires_at: float
    human_decision: str = "approve_once"

    def to_dict(self) -> Dict[str, object]:
        return {
            "receipt_id": self.receipt_id,
            "request_id": self.request_id,
            "request_digest": self.request_digest,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "tool_call_id": self.tool_call_id,
            "operation": self.operation,
            "vm_id": self.vm_id,
            "device": self.device,
            "fs_type": self.fs_type,
            "label": self.label,
            "incarnation_product_uuid": self.incarnation_product_uuid,
            "incarnation_boot_id": self.incarnation_boot_id,
            "incarnation_hostname": self.incarnation_hostname,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "human_decision": self.human_decision,
        }


# Process-local one-shot receipt store.  Like the authority secret, it lives
# only in the trusted process memory: a child execute_code process has its
# own (empty) store and cannot consume the parent's receipts.
_RECEIPTS: Dict[str, HumanApprovalReceipt] = {}


def _store_receipt(receipt: HumanApprovalReceipt) -> HumanApprovalReceipt:
    _RECEIPTS[receipt.receipt_id] = receipt
    return receipt


def peek_receipt(receipt_id: str) -> HumanApprovalReceipt:
    """Return a receipt WITHOUT consuming it.

    Used by ``issue_grant`` to validate the receipt against the requested
    operation/target/session BEFORE the atomic consume: a denied issuance
    must not burn the human decision.
    """
    receipt = _RECEIPTS.get(receipt_id)
    if receipt is None:
        raise ReceiptError(f"receipt {receipt_id!r} unknown or already consumed (replay denied)")
    return receipt


def consume_receipt(receipt_id: str) -> HumanApprovalReceipt:
    """Atomically consume a receipt (one-shot issuance).

    Raises ``ReceiptError`` if the receipt is unknown or already consumed.
    """
    receipt = _RECEIPTS.pop(receipt_id, None)
    if receipt is None:
        raise ReceiptError(f"receipt {receipt_id!r} unknown or already consumed (replay denied)")
    return receipt


def list_receipts() -> list:
    """Return live (unconsumed) receipts (tests/audit, no secrets)."""
    return list(_RECEIPTS.values())


class ReceiptError(Exception):
    """Raised when a receipt is missing, expired, or already consumed."""


# ---------------------------------------------------------------------------
# Approval gate for destructive grant issuance
# ---------------------------------------------------------------------------


def _build_request_text(
    *,
    operation: str,
    vm_id: str,
    device: str,
    fs_type: str,
    label: str,
    session_id: str,
    ttl_seconds: int,
    incarnation_product_uuid: str = "",
    incarnation_boot_id: str = "",
    incarnation_hostname: str = "",
) -> tuple:
    """Return (command, description) shown to the human in the approval UI.

    The observed guest incarnation is part of what the human approves: the
    request text shows a readable identity so the human can confirm the
    exact target generation before deciding.
    """
    command = (
        f"governed destructive operation: {operation} on VM {vm_id} "
        f"device {device} as {fs_type} label {label} "
        f"(session {session_id}, ttl {ttl_seconds}s)"
    )
    description = (
        f"One-shot destructive grant: {operation} on VM {vm_id} "
        f"device {device} as {fs_type} label {label}. "
        f"Observed guest identity: hostname={incarnation_hostname or '?'} "
        f"product_uuid={incarnation_product_uuid or '?'} "
        f"boot_id={incarnation_boot_id or '?'}. "
        f"This capability is consumed by exactly one governed operation "
        f"and expires automatically."
    )
    return command, description


def _request_digest(
    command: str,
    description: str,
    session_id: str,
    incarnation_product_uuid: str = "",
    incarnation_boot_id: str = "",
    incarnation_hostname: str = "",
) -> str:
    canonical = "|".join(
        [
            command,
            description,
            session_id,
            incarnation_product_uuid,
            incarnation_boot_id,
            incarnation_hostname,
        ]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def request_destructive_grant_approval(
    *,
    operation: str,
    vm_id: str,
    device: str,
    fs_type: str,
    label: str,
    session_id: str,
    ttl_seconds: int,
    incarnation_product_uuid: str = "",
    incarnation_boot_id: str = "",
    incarnation_hostname: str = "",
    turn_id: str = "",
    tool_call_id: str = "",
) -> HumanApprovalReceipt:
    """Ask the human through the REAL approval transport and return a
    one-shot receipt.

    Refuses every bypass context: yolo (process or session), approvals.mode
    = off, cron auto-approval, unattended platforms, single-query mode, and
    any non-interactive context with no human present.  Only an explicit
    human ``once`` decision is accepted — never session/permanent/smart
    approval.

    The receipt is produced HERE, after the decision, and is bound to the
    exact request the human saw (request_id + request_digest) plus the
    correlation ids of the calling context.
    """
    from tools.approval import (
        _await_gateway_decision,
        _get_approval_timeout,
        _is_cron_approval_context,
        _is_gateway_approval_context,
        _is_interactive_cli,
        _is_single_query_approval_context,
        _is_unattended_platform_approval_context,
        _present_with_selected_transport,
        get_current_session_key,
        is_approval_bypass_active,
        prompt_dangerous_approval,
        register_gateway_notify,
    )

    # 1. Bypass contexts are structurally refused: a destructive grant must
    #    never be minted under yolo / mode=off / cron approve / unattended
    #    approve / single-query approve.
    if is_approval_bypass_active():
        raise ReceiptError(
            "destructive grant issuance is refused while approval bypass is "
            "active (yolo / approvals.mode=off)"
        )
    if _is_cron_approval_context():
        raise ReceiptError(
            "destructive grant issuance is refused in cron sessions "
            "(no human present to approve)"
        )
    if _is_unattended_platform_approval_context():
        raise ReceiptError(
            "destructive grant issuance is refused on unattended platforms "
            "(no human present to approve)"
        )
    if _is_single_query_approval_context():
        raise ReceiptError(
            "destructive grant issuance is refused in single-query mode "
            "(no human present to approve)"
        )

    command, description = _build_request_text(
        operation=operation,
        vm_id=vm_id,
        device=device,
        fs_type=fs_type,
        label=label,
        session_id=session_id,
        ttl_seconds=ttl_seconds,
        incarnation_product_uuid=incarnation_product_uuid,
        incarnation_boot_id=incarnation_boot_id,
        incarnation_hostname=incarnation_hostname,
    )
    digest = _request_digest(
        command,
        description,
        session_id,
        incarnation_product_uuid,
        incarnation_boot_id,
        incarnation_hostname,
    )
    now = time.time()
    expires_at = now + ttl_seconds

    # 2. Preferred: an explicitly selected plugin transport (Telegram/WebUI/
    #    desktop/…).  The transport returns a request_id + request_digest
    #    bound to the exact request the human saw.
    try:
        from tools.approval import _present_with_selected_transport as _present
    except Exception:
        _present = None
    if _present is not None:
        try:
            presented = _present(
                command=command,
                description=description,
                pattern_key="governed_grant_issue",
                pattern_keys=["governed_grant_issue"],
                session_key=session_id,
                surface="cli",
                allow_session=False,
                allow_permanent=False,
            )
        except Exception as exc:
            raise ReceiptError(f"approval transport failed: {exc}") from exc

        if presented.get("selected"):
            if presented.get("choice") != "once" or presented.get("failure"):
                raise ReceiptError(
                    "human approval not granted (transport decision: "
                    f"{presented.get('choice') or presented.get('failure')})"
                )
            request_id = str(presented.get("request_id") or "")
            if not request_id:
                raise ReceiptError("approval transport returned no request_id")
            return _store_receipt(
                HumanApprovalReceipt(
                    receipt_id=uuid.uuid4().hex,
                    request_id=request_id,
                    request_digest=digest,
                    session_id=session_id,
                    turn_id=turn_id,
                    tool_call_id=tool_call_id,
                    operation=operation,
                    vm_id=vm_id,
                    device=device,
                    fs_type=fs_type,
                    label=label,
                    incarnation_product_uuid=incarnation_product_uuid,
                    incarnation_boot_id=incarnation_boot_id,
                    incarnation_hostname=incarnation_hostname,
                    issued_at=now,
                    expires_at=expires_at,
                )
            )

    # 3. Gateway round-trip: a notify callback registered for this session
    #    (Discord/Telegram/Slack buttons).  The request_id is minted by the
    #    approval layer and the decision comes from the human's button press.
    if _is_gateway_approval_context():
        notify_cb = None
        from tools.approval import _gateway_notify_cbs, _lock

        with _lock:
            notify_cb = _gateway_notify_cbs.get(session_id)
        if notify_cb is not None:
            request_id = uuid.uuid4().hex
            approval_data = {
                "command": command,
                "pattern_key": "governed_grant_issue",
                "pattern_keys": ["governed_grant_issue"],
                "description": description,
                "allow_permanent": False,
                "allow_session": False,
                "request_id": request_id,
            }
            decision = _await_gateway_decision(
                session_id, notify_cb, approval_data, surface="gateway"
            )
            if decision.get("notify_failed"):
                raise ReceiptError("failed to send approval request to the user")
            if not decision.get("resolved") or decision.get("choice") != "once":
                raise ReceiptError(
                    "human approval not granted (gateway decision: "
                    f"{decision.get('choice') or 'no response'})"
                )
            return _store_receipt(
                HumanApprovalReceipt(
                    receipt_id=uuid.uuid4().hex,
                    request_id=request_id,
                    request_digest=digest,
                    session_id=session_id,
                    turn_id=turn_id,
                    tool_call_id=tool_call_id,
                    operation=operation,
                    vm_id=vm_id,
                    device=device,
                    fs_type=fs_type,
                    label=label,
                    incarnation_product_uuid=incarnation_product_uuid,
                    incarnation_boot_id=incarnation_boot_id,
                    incarnation_hostname=incarnation_hostname,
                    issued_at=now,
                    expires_at=expires_at,
                )
            )

    # 4. Interactive CLI: the standard dangerous-command prompt, restricted
    #    to once/deny (no session, no permanent).  This is the SAME prompt
    #    every other dangerous action uses; the hardline blocklist prevents
    #    the model from reaching this path through the terminal tool.
    if _is_interactive_cli():
        choice = prompt_dangerous_approval(
            command,
            description,
            allow_permanent=False,
            allow_session=False,
        )
        if choice != "once":
            raise ReceiptError(
                "human approval not granted (CLI decision: "
                f"{choice or 'no response'})"
            )
        return _store_receipt(
            HumanApprovalReceipt(
                receipt_id=uuid.uuid4().hex,
                request_id=uuid.uuid4().hex,
                request_digest=digest,
                session_id=session_id,
                turn_id=turn_id,
                tool_call_id=tool_call_id,
                operation=operation,
                vm_id=vm_id,
                device=device,
                fs_type=fs_type,
                label=label,
                incarnation_product_uuid=incarnation_product_uuid,
                incarnation_boot_id=incarnation_boot_id,
                incarnation_hostname=incarnation_hostname,
                issued_at=now,
                expires_at=expires_at,
            )
        )

    # 5. No human surface at all: fail closed.  A grant is never issued
    #    unattended.
    raise ReceiptError(
        "destructive grant issuance requires an interactive human "
        "(gateway session, approval transport, or interactive CLI); "
        "refusing unattended issuance"
    )


# ---------------------------------------------------------------------------
# Canonical authenticated payload
# ---------------------------------------------------------------------------


def canonical_grant_payload(
    *,
    schema_version: int,
    grant_id: str,
    receipt_id: str,
    operation: str,
    vm_id: str,
    hostname: str,
    product_uuid: str,
    boot_id: str,
    device: str,
    fs_type: str,
    label: str,
    authorization_subject: str,
    authorization_source: str,
    session_id: str,
    turn_id: str,
    tool_call_id: str,
    issued_at: float,
    expires_at: float,
    nonce: str,
    authorization_evidence: Optional[Dict[str, object]] = None,
) -> bytes:
    """Canonical JSON of every immutable authority field.

    The correlated human approval receipt (``authorization_evidence``) is
    part of the authenticated payload: swapping or editing the receipt breaks
    the tag.  The execution lifecycle (claimed/settled/consumed) is
    deliberately NOT part of this payload: it is mutable state recorded
    beside the authority record, and mutating it must not invalidate the
    authentication tag.
    """
    payload = {
        "schema_version": schema_version,
        "grant_id": grant_id,
        "receipt_id": receipt_id,
        "operation": operation,
        "vm_id": vm_id,
        "hostname": hostname,
        "product_uuid": product_uuid,
        "boot_id": boot_id,
        "device": device,
        "fs_type": fs_type,
        "label": label,
        "authorization_subject": authorization_subject,
        "authorization_source": authorization_source,
        "session_id": session_id,
        "turn_id": turn_id,
        "tool_call_id": tool_call_id,
        "issued_at": issued_at,
        "expires_at": expires_at,
        "nonce": nonce,
        "authorization_evidence": authorization_evidence or {},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
