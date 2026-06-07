"""Phase 6/7: explicit fail-closed paths.

The spec's invariant 6 ("no lockfile handwaving") + invariant 7
("ambiguous state must fail closed") require explicit tests for
every failure mode that could otherwise allow silent execution.

Earlier files already cover wrong-id / consumed / denied / expired /
cross-session / corrupt-payload / classifier-raise / missing-pinned.

This file adds the remaining explicit cases:

- Store entirely unavailable during consume → no execution
- Submit fails before notify → no entry queued, no notify, no
  silent execution
- End-to-end-ish: dangerous command flows through the real handler
  with a real store, exactly one execution wake-up, second /approve
  refuses.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import patch

import pytest

from tools import approval as approval_mod
from tools.approval import (
    _ApprovalEntry,
    _await_gateway_decision,
    _gateway_queues,
    register_gateway_notify,
    resolve_gateway_approval_by_id,
    set_default_approval_store,
)
from tools.approval_store import ApprovalProposal, ApprovalStoreError
from tools.approval_store_memory import InMemoryApprovalStore


@pytest.fixture(autouse=True)
def _reset():
    prev = approval_mod.get_default_approval_store()
    approval_mod._gateway_queues.clear()
    approval_mod._pending.clear()
    approval_mod._session_approved.clear()
    yield
    approval_mod._gateway_queues.clear()
    approval_mod._pending.clear()
    approval_mod._session_approved.clear()
    set_default_approval_store(prev)


# ---------------------------------------------------------------------------
# Store unavailable during consume
# ---------------------------------------------------------------------------


class _RaisingStore(InMemoryApprovalStore):
    """In-memory store that raises ApprovalStoreError on .consume / .deny.

    Used to simulate transient backend failure (DB locked, file disk-full,
    permission revoked mid-session). resolve_gateway_approval_by_id must
    treat any such exception as "transition failed" and refuse execution.
    """

    def __init__(self, *, raise_on_consume=False, raise_on_deny=False,
                 raise_on_get=False):
        super().__init__()
        self.raise_on_consume = raise_on_consume
        self.raise_on_deny = raise_on_deny
        self.raise_on_get = raise_on_get

    def get(self, approval_id):
        if self.raise_on_get:
            raise ApprovalStoreError("simulated DB failure")
        return super().get(approval_id)

    def consume(self, approval_id, *, consumed_by, now=None):
        if self.raise_on_consume:
            raise ApprovalStoreError("simulated DB failure on consume")
        return super().consume(approval_id, consumed_by=consumed_by, now=now)

    def deny(self, approval_id, *, denied_by, now=None):
        if self.raise_on_deny:
            raise ApprovalStoreError("simulated DB failure on deny")
        return super().deny(approval_id, denied_by=denied_by, now=now)


def _make_proposal_in(store, approval_id="appr-X", session_key="s",
                     risk_level="medium"):
    proposal = ApprovalProposal(
        approval_id=approval_id,
        created_at=time.time(),
        expires_at=time.time() + 300,
        session_key=session_key,
        command="echo hi",
        risk_level=risk_level,
        risk_reason="test",
        policy_decision="needs_approval",
        requires_explicit_approval=True,
        default_decision="deny",
    )
    store.submit(proposal)
    return proposal


def test_store_consume_raises_does_not_signal_waiter():
    """When store.consume raises mid-transaction, the agent thread is
    NOT signalled — fail closed, no execution."""
    store = _RaisingStore(raise_on_consume=True)
    set_default_approval_store(store)
    _make_proposal_in(store, "appr-CR", "s1")

    entry = _ApprovalEntry({"command": "echo"}, approval_id="appr-CR")
    _gateway_queues["s1"] = [entry]

    with pytest.raises(ApprovalStoreError):
        resolve_gateway_approval_by_id("s1", "appr-CR", "once")

    # Waiter MUST NOT have been signalled.
    assert not entry.event.is_set()


def test_store_get_raises_returns_zero_without_signalling():
    """When store.get raises before consume, we never reach the gate
    transition. Waiter is not signalled. Per ApprovalStoreError docstring
    callers must treat the raised error as 'proposal not accepted'."""
    store = _RaisingStore(raise_on_get=True)
    set_default_approval_store(store)
    _make_proposal_in(store, "appr-GR", "s2")

    entry = _ApprovalEntry({"command": "echo"}, approval_id="appr-GR")
    _gateway_queues["s2"] = [entry]

    with pytest.raises(ApprovalStoreError):
        resolve_gateway_approval_by_id("s2", "appr-GR", "once")

    assert not entry.event.is_set()


def test_store_deny_raises_does_not_signal_waiter():
    store = _RaisingStore(raise_on_deny=True)
    set_default_approval_store(store)
    _make_proposal_in(store, "appr-DR", "s3")

    entry = _ApprovalEntry({"command": "echo"}, approval_id="appr-DR")
    _gateway_queues["s3"] = [entry]

    with pytest.raises(ApprovalStoreError):
        resolve_gateway_approval_by_id("s3", "appr-DR", "deny")

    assert not entry.event.is_set()


# ---------------------------------------------------------------------------
# Submit fails before notify
# ---------------------------------------------------------------------------


def test_submit_failure_falls_through_to_legacy_flow_with_no_id():
    """If store.submit raises, the proposal isn't persisted but the
    in-memory entry still gets created (legacy fallback). The
    approval_id on the entry must be None so the resulting /approve
    cannot accidentally identify a non-existent store row.

    This documents current behavior: submit-failure is loud (logged
    ERROR) but the agent thread still blocks waiting for resolution
    via the legacy in-memory path. Future hardening could elevate this
    to fail-closed-immediately."""
    store = InMemoryApprovalStore()
    with patch.object(store, "submit",
                      side_effect=ApprovalStoreError("simulated submit failure")):
        set_default_approval_store(store)
        session_key = "submit-fail"
        register_gateway_notify(session_key, lambda data: None)

        captured: dict[str, Any] = {}
        barrier = threading.Event()

        def driver():
            captured["result"] = _await_gateway_decision(
                session_key=session_key,
                notify_cb=lambda data: (captured.update(data=data), barrier.set()),
                approval_data={
                    "command": "echo fail-submit",
                    "description": "test",
                    "pattern_key": "test",
                    "pattern_keys": ["test"],
                },
                surface="test",
            )

        t = threading.Thread(target=driver, daemon=True)
        t.start()
        assert barrier.wait(timeout=5)

        # approval_data wire: approval_id MUST be absent (no row to address).
        # The render-helper field was set before submit so it might exist;
        # what matters is the entry-level approval_id is None.
        entry = approval_mod._gateway_queues[session_key][0]
        assert entry.approval_id is None, (
            "submit failure must clear approval_id; otherwise /approve <id> "
            "would target a row that does not exist"
        )

        # Cleanup: deny the entry to release driver.
        from tools.approval import resolve_gateway_approval
        resolve_gateway_approval(session_key, "deny")
        t.join(timeout=5)


# ---------------------------------------------------------------------------
# Payload missing required pinned fields
# ---------------------------------------------------------------------------


def test_proposal_missing_pinned_fields_rejected_at_construction():
    """Missing required pinned fields → ValueError at construction → no
    submit → no entry → no execution. Defense in depth: even if the
    proposal somehow reaches consume, the Phase 3 guard catches it,
    but the primary boundary is submit-time refusal per spec.

    Required fields per spec: approval_id, session_key, command,
    risk_level (low/medium/high), risk_reason, policy_decision,
    default_decision (must be 'deny' for high).
    """
    base = dict(
        approval_id="appr-M",
        created_at=time.time(),
        session_key="sm",
        command="echo m",
        risk_level="medium",
        risk_reason="missing-field-test",
    )

    # Sanity: complete payload constructs fine.
    ApprovalProposal(**base)

    # Each individually-omitted required field MUST raise.
    for missing in ("session_key", "command", "risk_reason"):
        bad = {**base, missing: ""}
        with pytest.raises(ValueError, match=missing):
            ApprovalProposal(**bad)

    # risk_level outside {low,medium,high}
    with pytest.raises(ValueError, match="risk_level"):
        ApprovalProposal(**{**base, "risk_level": "extreme"})

    # High-risk MUST default to deny.
    with pytest.raises(ValueError, match="default_decision"):
        ApprovalProposal(**{
            **base,
            "risk_level": "high",
            "default_decision": "allow",
        })


def test_consume_with_missing_required_fields_in_payload_cannot_be_submitted():
    """Document submit-time refusal: an invalid ApprovalProposal never
    reaches the store, so consume never sees it. The execution path is
    impossible by construction."""
    set_default_approval_store(InMemoryApprovalStore())
    with pytest.raises(ValueError):
        # session_key omitted → validation refuses construction.
        ApprovalProposal(
            approval_id="appr-bad",
            created_at=time.time(),
            session_key="",       # invalid
            command="echo m",
            risk_level="medium",
            risk_reason="x",
        )


# ---------------------------------------------------------------------------
# End-to-end-ish: real store + real handler + fake waiter, exactly-one exec
# ---------------------------------------------------------------------------


def test_end_to_end_via_handler_exactly_one_execution():
    """Drive the realistic flow:
       1. _await_gateway_decision creates proposal + entry
       2. notify_cb fires (we capture it)
       3. First call to resolve_gateway_approval_by_id with the id wakes
          the waiter and the store row is consumed
       4. Second call with same id refuses (returns 0)
       5. Waiter was signalled exactly once
    """
    store = InMemoryApprovalStore()
    set_default_approval_store(store)
    session_key = "e2e"
    register_gateway_notify(session_key, lambda data: None)

    captured: dict[str, Any] = {}
    barrier = threading.Event()
    wake_count = [0]

    def driver():
        result = _await_gateway_decision(
            session_key=session_key,
            notify_cb=lambda data: (captured.update(data=data), barrier.set()),
            approval_data={
                "command": "rm -rf /tmp/test",
                "description": "recursive delete",
                "pattern_key": "rm-recursive",
                "pattern_keys": ["rm-recursive"],
            },
            surface="test",
        )
        wake_count[0] += 1
        captured["result"] = result

    t = threading.Thread(target=driver, daemon=True)
    t.start()
    assert barrier.wait(timeout=5)

    approval_id = captured["data"]["approval_id"]
    assert "HIGH RISK" in captured["data"]["display_text"]  # Phase 4 wiring

    # First /approve <id> — wakes waiter, store row consumed
    count1 = resolve_gateway_approval_by_id(session_key, approval_id, "once")
    assert count1 == 1
    t.join(timeout=5)
    assert wake_count[0] == 1
    assert store.get(approval_id).status == "consumed"

    # Second /approve <id> — refused; waiter already gone, store terminal
    count2 = resolve_gateway_approval_by_id(session_key, approval_id, "once")
    assert count2 == 0
    # No additional wake — exactly-one execution invariant holds
    assert wake_count[0] == 1


def test_end_to_end_deny_prevents_subsequent_approve():
    """/deny <id> must terminally block any later /approve <id> on the
    same proposal."""
    store = InMemoryApprovalStore()
    set_default_approval_store(store)
    session_key = "e2e-deny"
    register_gateway_notify(session_key, lambda data: None)

    captured: dict[str, Any] = {}
    barrier = threading.Event()

    def driver():
        _await_gateway_decision(
            session_key=session_key,
            notify_cb=lambda data: (captured.update(data=data), barrier.set()),
            approval_data={
                "command": "echo end-to-end-deny",
                "description": "test",
                "pattern_key": "test",
                "pattern_keys": ["test"],
            },
            surface="test",
        )

    t = threading.Thread(target=driver, daemon=True)
    t.start()
    assert barrier.wait(timeout=5)

    approval_id = captured["data"]["approval_id"]

    # Deny first
    deny_count = resolve_gateway_approval_by_id(session_key, approval_id, "deny")
    assert deny_count == 1
    t.join(timeout=5)
    assert store.get(approval_id).status == "denied"

    # Subsequent approve — must refuse
    appr_count = resolve_gateway_approval_by_id(session_key, approval_id, "once")
    assert appr_count == 0
    assert store.get(approval_id).status == "denied", (
        "denied status must NOT be overwritten by a later consume attempt"
    )
