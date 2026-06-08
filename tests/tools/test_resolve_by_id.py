"""Tests for :func:`resolve_gateway_approval_by_id` — the per-id
atomic-consume security gate.

These tests are at the tools/approval.py layer (no gateway/run.py mock).
Integration tests at the gateway-handler layer come in a follow-up commit
covering /approve <id> end-to-end.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from tools import approval as approval_mod
from tools.approval import (
    _ApprovalEntry,
    _gateway_queues,
    resolve_gateway_approval_by_id,
    set_default_approval_store,
)
from tools.approval_store import ApprovalProposal
from tools.approval_store_memory import InMemoryApprovalStore


@pytest.fixture(autouse=True)
def _reset():
    prev_store = approval_mod.get_default_approval_store()
    approval_mod._gateway_queues.clear()
    approval_mod._pending.clear()
    approval_mod._session_approved.clear()
    yield
    approval_mod._gateway_queues.clear()
    approval_mod._pending.clear()
    approval_mod._session_approved.clear()
    set_default_approval_store(prev_store)


def _proposal(approval_id: str, session_key: str = "session-A",
              expires_in: float = 600) -> ApprovalProposal:
    now = time.time()
    return ApprovalProposal(
        approval_id=approval_id,
        created_at=now,
        expires_at=now + expires_in,
        session_key=session_key,
        command=f"echo {approval_id}",
        risk_reason="test",
        policy_decision="needs_approval",
        requires_explicit_approval=True,
        default_decision="deny",
    )


def test_resolve_by_id_consumes_via_store_and_signals_waiter():
    store = InMemoryApprovalStore()
    set_default_approval_store(store)
    store.submit(_proposal("appr-correct"))

    entry = _ApprovalEntry({"command": "echo correct"}, approval_id="appr-correct")
    _gateway_queues["session-A"] = [entry]

    count = resolve_gateway_approval_by_id("session-A", "appr-correct", "once")
    assert count == 1
    assert entry.event.is_set()
    assert entry.result == "once"

    proposal = store.get("appr-correct")
    assert proposal.status == "consumed"


def test_resolve_by_wrong_id_does_not_signal_anything():
    store = InMemoryApprovalStore()
    set_default_approval_store(store)
    store.submit(_proposal("appr-real"))

    entry = _ApprovalEntry({"command": "echo real"}, approval_id="appr-real")
    _gateway_queues["session-A"] = [entry]

    count = resolve_gateway_approval_by_id("session-A", "appr-fake", "once")
    assert count == 0
    assert not entry.event.is_set()
    # Real proposal still pending — wrong id MUST NOT touch it.
    assert store.get("appr-real").status == "pending"


def test_resolve_by_id_after_already_consumed_returns_zero():
    store = InMemoryApprovalStore()
    set_default_approval_store(store)
    store.submit(_proposal("appr-once"))
    # First consume succeeds.
    store.consume("appr-once", consumed_by="@earlier")

    entry = _ApprovalEntry({"command": "echo once"}, approval_id="appr-once")
    _gateway_queues["session-A"] = [entry]

    count = resolve_gateway_approval_by_id("session-A", "appr-once", "once")
    assert count == 0
    assert not entry.event.is_set()


def test_resolve_by_id_after_denied_returns_zero():
    store = InMemoryApprovalStore()
    set_default_approval_store(store)
    store.submit(_proposal("appr-denied"))
    store.deny("appr-denied", denied_by="@earlier")

    entry = _ApprovalEntry({"command": "echo denied"}, approval_id="appr-denied")
    _gateway_queues["session-A"] = [entry]

    count = resolve_gateway_approval_by_id("session-A", "appr-denied", "once")
    assert count == 0
    assert not entry.event.is_set()


def test_resolve_by_id_after_expired_returns_zero():
    store = InMemoryApprovalStore()
    set_default_approval_store(store)
    now = time.time()
    expired = ApprovalProposal(
        approval_id="appr-expired",
        created_at=now - 100,
        expires_at=now - 10,
        session_key="session-A",
        command="echo expired",
        risk_level="medium",
        risk_reason="expired-test",
    )
    store.submit(expired)

    entry = _ApprovalEntry({"command": "echo expired"}, approval_id="appr-expired")
    _gateway_queues["session-A"] = [entry]

    count = resolve_gateway_approval_by_id("session-A", "appr-expired", "once")
    assert count == 0
    assert not entry.event.is_set()


def test_resolve_by_id_cross_session_attempt_rejected():
    """Approval id owned by session-A cannot be approved from session-B."""
    store = InMemoryApprovalStore()
    set_default_approval_store(store)
    store.submit(_proposal("appr-A", session_key="session-A"))

    entry = _ApprovalEntry({"command": "echo A"}, approval_id="appr-A")
    _gateway_queues["session-A"] = [entry]

    # session-B tries to approve session-A's id.
    count = resolve_gateway_approval_by_id("session-B", "appr-A", "once")
    assert count == 0
    assert not entry.event.is_set()
    assert store.get("appr-A").status == "pending"


def test_resolve_by_id_deny_path():
    store = InMemoryApprovalStore()
    set_default_approval_store(store)
    store.submit(_proposal("appr-todeny"))

    entry = _ApprovalEntry({"command": "echo x"}, approval_id="appr-todeny")
    _gateway_queues["session-A"] = [entry]

    count = resolve_gateway_approval_by_id("session-A", "appr-todeny", "deny")
    assert count == 1
    assert entry.event.is_set()
    assert entry.result == "deny"
    assert store.get("appr-todeny").status == "denied"


def test_resolve_by_id_orphan_consumes_store_but_returns_minus_one():
    """Proposal exists in store but no live waiter (gateway restart scenario):
    consume the row anyway, but return -1 so the handler can surface a
    DISTINCT 'command was NOT executed' message to the user. Previously
    this returned 1 (success) which lied — handler showed '✅ approved'
    while no command actually ran.

    Also: execution_status is recorded as 'blocked_after_consume' with
    reason 'orphan_no_waiter' so audit retrospective can tell."""
    store = InMemoryApprovalStore()
    set_default_approval_store(store)
    store.submit(_proposal("appr-orphan"))
    # No entry in _gateway_queues — simulates post-restart state.

    count = resolve_gateway_approval_by_id("session-A", "appr-orphan", "once")
    assert count == -1  # Distinct from 1 (signalled) and 0 (rejected)
    proposal = store.get("appr-orphan")
    assert proposal.status == "consumed"
    assert proposal.execution_status == "blocked_after_consume"
    assert proposal.execution_reason == "orphan_no_waiter"


def test_resolve_by_id_falls_back_to_fifo_when_no_store():
    """When no store is configured, by-id call falls back to legacy FIFO."""
    set_default_approval_store(None)

    entry = _ApprovalEntry({"command": "echo fallback"})
    _gateway_queues["session-A"] = [entry]

    count = resolve_gateway_approval_by_id("session-A", "ignored-id", "once")
    # FIFO: oldest entry resolved regardless of id.
    assert count == 1
    assert entry.event.is_set()
