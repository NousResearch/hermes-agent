"""Wiring tests: when ``set_default_approval_store`` is used, the existing
gateway approval flow (``_await_gateway_decision``) must produce a
durable proposal in the store and mirror the resolution outcome onto it.

These tests don't claim the store is yet the security gate — that's a
later commit. They only assert that the shadow-persistence wiring works:

  - approval_id is generated and appears in approval_data passed to notify
  - ApprovalProposal is submitted with pinned-ish fields
  - approve choice → store.consume → status=consumed
  - deny choice → store.deny → status=denied
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from tools import approval as approval_mod
from tools.approval import (
    _await_gateway_decision,
    register_gateway_notify,
    resolve_gateway_approval,
    set_default_approval_store,
)
from tools.approval_store_memory import InMemoryApprovalStore


@pytest.fixture(autouse=True)
def _reset_state():
    """Snapshot+restore module state so wiring tests don't bleed."""
    # Snapshot the global store so we can restore after the test.
    prev_store = approval_mod.get_default_approval_store()
    approval_mod._gateway_queues.clear()
    approval_mod._gateway_notify_cbs.clear()
    approval_mod._pending.clear()
    yield
    approval_mod._gateway_queues.clear()
    approval_mod._gateway_notify_cbs.clear()
    approval_mod._pending.clear()
    set_default_approval_store(prev_store)


def _run_decision_with_resolution(
    store: InMemoryApprovalStore,
    *,
    choice: str,
) -> dict:
    """Drive ``_await_gateway_decision`` from a background thread, resolve
    from the main thread, return the decision dict.
    """
    set_default_approval_store(store)

    session_key = "tester:c1:u1"
    captured: dict[str, Any] = {}

    def notify(data: dict) -> None:
        # Capture for assertions, do NOT block.
        captured["data"] = data

    register_gateway_notify(session_key, notify)

    decision_holder: dict[str, dict] = {}

    def driver():
        decision_holder["result"] = _await_gateway_decision(
            session_key=session_key,
            notify_cb=notify,
            approval_data={
                # Use a description that maps to the same risk-level
                # the runtime classifier will assign — otherwise the
                # Phase 3 guard (correctly) fail-closes when pinned <
                # runtime. Real classifier flows always have matching
                # pinned/runtime descriptions because they come from
                # the SAME classifier; this test was originally written
                # before Phase 3 + Phase 4 risk-mapping, hence the
                # artificial mismatch.
                "command": "echo wiring-test",
                "description": "world/other-writable permissions",
                "pattern_key": "perm-loose",
                "pattern_keys": ["perm-loose"],
            },
            surface="test",
        )

    t = threading.Thread(target=driver, daemon=True)
    t.start()

    # Wait for the proposal to be visible in the queue (driver has
    # progressed past entry-creation).
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if approval_mod._gateway_queues.get(session_key):
            break
        time.sleep(0.02)
    assert approval_mod._gateway_queues.get(session_key), (
        "driver did not enqueue an entry in time"
    )

    resolved_count = resolve_gateway_approval(session_key, choice)
    assert resolved_count == 1

    t.join(timeout=5)
    assert "result" in decision_holder, "driver thread did not return"
    return {**decision_holder["result"], "_captured": captured}


def test_store_receives_proposal_with_pinned_metadata():
    store = InMemoryApprovalStore()
    res = _run_decision_with_resolution(store, choice="once")

    assert res["resolved"] is True
    assert res["choice"] == "once"

    # approval_id was generated and passed to notify_cb
    approval_id = res["_captured"]["data"]["approval_id"]
    assert approval_id and isinstance(approval_id, str)

    # Store row exists and pinned fields are present
    proposal = store.get(approval_id)
    assert proposal is not None
    assert proposal.command == "echo wiring-test"
    assert proposal.risk_reason == "world/other-writable permissions"
    assert proposal.session_key == "tester:c1:u1"
    assert proposal.policy_decision == "needs_approval"
    assert proposal.requires_explicit_approval is True
    assert proposal.default_decision == "deny"


def test_approve_choice_marks_proposal_consumed_in_store():
    store = InMemoryApprovalStore()
    res = _run_decision_with_resolution(store, choice="once")

    approval_id = res["_captured"]["data"]["approval_id"]
    proposal = store.get(approval_id)
    assert proposal is not None
    assert proposal.status == "consumed"
    assert proposal.consumed_by == "session:tester:c1:u1"


def test_deny_choice_marks_proposal_denied_in_store():
    store = InMemoryApprovalStore()
    res = _run_decision_with_resolution(store, choice="deny")

    approval_id = res["_captured"]["data"]["approval_id"]
    proposal = store.get(approval_id)
    assert proposal is not None
    assert proposal.status == "denied"


def test_legacy_flow_without_store_unchanged():
    """When no store is configured, approval_data must NOT carry an
    approval_id (wire format stays legacy)."""
    set_default_approval_store(None)

    session_key = "tester:c1:u1"
    captured: dict[str, Any] = {}

    def notify(data: dict) -> None:
        captured["data"] = data

    register_gateway_notify(session_key, notify)

    def driver():
        _await_gateway_decision(
            session_key=session_key,
            notify_cb=notify,
            approval_data={
                "command": "echo legacy",
                "description": "legacy pattern",
                "pattern_key": "legacy",
                "pattern_keys": ["legacy"],
            },
            surface="test",
        )

    t = threading.Thread(target=driver, daemon=True)
    t.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if approval_mod._gateway_queues.get(session_key):
            break
        time.sleep(0.02)
    resolve_gateway_approval(session_key, "once")
    t.join(timeout=5)

    assert "approval_id" not in captured["data"]
