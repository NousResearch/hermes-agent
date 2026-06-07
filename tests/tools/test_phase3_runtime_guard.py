"""Phase 3: stricter-runtime fail-closed guard.

The pinned-policy contract says: at execution-thread wake-up, runtime
classification may only override pinned policy when STRICTER (deny-tier
escalation). Runtime ranking the command as weaker than the pinned
proposal MUST NOT downgrade the pinned decision.

These tests pin a proposal at one risk level, mutate the live classifier
between submit and resolution, then drive the flow and assert the guard
fires (or does not fire) correctly.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import patch

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
def _reset():
    prev_store = approval_mod.get_default_approval_store()
    approval_mod._gateway_queues.clear()
    approval_mod._pending.clear()
    yield
    approval_mod._gateway_queues.clear()
    approval_mod._pending.clear()
    set_default_approval_store(prev_store)


def _drive_with_pinned_risk_override(
    store: InMemoryApprovalStore,
    *,
    pinned_risk: str,
    runtime_risk: str,
    user_choice: str = "once",
) -> dict:
    """Run _await_gateway_decision with a controlled pinned risk and a
    controlled runtime-classifier result.

    The pinned risk is set by monkeypatching ApprovalProposal construction
    inside the helper (we patch the submit step to write the desired
    risk_level). The runtime classifier is mocked via
    _classify_runtime_risk.
    """
    set_default_approval_store(store)
    session_key = "phase3-test"
    register_gateway_notify(session_key, lambda data: None)

    decision: dict[str, Any] = {}
    barrier = threading.Event()

    # Patch the proposal-creation path to force pinned_risk.
    original_submit = store.submit

    def submit_with_pinned_risk(proposal):
        from dataclasses import replace
        return original_submit(replace(proposal, risk_level=pinned_risk))

    # Patch _classify_runtime_risk to return runtime_risk at wake-up.
    def fake_runtime_classifier(cmd: str) -> str:
        return runtime_risk

    with patch.object(store, "submit", side_effect=submit_with_pinned_risk), \
         patch.object(approval_mod, "_classify_runtime_risk",
                      side_effect=fake_runtime_classifier):
        def driver():
            decision["result"] = _await_gateway_decision(
                session_key=session_key,
                notify_cb=lambda data: barrier.set(),
                approval_data={
                    "command": "phase3-test-command",
                    "description": "test pattern",
                    "pattern_key": "test",
                    "pattern_keys": ["test"],
                },
                surface="test",
            )

        t = threading.Thread(target=driver, daemon=True)
        t.start()

        # Wait until notify fires (= proposal was submitted, entry enqueued)
        assert barrier.wait(timeout=5), "driver never invoked notify"

        # Find the entry's approval_id from the queue
        entry = approval_mod._gateway_queues[session_key][0]
        approval_id = entry.approval_id
        assert approval_id is not None

        # Simulate user approving via the by-id path; this fires the guard
        # inside the driver thread after the event resolves.
        from tools.approval import resolve_gateway_approval_by_id
        resolve_gateway_approval_by_id(session_key, approval_id, user_choice)

        t.join(timeout=5)

    return {**decision["result"], "_approval_id": approval_id}


def test_pinned_high_runtime_low_does_not_downgrade():
    """pinned=high + runtime=low → choice stays approve (no downgrade)."""
    store = InMemoryApprovalStore()
    result = _drive_with_pinned_risk_override(
        store, pinned_risk="high", runtime_risk="low",
    )
    assert result["resolved"] is True
    assert result["choice"] == "once"  # NOT denied — pinned wins


def test_pinned_low_runtime_high_fails_closed():
    """pinned=low + runtime=high → choice overridden to deny."""
    store = InMemoryApprovalStore()
    result = _drive_with_pinned_risk_override(
        store, pinned_risk="low", runtime_risk="high",
    )
    assert result["resolved"] is True
    assert result["choice"] == "deny", (
        "Phase 3 guard must fail closed when runtime is stricter than pinned"
    )


def test_pinned_medium_runtime_high_fails_closed():
    """pinned=medium + runtime=high → fail closed."""
    store = InMemoryApprovalStore()
    result = _drive_with_pinned_risk_override(
        store, pinned_risk="medium", runtime_risk="high",
    )
    assert result["choice"] == "deny"


def test_matching_risk_levels_proceed_normally():
    """pinned=medium + runtime=medium → no override, proceed."""
    store = InMemoryApprovalStore()
    result = _drive_with_pinned_risk_override(
        store, pinned_risk="medium", runtime_risk="medium",
    )
    assert result["choice"] == "once"


def test_missing_pinned_risk_rejected_at_submit_time():
    """A proposal with risk_level='' MUST be rejected at construction,
    so nothing reaches the runtime guard at all.

    Defense in depth: even if validation were somehow bypassed, the
    Phase 3 guard ranks unknown risk at 999 (no runtime can be stricter
    than unknown). But the primary security boundary is now submit-time
    refusal — the spec's pinned-completeness invariant says missing
    fields MUST prevent proposal creation.
    """
    from tools.approval_store import ApprovalProposal
    with pytest.raises(ValueError, match="risk_level"):
        ApprovalProposal(
            approval_id="appr-no-risk",
            created_at=time.time(),
            session_key="s",
            command="echo x",
            risk_level="",          # invalid — must be low/medium/high
            risk_reason="test",
        )


def test_phase3_no_downgrade_on_runtime_failure():
    """If runtime classifier itself fails/raises, do not silently proceed."""
    store = InMemoryApprovalStore()
    set_default_approval_store(store)
    session_key = "phase3-runtime-fail"
    register_gateway_notify(session_key, lambda data: None)

    decision: dict[str, Any] = {}
    barrier = threading.Event()

    def raising_classifier(cmd: str) -> str:
        raise RuntimeError("classifier crashed")

    with patch.object(approval_mod, "_classify_runtime_risk",
                      side_effect=raising_classifier):
        def driver():
            decision["result"] = _await_gateway_decision(
                session_key=session_key,
                notify_cb=lambda data: barrier.set(),
                approval_data={
                    "command": "test-cmd",
                    "description": "test",
                    "pattern_key": "test",
                    "pattern_keys": ["test"],
                },
                surface="test",
            )

        t = threading.Thread(target=driver, daemon=True)
        t.start()
        assert barrier.wait(timeout=5)

        entry = approval_mod._gateway_queues[session_key][0]
        approval_id = entry.approval_id
        from tools.approval import resolve_gateway_approval_by_id
        resolve_gateway_approval_by_id(session_key, approval_id, "once")

        t.join(timeout=5)

    # Classifier raise → guard's explicit try/except catches → choice
    # overridden to deny. Thread returns cleanly with deny decision.
    assert "result" in decision, "driver thread crashed; guard should catch"
    assert decision["result"]["choice"] == "deny", (
        "runtime classifier exception MUST fail closed to deny"
    )
