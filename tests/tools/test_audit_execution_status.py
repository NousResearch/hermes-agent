"""Audit-trail tests for the post-consume execution_status distinction.

Pre-existing security spec invariant: the SQLite row's ``status`` column
records what the USER did (approve/deny/expire). The new
``execution_status`` column records what HAPPENED — whether the command
actually ran or was blocked by a post-consume guard.

Before these fixes, a Phase 3 override would leave status='consumed' on
the store row even though no execution occurred. Retrospective audit
six months later couldn't tell "user approved AND ran" from "user
approved BUT Phase 3 blocked".

These tests pin the post-consume audit semantics.
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
    resolve_gateway_approval_by_id,
    set_default_approval_store,
)
from tools.approval_store_memory import InMemoryApprovalStore


@pytest.fixture(autouse=True)
def _reset():
    prev = approval_mod.get_default_approval_store()
    approval_mod._gateway_queues.clear()
    approval_mod._pending.clear()
    yield
    approval_mod._gateway_queues.clear()
    approval_mod._pending.clear()
    set_default_approval_store(prev)


def _drive_to_resolution(*, pinned_risk_level: str, runtime_risk: str,
                        user_choice: str) -> tuple:
    """Run _await_gateway_decision in a thread, resolve via id-handler.

    Returns (approval_id, final_decision_dict, store).
    """
    store = InMemoryApprovalStore()
    set_default_approval_store(store)
    session_key = "audit-test"
    register_gateway_notify(session_key, lambda data: None)

    decision: dict = {}
    barrier = threading.Event()

    # Force the proposal's pinned risk_level to match the test scenario.
    from dataclasses import replace
    original_submit = store.submit

    def submit_with_pinned(proposal):
        return original_submit(replace(proposal, risk_level=pinned_risk_level))

    # Force the Phase 3 runtime classifier to return the test scenario.
    def fake_runtime(_cmd: str) -> str:
        return runtime_risk

    with patch.object(store, "submit", side_effect=submit_with_pinned), \
         patch.object(approval_mod, "_classify_runtime_risk",
                      side_effect=fake_runtime):

        def driver():
            decision["result"] = _await_gateway_decision(
                session_key=session_key,
                notify_cb=lambda data: (decision.update(data=data), barrier.set()),
                approval_data={
                    "command": "audit-test-command",
                    "description": "world/other-writable permissions",  # medium-tier pinned default
                    "pattern_key": "audit",
                    "pattern_keys": ["audit"],
                },
                surface="test",
            )

        t = threading.Thread(target=driver, daemon=True)
        t.start()
        assert barrier.wait(timeout=5)

        approval_id = decision["data"]["approval_id"]
        resolve_gateway_approval_by_id(session_key, approval_id, user_choice)
        t.join(timeout=5)

    return approval_id, decision["result"], store


def test_user_approve_then_executes_records_executed():
    """Approve + Phase 3 doesn't override → execution_status='executed'."""
    approval_id, decision, store = _drive_to_resolution(
        pinned_risk_level="medium",
        runtime_risk="medium",  # no override
        user_choice="once",
    )
    assert decision["choice"] == "once"  # not overridden
    proposal = store.get(approval_id)
    assert proposal.status == "consumed"           # user clicked /approve
    assert proposal.execution_status == "executed" # AND command actually ran (allowed)
    assert proposal.execution_reason is None
    assert proposal.execution_recorded_at is not None


def test_user_approve_blocked_by_phase3_records_blocked_after_consume():
    """Approve + Phase 3 stricter override → status=consumed but
    execution_status=blocked_after_consume. This is the critical audit
    distinction Risk 1 from review v2 demanded."""
    approval_id, decision, store = _drive_to_resolution(
        pinned_risk_level="medium",
        runtime_risk="high",  # stricter than pinned → override
        user_choice="once",
    )
    assert decision["choice"] == "deny"  # Phase 3 overrode
    proposal = store.get(approval_id)
    assert proposal.status == "consumed", (
        "user clicked /approve → row IS consumed; the audit-distinct field is "
        "execution_status, not status"
    )
    assert proposal.execution_status == "blocked_after_consume", (
        "Phase 3 blocked execution after consume — audit MUST record this"
    )
    assert proposal.execution_reason == "phase3_runtime_stricter"
    assert proposal.execution_recorded_at is not None


def test_user_deny_records_not_started():
    """Deny → status=denied, execution_status=not_started (never ran)."""
    approval_id, decision, store = _drive_to_resolution(
        pinned_risk_level="medium",
        runtime_risk="medium",
        user_choice="deny",
    )
    assert decision["choice"] == "deny"
    proposal = store.get(approval_id)
    assert proposal.status == "denied"
    assert proposal.execution_status == "not_started", (
        "user-denied proposals never reach execution path; execution_status "
        "must remain at default"
    )


def test_orphan_consumed_records_blocked_after_consume():
    """Orphan path (consume succeeds, no waiter) → execution_status=
    blocked_after_consume with reason='orphan_no_waiter'."""
    store = InMemoryApprovalStore()
    set_default_approval_store(store)
    from tools.approval_store import ApprovalProposal
    proposal = ApprovalProposal(
        approval_id="appr-orphan-audit",
        created_at=time.time(),
        expires_at=time.time() + 300,
        session_key="s",
        command="echo orphan",
        risk_level="medium",
        risk_reason="orphan-audit-test",
    )
    store.submit(proposal)
    # NO _ApprovalEntry registered — simulates post-restart state.

    count = resolve_gateway_approval_by_id("s", "appr-orphan-audit", "once")
    assert count == -1, "orphan must return -1 (distinct from 1 signalled)"

    stored = store.get("appr-orphan-audit")
    assert stored.status == "consumed"
    assert stored.execution_status == "blocked_after_consume"
    assert stored.execution_reason == "orphan_no_waiter"


def test_handler_orphan_message_signals_command_not_executed():
    """End-to-end via the gateway handler: orphan-consume must surface
    a DISTINCT user message that the command did NOT execute.
    Previously it returned '✅ Command approved. The agent is resuming...'
    which lied about a side effect."""
    # We can't easily exercise the full handler here without GatewayRunner,
    # but we can verify the locale key exists and the handler routes to it.
    # Direct check: gateway/run.py should return t("gateway.approve.orphan_consumed")
    # when count == -1.
    import re
    from pathlib import Path
    src = Path(__file__).parent.parent.parent / "gateway" / "run.py"
    text = src.read_text()
    # The handler must distinguish count == -1 and return the orphan key.
    pattern = re.compile(
        r"if\s+count\s*==\s*-1.*?return\s+t\(['\"]gateway\.approve\.orphan_consumed['\"]\)",
        re.DOTALL,
    )
    assert pattern.search(text), (
        "gateway/run.py must distinguish count == -1 (orphan) and return "
        "gateway.approve.orphan_consumed locale key, NOT the regular "
        "approve.once_singular which would falsely claim the command ran"
    )

    # Locale key exists
    locale = Path(__file__).parent.parent.parent / "locales" / "en.yaml"
    locale_text = locale.read_text()
    assert "orphan_consumed:" in locale_text
    assert "NOT executed" in locale_text or "not executed" in locale_text.lower()
