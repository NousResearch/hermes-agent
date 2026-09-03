"""Pure reconciliation-state tests for the Orca/Hermes bridge."""

from __future__ import annotations

import base64
import json
import time

from tools.orca_hermes_bridge.accounts import OrcaAccount, OrcaSnapshot
from tools.orca_hermes_bridge.state import (
    BridgeState,
    OrcaMutation,
    PoolMutation,
    reconcile,
)


def _jwt(provider_id: str) -> str:
    def part(value: dict) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    claims = {"https://api.openai.com/auth": {"chatgpt_account_id": provider_id}}
    return f"{part({'alg': 'none'})}.{part(claims)}.sig"


def _row(credential_id: str, provider_id: str, priority: int, *, usable: bool) -> dict:
    now = time.time()
    return {
        "id": credential_id,
        "label": credential_id,
        "auth_type": "oauth",
        "priority": priority,
        "source": "manual:device_code",
        "access_token": _jwt(provider_id),
        "refresh_token": f"refresh-{credential_id}",
        "last_status": "ok" if usable else "exhausted",
        "last_status_at": None if usable else now,
        "last_error_code": None if usable else 429,
        "last_error_reset_at": None if usable else now + 3600,
    }


def _rows(*, a_ok: bool, b_ok: bool) -> list[dict]:
    return [
        _row("a", "provider-a", 0, usable=a_ok),
        _row("b", "provider-b", 1, usable=b_ok),
    ]


def _snapshot(active_provider: str) -> OrcaSnapshot:
    system = OrcaAccount(None, "provider-a", "system@example.test")
    managed = OrcaAccount("managed-b", "provider-b", "managed@example.test")
    accounts = {"provider-a": system, "provider-b": managed}
    return OrcaSnapshot(active=accounts[active_provider], accounts_by_provider_id=accounts)


def test_startup_applies_current_orca_selection_as_manual_probe():
    decision = reconcile(_snapshot("provider-b"), _rows(a_ok=True, b_ok=True), BridgeState(), 100.0)

    assert decision.pool_mutation == PoolMutation("provider-b", clear_selected_status=True)
    assert decision.orca_mutation is None
    assert decision.state.last_seen_orca_provider_id == "provider-b"


def test_manual_orca_change_reorders_and_clears_only_selected_status():
    state = BridgeState(last_seen_orca_provider_id="provider-a")

    decision = reconcile(_snapshot("provider-b"), _rows(a_ok=True, b_ok=False), state, 100.0)

    assert decision.pool_mutation == PoolMutation("provider-b", clear_selected_status=True)
    assert decision.orca_mutation is None


def test_exhausted_displayed_account_selects_next_usable_account_in_orca():
    state = BridgeState(last_seen_orca_provider_id="provider-a")

    decision = reconcile(_snapshot("provider-a"), _rows(a_ok=False, b_ok=True), state, 100.0)

    assert decision.orca_mutation == OrcaMutation("managed-b", "provider-b")
    assert decision.pool_mutation == PoolMutation("provider-b", clear_selected_status=False)
    assert decision.state.pending_orca_provider_id == "provider-b"
    assert decision.state.pending_started_at == 100.0


def test_rpc_echo_does_not_clear_exhaustion_as_manual_probe():
    state = BridgeState(
        last_seen_orca_provider_id="provider-a",
        pending_orca_provider_id="provider-b",
        pending_started_at=99.0,
    )

    decision = reconcile(_snapshot("provider-b"), _rows(a_ok=False, b_ok=True), state, 100.0)

    assert decision.pool_mutation == PoolMutation("provider-b", clear_selected_status=False)
    assert decision.orca_mutation is None
    assert decision.state.pending_orca_provider_id is None
    assert decision.state.last_seen_orca_provider_id == "provider-b"


def test_system_default_is_selected_with_null_orca_account_id():
    rows = [
        _row("b", "provider-b", 0, usable=False),
        _row("a", "provider-a", 1, usable=True),
    ]
    state = BridgeState(last_seen_orca_provider_id="provider-b")

    decision = reconcile(_snapshot("provider-b"), rows, state, 100.0)

    assert decision.orca_mutation == OrcaMutation(None, "provider-a")


def test_all_codex_unavailable_notifies_once_and_keeps_orca_selection():
    state = BridgeState(last_seen_orca_provider_id="provider-a")

    first = reconcile(_snapshot("provider-a"), _rows(a_ok=False, b_ok=False), state, 100.0)
    second = reconcile(_snapshot("provider-a"), _rows(a_ok=False, b_ok=False), first.state, 101.0)

    assert first.notify_qwen is True
    assert first.orca_mutation is None
    assert first.pool_mutation is None
    assert second.notify_qwen is False
    assert second.orca_mutation is None


def test_codex_recovery_rearms_qwen_notice_without_automatic_jump():
    state = BridgeState(last_seen_orca_provider_id="provider-a", qwen_notified=True)

    decision = reconcile(_snapshot("provider-a"), _rows(a_ok=True, b_ok=False), state, 100.0)

    assert decision.state.qwen_notified is False
    assert decision.orca_mutation is None
    assert decision.pool_mutation is None


def test_unknown_orca_account_does_not_mutate_known_hermes_rows():
    unknown = OrcaAccount("managed-c", "provider-c", "unknown@example.test")
    snapshot = OrcaSnapshot(active=unknown, accounts_by_provider_id={"provider-c": unknown})
    state = BridgeState(last_seen_orca_provider_id="provider-a")

    decision = reconcile(snapshot, _rows(a_ok=True, b_ok=True), state, 100.0)

    assert decision.pool_mutation is None
    assert decision.orca_mutation is None
    assert decision.state.last_seen_orca_provider_id == "provider-c"


def test_pending_selection_expires_and_is_replanned_after_thirty_seconds():
    state = BridgeState(
        last_seen_orca_provider_id="provider-a",
        pending_orca_provider_id="provider-b",
        pending_started_at=60.0,
    )

    decision = reconcile(_snapshot("provider-a"), _rows(a_ok=False, b_ok=True), state, 100.0)

    assert decision.orca_mutation == OrcaMutation("managed-b", "provider-b")
    assert decision.state.pending_started_at == 100.0
