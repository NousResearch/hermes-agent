"""Daemon orchestration and lifecycle tests for the Orca/Hermes bridge."""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import pytest

from tools.orca_hermes_bridge.bridge import (
    Bridge,
    build_daemon_launch,
    load_state,
    retry_delay,
    save_state,
)
from tools.orca_hermes_bridge.rpc import OrcaRpcError
from tools.orca_hermes_bridge.state import BridgeState
from tools.orca_hermes_bridge.windows import AlreadyRunningError, SingletonLock


def _jwt(provider_id: str) -> str:
    def part(value: dict) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    claims = {"https://api.openai.com/auth": {"chatgpt_account_id": provider_id}}
    return f"{part({'alg': 'none'})}.{part(claims)}.sig"


def _rows(*, a_ok: bool, b_ok: bool) -> list[dict]:
    now = time.time()
    return [
        {
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
        for credential_id, provider_id, priority, usable in (
            ("a", "provider-a", 0, a_ok),
            ("b", "provider-b", 1, b_ok),
        )
    ]


def _snapshot(active_id: str = "managed-a") -> dict:
    return {
        "codex": {
            "accounts": [
                {"id": "managed-a", "email": "a@example.test", "providerAccountId": "provider-a"},
                {"id": "managed-b", "email": "b@example.test", "providerAccountId": "provider-b"},
            ],
            "activeAccountId": active_id,
            "activeAccountIdsByRuntime": {"host": active_id, "wsl": {}},
            "systemDefault": None,
        }
    }


class FakeRpc:
    def __init__(self, snapshot: dict, on_select=None, error: Exception | None = None):
        self.snapshot = snapshot
        self.on_select = on_select
        self.error = error
        self.selections: list[str | None] = []

    def list_accounts(self) -> dict:
        return self.snapshot

    def select_host_codex(self, account_id: str | None) -> dict:
        self.selections.append(account_id)
        if self.on_select:
            self.on_select()
        if self.error:
            raise self.error
        return self.snapshot


def _bridge(
    tmp_path: Path,
    *,
    rows: list[dict],
    rpc: FakeRpc,
    state: BridgeState,
    mutations: list[tuple[str, bool]] | None = None,
    notifications: list[str] | None = None,
) -> Bridge:
    state_path = tmp_path / "state.json"
    save_state(state_path, state)
    mutation_events = mutations if mutations is not None else []
    notification_events = notifications if notifications is not None else []
    return Bridge(
        state_path=state_path,
        rpc=rpc,
        pool_reader=lambda: rows,
        pool_mutator=lambda provider_id, *, clear_selected_status: mutation_events.append(
            (provider_id, clear_selected_status)
        ) or True,
        notifier=lambda: notification_events.append("qwen"),
        clock=lambda: 100.0,
    )


def test_tick_persists_pending_before_orca_rpc(tmp_path):
    observed: list[BridgeState] = []
    state_path = tmp_path / "state.json"
    rpc = FakeRpc(_snapshot(), on_select=lambda: observed.append(load_state(state_path)))
    mutations: list[tuple[str, bool]] = []
    bridge = _bridge(
        tmp_path,
        rows=_rows(a_ok=False, b_ok=True),
        rpc=rpc,
        state=BridgeState(last_seen_orca_provider_id="provider-a"),
        mutations=mutations,
    )

    bridge.tick()

    assert observed[0].pending_orca_provider_id == "provider-b"
    assert rpc.selections == ["managed-b"]
    assert mutations == [("provider-b", False)]


def test_tick_applies_manual_selection_to_pool(tmp_path):
    mutations: list[tuple[str, bool]] = []
    bridge = _bridge(
        tmp_path,
        rows=_rows(a_ok=True, b_ok=False),
        rpc=FakeRpc(_snapshot("managed-b")),
        state=BridgeState(last_seen_orca_provider_id="provider-a"),
        mutations=mutations,
    )

    bridge.tick()

    assert mutations == [("provider-b", True)]


def test_rpc_failure_clears_pending_and_does_not_mutate_pool(tmp_path):
    mutations: list[tuple[str, bool]] = []
    bridge = _bridge(
        tmp_path,
        rows=_rows(a_ok=False, b_ok=True),
        rpc=FakeRpc(_snapshot(), error=OrcaRpcError("runtime_unavailable", "failed")),
        state=BridgeState(last_seen_orca_provider_id="provider-a"),
        mutations=mutations,
    )

    with pytest.raises(OrcaRpcError):
        bridge.tick()

    assert mutations == []
    assert load_state(bridge.state_path).pending_orca_provider_id is None


def test_malformed_sidecar_does_not_replace_last_known_good_memory(tmp_path):
    bridge = _bridge(
        tmp_path,
        rows=_rows(a_ok=True, b_ok=False),
        rpc=FakeRpc(_snapshot()),
        state=BridgeState(last_seen_orca_provider_id="provider-a"),
    )
    bridge.state_path.write_text("{", encoding="utf-8")

    bridge.tick()

    assert bridge.state.last_seen_orca_provider_id == "provider-a"


def test_qwen_notification_is_emitted_once_across_repeated_ticks(tmp_path):
    notifications: list[str] = []
    bridge = _bridge(
        tmp_path,
        rows=_rows(a_ok=False, b_ok=False),
        rpc=FakeRpc(_snapshot()),
        state=BridgeState(last_seen_orca_provider_id="provider-a"),
        notifications=notifications,
    )

    bridge.tick()
    bridge.tick()

    assert notifications == ["qwen"]
    assert load_state(bridge.state_path).qwen_notified is True


def test_save_state_is_atomic_and_round_trips(tmp_path):
    path = tmp_path / "state.json"
    expected = BridgeState(last_seen_orca_provider_id="provider-a", qwen_notified=True)

    save_state(path, expected)

    assert load_state(path) == expected
    assert list(tmp_path.glob("state.json.tmp.*")) == []


def test_singleton_lock_rejects_second_owner(tmp_path):
    path = tmp_path / "bridge.lock"

    with SingletonLock(path):
        with pytest.raises(AlreadyRunningError):
            with SingletonLock(path):
                raise AssertionError("second owner entered the lock")


def test_retry_delay_is_exponential_and_capped():
    assert [retry_delay(failures) for failures in range(1, 7)] == [2, 4, 8, 16, 30, 30]


def test_build_daemon_launch_uses_current_python_and_repo_cwd(tmp_path):
    python = Path(sys.executable)

    spec = build_daemon_launch(tmp_path, python)

    assert spec.argv == [
        str(python), "-m", "tools.orca_hermes_bridge.bridge", "--daemon"
    ]
    assert spec.cwd == tmp_path
