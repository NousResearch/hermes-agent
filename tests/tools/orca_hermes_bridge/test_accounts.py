"""Account identity and Hermes pool mutation tests for the Orca bridge."""

from __future__ import annotations

import base64
import json
import time

import pytest

from tools.orca_hermes_bridge.accounts import (
    DuplicateProviderAccountError,
    chatgpt_account_id,
    first_usable_provider_id,
    mapped_pool_rows,
    parse_orca_accounts,
    reorder_codex_pool,
)


def _part(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _jwt(provider_account_id: str) -> str:
    claims = {
        "https://api.openai.com/auth": {
            "chatgpt_account_id": provider_account_id,
        }
    }
    return f"{_part({'alg': 'none', 'typ': 'JWT'})}.{_part(claims)}.sig"


def _row(
    credential_id: str,
    provider_account_id: str,
    priority: int,
    *,
    status: str = "ok",
    reset_at: float | None = None,
) -> dict:
    return {
        "id": credential_id,
        "label": credential_id,
        "auth_type": "oauth",
        "priority": priority,
        "source": "manual:device_code",
        "access_token": _jwt(provider_account_id),
        "refresh_token": f"refresh-{credential_id}",
        "last_status": status,
        "last_status_at": time.time() if status == "exhausted" else None,
        "last_error_code": 429 if status == "exhausted" else None,
        "last_error_reason": "rate_limit" if status == "exhausted" else None,
        "last_error_message": "limited" if status == "exhausted" else None,
        "last_error_reset_at": reset_at,
        "request_count": 0,
    }


def _orca_payload(active_id: str | None = "managed-1") -> dict:
    return {
        "codex": {
            "accounts": [
                {
                    "id": "managed-1",
                    "email": "managed@example.test",
                    "providerAccountId": "provider-managed",
                }
            ],
            "activeAccountId": active_id,
            "activeAccountIdsByRuntime": {"host": active_id, "wsl": {}},
            "systemDefault": {
                "hasAuth": True,
                "authKind": "oauth",
                "email": "system@example.test",
                "providerAccountId": "provider-system",
            },
        }
    }


def _write_store(tmp_path, rows: list[dict]) -> None:
    home = tmp_path / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "providers": {}, "credential_pool": {"openai-codex": rows}}, indent=2),
        encoding="utf-8",
    )


def test_parse_snapshot_maps_managed_and_system_accounts():
    managed = parse_orca_accounts(_orca_payload(active_id="managed-1"))
    system = parse_orca_accounts({"result": _orca_payload(active_id=None)})

    assert managed.active.provider_account_id == "provider-managed"
    assert managed.accounts_by_provider_id["provider-managed"].account_id == "managed-1"
    assert managed.accounts_by_provider_id["provider-system"].account_id is None
    assert system.active.provider_account_id == "provider-system"


def test_jwt_mapping_uses_provider_account_id_and_rejects_invalid_tokens():
    assert chatgpt_account_id(_jwt("provider-a")) == "provider-a"
    assert chatgpt_account_id("not-a-jwt") is None


def test_duplicate_hermes_provider_identity_fails_closed():
    rows = [_row("a", "same", 0), _row("b", "same", 1)]

    with pytest.raises(DuplicateProviderAccountError):
        mapped_pool_rows(rows)


def test_first_usable_provider_respects_priority_dead_and_cooldown():
    now = time.time()
    rows = [
        _row("a", "provider-a", 0, status="dead"),
        _row("b", "provider-b", 1, status="exhausted", reset_at=now + 3600),
        _row("c", "provider-c", 2),
    ]

    assert first_usable_provider_id(rows, now=now) == "provider-c"
    rows[1]["last_error_reset_at"] = now - 1
    assert first_usable_provider_id(rows, now=now) == "provider-b"


def test_reorder_moves_selected_first_and_clears_only_selected_status(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    now = time.time()
    rows = [
        _row("a", "provider-a", 0, status="exhausted", reset_at=now + 3600),
        _row("b", "provider-b", 1, status="exhausted", reset_at=now + 3600),
    ]
    _write_store(tmp_path, rows)
    before_tokens = {
        row["id"]: (row["access_token"], row["refresh_token"])
        for row in rows
    }

    assert reorder_codex_pool("provider-b", clear_selected_status=True) is True

    persisted = json.loads((tmp_path / "hermes" / "auth.json").read_text(encoding="utf-8"))
    after = persisted["credential_pool"]["openai-codex"]
    assert [(row["id"], row["priority"]) for row in after] == [("b", 0), ("a", 1)]
    assert after[0]["last_status"] == "ok"
    assert all(after[0][field] is None for field in (
        "last_status_at", "last_error_code", "last_error_reason",
        "last_error_message", "last_error_reset_at",
    ))
    assert after[1]["last_status"] == "exhausted"
    assert {
        row["id"]: (row["access_token"], row["refresh_token"])
        for row in after
    } == before_tokens


def test_missing_selected_identity_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    rows = [_row("a", "provider-a", 0)]
    _write_store(tmp_path, rows)
    auth_path = tmp_path / "hermes" / "auth.json"
    before = auth_path.read_bytes()

    assert reorder_codex_pool("provider-missing", clear_selected_status=True) is False
    assert auth_path.read_bytes() == before
