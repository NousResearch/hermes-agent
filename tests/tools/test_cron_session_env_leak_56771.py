"""Regression tests for #56771 — leaked HERMES_CRON_SESSION env blocking interactive sessions."""

from __future__ import annotations

import pytest

from tools import approval as A


@pytest.fixture
def manual_approval(monkeypatch):
    monkeypatch.setattr(A, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(A, "_get_cron_approval_mode", lambda: "deny")


def test_leaked_cron_env_does_not_block_gateway_execute_code(manual_approval, monkeypatch):
    """Interactive gateway with leaked HERMES_CRON_SESSION must use gateway path."""
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    session_key = "leak-test-gateway"
    token = A.set_current_session_key(session_key)
    try:
        assert A._is_cron_approval_context() is False
        assert A._is_gateway_approval_context() is True
        res = A.check_execute_code_guard("print('hello')", "local")
        assert res["approved"] is False
        assert res.get("outcome") != "blocked"
        assert res.get("status") in ("approval_required", "pending_approval")
    finally:
        A.reset_current_session_key(token)


def test_leaked_cron_env_does_not_block_contextvar_gateway_execute_code(
    manual_approval, monkeypatch
):
    """Modern gateway paths bind session_key/platform without HERMES_GATEWAY_SESSION."""
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    from gateway.session_context import clear_session_vars, set_session_vars

    tokens = set_session_vars(platform="telegram", chat_id="123", session_key="tg-session")
    try:
        assert A._is_cron_approval_context() is False
        assert A._is_gateway_approval_context() is True
        res = A.check_execute_code_guard("print('hello')", "local")
        assert res["approved"] is False
        assert res.get("outcome") != "blocked"
        assert res.get("status") in ("approval_required", "pending_approval")
    finally:
        clear_session_vars(tokens)


def test_real_cron_still_blocks_execute_code(manual_approval, monkeypatch):
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    from gateway.session_context import clear_session_vars, set_session_vars

    tokens = set_session_vars(platform="telegram", chat_id="123")
    try:
        assert A._is_cron_approval_context() is True
        assert A._is_gateway_approval_context() is False
        res = A.check_execute_code_guard("print('hello')", "local")
        assert res["approved"] is False
        assert res["outcome"] == "blocked"
        assert "Cron jobs run without a user present" in res["message"]
    finally:
        clear_session_vars(tokens)
