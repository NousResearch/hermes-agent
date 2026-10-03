"""Unattended approval contexts never resolve as interactive (#110932).

A gateway sets HERMES_EXEC_ASK=1 at startup and hands its environ to every external cron
worker; interactive launches export HERMES_INTERACTIVE=1. Inside cron nobody can answer the
card, so ``_presence()`` must clear the trio and let the gate resolve from
``approvals.cron_mode``. Unattended platforms also clear it: an external request has no
interactive approver, so a dangerous command must resolve from ``approvals.unattended_mode``.
"""

import pytest

from tools import approval as approval_mod


@pytest.fixture
def leaked_presence(monkeypatch):
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)


def test_cron_context_clears_leaked_presence(monkeypatch, leaked_presence):
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    _, is_cli, is_gateway, is_ask = approval_mod._presence()
    assert (is_cli, is_gateway, is_ask) == (False, False, False)


def test_interactive_session_keeps_presence(monkeypatch, leaked_presence):
    _, is_cli, is_gateway, is_ask = approval_mod._presence()
    assert (is_cli, is_gateway, is_ask) == (True, True, True)


def test_api_server_platform_clears_exec_ask_and_fails_fast(monkeypatch, leaked_presence):
    """api_server has no interactive approver, even if the gateway environment leaked ask mode."""
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "api_server")
    _, _, _, is_ask = approval_mod._presence()
    assert is_ask is False
