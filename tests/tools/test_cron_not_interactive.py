"""Cron approval context must never resolve as interactive (layer-2 fix for #110932).

Layer 1 (#110942) strips presence vars at the launch path. This test pins the
deeper invariant: even if HERMES_INTERACTIVE / HERMES_EXEC_ASK leak into a cron
worker by some other route, `_approval_transport()` must not treat the context
as CLI-interactive — nobody can answer the card, so the gate must fall through
to `approvals.cron_mode` instead of hanging.
"""

import os

import pytest

from tools import approval as approval_mod
from tools.approval_context import set_hermes_interactive_context


@pytest.fixture
def leaked_presence(monkeypatch):
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    # Explicitly NOT a gateway session: gateway handling is a separate path.
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    # Mark the approval decision as running inside cron (session env is the
    # fallback path used by external workers; contextvars win in-process).
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")


def test_cron_context_is_not_cli_even_with_leaked_interactive(monkeypatch, leaked_presence):
    callback, is_cli, is_gateway, is_ask = approval_mod._presence()
    assert is_cli is False, "cron context must not resolve as interactive CLI"
    assert is_gateway is False, "cron context must not resolve as gateway"
    assert is_ask is False, "HERMES_EXEC_ASK leak must not make cron ask"


def test_non_cron_interactive_still_cli(monkeypatch):
    # Guard against over-broad fix: a real interactive session stays interactive.
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    callback, is_cli, is_gateway, is_ask = approval_mod._presence()
    assert is_cli is True


def test_cron_contextvar_interactive_also_not_cli(monkeypatch, leaked_presence):
    # The contextvar binding is stronger than env; cron must win over it too.
    token = set_hermes_interactive_context(True)
    try:
        callback, is_cli, is_gateway, is_ask = approval_mod._presence()
        assert is_cli is False
    finally:
        from tools.approval_context import reset_hermes_interactive_context
        reset_hermes_interactive_context(token)
