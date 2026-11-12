"""Native app approval works in a local viewer without a masked-password callback."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("surface,allowed", [
    ("cli", True), ("tui", True), ("none", False), ("cron", False),
    ("single_query", False), ("api_server", False), ("webhook", False),
])
def test_native_approval_uses_local_authority_not_password_prompt(monkeypatch, surface, allowed):
    from agent.vault_backends import unlock
    from tools import approval_context, browser_vault_tool

    calls = []
    backend = SimpleNamespace(
        name="onepassword", display_name="1Password", needs_unlock=True,
        supports_app_unlock=True, is_unlocked=lambda: False,
        unlock=lambda password: calls.append(password),
    )
    monkeypatch.setattr("agent.vault_backends.enabled_backends", lambda: [backend])
    monkeypatch.setattr(unlock, "get_unlock_prompt_callback", lambda: None)
    monkeypatch.setattr(approval_context, "_is_interactive_cli", lambda: surface == "cli")
    # An unattended turn must stay blocked even if a local-session predicate is true.
    monkeypatch.setattr(approval_context, "is_local_interactive_session", lambda: surface != "none")
    monkeypatch.setattr(approval_context, "_is_single_query_approval_context", lambda: surface == "single_query")
    monkeypatch.setattr(approval_context, "_is_cron_approval_context", lambda: surface == "cron")
    monkeypatch.setattr(approval_context, "_is_unattended_platform_approval_context",
                        lambda: surface in {"api_server", "webhook"})

    listed = json.loads(browser_vault_tool.browser_vault_list())
    assert listed["locked"][0]["unlock"] == (
        "browser_vault action=unlock" if allowed else "unavailable_in_this_session")
    result = json.loads(browser_vault_tool._handle_vault({"action": "unlock", "backend": "onepassword"}))
    assert result["success"] is allowed
    assert calls == ([""] if allowed else [])
    if not allowed:
        assert result["error_type"] == "unlock_unavailable"


def test_manual_method_still_requires_masked_password_callback(monkeypatch):
    from agent.vault_backends import unlock
    from tools import approval_context, browser_vault_tool

    backend = SimpleNamespace(
        name="onepassword", display_name="1Password", needs_unlock=True,
        supports_app_unlock=True, is_unlocked=lambda: False,
        unlock=lambda password: pytest.fail("manual unlock cannot run without its prompt"),
    )
    monkeypatch.setattr("agent.vault_backends.enabled_backends", lambda: [backend])
    monkeypatch.setattr(unlock, "get_unlock_prompt_callback", lambda: None)
    monkeypatch.setattr(approval_context, "_is_interactive_cli", lambda: True)
    monkeypatch.setattr(approval_context, "_no_user_can_answer", lambda: False)
    result = json.loads(browser_vault_tool._handle_vault(
        {"action": "unlock", "backend": "onepassword", "method": "password"}))
    assert result["success"] is False
    assert result["error_type"] == "unlock_unavailable"
