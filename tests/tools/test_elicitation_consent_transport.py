"""MCP elicitation consent honours the selected approval transport (#157506, #133946)."""

from __future__ import annotations

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tools import approval, approval_context, approval_gateway_wait, approval_prompt


def _manager(present):
    manager = PluginManager()
    manifest = PluginManifest(name="fixture-approval", version="1.0.0", description="fixture", source="user",
                              key="fixture-approval")
    PluginContext(manifest, manager).register_approval_transport("phone", present)
    return manager


@pytest.fixture
def gateway(monkeypatch):
    """A gateway session with a notifier; the built-in decision is recorded, never awaited."""
    builtin = []
    monkeypatch.setattr(approval_context, "get_current_session_key", lambda *a, **k: "session-a")
    monkeypatch.setattr(approval_context, "_is_single_query_approval_context", lambda: False)
    monkeypatch.setattr(approval_context, "_is_gateway_approval_context", lambda: True)
    monkeypatch.setattr(approval_context, "_get_session_platform", lambda: "telegram")
    monkeypatch.setattr(approval, "_gateway_notify_cb", lambda key: (lambda data: None))
    monkeypatch.setattr(approval_gateway_wait, "_await_gateway_decision",
                        lambda key, cb, data, surface: builtin.append(data) or {"resolved": True, "choice": "once"})
    return builtin


def _select(monkeypatch, manager, fallback=None):
    monkeypatch.setattr(approval_prompt, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(approval_context, "_get_approval_transport_config", lambda: ("phone", fallback))


def test_selected_transport_answers_elicitation_consent(monkeypatch, gateway):
    seen = []
    _select(monkeypatch, _manager(lambda request: seen.append(request) or request.respond("once")))

    assert approval_prompt.request_elicitation_consent("Send the email?", "outbound email", surface="publish") == "accept"
    assert gateway == [], "a selected transport replaces the built-in gateway prompt"
    assert len(seen) == 1
    assert seen[0].pattern_key == "mcp_elicitation" and seen[0].surface == "publish"
    assert seen[0].allowed_choices == ("once", "deny"), "without remember_key only once/deny are offered"


def test_transport_deny_declines(monkeypatch, gateway):
    _select(monkeypatch, _manager(lambda request: request.respond("deny")))

    assert approval_prompt.request_elicitation_consent("Post this?", "social post") == "decline"
    assert gateway == []


def test_transport_failure_declines_unless_builtin_fallback(monkeypatch, gateway):
    def broken(request):
        raise RuntimeError("phone offline")

    _select(monkeypatch, _manager(broken))
    assert approval_prompt.request_elicitation_consent("Pay the invoice?", "payment") == "decline"
    assert gateway == [], "a failed transport never silently reaches a built-in surface"

    _select(monkeypatch, _manager(broken), fallback="builtin")
    assert approval_prompt.request_elicitation_consent("Pay the invoice?", "payment") == "accept"
    assert len(gateway) == 1, "the explicit builtin fallback uses the gateway prompt"


def test_transport_scoped_answer_is_remembered_under_the_key(monkeypatch, gateway):
    persisted = []
    monkeypatch.setattr(approval, "_persist_choice", lambda key, choice, keys: persisted.append((key, choice, keys)))
    seen = []
    _select(monkeypatch, _manager(lambda request: seen.append(request) or request.respond("always")))

    assert approval_prompt.request_elicitation_consent("Pause the campaign?", "ad pause", remember_key="ads:1") == "accept"
    assert seen[0].allowed_choices == ("once", "session", "always", "deny")
    assert persisted == [("session-a", "always", ["ads:1"])]


def test_builtin_selection_keeps_the_gateway_prompt(monkeypatch, gateway):
    monkeypatch.setattr(approval_context, "_get_approval_transport_config", lambda: ("builtin", None))

    assert approval_prompt.request_elicitation_consent("Send the email?", "outbound email") == "accept"
    assert len(gateway) == 1


def test_no_transport_when_nobody_could_answer(monkeypatch, gateway):
    seen = []
    _select(monkeypatch, _manager(lambda request: seen.append(request) or request.respond("once")))
    monkeypatch.setattr(approval_context, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(approval_context, "_get_session_platform", lambda: "")
    monkeypatch.setattr(approval_context, "_no_user_can_answer", lambda: True)

    assert approval_prompt.request_elicitation_consent("Send the email?", "outbound email") == "decline"
    assert seen == [], "a cron or unattended worker still fails closed; the transport is not a new answerer"
