"""``/usage`` before the first turn reads the route that turn builds on, in the session's own profile.

``session.usage`` and ``slash.exec /usage`` on a not-yet-built session guessed the route from the
profile's ``model.provider``: ``auto`` read as no provider at all (a Nous login lost its balance),
the session's composer pick was ignored (a Codex pick under a Nous profile got the Nous CTA),
and the read ran outside the session's profile scope. Both now resolve the route the agent build
resolves (``_resolve_agent_model_runtime``) inside ``_session_profile_runtime_scope``.
"""
from __future__ import annotations

import contextlib
import threading

import pytest

from tui_gateway import server

NOUS_URL = "https://inference-api.nousresearch.com/v1"


@pytest.fixture
def rig(monkeypatch):
    scoped: list[str] = []
    resolved: list[object] = []

    @contextlib.contextmanager
    def scope(session, **kw):
        scoped.append(session.get("profile_home"))
        yield

    def resolve(*, requested=None, **kw):
        assert scoped, "resolved outside the session's profile scope"
        resolved.append(requested)
        # The profile's saved logins: ``auto`` selects Nous; a Codex pick resolves Codex.
        if requested == "openai-codex":
            return {"provider": "openai-codex", "base_url": "https://chatgpt.com/backend-api/codex", "api_key": "c"}
        return {"provider": "nous", "base_url": NOUS_URL, "api_key": "n"}

    monkeypatch.setattr(server, "_session_profile_runtime_scope", scope)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {"provider": "auto", "default": "m"}})
    monkeypatch.setattr(server, "_session_usage_snapshot", lambda session: {})
    monkeypatch.setattr("agent.account_usage.nous_credits_view", lambda **kw: (["Nous credits: 7 remaining"], False))
    monkeypatch.setattr("agent.account_usage.fetch_account_usage", lambda *a, **kw: None)
    return scoped, resolved


def _cold(**extra):
    return {"agent": None, "profile_home": "/profiles/coder", "history": [], "history_lock": threading.Lock(),
            "session_key": "k", **extra}


def _usage(session):
    with server._sessions_lock:
        server._sessions["sid-cold"] = session
    try:
        return server.handle_request({"id": 1, "method": "session.usage", "params": {"session_id": "sid-cold"}})
    finally:
        with server._sessions_lock:
            server._sessions.pop("sid-cold", None)


def test_an_auto_profile_logged_in_to_nous_is_a_nous_route_before_the_first_turn(rig):
    scoped, _ = rig
    reply = _usage(_cold())
    assert reply["result"]["nous_route"] is True, reply
    assert reply["result"]["credits_lines"] == ["Nous credits: 7 remaining"]
    assert scoped == ["/profiles/coder"]
    assert "Nous credits: 7 remaining" in server._format_live_usage_output("sid-cold", _cold(), "")


def test_a_codex_pick_under_a_nous_profile_is_not_a_nous_route(rig):
    from agent.account_usage import NOUS_OTHER_ROUTE_HEADING
    _, resolved = rig
    pick = {"model_override": {"model": "gpt-5.3-codex", "provider": "openai-codex"}}
    reply = _usage(_cold(**pick))
    assert reply["result"]["nous_route"] is False, reply
    assert reply["result"]["credits_lines"][0] == NOUS_OTHER_ROUTE_HEADING  # the balance, as its own section
    assert resolved == ["openai-codex"]
    out = server._format_live_usage_output("sid-cold", _cold(**pick), "")
    assert NOUS_OTHER_ROUTE_HEADING in out
