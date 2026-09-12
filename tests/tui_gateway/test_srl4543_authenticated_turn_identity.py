"""SRL-4543 Tarefa 1: per-turn authenticated identity must not leak across concurrent sessions.

``_set_session_context`` (tui_gateway/server.py) already resolves ``browser_control_principal``
from ``WSTransport.auth_identity`` at admission time, but never forwards the authenticated
``user_id`` into ``set_session_vars``/``HERMES_SESSION_USER_ID`` — so a subprocess or tool call
made mid-turn cannot see which authenticated user owns the turn. This reproduces two concurrent
admissions (distinct ticket-authenticated identities, same pattern as the dashboard WS upgrade)
and asserts each turn's ``HERMES_SESSION_USER_ID`` matches ITS OWN owner, never the other's and
never empty.
"""
import concurrent.futures
import threading
from types import SimpleNamespace

from gateway.session_context import get_session_env


def test_concurrent_authenticated_turns_see_own_user_id(monkeypatch):
    from hermes_cli import web_server
    from hermes_cli.web_server_chat import _ws_auth_ok
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket
    from tui_gateway import server
    from tui_gateway.ws import WSTransport

    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)
    owners = ("alice@example.invalid", "bob@example.invalid")
    sessions = {}
    for owner in owners:
        ticket = mint_ticket(user_id=owner, provider="fixture-provider")
        ws = SimpleNamespace(
            query_params={"ticket": ticket}, headers={},
            client=SimpleNamespace(host="127.0.0.1"),
            url=SimpleNamespace(path="/api/ws"))
        assert _ws_auth_ok(ws)
        transport = WSTransport(ws, SimpleNamespace(), auth_identity=ws._hermes_auth_identity)
        sessions[owner] = {
            "transport": transport, "session_key": owner,
            "profile": "default", "agent": SimpleNamespace(session_id=owner)}
    monkeypatch.setattr(server, "_sessions", sessions)

    barrier = threading.Barrier(2)

    def read_context(owner):
        tokens = server._set_session_context(owner)
        try:
            barrier.wait(timeout=10)
            return owner, get_session_env("HERMES_SESSION_USER_ID")
        finally:
            server._clear_session_context(tokens)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(pool.map(read_context, owners))

    assert all(results.values()), results
    assert results[owners[0]] != results[owners[1]], results
    assert results == {owner: owner for owner in owners}, results


def test_unauthenticated_session_gets_empty_user_id(monkeypatch):
    """No auth identity (or session_key not in ``_sessions``) must never fall back to an
    OS user, hostname, or config-derived identity."""
    from tui_gateway import server

    monkeypatch.setattr(server, "_sessions", {})
    tokens = server._set_session_context("unknown-session-key")
    try:
        assert get_session_env("HERMES_SESSION_USER_ID") == ""
    finally:
        server._clear_session_context(tokens)


def test_clear_session_context_does_not_leak_user_id_to_next_turn(monkeypatch):
    """After ``_clear_session_context`` runs (e.g. a cancelled/interleaved turn), the next
    read on the same thread/task must not observe the cleared turn's authenticated user_id."""
    from hermes_cli import web_server
    from hermes_cli.web_server_chat import _ws_auth_ok
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket
    from tui_gateway import server
    from tui_gateway.ws import WSTransport

    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)
    owner = "carol@example.invalid"
    ticket = mint_ticket(user_id=owner, provider="fixture-provider")
    ws = SimpleNamespace(
        query_params={"ticket": ticket}, headers={},
        client=SimpleNamespace(host="127.0.0.1"),
        url=SimpleNamespace(path="/api/ws"))
    assert _ws_auth_ok(ws)
    transport = WSTransport(ws, SimpleNamespace(), auth_identity=ws._hermes_auth_identity)
    sessions = {owner: {
        "transport": transport, "session_key": owner,
        "profile": "default", "agent": SimpleNamespace(session_id=owner)}}
    monkeypatch.setattr(server, "_sessions", sessions)

    tokens = server._set_session_context(owner)
    assert get_session_env("HERMES_SESSION_USER_ID") == owner
    server._clear_session_context(tokens)

    assert get_session_env("HERMES_SESSION_USER_ID") == ""
