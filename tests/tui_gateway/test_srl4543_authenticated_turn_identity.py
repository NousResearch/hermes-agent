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


def _fixture_transport(monkeypatch, owner: str):
    """A real WSTransport carrying a server-minted ``{user_id, provider}`` identity for
    ``owner``, built through the same ticket-mint/verify path a real WS upgrade uses."""
    from hermes_cli import web_server
    from hermes_cli.web_server_chat import _ws_auth_ok
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket
    from tui_gateway.ws import WSTransport

    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)
    ticket = mint_ticket(user_id=owner, provider="fixture-provider")
    ws = SimpleNamespace(
        query_params={"ticket": ticket}, headers={},
        client=SimpleNamespace(host="127.0.0.1"),
        url=SimpleNamespace(path="/api/ws"))
    assert _ws_auth_ok(ws)
    return WSTransport(ws, SimpleNamespace(), auth_identity=ws._hermes_auth_identity)


def test_reattach_during_turn_does_not_change_admitted_identity(monkeypatch):
    """A turn is admitted for alice (outer ``_set_session_context``, mirrors
    ``_prepare_turn_input``'s per-turn admission). While that turn is still running, a
    different principal (carol) reattaches to the SAME session_key — exactly how
    ``session_transports._attach_session_transport`` swaps ``session["transport"]`` in place
    when the prior peer is no longer live. A NESTED ``_set_session_context`` call in the same
    task (mirrors ``_persist_live_session_system_prompt``/model_switch calling it again before
    the turn's own ``_clear_session_context`` runs) must still see alice, the admitting
    principal — never carol, and the outer admission must survive the nested call's cleanup."""
    from tui_gateway import server

    owner, reattacher = "alice@example.invalid", "carol@example.invalid"
    session = {
        "transport": _fixture_transport(monkeypatch, owner), "session_key": "shared-session-key",
        "profile": "default", "agent": SimpleNamespace(session_id="shared-session-key")}
    monkeypatch.setattr(server, "_sessions", {"shared-session-key": session})

    outer_tokens = server._set_session_context("shared-session-key")
    try:
        assert get_session_env("HERMES_SESSION_USER_ID") == owner

        # Reattach: session["transport"] is mutated in place while alice's turn is still open.
        session["transport"] = _fixture_transport(monkeypatch, reattacher)

        inner_tokens = server._set_session_context("shared-session-key")
        try:
            assert get_session_env("HERMES_SESSION_USER_ID") == owner
        finally:
            server._clear_session_context(inner_tokens)

        # The outer (turn-admitting) identity must still be intact after the nested call.
        assert get_session_env("HERMES_SESSION_USER_ID") == owner
    finally:
        server._clear_session_context(outer_tokens)


def test_fresh_admission_after_reattach_reflects_new_owner_not_original(monkeypatch):
    """Reattach is not silent identity inheritance: once alice's turn has fully ended (context
    cleared) and carol has taken over the session_key, the NEXT fresh admission (a new turn, no
    already-bound context for this session_key in this task) must reflect carol — never keep
    serving alice's identity just because she opened the session first."""
    from tui_gateway import server

    owner, reattacher = "alice@example.invalid", "carol@example.invalid"
    session = {
        "transport": _fixture_transport(monkeypatch, owner), "session_key": "resumed-session-key",
        "profile": "default", "agent": SimpleNamespace(session_id="resumed-session-key")}
    monkeypatch.setattr(server, "_sessions", {"resumed-session-key": session})

    tokens = server._set_session_context("resumed-session-key")
    assert get_session_env("HERMES_SESSION_USER_ID") == owner
    server._clear_session_context(tokens)

    session["transport"] = _fixture_transport(monkeypatch, reattacher)
    tokens = server._set_session_context("resumed-session-key")
    try:
        assert get_session_env("HERMES_SESSION_USER_ID") == reattacher
    finally:
        server._clear_session_context(tokens)


def test_legacy_session_without_transport_never_backfills_or_inherits_other_session(monkeypatch):
    """A session created before this fix (no ``transport`` key at all) must get
    ``user_id=""`` forever: never from os.environ, never from config, and never from a
    DIFFERENT session_key admitted earlier in the same task (cross-session-key isolation for
    the same nested-identity guard that protects same-key reattach)."""
    from tui_gateway import server

    monkeypatch.setenv("HERMES_SESSION_USER_ID", "leaked-from-os-environ")
    fresh_owner = "dave@example.invalid"
    fresh_session = {
        "transport": _fixture_transport(monkeypatch, fresh_owner), "session_key": "fresh-key",
        "profile": "default", "agent": SimpleNamespace(session_id="fresh-key")}
    legacy_session = {"session_key": "legacy-key", "profile": "default"}  # no "transport" key
    monkeypatch.setattr(server, "_sessions", {"fresh-key": fresh_session, "legacy-key": legacy_session})

    fresh_tokens = server._set_session_context("fresh-key")
    try:
        assert get_session_env("HERMES_SESSION_USER_ID") == fresh_owner

        # A DIFFERENT session_key, looked up while fresh-key's context is still bound in this
        # same task, must derive its own (empty) identity — not inherit fresh_owner.
        legacy_tokens = server._set_session_context("legacy-key")
        try:
            assert get_session_env("HERMES_SESSION_USER_ID") == ""
        finally:
            server._clear_session_context(legacy_tokens)

        assert get_session_env("HERMES_SESSION_USER_ID") == fresh_owner
    finally:
        server._clear_session_context(fresh_tokens)


def test_concurrent_turns_reach_real_tool_with_own_user_id(monkeypatch):
    """Not just a raw ContextVar read: a real tool-facing function
    (``tools.kanban_tools._resolve_notify_target``, used to address kanban notify/wake
    deliveries) must resolve to the ADMITTING turn's own user_id for each of two concurrent
    admissions, never the other's."""
    from tui_gateway import server
    from tools.kanban_tools import _resolve_notify_target

    monkeypatch.setenv("HERMES_PROFILE", "fixture-profile")  # skip get_active_profile_name() I/O
    owners = ("frank@example.invalid", "grace@example.invalid")
    sessions = {
        owner: {
            "transport": _fixture_transport(monkeypatch, owner), "session_key": owner,
            "profile": "default", "agent": SimpleNamespace(session_id=owner)}
        for owner in owners}
    monkeypatch.setattr(server, "_sessions", sessions)

    barrier = threading.Barrier(2)

    def read_via_tool(owner):
        tokens = server._set_session_context(owner)
        try:
            barrier.wait(timeout=10)
            target = _resolve_notify_target()
            return owner, (target or {}).get("user_id")
        finally:
            server._clear_session_context(tokens)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(pool.map(read_via_tool, owners))

    assert all(results.values()), results
    assert results == {owner: owner for owner in owners}, results


def test_cancel_of_one_turn_does_not_disturb_a_concurrently_active_turn(monkeypatch):
    """A turn cancelled mid-flight (its ``finally`` runs ``_clear_session_context``) must not
    disturb a DIFFERENT turn concurrently active on another thread: ContextVars are per-task,
    so clearing one thread's vars must never be observable on another thread's."""
    from tui_gateway import server

    owner_a, owner_b = "henry@example.invalid", "irene@example.invalid"
    sessions = {
        owner: {
            "transport": _fixture_transport(monkeypatch, owner), "session_key": owner,
            "profile": "default", "agent": SimpleNamespace(session_id=owner)}
        for owner in (owner_a, owner_b)}
    monkeypatch.setattr(server, "_sessions", sessions)

    a_cancelled = threading.Event()
    errors: list[str] = []

    def run_a():
        tokens = server._set_session_context(owner_a)
        try:
            if get_session_env("HERMES_SESSION_USER_ID") != owner_a:
                errors.append("A did not see its own identity before cancellation")
        finally:
            server._clear_session_context(tokens)  # simulate cancellation's finally
            a_cancelled.set()

    def run_b():
        tokens = server._set_session_context(owner_b)
        try:
            if get_session_env("HERMES_SESSION_USER_ID") != owner_b:
                errors.append("B did not see its own identity before A cancelled")
            if not a_cancelled.wait(timeout=10):
                errors.append("timed out waiting for A's cancellation")
            if get_session_env("HERMES_SESSION_USER_ID") != owner_b:
                errors.append("B's identity changed after A's concurrent cancellation")
        finally:
            server._clear_session_context(tokens)

    ta, tb = threading.Thread(target=run_a), threading.Thread(target=run_b)
    ta.start(); tb.start()
    ta.join(10); tb.join(10)

    assert not errors, errors
