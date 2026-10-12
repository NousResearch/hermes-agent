"""Contract test: tui_gateway._set_session_context must bind the signed-in
dashboard user into HERMES_SESSION_USER_ID / HERMES_SESSION_USER_NAME.

Regression for the bug where _set_session_context called set_session_vars
WITHOUT user_id, so every tool that attributes work to a person (kanban_tools,
send_message_tool, cronjob_job_args, the terminal_tool_background watcher
fields) read an empty identity on the dashboard/WebSocket route, even though
the record was admitted under a real login and _make_agent already builds the
agent with that same user id.

An ungated gateway names no login and must keep binding "" — the vars carry a
verified identity or nothing at all.
"""
import types

import pytest

from gateway.session_context import (
    _UNSET,
    _VAR_MAP,
    get_session_env,
)
import tui_gateway.server as server


@pytest.fixture(autouse=True)
def _reset_contextvars():
    """Reset all session contextvars to _UNSET between tests (see
    test_session_id_injection.py: tests share one thread context)."""
    yield
    for var in _VAR_MAP.values():
        var.set(_UNSET)


class _FakeAgent:
    def __init__(self, session_id):
        self.session_id = session_id


def _install_session(monkeypatch, *, session_key, transport=None, **extra):
    """Register a fake session in server._sessions for the duration of a test."""
    sess = {
        "session_key": session_key,
        "source": "desktop",
        "agent": _FakeAgent("20260921_cafebabe"),
        "cwd": "/home/user",
        **extra,
    }
    if transport is not None:
        sess["transport"] = transport
    monkeypatch.setattr(server, "_sessions", {session_key: sess}, raising=False)
    return sess


def _transport(provider, user_id):
    return types.SimpleNamespace(auth_identity={"provider": provider, "user_id": user_id})


def test_binds_signed_in_user_from_the_transport_identity(monkeypatch):
    """The provider-prefixed login reaches the tools, and the bare login id is
    the display name (the minted identity carries no email or display name)."""
    _install_session(
        monkeypatch, session_key="skey-oidc", transport=_transport("oidc", "carol")
    )

    tokens = server._set_session_context("skey-oidc")
    try:
        assert get_session_env("HERMES_SESSION_USER_ID") == "oidc:carol"
        assert get_session_env("HERMES_SESSION_USER_NAME") == "carol"
        # No second platform id exists for a dashboard login.
        assert get_session_env("HERMES_SESSION_USER_ID_ALT") == ""
    finally:
        server._clear_session_context(tokens)

    assert get_session_env("HERMES_SESSION_USER_ID") == ""
    assert get_session_env("HERMES_SESSION_USER_NAME") == ""


def test_bound_user_matches_the_id_the_agent_is_built_with(monkeypatch):
    """HERMES_SESSION_USER_ID and _make_agent's user_id must not diverge, or a
    tool's attribution disagrees with the agent's own runtime identity."""
    sess = _install_session(
        monkeypatch, session_key="skey-basic", transport=_transport("basic", "alice")
    )

    tokens = server._set_session_context("skey-basic")
    try:
        assert get_session_env("HERMES_SESSION_USER_ID") == server._session_auth_user_id(sess)
    finally:
        server._clear_session_context(tokens)


def test_prefers_the_login_stamped_on_the_record(monkeypatch):
    """A second window turns the transport slot into a FanoutTransport, which
    names no login; the record's own ``auth_user_id`` still does."""
    _install_session(
        monkeypatch, session_key="skey-fanout", transport=types.SimpleNamespace(),
        auth_user_id="oidc:dave",
    )

    tokens = server._set_session_context("skey-fanout")
    try:
        assert get_session_env("HERMES_SESSION_USER_ID") == "oidc:dave"
        assert get_session_env("HERMES_SESSION_USER_NAME") == "dave"
    finally:
        server._clear_session_context(tokens)


@pytest.mark.parametrize(
    "transport",
    [
        None,
        types.SimpleNamespace(auth_identity=None),
        types.SimpleNamespace(
            auth_identity={"provider": "server-internal", "user_id": "server-internal"}
        ),
    ],
    ids=["no-transport", "ungated", "server-internal-credential"],
)
def test_no_login_binds_nothing(monkeypatch, transport):
    """An ungated gateway, and the PTY child's server-internal credential, name
    no person: the identity vars stay empty rather than inventing a user."""
    _install_session(monkeypatch, session_key="skey-open", transport=transport)

    tokens = server._set_session_context("skey-open")
    try:
        assert get_session_env("HERMES_SESSION_USER_ID") == ""
        assert get_session_env("HERMES_SESSION_USER_NAME") == ""
    finally:
        server._clear_session_context(tokens)


def test_unknown_session_key_binds_nothing(monkeypatch):
    """Ephemeral task ids are not in ``_sessions`` and carry no identity."""
    monkeypatch.setattr(server, "_sessions", {}, raising=False)

    tokens = server._set_session_context("task-ephemeral")
    try:
        assert get_session_env("HERMES_SESSION_USER_ID") == ""
        assert get_session_env("HERMES_SESSION_USER_NAME") == ""
    finally:
        server._clear_session_context(tokens)
