"""Regression test for #128675: Desktop app never passes gateway_session_key to memory providers.

When a session that started on a messaging platform (e.g. a Telegram thread) is continued
in the desktop app, the desktop backend (tui_gateway) must restore the gateway_session_key
from the stored session row so memory providers scope to the same chat.
"""
from types import SimpleNamespace


def test_stored_session_runtime_overrides_restores_gateway_session_key():
    """A stored row with a gateway routing key (agent:main:telegram:dm:123) must surface
    gateway_session_key in the overrides so _make_agent passes it to AIAgent."""
    from tui_gateway.server import _stored_session_runtime_overrides

    row = {
        "model": "claude-test",
        "model_config": {"model": "claude-test", "provider": "anthropic"},
        "session_key": "agent:main:telegram:dm:42",
    }
    overrides = _stored_session_runtime_overrides(row)
    assert overrides.get("gateway_session_key") == "agent:main:telegram:dm:42"


def test_stored_session_runtime_overrides_skips_server_minted_key():
    """A server-minted stored id (20260930_184450_abcdef) is NOT a gateway routing key —
    native desktop/TUI sessions must not get a bogus per-chat scope (#128675)."""
    from tui_gateway.server import _stored_session_runtime_overrides

    row = {
        "model": "claude-test",
        "model_config": {"model": "claude-test", "provider": "anthropic"},
        "session_key": "20260930_184450_abcdef",
    }
    overrides = _stored_session_runtime_overrides(row)
    assert "gateway_session_key" not in overrides


def test_stored_session_runtime_overrides_no_session_key():
    """A row with no session_key at all must not surface gateway_session_key."""
    from tui_gateway.server import _stored_session_runtime_overrides

    row = {
        "model": "claude-test",
        "model_config": {"model": "claude-test", "provider": "anthropic"},
    }
    overrides = _stored_session_runtime_overrides(row)
    assert "gateway_session_key" not in overrides


def test_session_gateway_key_returns_routing_key():
    """_session_gateway_key returns the gateway routing key from a live record's session_key
    when it's a real routing key (not server-minted)."""
    from tui_gateway.server import _session_gateway_key

    assert _session_gateway_key({"session_key": "agent:main:telegram:dm:42"}) == "agent:main:telegram:dm:42"
    assert _session_gateway_key({"session_key": "20260930_184450_abcdef"}) is None
    assert _session_gateway_key({"session_key": ""}) is None
    assert _session_gateway_key({}) is None
    assert _session_gateway_key(None) is None


def test_session_gateway_key_prefers_explicit_slot():
    """When a live record carries an explicit gateway_session_key slot, it wins."""
    from tui_gateway.server import _session_gateway_key

    session = {"session_key": "20260930_184450_abcdef", "gateway_session_key": "agent:main:telegram:dm:42"}
    assert _session_gateway_key(session) == "agent:main:telegram:dm:42"


def test_deferred_build_agent_kwargs_passes_gateway_session_key():
    """_deferred_build_agent_kwargs must pass gateway_session_key through when it's in
    resume_runtime_overrides, even when no routable provider override is present."""
    from tui_gateway.server import _deferred_build_agent_kwargs

    current = {
        "resume_session_id": "stored-session-id",
        "resume_runtime_overrides": {
            "gateway_session_key": "agent:main:telegram:dm:42",
            "model_override": {"model": "claude-test", "provider": "anthropic",
                               "base_url": None, "api_mode": None},
        },
        "model_override": None,
    }
    kw = _deferred_build_agent_kwargs(current, session_db=None)
    assert kw.get("gateway_session_key") == "agent:main:telegram:dm:42"
