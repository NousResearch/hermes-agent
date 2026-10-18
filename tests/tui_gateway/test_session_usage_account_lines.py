"""Desktop/TUI ``/usage`` (``slash.exec``) shows provider account limits before the first turn.

Account limits need only the session profile's configured provider and its credentials, never a
built agent, and the fetch must run inside that profile's secret scope: on a multiplexed backend a
second profile's key is unreadable outside it (#132422).
"""
from datetime import UTC, datetime
from unittest.mock import patch

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow


def _snapshot() -> AccountUsageSnapshot:
    return AccountUsageSnapshot(
        provider="openrouter", source="usage_api", fetched_at=datetime.now(UTC),
        windows=(AccountUsageWindow(label="API key quota", used_percent=2.0),),
    )


def test_usage_slash_shows_account_limits_before_first_turn_inside_session_scope():
    from tui_gateway import server

    # Fresh chat whose composer picked Codex under an OpenRouter-default profile.
    session = {"agent": None, "profile_home": "/nonexistent/hermes-profiles/coder", "history": [],
               "running": False, "session_key": "sess-usage",
               "model_override": {"model": "gpt-5.3-codex", "provider": "openai-codex"}}
    fetched: list[str] = []
    scoped: list[str] = []

    class _Scope:
        def __enter__(self):
            scoped.append("enter")

        def __exit__(self, *exc):
            scoped.pop()
            return False

    def _fetch(provider, *, base_url=None, api_key=None):
        assert scoped == ["enter"], scoped  # the profile's key is only readable inside its scope
        fetched.append(provider)
        return _snapshot()

    def _credits(**kw):
        assert scoped == ["enter"], scoped  # the profile's own Nous login, not the launch profile's
        return ["Nous credits: 100 remaining"]

    with (
        patch.object(server, "_session_usage_snapshot", return_value={}),
        patch.object(server, "_session_profile_runtime_scope", lambda sess, **kw: _Scope()),
        patch.object(server, "_config_model_target", lambda: ("openai/gpt-5-mini", "openrouter")),
        patch("agent.account_usage.fetch_account_usage", _fetch),
        patch("agent.account_usage.nous_credits_lines", _credits),
    ):
        out = server._live_slash_command_output("sid-usage", session, "usage", "")

    assert scoped == []
    assert fetched == ["openai-codex"]
    assert "Account limits" in out and "Nous credits: 100 remaining" in out, out
    assert "send a message first" not in out
