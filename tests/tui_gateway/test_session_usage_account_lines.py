"""``session.usage`` RPC (Desktop usage feed) carries the provider account-limits block.

The CLI/TUI slash worker and gateway ``/usage`` render Codex quota windows via
``render_account_usage_lines``; the Desktop feed reads ``session.usage`` instead, so the RPC
must ship the same lines (``account_lines``) or that surface silently omits them.
"""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow


def _codex_snapshot() -> AccountUsageSnapshot:
    return AccountUsageSnapshot(
        provider="openai-codex", source="usage_api", fetched_at=datetime.now(timezone.utc), plan="Plus",
        windows=(AccountUsageWindow(label="Weekly", used_percent=12.0),),
    )


def test_session_usage_rpc_ships_account_lines_for_the_live_route():
    from tui_gateway import server

    agent = SimpleNamespace(provider="openai-codex", base_url="https://chatgpt.example/backend-api",
                            api_key="tok", model="gpt-5.3-codex")
    session = {"agent": agent, "history": [], "running": False, "session_key": "sess-usage"}
    sid = "sid-usage-account"
    server._sessions[sid] = session
    seen: list[tuple] = []

    def _fetch(provider, *, base_url=None, api_key=None):
        seen.append((provider, base_url, api_key))
        return _codex_snapshot()

    try:
        with (
            patch.object(server, "_get_usage", return_value={"calls": 1, "input": 10, "output": 20, "total": 30}),
            patch("agent.account_usage.fetch_account_usage", _fetch),
            patch("agent.account_usage.nous_credits_lines", lambda **kw: []),
        ):
            r = server._methods["session.usage"]("r1", {"session_id": sid})
    finally:
        server._sessions.pop(sid, None)

    assert "error" not in r, r
    result = r["result"]
    assert result["total"] == 30 and "credits_lines" not in result
    # Fetched against the session's own route, not a default endpoint.
    assert seen == [("openai-codex", "https://chatgpt.example/backend-api", "tok")]
    assert result["account_lines"]
