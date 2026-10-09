"""shared_metrics.slash_command: the TUI/Desktop client reports each typed slash command once, and the
gateway no longer counts the slash.exec / command.dispatch hops a command may or may not take."""

from __future__ import annotations

import hermes_cli.observability.shared_metrics_events as events
import tui_gateway.server as server


def _request(method: str, params: dict) -> dict:
    return server.handle_request({"jsonrpc": "2.0", "id": "r1", "method": method, "params": params})


def test_client_reports_the_command_and_gateway_hops_do_not_count(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(events, "record_slash_command", lambda **kw: calls.append(kw))
    monkeypatch.setenv("HERMES_DESKTOP", "1")
    monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)
    monkeypatch.setitem(server._methods, "slash.exec", lambda rid, params: server._ok(rid, {"output": "ok"}))

    assert _request("shared_metrics.slash_command", {"command": "branch"})["result"] == {"ok": True}
    assert _request("slash.exec", {"command": "branch", "session_id": "s1"})["result"] == {"output": "ok"}

    assert calls == [{"command": "branch", "surface": "desktop"}]
