"""``set_theme`` picker bridge: the model can tell a pick, a dismissed card and an expired one apart.
See tui_gateway.theme_bridge.theme_request."""

from types import SimpleNamespace

import pytest

from tui_gateway import server, server_requests, theme_bridge


@pytest.mark.parametrize(("answer", "elapsed", "result"), [
    ({"picked": {"accent": "violet"}}, 4.0, {"success": True, "picked": {"accent": "violet"}}),
    ({"said": " make it pink "}, 4.0, {"said": "make it pink"}),
    ({"cancelled": True}, 4.0, {"cancelled": True}),
    (None, 4.0, {"cancelled": True}),
    (None, theme_bridge.THEME_PICK_TIMEOUT_S, {"timed_out": True}),
])
def test_pick_outcomes(monkeypatch, answer, elapsed, result):
    clock = iter((0.0, elapsed))
    sent = []

    def fake_send(method, sid, params, *, timeout, **_kw):
        sent.append((method, sid, params, timeout))
        return answer

    monkeypatch.setattr(server_requests, "send", fake_send)
    monkeypatch.setattr(theme_bridge, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    monkeypatch.setitem(server._sessions, "s1", {})

    assert theme_bridge.theme_request("s1", {"picker": "accent"}) == result
    assert sent == [("theme.pick", "s1", {"picker": "accent"}, theme_bridge.THEME_PICK_TIMEOUT_S)]


def test_the_session_callbacks_reach_the_bridge(monkeypatch):
    """Built the way production builds them: server._agent_cbs, rebound to server.py's globals."""
    calls = []
    monkeypatch.setattr(theme_bridge, "theme_request", lambda sid, payload: calls.append((sid, payload)) or {"success": True})

    assert server._agent_cbs("s1")["theme_callback"]({"accent": "violet"}) == {"success": True}
    assert calls == [("s1", {"accent": "violet"})]
