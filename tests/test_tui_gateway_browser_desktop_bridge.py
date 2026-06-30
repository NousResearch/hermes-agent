import threading

from tui_gateway import server


def _cleanup():
    server._sessions.pop("visible-sid", None)
    if hasattr(server, "_browser_command_pending"):
        server._browser_command_pending.clear()


def test_browser_desktop_command_emits_request_and_waits_for_renderer_response(monkeypatch):
    server._sessions["visible-sid"] = {"session_key": "stored-visible"}
    emitted = []

    def fake_emit(event, sid, payload):
        emitted.append((event, sid, dict(payload)))

        def respond():
            server._methods["browser.desktop.respond"](
                "respond-1",
                {
                    "request_id": payload["request_id"],
                    "ok": True,
                    "result": {"title": "Visible", "url": "https://example.com"},
                },
            )

        threading.Timer(0.01, respond).start()

    monkeypatch.setattr(server, "_emit", fake_emit)

    try:
        response = server._methods["browser.desktop.command"](
            "cmd-1",
            {
                "session_id": "visible-sid",
                "command": "getState",
                "params": {"probe": True},
                "timeout": 1,
            },
        )
    finally:
        _cleanup()

    assert response["result"] == {
        "ok": True,
        "request_id": emitted[0][2]["request_id"],
        "result": {"title": "Visible", "url": "https://example.com"},
    }
    assert emitted == [
        (
            "browser.command.request",
            "visible-sid",
            {
                "command": "getState",
                "params": {"probe": True},
                "request_id": emitted[0][2]["request_id"],
            },
        )
    ]


def test_browser_desktop_command_times_out_and_cleans_pending(monkeypatch):
    server._sessions["visible-sid"] = {"session_key": "stored-visible"}
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)

    try:
        response = server._methods["browser.desktop.command"](
            "cmd-timeout",
            {"session_id": "visible-sid", "command": "getState", "timeout": 0.01},
        )
    finally:
        _cleanup()

    assert response["error"]["message"] == "visible browser command timed out"
    assert not getattr(server, "_browser_command_pending", {})


def test_browser_desktop_respond_rejects_unknown_request_id():
    response = server._methods["browser.desktop.respond"](
        "respond-missing",
        {"request_id": "missing", "ok": True, "result": {}},
    )

    assert response["error"]["message"] == "unknown visible browser request"
