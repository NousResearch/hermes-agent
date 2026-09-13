"""The gateway sudo request carries the command being authorized (#79874)."""

from tools.terminal_tool import _get_sudo_password_callback, set_sudo_password_callback


def test_sudo_request_carries_command(monkeypatch):
    from tui_gateway import server

    captured = {}

    def _fake_ask(method, sid, params, timeout=300):
        captured["method"] = method
        captured["sid"] = sid
        captured["params"] = params
        return "pw"

    monkeypatch.setattr(server, "_ask", _fake_ask)
    set_sudo_password_callback(None)
    try:
        server._wire_callbacks("sid-1")
        cb = _get_sudo_password_callback()
        assert cb is not None
        assert cb("sudo whoami") == "pw"
        assert captured == {
            "method": "sudo",
            "sid": "sid-1",
            "params": {"command": "sudo whoami"},
        }
    finally:
        set_sudo_password_callback(None)
