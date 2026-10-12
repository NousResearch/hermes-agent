"""Doctor must diagnose Hermes' private standard runtime, not a global install."""

from types import SimpleNamespace

from tools.computer_use import cua_backend_daemon, doctor


def test_doctor_owns_standard_daemon_until_proxy_is_closed(monkeypatch):
    events = []
    daemon = SimpleNamespace(
        start=lambda: events.append("start"), stop=lambda: events.append("stop"),
        proxy_invocation=lambda: ("selected-driver", ["mcp", "--embedded", "--socket", "private"]),
        child_env=lambda: {"CUA_DRIVER_PERMISSION_MODE": "standard"},
    )
    def create(binary, mode):
        assert binary == "selected-driver" and mode == "standard"
        return daemon
    monkeypatch.setattr(cua_backend_daemon, "_EmbeddedCuaDaemon", create)
    def open_proxy(binary, **kwargs):
        assert binary == "selected-driver"
        assert kwargs["args"] == ["mcp", "--embedded", "--socket", "private"]
        assert kwargs["env"] == daemon.child_env()
        events.append("proxy")
        return SimpleNamespace(stdin=SimpleNamespace(close=lambda: events.append("close")),
                               wait=lambda **kwargs: events.append("wait"))
    monkeypatch.setattr(doctor, "_open_mcp", open_proxy)
    with doctor._mcp_session("selected-driver", 1.0):
        events.append("probe")
    assert events == ["start", "proxy", "probe", "close", "wait", "stop"]
