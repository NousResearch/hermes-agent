import uvicorn

from hermes_cli import web_server


def test_start_server_enables_ws_ping_for_half_open_detection(monkeypatch):
    """WS ping must be configured so half-open connections (reverse-proxy 524,
    dropped tunnels) raise WebSocketDisconnect into the reaping path (#32377)."""
    captured = {}
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: captured.update(kwargs))

    # Loopback bind => no auth gate, so this reaches uvicorn.run without setup.
    web_server.start_server(host="127.0.0.1", port=0, open_browser=False)

    assert captured["ws_ping_interval"] == 20.0
    assert captured["ws_ping_timeout"] == 20.0
