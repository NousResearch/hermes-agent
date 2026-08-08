import os
import sys
import types
from unittest.mock import patch

from hermes_cli.browser_connect import ChromeDebugLaunch
from tui_gateway import server


# ── browser.manage connect/disconnect lifecycle ──────────────────────


def _stub_urlopen(monkeypatch, *, ok: bool):
    """Patch urllib.request.urlopen used by browser.manage to short-circuit probes."""

    class _Resp:
        status = 200 if ok else 503

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def _opener(_url, timeout=2.0):  # noqa: ARG001 — match urllib signature
        if not ok:
            raise OSError("probe failed")
        return _Resp()

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", _opener)


def test_browser_manage_connect_squatted_port_launches_on_alternate(monkeypatch):
    """When neither loopback speaks CDP but the port is held by another
    application, connect must pick an alternate port for the launch and
    say so — never fight the squatter for 9222."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.setattr(server.time, "sleep", lambda _seconds: None)
    fake = types.SimpleNamespace(
        cleanup_all_browsers=lambda: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def _opener(url, timeout=2.0):  # noqa: ARG001 — match urllib signature
        if ":9223" in url and "127.0.0.1" in url:
            return _Resp()  # relaunched browser comes up on the alternate port
        raise OSError("9222 squatted / nothing else listening")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", _opener)
    launch_ports: list[int] = []

    def _launch(port, _system):
        launch_ports.append(port)
        return ChromeDebugLaunch(launched=True)

    with patch.dict(sys.modules, {"tools.browser_tool": fake}):
        with (
            patch("hermes_cli.browser_connect.launch_chrome_debug", side_effect=_launch),
            patch("hermes_cli.browser_connect.local_port_in_use", return_value=True),
            patch("hermes_cli.browser_connect.find_free_debug_port", return_value=9223),
        ):
            resp = server.handle_request(
                {"id": "1", "method": "browser.manage", "params": {"action": "connect"}}
            )

    assert launch_ports == [9223]
    assert resp["result"]["connected"] is True
    assert resp["result"]["url"] == "http://127.0.0.1:9223"
    assert os.environ["BROWSER_CDP_URL"] == "http://127.0.0.1:9223"
    assert any("occupied by another application" in m for m in resp["result"]["messages"])


def test_browser_manage_connect_rejects_unreachable_endpoint(monkeypatch):
    """An unreachable endpoint must NOT mutate the env or reap sessions."""
    monkeypatch.setenv("BROWSER_CDP_URL", "http://existing:9222")
    cleanup_calls: list[str] = []
    fake = types.SimpleNamespace(
        cleanup_all_browsers=lambda: cleanup_calls.append(
            os.environ.get("BROWSER_CDP_URL", "")
        ),
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    with patch.dict(sys.modules, {"tools.browser_tool": fake}):
        _stub_urlopen(monkeypatch, ok=False)
        resp = server.handle_request(
            {
                "id": "1",
                "method": "browser.manage",
                "params": {"action": "connect", "url": "http://unreachable:9222"},
            }
        )

    assert "error" in resp
    # Env preserved; nothing reaped.
    assert os.environ["BROWSER_CDP_URL"] == "http://existing:9222"
    assert cleanup_calls == []


def test_browser_manage_connect_normalizes_bare_host_port(monkeypatch):
    """Persist a parsed `scheme://host:port` URL so `_get_cdp_override`
    can normalize it; storing a bare host:port would break subsequent
    tool calls (Copilot review on #17120)."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    fake = types.SimpleNamespace(
        cleanup_all_browsers=lambda: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    with patch.dict(sys.modules, {"tools.browser_tool": fake}):
        _stub_urlopen(monkeypatch, ok=True)
        resp = server.handle_request(
            {
                "id": "1",
                "method": "browser.manage",
                "params": {"action": "connect", "url": "127.0.0.1:9222"},
            }
        )

    assert resp["result"]["connected"] is True
    # Bare host:port got promoted to a full URL with explicit scheme.
    assert resp["result"]["url"].startswith("http://")
    assert os.environ["BROWSER_CDP_URL"].startswith("http://")


def test_browser_manage_connect_strips_discovery_path(monkeypatch):
    """User-supplied discovery paths like `/json` or `/json/version`
    must collapse to bare `scheme://host:port`; otherwise
    ``_resolve_cdp_override`` will append ``/json/version`` again and
    produce a duplicate path (Copilot review round-2 on #17120)."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    fake = types.SimpleNamespace(
        cleanup_all_browsers=lambda: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    with patch.dict(sys.modules, {"tools.browser_tool": fake}):
        _stub_urlopen(monkeypatch, ok=True)
        resp = server.handle_request(
            {
                "id": "1",
                "method": "browser.manage",
                "params": {"action": "connect", "url": "http://127.0.0.1:9222/json"},
            }
        )

    assert resp["result"]["connected"] is True
    assert resp["result"]["url"] == "http://127.0.0.1:9222"
    assert os.environ["BROWSER_CDP_URL"] == "http://127.0.0.1:9222"


def test_browser_manage_connect_preserves_devtools_browser_endpoint(monkeypatch):
    """Concrete devtools websocket endpoints (e.g. Browserbase) must
    survive verbatim — we only collapse discovery-style paths."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    fake = types.SimpleNamespace(
        cleanup_all_browsers=lambda: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    concrete = "ws://browserbase.example/devtools/browser/abc123"

    class _OkSocket:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch.dict(sys.modules, {"tools.browser_tool": fake}):
        # If urlopen is reached for a concrete ws endpoint, the test
        # would still pass because _stub_urlopen returned ok=True before;
        # patch it to assert-fail so we prove the HTTP probe is skipped.
        with patch(
            "urllib.request.urlopen", side_effect=AssertionError("urlopen called")
        ):
            with patch("socket.create_connection", return_value=_OkSocket()):
                resp = server.handle_request(
                    {
                        "id": "1",
                        "method": "browser.manage",
                        "params": {"action": "connect", "url": concrete},
                    }
                )

    assert resp["result"]["connected"] is True
    assert resp["result"]["url"] == concrete
    assert os.environ["BROWSER_CDP_URL"] == concrete


def test_browser_manage_connect_local_devtools_ws_preserves_path(monkeypatch):
    """Regression: ``ws://127.0.0.1:9222/devtools/browser/<id>`` is a real
    connectable endpoint; default-local normalization must not strip the
    ``/devtools/browser/...`` path or it breaks valid local CDP connects."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    fake = types.SimpleNamespace(
        cleanup_all_browsers=lambda: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    concrete = "ws://127.0.0.1:9222/devtools/browser/abc123"

    class _OkSocket:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch.dict(sys.modules, {"tools.browser_tool": fake}):
        with patch("socket.create_connection", return_value=_OkSocket()):
            resp = server.handle_request(
                {
                    "id": "1",
                    "method": "browser.manage",
                    "params": {"action": "connect", "url": concrete},
                }
            )

    assert resp["result"]["connected"] is True
    assert resp["result"]["url"] == concrete
    assert os.environ["BROWSER_CDP_URL"] == concrete


def test_browser_manage_connect_rejects_invalid_port(monkeypatch):
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    resp = server.handle_request(
        {
            "id": "1",
            "method": "browser.manage",
            "params": {"action": "connect", "url": "http://localhost:abc"},
        }
    )

    assert resp["error"]["code"] == 4015
    assert "invalid port" in resp["error"]["message"]
    assert "BROWSER_CDP_URL" not in os.environ


def test_browser_manage_connect_rejects_missing_host(monkeypatch):
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    resp = server.handle_request(
        {
            "id": "1",
            "method": "browser.manage",
            "params": {"action": "connect", "url": "http://:9222"},
        }
    )

    assert resp["error"]["code"] == 4015
    assert "missing host" in resp["error"]["message"]
    assert "BROWSER_CDP_URL" not in os.environ


def test_browser_manage_connect_concrete_ws_skips_http_probe(monkeypatch):
    """Regression for round-2 Copilot review: a hosted CDP endpoint
    (no HTTP discovery) must connect via TCP-only reachability check.
    The HTTP probe used to reject these even though they're valid."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    fake = types.SimpleNamespace(
        cleanup_all_browsers=lambda: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    concrete = "wss://chrome.browserless.io/devtools/browser/sess-1"

    seen_targets: list[tuple[str, int]] = []

    class _OkSocket:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_create_connection(addr, timeout=None):
        seen_targets.append(addr)
        return _OkSocket()

    with patch.dict(sys.modules, {"tools.browser_tool": fake}):
        # urlopen would 404/ECONNREFUSED on a real hosted CDP endpoint;
        # asserting it's never called proves the probe was skipped.
        with patch(
            "urllib.request.urlopen", side_effect=AssertionError("urlopen called")
        ):
            with patch("socket.create_connection", side_effect=_fake_create_connection):
                resp = server.handle_request(
                    {
                        "id": "1",
                        "method": "browser.manage",
                        "params": {"action": "connect", "url": concrete},
                    }
                )

    assert resp["result"] == {"connected": True, "url": concrete}
    # wss → port 443, host preserved verbatim.
    assert seen_targets == [("chrome.browserless.io", 443)]


def test_browser_manage_connect_concrete_ws_tcp_unreachable(monkeypatch):
    """If the TCP reachability check fails for a concrete ws endpoint,
    return a clear 5031 error — no fallback to the HTTP probe (which
    can never succeed for these URLs anyway)."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    fake = types.SimpleNamespace(
        cleanup_all_browsers=lambda: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    concrete = "ws://offline.example/devtools/browser/missing"

    with patch.dict(sys.modules, {"tools.browser_tool": fake}):
        with patch("socket.create_connection", side_effect=OSError("ECONNREFUSED")):
            resp = server.handle_request(
                {
                    "id": "1",
                    "method": "browser.manage",
                    "params": {"action": "connect", "url": concrete},
                }
            )

    assert "error" in resp
    assert resp["error"]["code"] == 5031


def test_browser_manage_disconnect_drops_env_and_cleans(monkeypatch):
    monkeypatch.setenv("BROWSER_CDP_URL", "http://127.0.0.1:9222")
    cleanup_count = {"n": 0}
    fake = types.SimpleNamespace(
        cleanup_all_browsers=lambda: cleanup_count.__setitem__(
            "n", cleanup_count["n"] + 1
        ),
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    with patch.dict(sys.modules, {"tools.browser_tool": fake}):
        resp = server.handle_request(
            {"id": "1", "method": "browser.manage", "params": {"action": "disconnect"}}
        )

    assert resp["result"] == {"connected": False}
    assert "BROWSER_CDP_URL" not in os.environ
    # Two cleanups: once before env removal, once after, matching connect.
    assert cleanup_count["n"] == 2
