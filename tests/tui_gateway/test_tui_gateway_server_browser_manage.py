"""``browser.manage`` RPCs: connect/disconnect/status (split from ``test_tui_gateway_server.py``).

Since CLI-9 (#134889) the live ``/browser connect`` endpoint is the requesting profile's (a per-home
map in ``tools.browser_session_cdp``), and only that profile's browsers are reaped.
"""
import json
import os
import sys
import time
import types
from unittest.mock import patch

import pytest

from hermes_cli.browser_connect import ChromeDebugLaunch
from tui_gateway import server


def _stub_urlopen(monkeypatch, *, ok: bool):
    """Patch the loopback-aware opener browser.manage probes through (#110565) to short-circuit probes."""

    class _Resp:
        status = 200 if ok else 503

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def _opener(_url, timeout=2.0):
        if not ok:
            raise OSError("probe failed")
        return _Resp()

    import urllib.request

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", lambda _self, url, *a, timeout=2.0, **k: _opener(url, timeout=timeout))


def _stub_urlopen_capture(monkeypatch, *, ok: bool):
    urls: list[str] = []

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def _opener(url, timeout=2.0):
        urls.append(url)
        if not ok:
            raise OSError("probe failed")
        return _Resp()

    import urllib.request

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", lambda _self, url, *a, timeout=2.0, **k: _opener(url, timeout=timeout))
    return urls


def test_browser_manage_status_reads_env_var(monkeypatch):
    """Status returns the env var verbatim (no network I/O)."""
    monkeypatch.setenv("BROWSER_CDP_URL", "http://127.0.0.1:9222")

    resp = server.handle_request(
        {"id": "1", "method": "browser.manage", "params": {"action": "status"}}
    )

    assert resp["result"]["connected"] is True
    assert resp["result"]["url"] == "http://127.0.0.1:9222"


def test_browser_manage_status_falls_back_to_config_cdp_url(monkeypatch):
    """When env is unset, status surfaces ``browser.cdp_url`` from
    config.yaml so users see what the next tool call will read."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)

    with patch("hermes_cli.config_effective.load_user_config_effective",
               lambda: {"browser": {"cdp_url": "http://lan:9222"}}):
        resp = server.handle_request(
            {"id": "1", "method": "browser.manage", "params": {"action": "status"}}
        )

    assert {k: resp["result"][k] for k in ("connected", "url")} == {"connected": True, "url": "http://lan:9222"}


def test_browser_manage_has_no_use_action_and_status_reports_no_mode():
    """``browser.manage`` ``use`` (the old Browser Use / built-in tools switch) is gone: browser_exec is
    the only core browser path, so the action is refused and ``status`` carries no mode flag."""
    resp = server.handle_request({"id": "1", "method": "browser.manage", "params": {"action": "use"}})
    assert "error" in resp and "result" not in resp
    status = server.handle_request({"id": "2", "method": "browser.manage", "params": {"action": "status"}})
    assert "browser_use" not in status["result"]


@pytest.fixture(autouse=True)
def _no_live_cdp(monkeypatch):
    """Each test starts with no profile's live ``/browser connect`` endpoint."""
    from tools import browser_session_cdp
    monkeypatch.setattr(browser_session_cdp, "_live_overrides", {})


def _live_cdp() -> str:
    """The launch profile's live ``/browser connect`` endpoint (per profile since CLI-9)."""
    from tools.browser_session_cdp import live_cdp_override
    return live_cdp_override()


def test_browser_manage_connect_sets_env_and_cleans_twice(monkeypatch):
    """`/browser connect` must reach the live process: set this profile's endpoint, reap its browser
    sessions before AND after publishing the new URL.  The double-cleanup
    closes the supervisor swap window where ``_ensure_cdp_supervisor``
    could re-attach to the *old* CDP endpoint between steps."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    cleanup_calls: list[str] = []

    def _cleanup_all(_key):
        cleanup_calls.append(_live_cdp())

    fake = types.SimpleNamespace(
        cleanup_profile_browsers=_cleanup_all,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    with patch.dict(sys.modules, {"tools.browser_session_lifecycle": fake}):
        _stub_urlopen(monkeypatch, ok=True)
        resp = server.handle_request(
            {
                "id": "1",
                "method": "browser.manage",
                "params": {"action": "connect", "url": "http://127.0.0.1:9222"},
            }
        )

    assert resp["result"]["connected"] is True
    assert resp["result"]["url"] == "http://127.0.0.1:9222"
    assert _live_cdp() == "http://127.0.0.1:9222"
    # First cleanup runs against the OLD env (none here), second against the NEW.
    assert cleanup_calls == ["", "http://127.0.0.1:9222"]


def test_browser_manage_connect_defaults_to_loopback(monkeypatch):
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    fake = types.SimpleNamespace(
        cleanup_profile_browsers=lambda _key: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    with patch.dict(sys.modules, {"tools.browser_session_lifecycle": fake}):
        urls = _stub_urlopen_capture(monkeypatch, ok=True)
        resp = server.handle_request(
            {"id": "1", "method": "browser.manage", "params": {"action": "connect"}}
        )

    assert resp["result"]["connected"] is True
    assert resp["result"]["url"] == "http://127.0.0.1:9222"
    assert urls[0] == "http://127.0.0.1:9222/json/version"


def test_browser_manage_connect_default_local_reports_launch_hint(monkeypatch):
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    # No ``platform.system`` fake: the resolved system string only flows into
    # ``launch_chrome_debug`` / ``manual_chrome_debug_command`` /
    # ``get_chrome_debug_candidates``, all of which are mocked below — the
    # host's real value never reaches unmocked code.
    emitted: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda evt, sid, payload=None: emitted.append((evt, payload or {})),
    )
    fake = types.SimpleNamespace(
        cleanup_profile_browsers=lambda _key: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    with patch.dict(sys.modules, {"tools.browser_session_lifecycle": fake}):
        _stub_urlopen(monkeypatch, ok=False)
        with (
            patch(
                "hermes_cli.browser_connect.launch_chrome_debug",
                return_value=ChromeDebugLaunch(),
            ),
            patch("hermes_cli.browser_connect.local_port_in_use", return_value=False),
            patch("hermes_cli.browser_connect.manual_chrome_debug_command", return_value=None),
            patch(
                "hermes_cli.browser_connect.get_chrome_debug_candidates",
                return_value=[],
            ),
        ):
            resp = server.handle_request(
                {
                    "id": "1",
                    "method": "browser.manage",
                    "params": {
                        "action": "connect",
                        "session_id": "sess-1",
                        "url": "http://localhost:9222",
                    },
                }
            )

    assert resp["result"]["connected"] is False
    assert resp["result"]["url"] == "http://127.0.0.1:9222"
    assert resp["result"]["messages"]
    assert "BROWSER_CDP_URL" not in os.environ
    progress = [p["message"] for evt, p in emitted if evt == "browser.progress"]
    assert progress == resp["result"]["messages"]


def test_browser_manage_connect_no_session_skips_progress_events(monkeypatch):
    """Without a session_id the TUI prints messages from the response;
    emitting ``browser.progress`` events would double-render. Gate the
    emit so callers without a session see the bundled list only."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    emitted: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda evt, sid, payload=None: emitted.append((evt, payload or {})),
    )
    fake = types.SimpleNamespace(
        cleanup_profile_browsers=lambda _key: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    with patch.dict(sys.modules, {"tools.browser_session_lifecycle": fake}):
        _stub_urlopen(monkeypatch, ok=False)
        with (
            patch(
                "hermes_cli.browser_connect.launch_chrome_debug",
                return_value=ChromeDebugLaunch(),
            ),
            patch("hermes_cli.browser_connect.manual_chrome_debug_command", return_value=None),
            patch(
                "hermes_cli.browser_connect.get_chrome_debug_candidates",
                return_value=[],
            ),
        ):
            resp = server.handle_request(
                {
                    "id": "1",
                    "method": "browser.manage",
                    "params": {"action": "connect", "url": "http://localhost:9222"},
                }
            )

    assert resp["result"]["connected"] is False
    assert resp["result"]["messages"]  # bundled list still populated
    assert [evt for evt, _ in emitted if evt == "browser.progress"] == []


def test_browser_manage_connect_handles_null_url(monkeypatch):
    """Explicit ``{"url": null}`` (or empty string) must fall back to the
    default loopback URL instead of raising a TypeError that gets swallowed
    by the outer 5031 catch."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    fake = types.SimpleNamespace(
        cleanup_profile_browsers=lambda _key: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    with patch.dict(sys.modules, {"tools.browser_session_lifecycle": fake}):
        _stub_urlopen(monkeypatch, ok=True)
        resp = server.handle_request(
            {
                "id": "1",
                "method": "browser.manage",
                "params": {"action": "connect", "url": None},
            }
        )

    assert resp["result"]["connected"] is True
    assert resp["result"]["url"] == "http://127.0.0.1:9222"


def test_browser_manage_connect_rejects_non_string_url(monkeypatch):
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    resp = server.handle_request(
        {
            "id": "1",
            "method": "browser.manage",
            "params": {"action": "connect", "url": 9222},
        }
    )

    assert resp["error"]["code"] == 4015
    assert "BROWSER_CDP_URL" not in os.environ


def test_browser_manage_connect_default_local_retries_after_launch(monkeypatch):
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.setattr(server.time, "sleep", lambda _seconds: None)
    fake = types.SimpleNamespace(
        cleanup_profile_browsers=lambda _key: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    # IPv4 answers only from the 3rd probe onwards (browser still starting);
    # the IPv6 loopback never answers.
    attempts = {"n": 0}

    def _opener(url, timeout=2.0):
        if "[::1]" in url:
            raise OSError("no IPv6 listener")
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise OSError("not ready")
        return _Resp()

    import urllib.request

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", lambda _self, url, *a, timeout=2.0, **k: _opener(url, timeout=timeout))
    launched = ChromeDebugLaunch(launched=True)
    with patch.dict(sys.modules, {"tools.browser_session_lifecycle": fake}):
        with (
            patch(
                "hermes_cli.browser_connect.launch_chrome_debug",
                return_value=launched,
            ),
            patch("hermes_cli.browser_connect.local_port_in_use", return_value=False),
        ):
            resp = server.handle_request(
                {"id": "1", "method": "browser.manage", "params": {"action": "connect"}}
            )

    assert resp["result"]["connected"] is True
    assert resp["result"]["url"] == "http://127.0.0.1:9222"
    assert _live_cdp() == "http://127.0.0.1:9222"


def test_browser_manage_connect_finds_ipv6_only_browser(monkeypatch):
    """Regression: an IDE debugger squatting 127.0.0.1:9222 pushes the debug
    browser onto [::1]:9222. Connect must discover and adopt the IPv6
    endpoint instead of timing out against the squatter."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    fake = types.SimpleNamespace(
        cleanup_profile_browsers=lambda _key: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def _opener(url, timeout=2.0):
        if "[::1]" in url:
            return _Resp()
        raise OSError("IPv4 loopback held by a non-CDP squatter")

    import urllib.request

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", lambda _self, url, *a, timeout=2.0, **k: _opener(url, timeout=timeout))
    with patch.dict(sys.modules, {"tools.browser_session_lifecycle": fake}):
        resp = server.handle_request(
            {"id": "1", "method": "browser.manage", "params": {"action": "connect"}}
        )

    assert resp["result"]["connected"] is True
    assert resp["result"]["url"] == "http://[::1]:9222"
    assert _live_cdp() == "http://[::1]:9222"


def test_browser_manage_connect_squatted_port_launches_on_alternate(monkeypatch):
    """When neither loopback speaks CDP but the port is held by another
    application, connect must pick an alternate port for the launch and
    say so — never fight the squatter for 9222."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.setattr(server.time, "sleep", lambda _seconds: None)
    fake = types.SimpleNamespace(
        cleanup_profile_browsers=lambda _key: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def _opener(url, timeout=2.0):
        if ":9223" in url and "127.0.0.1" in url:
            return _Resp()  # relaunched browser comes up on the alternate port
        raise OSError("9222 squatted / nothing else listening")

    import urllib.request

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", lambda _self, url, *a, timeout=2.0, **k: _opener(url, timeout=timeout))
    launch_ports: list[int] = []

    def _launch(port, _system):
        launch_ports.append(port)
        return ChromeDebugLaunch(launched=True)

    with patch.dict(sys.modules, {"tools.browser_session_lifecycle": fake}):
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
    assert _live_cdp() == "http://127.0.0.1:9223"


def test_browser_manage_connect_rejects_unreachable_endpoint(monkeypatch):
    """An unreachable endpoint must NOT mutate the env or reap sessions."""
    from tools.browser_session_cdp import set_live_cdp_override
    set_live_cdp_override("http://existing:9222")
    cleanup_calls: list[str] = []
    fake = types.SimpleNamespace(
        cleanup_profile_browsers=lambda _key: cleanup_calls.append(
            _live_cdp()
        ),
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    with patch.dict(sys.modules, {"tools.browser_session_lifecycle": fake}):
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
    assert _live_cdp() == "http://existing:9222"
    assert cleanup_calls == []


def test_browser_manage_connect_normalizes_bare_host_port(monkeypatch):
    """Persist a parsed `scheme://host:port` URL so `_get_cdp_override`
    can normalize it; storing a bare host:port would break subsequent
    tool calls (Copilot review on #17120)."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    fake = types.SimpleNamespace(
        cleanup_profile_browsers=lambda _key: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    with patch.dict(sys.modules, {"tools.browser_session_lifecycle": fake}):
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
    assert _live_cdp().startswith("http://")


def test_browser_manage_connect_strips_discovery_path(monkeypatch):
    """User-supplied discovery paths like `/json` or `/json/version`
    must collapse to bare `scheme://host:port`; otherwise
    ``_resolve_cdp_override`` will append ``/json/version`` again and
    produce a duplicate path (Copilot review round-2 on #17120)."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    fake = types.SimpleNamespace(
        cleanup_profile_browsers=lambda _key: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    with patch.dict(sys.modules, {"tools.browser_session_lifecycle": fake}):
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
    assert _live_cdp() == "http://127.0.0.1:9222"


def test_browser_manage_connect_preserves_devtools_browser_endpoint(monkeypatch):
    """Concrete devtools websocket endpoints (e.g. Browserbase) must
    survive verbatim — we only collapse discovery-style paths."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    fake = types.SimpleNamespace(
        cleanup_profile_browsers=lambda _key: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    concrete = "ws://browserbase.example/devtools/browser/abc123"

    class _OkSocket:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch.dict(sys.modules, {"tools.browser_session_lifecycle": fake}):
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
    assert _live_cdp() == concrete


def test_browser_manage_connect_local_devtools_ws_preserves_path(monkeypatch):
    """Regression: ``ws://127.0.0.1:9222/devtools/browser/<id>`` is a real
    connectable endpoint; default-local normalization must not strip the
    ``/devtools/browser/...`` path or it breaks valid local CDP connects."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    fake = types.SimpleNamespace(
        cleanup_profile_browsers=lambda _key: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    concrete = "ws://127.0.0.1:9222/devtools/browser/abc123"

    class _OkSocket:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch.dict(sys.modules, {"tools.browser_session_lifecycle": fake}):
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
    assert _live_cdp() == concrete


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
    assert "BROWSER_CDP_URL" not in os.environ


def test_browser_manage_connect_concrete_ws_skips_http_probe(monkeypatch):
    """Regression for round-2 Copilot review: a hosted CDP endpoint
    (no HTTP discovery) must connect via TCP-only reachability check.
    The HTTP probe used to reject these even though they're valid."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    fake = types.SimpleNamespace(
        cleanup_profile_browsers=lambda _key: None,
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

    with patch.dict(sys.modules, {"tools.browser_session_lifecycle": fake}):
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
        cleanup_profile_browsers=lambda _key: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    concrete = "ws://offline.example/devtools/browser/missing"

    with patch.dict(sys.modules, {"tools.browser_session_lifecycle": fake}):
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
    from tools.browser_session_cdp import set_live_cdp_override
    monkeypatch.setenv("BROWSER_CDP_URL", "http://operator:9222")  # an operator's env is not the live connect
    set_live_cdp_override("http://127.0.0.1:9222")
    cleanup_count = {"n": 0}
    fake = types.SimpleNamespace(
        cleanup_profile_browsers=lambda _key: cleanup_count.__setitem__(
            "n", cleanup_count["n"] + 1
        ),
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    with patch.dict(sys.modules, {"tools.browser_session_lifecycle": fake}):
        resp = server.handle_request(
            {"id": "1", "method": "browser.manage", "params": {"action": "disconnect"}}
        )

    assert resp["result"] == {"connected": False}
    assert _live_cdp() == "" and os.environ["BROWSER_CDP_URL"] == "http://operator:9222"
    # Two cleanups: once before the override is dropped, once after, matching connect.
    assert cleanup_count["n"] == 2
