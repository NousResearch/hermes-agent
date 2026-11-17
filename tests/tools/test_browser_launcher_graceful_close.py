"""A browser this process spawned is asked to quit over CDP ``Browser.close`` before any signal.

macOS Google Chrome copies its app bundle to ``$TMPDIR/../X/com.google.Chrome.code_sign_clone/`` at every
launch and removes that copy only on a normal exit: a SIGTERM teardown leaves one behind per launch.
"""

from __future__ import annotations

import json
import subprocess
import threading

import pytest
from websockets.sync.server import serve

from tools import browser_launcher as launcher


class _Proc:
    """Popen stand-in: ``exits`` says whether ``wait`` sees the browser quit within the grace period."""

    def __init__(self, exits: bool, running: bool = True):
        self.exits, self.running, self.waits = exits, running, []

    def poll(self):
        return None if self.running else 0

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if not self.exits:
            raise subprocess.TimeoutExpired("chromium", timeout)
        self.running = False
        return 0


def _browser(cdp_url: str) -> launcher.LaunchedBrowser:
    return launcher.LaunchedBrowser(key="k", pid=4242, start_time=1, cdp_url=cdp_url, http_url="",
                                    user_data_dir="/nonexistent", owned=True, ephemeral=True, home_key="h")


@pytest.fixture
def cdp_server():
    """A loopback CDP endpoint that records each command and drops the socket on ``Browser.close``."""
    received: list[dict] = []

    def handler(ws):
        for raw in ws:
            msg = json.loads(raw)
            received.append(msg)
            if msg.get("method") == "Browser.close":
                return  # Chromium tears the socket down while it quits

    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.socket.getsockname()[1]
        yield f"ws://127.0.0.1:{port}/devtools/browser/fake", received
        server.shutdown()
        thread.join(timeout=5)


def test_a_browser_that_quits_on_browser_close_is_never_signalled(cdp_server):
    url, received = cdp_server
    proc = _Proc(exits=True)
    assert launcher._closed_gracefully(_browser(url), proc) is True
    assert [m["method"] for m in received] == ["Browser.close"]
    assert proc.waits == [launcher._CLOSE_GRACE_S]


def test_a_browser_that_ignores_browser_close_falls_back_to_the_signals(cdp_server):
    url, received = cdp_server
    assert launcher._closed_gracefully(_browser(url), _Proc(exits=False)) is False
    assert [m["method"] for m in received] == ["Browser.close"]


def test_a_browser_this_process_did_not_spawn_is_not_sent_browser_close(cdp_server):
    """No ``proc``: an attached or inherited browser keeps the plain signal teardown."""
    url, received = cdp_server
    assert launcher._closed_gracefully(_browser(url), None) is False
    assert received == []


def test_an_already_exited_browser_is_not_sent_browser_close(cdp_server):
    url, received = cdp_server
    assert launcher._closed_gracefully(_browser(url), _Proc(exits=True, running=False)) is False
    assert received == []


def test_an_unreachable_endpoint_is_not_a_graceful_close():
    """Nothing listens on the port: the request fails and the caller signals the browser."""
    with serve(lambda ws: None, "127.0.0.1", 0) as server:
        port = server.socket.getsockname()[1]
    proc = _Proc(exits=True)
    assert launcher._closed_gracefully(_browser(f"ws://127.0.0.1:{port}/devtools/browser/x"), proc) is False
    assert proc.waits == []


def test_a_non_websocket_endpoint_is_never_dialled():
    assert launcher._request_close("http://127.0.0.1:9/json/version") is False


def test_stop_browser_skips_the_signals_after_a_graceful_close(monkeypatch):
    browser, proc, signalled = _browser("ws://127.0.0.1:9/devtools/browser/x"), _Proc(exits=True), []
    monkeypatch.setattr(launcher, "_request_close", lambda url: True)
    monkeypatch.setattr(launcher, "_terminate", lambda pid, start_time, p: signalled.append(pid))
    monkeypatch.setitem(launcher._browsers, "k", browser)
    monkeypatch.setitem(launcher._procs, "k", proc)
    launcher.stop_browser("k")
    assert signalled == []
    assert launcher.get_browser("k") is None


def test_stop_browser_signals_a_browser_that_ignored_browser_close(monkeypatch):
    browser, proc, signalled = _browser("ws://127.0.0.1:9/devtools/browser/x"), _Proc(exits=False), []
    monkeypatch.setattr(launcher, "_request_close", lambda url: True)
    monkeypatch.setattr(launcher, "_terminate", lambda pid, start_time, p: signalled.append(pid))
    monkeypatch.setitem(launcher._browsers, "k", browser)
    monkeypatch.setitem(launcher._procs, "k", proc)
    launcher.stop_browser("k")
    assert signalled == [4242]
