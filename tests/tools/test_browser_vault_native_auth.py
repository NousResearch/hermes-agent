"""Native Basic auth contracts; all credentials and endpoints are isolated fixtures."""
from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.waiting import wait_until
from tools.browser_supervisor import CDPSupervisor, SUPERVISOR_REGISTRY, _schedule
from tools import browser_vault_tool as vault


@pytest.fixture
def native_browser(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    chrome = os.environ.get("HERMES_TEST_CHROME") or shutil.which("chrome")
    if not chrome:
        pytest.skip("Chromium not available for native-auth integration (add chrome to PATH)")
    expected = "Basic " + base64.b64encode(b"fixture-user:synthetic-password-493").decode()
    state = SimpleNamespace(authenticated=0, foreign_auth=0, redirect="", reject=False)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", state.redirect)
                self.end_headers()
                return
            if state.redirect and self.path == "/":
                self.send_response(302)
                self.send_header("Location", state.redirect)
                self.end_headers()
                return
            if self.headers.get("Authorization") == expected and not state.reject:
                state.authenticated += 1
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"<html><title>Authenticated fixture</title>native success</html>")
                return
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="Synthetic fixture"')
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    profile = tmp_path / "chrome"
    process = subprocess.Popen([chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
                                "--no-proxy-server", "--remote-debugging-port=0",
                                f"--user-data-dir={profile}", "about:blank"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    port_file = profile / "DevToolsActivePort"
    supervisor = None
    try:
        wait_until(port_file.exists, what="isolated Chromium CDP")
        port, socket = port_file.read_text().splitlines()[:2]
        supervisor = CDPSupervisor("native-fixture", f"ws://127.0.0.1:{port}{socket}")
        supervisor.start()
        monkeypatch.setattr(vault, "_ensure_supervisor", lambda task: supervisor)
        monkeypatch.setattr(SUPERVISOR_REGISTRY, "get", lambda task: supervisor)
        from agent.vault_backends.base import LoginBackend

        class SyntheticBackend(LoginBackend):
            name = "synthetic"

            def list_items(self):
                return []

            def get_meta(self, handle):
                return self.meta

            def resolve_password(self, handle):
                return "synthetic-password-493"

        backend = SyntheticBackend()
        backend.get_meta = Mock(return_value=SimpleNamespace(kind="login", origin=origin,
                                                             allowed_origins=[origin], identifier="fixture-user"))
        backend.resolve_password = Mock(return_value="synthetic-password-493")
        monkeypatch.setattr("agent.vault_backends.backend_for_handle", lambda handle: backend)

        def navigate(path="/"):
            _schedule(supervisor._cdp("Page.navigate", {"url": origin + path},
                                      session_id=supervisor._page_session_id), supervisor._loop, timeout=15)
            wait_until(lambda: supervisor.evaluate_runtime("location.href").get("result") ==
                       "chrome-error://chromewebdata/", what="real 401 error renderer")

        yield SimpleNamespace(supervisor=supervisor, backend=backend, state=state, origin=origin,
                              navigate=navigate, handler=Handler)
    finally:
        if supervisor:
            supervisor.stop()
        process.terminate()
        process.wait(timeout=15)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_fill_authenticates_native_challenge_from_error_renderer(native_browser, caplog):
    fixture = native_browser
    fixture.navigate()
    result = json.loads(vault.browser_vault_fill("synthetic-handle", "native-fixture"))
    assert result.get("success"), result
    assert result["authentication"] == "http_basic"
    assert result["origin"] == fixture.origin
    assert fixture.state.authenticated >= 1
    assert fixture.backend.resolve_password.call_count == 1
    wait_until(lambda: fixture.supervisor.evaluate_runtime("document.title").get("result") ==
               "Authenticated fixture", what="authenticated native browser document")
    assert "synthetic-password-493" not in json.dumps(result) + caplog.text
    from agent.redact import redact_registered_vault_values
    assert "synthetic-password-493" not in redact_registered_vault_values("synthetic-password-493")


def test_wrong_saved_origin_never_resolves(native_browser):
    fixture = native_browser
    fixture.navigate()
    meta = fixture.backend.get_meta.return_value
    meta.origin = "http://127.0.0.1:1"
    meta.allowed_origins = [meta.origin]
    result = json.loads(vault.browser_vault_fill("synthetic-handle", "native-fixture"))
    assert not result["success"]
    fixture.backend.resolve_password.assert_not_called()
    assert not fixture.supervisor._vault_auth_events


def test_wrong_password_has_single_attempt_and_cleans_interception(native_browser):
    fixture = native_browser
    fixture.state.reject = True
    fixture.navigate()
    result = json.loads(vault.browser_vault_fill("synthetic-handle", "native-fixture"))
    assert not result["success"]
    assert fixture.backend.resolve_password.call_count == 1
    assert fixture.supervisor.evaluate_runtime("1 + 1")["result"] == 2
    assert not fixture.supervisor._vault_auth_events


def test_cross_origin_redirect_cannot_resolve_or_forward_credentials(native_browser):
    fixture = native_browser
    received = []

    class Foreign(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            received.append(self.headers.get("Authorization"))
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="Foreign fixture"')
            self.end_headers()

    foreign = ThreadingHTTPServer(("127.0.0.1", 0), Foreign)
    worker = threading.Thread(target=foreign.serve_forever, daemon=True)
    worker.start()
    try:
        fixture.navigate()
        fixture.state.redirect = f"http://127.0.0.1:{foreign.server_port}/"
        result = json.loads(vault.browser_vault_fill("synthetic-handle", "native-fixture"))
        assert not result["success"]
        fixture.backend.resolve_password.assert_not_called()
        assert not any(received)
        assert not fixture.supervisor._vault_auth_events
    finally:
        foreign.shutdown()
        foreign.server_close()
        worker.join(timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [
    {"authChallenge": {"source": "Proxy", "scheme": "basic", "origin": "http://fixture.test"}},
    {"authChallenge": {"source": "Server", "scheme": "digest", "origin": "http://fixture.test"}},
    {"authChallenge": {"source": "Server", "scheme": "basic", "origin": "http://foreign.test"}},
    {"request": {"url": "http://foreign.test/"}},
    {"frameId": "child-frame"},
    {"resourceType": "XHR"},
    {"requestId": "unsolicited"},
])
async def test_untrusted_challenge_never_resolves(change):
    import asyncio
    from tools.browser_supervisor_auth import NativeAuthMixin
    supervisor = NativeAuthMixin()
    calls = []

    async def cdp(method, params, **kwargs):
        calls.append((method, params))
        return {}

    supervisor._cdp = cdp
    queue = asyncio.Queue()
    request = {"requestId": "request", "frameId": "top", "resourceType": "Document",
               "request": {"url": "http://fixture.test/"}, "networkId": "network"}
    challenge = {**request, "authChallenge": {"source": "Server", "scheme": "basic",
                                             "origin": "http://fixture.test"}, **change}
    queue.put_nowait(("Fetch.requestPaused", request))
    queue.put_nowait(("Fetch.authRequired", challenge))
    resolve = Mock()
    result = await supervisor._vault_auth_events_loop(queue, "session", "http://fixture.test", "top", resolve, Mock())
    assert not result["success"]
    resolve.assert_not_called()
    assert calls[-1][1]["authChallengeResponse"] == {"response": "CancelAuth"}


@pytest.mark.asyncio
async def test_timeout_disables_fetch_and_detaches_without_resolving():
    from tools.browser_supervisor_auth import NativeAuthMixin
    supervisor = NativeAuthMixin()
    supervisor._vault_auth_events = {}
    calls = []

    async def cdp(method, params=None, **kwargs):
        calls.append(method)
        if method == "Target.getTargets":
            return {"result": {"targetInfos": [{"type": "page", "url": "http://fixture.test/", "targetId": "target"}]}}
        if method == "Target.attachToTarget":
            return {"result": {"sessionId": "session"}}
        if method == "Page.getFrameTree":
            return {"result": {"frameTree": {"frame": {"id": "top"}}}}
        return {}

    supervisor._cdp = cdp
    resolve = Mock()
    result = await supervisor._vault_basic(["http://fixture.test"], resolve, 0.01, Mock())
    assert result["error_type"] == "native_auth_timeout"
    resolve.assert_not_called()
    assert calls[-2:] == ["Fetch.disable", "Target.detachFromTarget"]
    assert not supervisor._vault_auth_events
    assert not supervisor._vault_auth_busy
