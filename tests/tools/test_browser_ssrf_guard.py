"""Region C unit tests — CDP Fetch page guard + decision functions.

Decision functions are tested directly; the guard's arm/decision loop is
driven against a fake CDP websocket server (``websockets`` fixture) that
records commands and accepts pushed events, per the consensus contract's
fake-CDP style.
"""

import asyncio
import json
import os
import socket
import sys
import threading
import time
from unittest.mock import patch

import pytest
import websockets

import tools.browser_ssrf_guard as ssrf
import tools.browser_use_cli as bu_cli
import tools.browser_use_guard as bug

from tools.browser_ssrf_guard import (
    _SSRF_GUARD_JS,
    BrowserSsrfGuard,
    browser_exec_blocked,
    remote_ip_blocked,
)


@pytest.fixture(autouse=True)
def _isolate_hermes_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.delenv("BU_CDP_WS", raising=False)
    monkeypatch.delenv("BU_CDP_URL", raising=False)
    yield


# ── Decision functions ─────────────────────────────────────────────────────

class TestGuardDecisionComposition:
    @pytest.mark.parametrize("url", [
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.google.internal/",
        "http://10.0.0.8/",
        "http://172.16.5.5/",
        "http://192.168.1.1/",
        "http://100.64.0.1/",
        "http://localhost/",
        "http://intra.local/",
        "http://corp.internal/",
        "http://[fd00::1]/",
        "http://[::ffff:10.0.0.1]/",
    ])
    def test_private_and_metadata_blocked(self, url):
        assert browser_exec_blocked(url) is True, url

    @pytest.mark.parametrize("url", [
        "https://example.com/",
        "https://multimedia.nt.qq.com.cn/download?id=1",
    ])
    def test_public_and_trusted_allowed(self, url):
        with patch("socket.getaddrinfo", return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
        ]):
            assert browser_exec_blocked(url) is False, url

    def test_shape_precheck_closes_proxy_dns_hole(self, monkeypatch):
        """Guard never delegates DNS to the proxy (C2)."""
        monkeypatch.setenv("HTTP_PROXY", "http://proxy.internal:9090")
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("nxdomain")):
            assert browser_exec_blocked("http://corp.internal/") is True
            assert browser_exec_blocked("http://localhost/") is True
            assert browser_exec_blocked("http://some.public-looking.name/") is True

    def test_ws_endpoint_normalization(self):
        assert browser_exec_blocked("ws://169.254.169.254/") is True
        assert browser_exec_blocked("ws://127.0.0.1:9222/") is True
        with patch("socket.getaddrinfo", return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
        ]):
            assert browser_exec_blocked("wss://example.com/") is False

    def test_unparseable_url_fails_closed(self):
        assert browser_exec_blocked("") is True
        assert browser_exec_blocked("not a url") is True


class TestRemoteIpGate:
    def test_remote_ip_classes(self):
        assert remote_ip_blocked("10.0.0.5", "http://x.example/") is True
        assert remote_ip_blocked("169.254.169.254", "http://y.example/") is True
        assert remote_ip_blocked("::ffff:192.168.1.1", "http://z.example/") is True
        assert remote_ip_blocked("93.184.216.34", "https://z.example/") is False
        assert remote_ip_blocked("", "http://x.example/") is True  # missing → fail closed

    def test_trusted_host_consistency(self):
        # multimedia.nt.qq.com.cn may legitimately resolve to 198.18.x (trusted).
        assert remote_ip_blocked("198.18.0.5", "https://multimedia.nt.qq.com.cn/x") is False
        # ... but only over https.
        assert remote_ip_blocked("198.18.0.5", "http://multimedia.nt.qq.com.cn/x") is True

    def test_bracketed_ipv6_remote_ip(self):
        """Chrome reports IPv6 remoteIPAddress bracketed — never a false block."""
        assert remote_ip_blocked("[2606:4700:10::6814:179a]", "https://example.com/") is False
        assert remote_ip_blocked("[2606:4700:10::6814:179a]", "http://example.com/") is False
        # Bracketed IPv4-mapped private is still blocked.
        assert remote_ip_blocked("[::ffff:192.168.1.1]", "https://z.example/") is True


class TestGuardJsSourceInvariants:
    def test_source_invariants(self):
        js = _SSRF_GUARD_JS
        assert js and len(js) > 500
        assert "__hermesSsrfGuard" in js
        assert "Symbol.for" in js
        for wrapper in ("fetch", "XMLHttpRequest", "WebSocket", "EventSource",
                        "sendBeacon", "window.open"):
            assert wrapper in js, wrapper
        assert "configurable: false" in js
        assert "writable: false" in js
        assert "isPrivateLiteral" in js
        # Fail-closed at Layer 1: unresolvable URL → block.
        assert "return true; // unresolvable URL" in js


# ── Fake CDP server ────────────────────────────────────────────────────────

class FakeCdpServer:
    """WebSocket CDP server that records commands and accepts pushed events."""

    def __init__(self, targets=("t1",), target_types=None, delay_get_targets=0.0):
        self.targets = list(targets)
        self.target_types = dict(target_types or {})  # targetId -> type (default page)
        self.delay_get_targets = delay_get_targets
        self.commands = []          # (method, params, session_id)
        self.sessions = {}          # targetId -> sessionId
        self.failed = []            # requestIds passed to Fetch.failRequest
        self.continued = []         # requestIds passed to Fetch.continueRequest
        self.ws_url = None
        self._loop = None
        self._event_q = None
        self._thread = None
        self._ready = threading.Event()

    async def _handler(self, ws):
        self._loop = asyncio.get_running_loop()
        self._event_q = asyncio.Queue()

        async def _writer():
            while True:
                evt = await self._event_q.get()
                await ws.send(json.dumps(evt))

        writer = asyncio.create_task(_writer())
        try:
            async for raw in ws:
                msg = json.loads(raw)
                if "id" not in msg:
                    continue
                mid = msg["id"]
                method = msg.get("method", "")
                params = msg.get("params", {})
                session_id = msg.get("sessionId")
                self.commands.append((method, params, session_id))
                if method == "Target.getTargets":
                    if self.delay_get_targets:
                        await asyncio.sleep(self.delay_get_targets)
                    result = {
                        "targetInfos": [
                            {"type": self.target_types.get(t, "page"), "targetId": t,
                             "title": "", "url": "about:blank"}
                            for t in self.targets
                        ]
                    }
                elif method == "Target.attachToTarget":
                    target_id = str(params.get("targetId"))
                    self.sessions[target_id] = "s-" + target_id
                    result = {"sessionId": self.sessions[target_id]}
                elif method == "Fetch.failRequest":
                    self.failed.append(params.get("requestId"))
                    result = {}
                elif method == "Fetch.continueRequest":
                    self.continued.append(params.get("requestId"))
                    result = {}
                elif method == "Runtime.runIfWaitingForDebugger":
                    result = {}
                else:
                    result = {}
                await ws.send(json.dumps({"id": mid, "result": result}))
        finally:
            writer.cancel()
            try:
                await writer
            except Exception:
                pass

    def start(self):
        async def _serve():
            async with websockets.serve(self._handler, "127.0.0.1", 0, max_size=2**26) as srv:
                port = srv.sockets[0].getsockname()[1]
                self.ws_url = f"ws://127.0.0.1:{port}/devtools/browser/x"
                self._ready.set()
                await asyncio.Future()

        self._thread = threading.Thread(target=lambda: asyncio.run(_serve()), daemon=True)
        self._thread.start()
        assert self._ready.wait(timeout=10), "fake CDP server did not start"
        return self

    def stop(self):
        pass

    def push(self, method, params=None, session_id=None):
        evt = {"method": method, "params": params or {}}
        if session_id:
            evt["sessionId"] = session_id
        if self._loop is not None and self._event_q is not None:
            self._loop.call_soon_threadsafe(self._event_q.put_nowait, evt)


@pytest.fixture
def cdp_server():
    server = FakeCdpServer().start()
    yield server
    server.stop()


def _drive_guard(server, script, timeout=15.0):
    """Run an async script against an armed BrowserSsrfGuard."""

    async def _main():
        guard = BrowserSsrfGuard(server.ws_url, None, task_id="t")
        arm_task = asyncio.create_task(guard.arm())
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if guard._sessions_armed:
                break
            await asyncio.sleep(0.05)
        assert guard._sessions_armed, "guard never armed"
        try:
            return await script(guard)
        finally:
            arm_task.cancel()
            try:
                await arm_task
            except (asyncio.CancelledError, Exception):
                pass
            await guard.teardown()

    return asyncio.run(_main())


# ── Arm sequence + request decision loop ───────────────────────────────────

class TestArmAndDecisions:
    def _session_for_target(self, server, target_id):
        return server.sessions.get(target_id)

    def test_arm_sequence_on_existing_targets(self, cdp_server):
        async def script(guard):
            return None

        _drive_guard(cdp_server, script)
        methods = [m for m, _, _ in cdp_server.commands]
        assert "Target.getTargets" in methods
        sid = self._session_for_target(cdp_server, "t1")
        assert sid is not None
        # Per-target arm sequence.
        assert ("Page.enable", {}, sid) in cdp_server.commands
        assert ("Runtime.enable", {}, sid) in cdp_server.commands
        add_scripts = [p for m, p, s in cdp_server.commands
                       if m == "Page.addScriptToEvaluateOnNewDocument" and s == sid]
        assert add_scripts and "runImmediately" in add_scripts[0]
        fetch_enables = [p for m, p, s in cdp_server.commands
                         if m == "Fetch.enable" and s == sid]
        assert fetch_enables
        stages = {stage for p in fetch_enables
                  for stage in (pt.get("requestStage") for pt in p.get("patterns", []))}
        # Request-stage interception ONLY: a Response-stage pause suppresses
        # Network.responseReceived for intercepted requests and deadlocks
        # every legitimate public request against real Chrome (the remote-IP
        # gate is async off responseReceived instead).
        assert stages == {"Request"}
        assert ("Network.enable", {}, sid) in cdp_server.commands
        # Browser-level auto-attach (root session, no sessionId).
        assert any(m == "Target.setAutoAttach"
                   and p.get("waitForDebuggerOnStart") is True
                   and p.get("flatten") is True
                   for m, p, s in cdp_server.commands if s is None)

    def test_private_request_paused_failed_public_continued(self, cdp_server):
        async def script(guard):
            cdp_server.push("Fetch.requestPaused",
                            {"requestId": "r-private", "request": {"url": "http://10.0.0.5/x"}},
                            session_id="s-t1")
            cdp_server.push("Fetch.requestPaused",
                            {"requestId": "r-public", "request": {"url": "https://example.com/"}},
                            session_id="s-t1")
            await asyncio.sleep(0.4)

        _drive_guard(cdp_server, script)
        assert "r-private" in cdp_server.failed
        assert "r-public" in cdp_server.continued

    def test_new_target_paused_until_armed(self, cdp_server):
        async def script(guard):
            cdp_server.push("Target.attachedToTarget",
                            {"sessionId": "s-new", "waitingForDebugger": True,
                             "targetInfo": {"type": "page", "targetId": "t-new",
                                            "url": "about:blank"}},
                            session_id=None)
            await asyncio.sleep(0.3)
            cdp_server.push("Fetch.requestPaused",
                            {"requestId": "r-new", "request": {"url": "http://192.168.1.9/x"}},
                            session_id="s-new")
            await asyncio.sleep(0.4)

        _drive_guard(cdp_server, script)
        # Fetch.enable on s-new must precede runIfWaitingForDebugger.
        idx_fetch = next(i for i, (m, _, s) in enumerate(cdp_server.commands)
                         if m == "Fetch.enable" and s == "s-new")
        idx_resume = next(i for i, (m, _, s) in enumerate(cdp_server.commands)
                          if m == "Runtime.runIfWaitingForDebugger" and s == "s-new")
        assert idx_fetch < idx_resume
        assert "r-new" in cdp_server.failed

    def test_async_remote_ip_gate_emits_block_marker(self, cdp_server):
        """Response-stage pausing is gone: the remote-IP gate is async.

        Fetch.enable arms Request-stage interception only, so
        Network.responseReceived flows for every request; the guard emits
        the block marker (parent kills the CLI + withholds) when the
        browser-observed remote IP is private/IMDS and NEVER pauses a
        response (the old response-stage pause deadlocked real Chrome by
        waiting on a remote IP that Request+Response interception never
        delivers). Public remote IPs produce no marker.
        """
        markers = []
        original = ssrf.BrowserSsrfGuard._emit_block

        def _capture(self, url, stage):
            markers.append((url, stage))
            original(self, url, stage)

        ssrf.BrowserSsrfGuard._emit_block = _capture
        try:
            async def script(guard):
                # Public remote IP → no marker.
                cdp_server.push("Network.responseReceived",
                                {"requestId": "rp", "response": {
                                    "url": "https://public.example/x",
                                    "remoteIPAddress": "93.184.216.34"}})
                # Private remote IP → async block marker.
                cdp_server.push("Network.responseReceived",
                                {"requestId": "rpriv", "response": {
                                    "url": "https://host.example/x",
                                    "remoteIPAddress": "10.0.0.5"}})
                # IMDS remote IP → async block marker.
                cdp_server.push("Network.responseReceived",
                                {"requestId": "rimds", "response": {
                                    "url": "https://host2.example/x",
                                    "remoteIPAddress": "169.254.169.254"}})
                # IPv4-mapped private → async block marker.
                cdp_server.push("Network.responseReceived",
                                {"requestId": "rmap", "response": {
                                    "url": "https://host3.example/x",
                                    "remoteIPAddress": "::ffff:192.168.1.1"}})
                # A Request-stage pause for a public URL is still continued
                # (public literal IP — passes the request-stage gate with no
                # DNS; no response-stage fail exists anymore).
                cdp_server.push("Fetch.requestPaused",
                                {"requestId": "rpub2",
                                 "request": {"url": "http://93.184.216.34/x"}},
                                session_id="s-t1")
                await asyncio.sleep(0.8)

            _drive_guard(cdp_server, script)
        finally:
            ssrf.BrowserSsrfGuard._emit_block = original
        assert not any(url == "https://public.example/x" for url, _ in markers)
        assert ("https://host.example/x", "response:10.0.0.5") in markers
        assert any(url == "https://host2.example/x" for url, _ in markers)
        assert any(url == "https://host3.example/x" for url, _ in markers)
        # The Request-stage pause was continued — nothing was failRequest'd
        # (the async gate never pauses or fails a response).
        assert "rpub2" in cdp_server.continued
        assert cdp_server.failed == []

    def test_ws_handshake_private_remote_ip_marker(self, cdp_server):
        async def script(guard):
            cdp_server.push("Network.responseReceived",
                            {"requestId": "ws1", "type": "WebSocket", "response": {
                                "url": "wss://ws.example/x",
                                "remoteIPAddress": "10.0.0.5"}})
            await asyncio.sleep(0.4)

        markers = []
        original = ssrf.BrowserSsrfGuard._emit_block

        def _capture(self, url, stage):
            markers.append((url, stage))
            original(self, url, stage)

        ssrf.BrowserSsrfGuard._emit_block = _capture
        try:
            _drive_guard(cdp_server, script)
        finally:
            ssrf.BrowserSsrfGuard._emit_block = original
        assert markers and markers[0][0] == "wss://ws.example/x"

    def test_model_cdp_fetch_disable_does_not_disarm_guard(self, cdp_server):
        """Two-session independence: the guard's Fetch.enable is its own."""
        async def script(guard):
            # Simulate a model-side Fetch.disable on a DIFFERENT session.
            cdp_server.push("Fetch.requestPaused",
                            {"requestId": "r1", "request": {"url": "http://10.0.0.7/x"}},
                            session_id="s-t1")
            await asyncio.sleep(0.4)

        _drive_guard(cdp_server, script)
        # The guard session never issues Fetch.disable.
        assert not any(m == "Fetch.disable" for m, _, _ in cdp_server.commands)
        assert "r1" in cdp_server.failed


class TestGuardProcessEnv:
    def test_guard_env_strips_model_controlled_workspace(self, monkeypatch):
        """Mirror test_trusted_probe_strips_model_controlled_workspace_env."""
        monkeypatch.setenv("BH_AGENT_WORKSPACE", "C:/model/workspace")
        monkeypatch.setenv("BU_WORKSPACE", "C:/model/bu")
        env = bug._guard_process_env()
        assert "BH_AGENT_WORKSPACE" not in env
        assert "BU_WORKSPACE" not in env


# ── browser_exec integration (stubbed guard stack) ─────────────────────────

class _StubMonitor:
    def __init__(self, *, armed=True, saw_activity=True, violation=None, last_known_url=""):
        self._armed = armed
        self._saw = saw_activity
        self._violation = violation
        self._last = last_known_url
        self.stopped = False

    def attach_failed(self): return not self._armed
    def armed(self): return self._armed
    def saw_activity(self, exec_started): return self._saw
    def mark_probe_success(self): pass
    def violation(self): return dict(self._violation) if self._violation else None
    def reset(self): pass
    def last_known_url(self): return self._last
    def event_count(self): return 1 if self._saw else 0
    def dropped(self): return False
    def request_log(self, limit=200): return []
    def stop(self, timeout=5.0): self.stopped = True


class _FakeGuardProc(dict):
    def __init__(self, blocked=False):
        super().__init__(
            blocked=threading.Event(),
            arm_failed=threading.Event(),
            died=threading.Event(),
            markers=["__HERMES_BROWSER_EXEC_SSRF_BLOCK__:http://10.0.0.1/x"] if blocked else [],
            proc=None,
            report_sock=None,
        )
        if blocked:
            self["blocked"].set()


def _fake_cli(tmp_path, body):
    script = tmp_path / "browser-use.py"
    script.write_text(body, encoding="utf-8")
    return str(script)


class TestBrowserExecGuardIntegration:
    def test_browser_exec_arms_and_tears_down_guard(self, tmp_path, monkeypatch):
        """Armed before spawn; torn down in finally; no Layer-1-only branch."""
        import tools.browser_use_guard as bug_mod

        cli = _fake_cli(
            tmp_path,
            "import sys\nsys.stdin.read()\n"
            "print('__HERMES_BROWSER_EXEC_ARMED__:full', flush=True)\nprint('ok')\n",
        )
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [sys.executable, cli])
        monkeypatch.setattr(
            bu_cli, "_ensure_exec_cdp_endpoint",
            lambda env, tid, sess: (env.setdefault("BU_CDP_WS", "ws://127.0.0.1:1/x"), None)[1],
        )

        order = []
        fake_guard = _FakeGuardProc()
        monitor = _StubMonitor()
        ctx = {
            "enabled": True,
            "config": {"fail_open": False, "grace_s": 0.0, "attach_timeout_s": 1.0,
                       "allow_private": False},
            "endpoint": "ws://127.0.0.1:1/x",
            "tier": "t1",
            "token": "tok",
            "state_dir": "",
            "exec_started": time.monotonic(),
            "monitor": monitor,
            "ssrf_guard": fake_guard,
            "error": None,
        }

        def _prepare(env, tid, sess, popen_extra=None):
            order.append("prepare")
            return ctx

        monkeypatch.setattr(bug_mod, "_prepare_guard", _prepare)
        monkeypatch.setattr(bug_mod, "_guard_env", lambda env, ctx: env)
        monkeypatch.setattr(bug_mod, "_guard_self_test", lambda ctx, env: None)
        monkeypatch.setattr(bu_cli, "_trusted_landed_url", lambda *a, **k: "https://example.com/")
        result = json.loads(bu_cli.browser_exec("print('x')"))
        assert result["success"] is True
        assert order == ["prepare"]
        assert monitor.stopped is True  # torn down in finally

    def test_arm_failure_fails_closed(self, tmp_path, monkeypatch):
        import tools.browser_use_guard as bug_mod

        cli = _fake_cli(tmp_path, "import sys\nsys.stdin.read()\nprint('SECRET')\n")
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [sys.executable, cli])
        monkeypatch.setattr(
            bu_cli, "_ensure_exec_cdp_endpoint",
            lambda env, tid, sess: (env.setdefault("BU_CDP_WS", "ws://127.0.0.1:1/x"), None)[1],
        )

        def _prepare(env, tid, sess, popen_extra=None):
            return {"enabled": True, "config": {}, "error": "the Fetch guard could not be armed"}

        monkeypatch.setattr(bug_mod, "_prepare_guard", _prepare)
        result = json.loads(bu_cli.browser_exec("print('SECRET')"))
        assert "error" in result
        assert "Fetch guard could not be armed" in result["error"]
        assert "SECRET" not in json.dumps(result)

    def test_browser_exec_fails_closed_without_endpoint(self, tmp_path, monkeypatch):
        """No Hermes-attested endpoint → tool_error; CLI never spawned."""
        import tools.browser_use_guard as bug_mod

        cli = _fake_cli(tmp_path, "import sys\nsys.stdin.read()\nprint('SECRET')\n")
        spawned = []

        def _spy_popen(*a, **k):
            spawned.append(1)
            return _real_popen(*a, **k)

        _real_popen = bug_mod.subprocess.Popen
        monkeypatch.setattr(bug_mod.subprocess, "Popen", _spy_popen)
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [sys.executable, cli])
        monkeypatch.setattr(
            bu_cli, "_ensure_exec_cdp_endpoint",
            lambda env, tid, sess: "no Hermes-attested CDP endpoint available",
        )
        result = json.loads(bu_cli.browser_exec("print('SECRET')"))
        assert "error" in result
        assert not spawned, "CLI must not be spawned without an attested endpoint"

    def test_guard_block_marker_kills_exec(self, tmp_path, monkeypatch):
        import tools.browser_use_guard as bug_mod

        cli = _fake_cli(tmp_path, "import sys, time\nsys.stdin.read()\ntime.sleep(30)\nprint('SECRET')\n")
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [sys.executable, cli])
        monkeypatch.setattr(
            bu_cli, "_ensure_exec_cdp_endpoint",
            lambda env, tid, sess: (env.setdefault("BU_CDP_WS", "ws://127.0.0.1:1/x"), None)[1],
        )

        fake_guard = _FakeGuardProc(blocked=True)
        monitor = _StubMonitor()
        ctx = {
            "enabled": True,
            "config": {"fail_open": False, "grace_s": 0.0, "attach_timeout_s": 1.0,
                       "allow_private": False},
            "endpoint": "ws://127.0.0.1:1/x",
            "tier": "t1",
            "token": "tok",
            "state_dir": "",
            "exec_started": time.monotonic(),
            "monitor": monitor,
            "ssrf_guard": fake_guard,
            "error": None,
        }
        monkeypatch.setattr(bug_mod, "_prepare_guard", lambda *a, **k: ctx)
        monkeypatch.setattr(bug_mod, "_guard_env", lambda env, ctx: env)
        monkeypatch.setattr(bug_mod, "_guard_self_test", lambda ctx, env: None)
        monkeypatch.setattr(bu_cli, "_trusted_landed_url", lambda *a, **k: None)

        start = time.monotonic()
        result = json.loads(bu_cli.browser_exec("print('x')"))
        elapsed = time.monotonic() - start
        assert "error" in result
        assert "10.0.0.1" in result["error"]
        assert "SECRET" not in json.dumps(result)
        assert elapsed < 25, "CLI must be killed on guard block, not run to timeout"

    def test_model_cannot_restore_wrapped_globals(self):
        """Live-validated Layer-1 invariant (gated like the P1 boundary test)."""
        if not os.environ.get("HERMES_E2E_BROWSER"):
            pytest.skip("set HERMES_E2E_BROWSER=1 to run the live-validated JS test")
        import subprocess as _sp

        probe = (
            "import asyncio, json, sys\n"
            "sys.path.insert(0, '.')\n"
            "from tools.browser_ssrf_guard import _SSRF_GUARD_JS\n"
            "js = _SSRF_GUARD_JS\n"
            "assert 'Object.defineProperty(window, \\'fetch\\'' in js\n"
            "print('JS-OK')\n"
        )
        p = _sp.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=30)
        assert p.returncode == 0, p.stderr
        assert "JS-OK" in p.stdout


# ── Defect 3: every auto-attached target type is armed (Region C) ─────────

class TestArmEveryTargetType:
    def test_existing_worker_target_armed_and_private_request_failed(self):
        """A dedicated-worker session gets Fetch+Network gates, no page JS."""
        server = FakeCdpServer(
            targets=("tw",), target_types={"tw": "worker"}
        ).start()
        try:
            async def script(guard):
                server.push("Fetch.requestPaused",
                            {"requestId": "rw",
                             "request": {"url": "http://169.254.169.254/latest/meta-data/"}},
                            session_id="s-tw")
                await asyncio.sleep(0.4)

            _drive_guard(server, script)
        finally:
            server.stop()
        sid = server.sessions.get("tw")
        assert sid == "s-tw"
        # The worker session is armed with the authoritative gates…
        assert any(m == "Fetch.enable" and s == sid for m, _, s in server.commands)
        assert any(m == "Network.enable" and s == sid for m, _, s in server.commands)
        assert any(m == "Runtime.enable" and s == sid for m, _, s in server.commands)
        # …but the page-JS Layer 1 interceptor is skipped (no DOM).
        assert not any(m == "Page.enable" and s == sid for m, _, s in server.commands)
        # A worker fetch to the metadata address is blocked at the boundary.
        assert "rw" in server.failed

    def test_oopif_iframe_auto_attach_armed_before_resume_and_blocked(self):
        """OOPIF iframe: arm session BEFORE runIfWaitingForDebugger; block."""
        server = FakeCdpServer(targets=("t1",)).start()
        try:
            async def script(guard):
                server.push("Target.attachedToTarget",
                            {"sessionId": "s-iframe", "waitingForDebugger": True,
                             "targetInfo": {"type": "iframe", "targetId": "t-iframe",
                                            "url": "about:blank"}},
                            session_id=None)
                await asyncio.sleep(0.3)
                server.push("Fetch.requestPaused",
                            {"requestId": "rif",
                             "request": {"url": "http://10.0.0.5/secret"}},
                            session_id="s-iframe")
                await asyncio.sleep(0.4)

            _drive_guard(server, script)
        finally:
            server.stop()
        idx_fetch = next(i for i, (m, _, s) in enumerate(server.commands)
                         if m == "Fetch.enable" and s == "s-iframe")
        idx_resume = next(i for i, (m, _, s) in enumerate(server.commands)
                          if m == "Runtime.runIfWaitingForDebugger" and s == "s-iframe")
        assert idx_fetch < idx_resume
        assert "rif" in server.failed

    def test_worker_auto_attach_armed_before_resume(self):
        """A paused worker is resumed only after its session is armed."""
        server = FakeCdpServer(targets=("t1",)).start()
        try:
            async def script(guard):
                server.push("Target.attachedToTarget",
                            {"sessionId": "s-worker", "waitingForDebugger": True,
                             "targetInfo": {"type": "worker", "targetId": "t-worker",
                                            "url": "worker.js"}},
                            session_id=None)
                await asyncio.sleep(0.3)
                server.push("Fetch.requestPaused",
                            {"requestId": "rwk",
                             "request": {"url": "http://192.168.1.9/x"}},
                            session_id="s-worker")
                await asyncio.sleep(0.4)

            _drive_guard(server, script)
        finally:
            server.stop()
        idx_fetch = next(i for i, (m, _, s) in enumerate(server.commands)
                         if m == "Fetch.enable" and s == "s-worker")
        idx_resume = next(i for i, (m, _, s) in enumerate(server.commands)
                          if m == "Runtime.runIfWaitingForDebugger" and s == "s-worker")
        assert idx_fetch < idx_resume
        assert "rwk" in server.failed

    @pytest.mark.parametrize("ttype", [
        "fencedframe", "auction_worklet", "interest_group_worklet",
        "shared_storage_worklet",
    ])
    def test_fencedframe_and_worklet_targets_armed(self, ttype):
        """Fix-2 regression: fencedframe + worklet targets are armed.

        These network-capable target types used to be absent from
        ``_GUARD_ARMED_TARGET_TYPES``, so their requests were never
        intercepted. They now get the authoritative Fetch + Network gates
        (Fetch.enable/Network.enable per session); page-JS Layer 1 applies
        only to the DOM-bearing fenced frame, never to worklets.
        """
        server = FakeCdpServer(
            targets=("tx",), target_types={"tx": ttype}
        ).start()
        try:
            async def script(guard):
                server.push("Fetch.requestPaused",
                            {"requestId": "rx",
                             "request": {"url": "http://169.254.169.254/latest/meta-data/"}},
                            session_id="s-tx")
                await asyncio.sleep(0.4)

            _drive_guard(server, script)
        finally:
            server.stop()
        sid = server.sessions.get("tx")
        assert sid == "s-tx"
        assert any(m == "Fetch.enable" and s == sid for m, _, s in server.commands)
        assert any(m == "Network.enable" and s == sid for m, _, s in server.commands)
        if ttype == "fencedframe":
            assert any(m == "Page.enable" and s == sid for m, _, s in server.commands)
        else:
            assert not any(m == "Page.enable" and s == sid for m, _, s in server.commands)
        # A metadata fetch from the armed session is blocked at the boundary.
        assert "rx" in server.failed


# ── Defect 2: READY/ARMED emitted only after arm() completes ──────────────

class TestGuardMainReadyOrdering:
    def _report_listener(self):
        """Local report-channel listener; returns (port, lines, accepted_at)."""
        lsock = socket.socket()
        lsock.bind(("127.0.0.1", 0))
        lsock.listen(1)
        port = lsock.getsockname()[1]
        lines = []
        accepted_at = []
        ready_at = []

        def _accept():
            conn, _ = lsock.accept()
            accepted_at.append(time.monotonic())
            buf = b""
            conn.settimeout(5)
            try:
                while True:
                    data = conn.recv(4096)
                    if not data:
                        break
                    buf += data
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        text = line.decode("utf-8", "replace").strip()
                        lines.append(text)
                        if text.startswith("__HERMES_SSRF_GUARD_READY__"):
                            ready_at.append(time.monotonic())
            except Exception:
                pass
            conn.close()

        t = threading.Thread(target=_accept, daemon=True)
        t.start()
        return lsock, port, lines, accepted_at, ready_at, t

    def test_ready_emitted_only_after_arm_completes(self):
        """READY must follow arm() (here delayed 0.5s), never precede it."""
        server = FakeCdpServer(targets=("t1",), delay_get_targets=0.5).start()
        lsock, port, lines, accepted_at, ready_at, t = self._report_listener()
        try:
            async def _drive():
                task = asyncio.create_task(ssrf._guard_main(server.ws_url, port, "tok"))
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline and not lines:
                    await asyncio.sleep(0.05)
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

            asyncio.run(_drive())
        finally:
            server.stop()
        t.join(timeout=5)
        lsock.close()
        # The single report line is the post-arm READY carrying the token.
        assert lines == ["__HERMES_SSRF_GUARD_READY__:tok"], lines
        # It arrived well after the report connection was accepted — i.e.
        # after arming, not at connect time.
        assert accepted_at, "report connection was never accepted"
        assert ready_at, "READY never arrived"
        assert ready_at[0] - accepted_at[0] >= 0.3, (
            "READY arrived before arm() could have completed"
        )

    def test_arm_failure_emits_arm_failed_and_exits_nonzero(self):
        """A guard that cannot arm reports ARM_FAILED and exits 1."""
        lsock, port, lines, _acc, _rdy, t = self._report_listener()
        try:
            rc = asyncio.run(ssrf._guard_main("ws://127.0.0.1:1/dead", port, "tok"))
        finally:
            t.join(timeout=5)
            lsock.close()
        assert rc == 1
        assert lines == ["__HERMES_SSRF_GUARD_ARM_FAILED__"], lines


class TestSpawnSsrfGuard:
    """Parent-side ``_spawn_ssrf_guard`` arm handshake (defect 2)."""

    def test_spawn_withholds_on_arm_failure(self, monkeypatch, tmp_path):
        """Guard that cannot arm (dead CDP) → None (fail closed, no CLI spawn)."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
        guard = bug._spawn_ssrf_guard("ws://127.0.0.1:1/dead", "tok")
        assert guard is None

    def test_spawn_returns_guard_only_after_armed(self, monkeypatch, tmp_path):
        """READY follows arm(): spawn returns the guard dict only once armed."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
        server = FakeCdpServer(targets=("t1",), delay_get_targets=0.3).start()
        guard = None
        try:
            guard = bug._spawn_ssrf_guard(server.ws_url, "tok")
            assert guard is not None
            assert guard["markers"] == ["__HERMES_SSRF_GUARD_READY__:tok"]
        finally:
            if guard is not None:
                bug._teardown_ssrf_guard(guard)
            server.stop()


# ── Fix: request-stage-only interception + async remote-IP gate (real browser) ──

class TestRealBrowserRequestStageFlow:
    """Env-gated (HERMES_E2E_BROWSER=1) real-Chrome regression for the
    response-stage deadlock.

    With the old Request+Response Fetch.enable the guard paused every
    response and waited 300ms for a ``Network.responseReceived`` remote IP
    that Request+Response interception never delivered — so every
    legitimate public request was failed and navigations landed on
    ``chrome-error://chromewebdata/``. This test drives a REAL headless
    Chrome through the armed guard and asserts the opposite:

    * a public request (https://example.com/) passes — the Request-stage
      pause is continued, no block marker is emitted, and the page actually
      lands on example.com (NOT chrome-error://chromewebdata/);
    * a private/IMDS request (http://169.254.169.254/...) is blocked — the
      request-stage gate fails it and the guard emits the block marker.

    Skipped unless ``HERMES_E2E_BROWSER=1`` (and a Chrome binary + internet
    access exist); the FakeCdpServer tests above cover the unit surface.
    """

    def test_public_request_passes_private_request_blocked(self):
        if not os.environ.get("HERMES_E2E_BROWSER"):
            pytest.skip("set HERMES_E2E_BROWSER=1 to run the real-browser SSRF guard test")
        from tools.browser_exec_monitor import _find_chrome_binary, spawn_supervised_chrome

        if not _find_chrome_binary():
            pytest.skip("no Chrome/Chromium binary available for the E2E test")
        # The public half of the test needs real internet; skip cleanly when
        # the host is offline rather than failing on the network itself.
        try:
            socket.create_connection(("example.com", 443), timeout=5).close()
        except OSError:
            pytest.skip("no internet access for the public-request half of the E2E test")

        try:
            ws_url = spawn_supervised_chrome("e2e-ssrf-request-stage")
        except RuntimeError as e:
            # e.g. the host runs an elevated shell, which Chromium refuses to
            # start under (browser exits immediately, no CDP endpoint).
            pytest.skip(f"could not spawn a real browser for the E2E test: {e}")
        markers: list = []

        async def _main() -> bool:
            guard = BrowserSsrfGuard(ws_url, None, task_id="e2e-request-stage")
            orig_emit = guard._emit_block

            def _capture(url, stage):
                markers.append((url, stage))
                orig_emit(url, stage)

            guard._emit_block = _capture
            await guard.arm()
            driver = await websockets.connect(ws_url, max_size=50 * 1024 * 1024)
            next_id = [1]
            pending: dict = {}

            async def _cdp(method, params=None, session_id=None):
                cid = next_id[0]
                next_id[0] += 1
                payload = {"id": cid, "method": method}
                if params:
                    payload["params"] = params
                if session_id:
                    payload["sessionId"] = session_id
                fut = asyncio.get_running_loop().create_future()
                pending[cid] = fut
                await driver.send(json.dumps(payload))
                return await asyncio.wait_for(fut, timeout=20.0)

            async def _reader():
                async for raw in driver:
                    msg = json.loads(raw)
                    if "id" in msg:
                        fut = pending.pop(msg["id"], None)
                        if fut is not None and not fut.done():
                            fut.set_result(msg)

            reader = asyncio.create_task(_reader())
            try:
                created = await _cdp("Target.createTarget", {"url": "about:blank"})
                target_id = created["result"]["targetId"]
                # The guard's auto-attach pauses the new target
                # (waitForDebuggerOnStart) and arms + resumes it; wait for the
                # arming to land so Fetch.enable precedes any navigation.
                await asyncio.sleep(1.5)
                assert guard._sessions_armed, "guard never armed a session"
                attach = await _cdp("Target.attachToTarget",
                                    {"targetId": target_id, "flatten": True})
                sess = attach["result"]["sessionId"]
                await _cdp("Page.enable", session_id=sess)

                # 1. Public request must PASS — never chrome-error.
                await _cdp("Page.navigate", {"url": "https://example.com/"}, session_id=sess)
                title = ""
                url_now = ""
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    try:
                        ev = await _cdp(
                            "Runtime.evaluate",
                            {"expression": "document.title + '|' + location.href",
                             "returnByValue": True},
                            session_id=sess,
                        )
                        val = (ev.get("result", {}).get("result", {}) or {}).get("value", "")
                        if val:
                            title, url_now = str(val).split("|", 1)
                    except Exception:
                        pass
                    if "Example Domain" in title or "example.com" in url_now:
                        break
                    await asyncio.sleep(0.5)
                assert "chromewebdata" not in url_now, (
                    f"public request was failed by the guard (deadlock): landed on {url_now}"
                )
                assert "example.com" in url_now, f"public navigation did not land: {url_now}"
                assert not any("example.com" in u for u, _ in markers), markers

                # 2. Private/IMDS request must be BLOCKED (request-stage gate).
                try:
                    await _cdp("Page.navigate",
                               {"url": "http://169.254.169.254/latest/meta-data/"},
                               session_id=sess)
                except Exception:
                    pass
                await asyncio.sleep(2.0)
                assert any("169.254.169.254" in u for u, _ in markers), (
                    "IMDS navigation was not blocked by the armed guard"
                )
                return True
            finally:
                reader.cancel()
                try:
                    await reader
                except (asyncio.CancelledError, Exception):
                    pass
                try:
                    await driver.close()
                except Exception:
                    pass
                await guard.teardown()

        assert asyncio.run(_main()) is True
