"""Region A unit tests — NetworkExecMonitor + exec_url_violation.

Synthetic CDP frames are driven through ``_on_event(method, params,
session_id)`` per the consensus contract; the start()/armed() path uses a
tiny fake CDP websocket server (websockets fixture). Browser_exec
withhold tests (15-17) reuse the fake-CLI pattern from
``test_browser_use_cli.py`` with the guard stack stubbed to the monitor.
"""

import asyncio
import json
import socket
import sys
import threading
import time
from unittest.mock import patch

import pytest
import websockets

import tools.browser_exec_monitor as bem
import tools.browser_use_cli as bu_cli

from tools.browser_exec_monitor import (
    NetworkExecMonitor,
    exec_url_violation,
)

# Monkeypatch socket.getaddrinfo so the monitor's DNS never hits the network
# in these tests (private literal IPs need no DNS anyway).
_ORIG_GETADDRINFO = socket.getaddrinfo


@pytest.fixture(autouse=True)
def _isolate_hermes_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.delenv("BU_CDP_WS", raising=False)
    monkeypatch.delenv("BU_CDP_URL", raising=False)
    yield


async def _feed(monitor, method, params, session_id=None):
    await monitor._on_event(method, params, session_id)


def _rwbs(params):
    return {
        "requestId": params.get("requestId", "req1"),
        "frameId": params.get("frameId", "frame1"),
        "url": params.get("url", "https://example.com/"),
        "type": params.get("type", "Document"),
    }


def _make_monitor(cdp_url="ws://127.0.0.1:1/devtools/browser/x"):
    return NetworkExecMonitor(cdp_url, task_id="t1")


# ── exec_url_violation unit tests ──────────────────────────────────────────

class TestExecUrlViolation:
    def test_request_will_be_sent_private_url_latches_policy(self):
        assert exec_url_violation("http://127.0.0.1:8080/secret") == "private"

    def test_metadata_url_latches_always_blocked(self):
        assert exec_url_violation("http://169.254.169.254/latest/meta-data/") == "metadata"

    def test_redirect_hop_url_validated(self):
        # Redirect hops are validated by redirectResponse.url (private hop).
        assert exec_url_violation("http://10.0.0.5/admin") == "private"

    def test_validate_url_scheme_filter(self):
        for url in ("data:text/html,hi", "blob:https://example.com/uuid",
                    "about:blank", "chrome://settings", "file:///etc/passwd"):
            assert exec_url_violation(url) is None, url
        assert exec_url_violation("ws://127.0.0.1:8080/x") == "private"

    def test_encoded_separator_host_fails_closed(self):
        assert exec_url_violation("http://127.0.0.1%2fevil.com/latest/meta-data/") == "malformed"
        assert exec_url_violation("http://127.0.0.1%5cevil.com/") == "malformed"
        assert exec_url_violation("http://\\user@evil.com/") == "malformed"
        assert exec_url_violation("http://user@evil.com/") == "malformed"

    def test_proxy_carveout_no_longer_fails_open(self, monkeypatch):
        """gaierror on a non-literal host → private even with a proxy set."""
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:9090")
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("nxdomain")):
            assert exec_url_violation("http://some.public-looking.name/") == "private"

    def test_ws_dns_failure_fails_closed(self, monkeypatch):
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:9090")
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("nxdomain")):
            assert exec_url_violation("ws://ws.example.com/socket") == "private"

    def test_global_toggle_ignored(self, monkeypatch):
        """The monitor's predicate is ungated: allow_private must not matter."""
        import tools.url_safety as us

        monkeypatch.setattr(us, "_global_allow_private_urls", lambda: True)
        assert exec_url_violation("http://10.0.0.1/x") == "private"

    def test_public_passes(self):
        # Public literal IP → no DNS needed.
        assert exec_url_violation("http://93.184.216.34/") is None
        with patch("socket.getaddrinfo", return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
        ]):
            assert exec_url_violation("https://example.com/") is None

    def test_private_hostname_suffixes(self):
        assert exec_url_violation("http://intra.local/x") == "private"
        assert exec_url_violation("http://printer.lan/x") == "private"
        assert exec_url_violation("http://localhost:8080/x") == "private"
        # .internal names and metadata hostnames hit the DNS-independent
        # metadata floor FIRST (contract §3 step 3 before step 5).
        assert exec_url_violation("http://corp.internal/x") == "metadata"
        assert exec_url_violation("http://metadata.google.internal/") == "metadata"


# ── Monitor latch tests (synthetic frames) ─────────────────────────────────

class TestMonitorLatches:
    def test_request_will_be_sent_private_url_latches(self):
        m = _make_monitor()
        asyncio.run(_feed(m, "Network.requestWillBeSent", _rwbs({"url": "http://127.0.0.1:8080/secret"})))
        v = m.violation()
        assert v is not None
        assert v["policy"] == "private"
        assert v["url"] == "http://127.0.0.1:8080/secret"

    def test_metadata_url_latches_always_blocked(self):
        m = _make_monitor()
        asyncio.run(_feed(m, "Network.requestWillBeSent",
                          _rwbs({"url": "http://169.254.169.254/latest/meta-data/"})))
        assert m.violation()["policy"] == "metadata"

    def test_redirect_hop_url_validated(self):
        m = _make_monitor()
        params = _rwbs({"url": "http://93.184.216.34/"})  # public literal, no DNS
        params["redirectResponse"] = {"url": "http://10.0.0.5/admin"}
        asyncio.run(_feed(m, "Network.requestWillBeSent", params))
        v = m.violation()
        assert v is not None and v["policy"] == "private"
        assert v["url"] == "http://10.0.0.5/admin"

    def test_response_received_url_validated(self):
        m = _make_monitor()
        params = {"requestId": "r9", "response": {"url": "http://192.168.1.1/x"}}
        asyncio.run(_feed(m, "Network.responseReceived", params))
        assert m.violation()["policy"] == "private"

    def test_request_served_from_cache_private_url_validated(self):
        # A cache-served response URL is validated even though no new wire
        # traffic occurred (fromDiskCache/fromServiceWorker path).
        m = _make_monitor()
        asyncio.run(_feed(m, "Network.requestWillBeSent",
                          _rwbs({"requestId": "c1", "url": "http://93.184.216.34/cached"})))
        assert m.violation() is None
        asyncio.run(_feed(m, "Network.requestServedFromCache", {"requestId": "c1"}))
        assert m.violation() is None  # public cache hit passes
        params = {"requestId": "c2",
                  "response": {"url": "http://10.0.0.9/cached-private", "fromDiskCache": True}}
        asyncio.run(_feed(m, "Network.responseReceived", params))
        assert m.violation() is not None
        assert m.violation()["policy"] == "private"

    def test_new_target_session_enables_network_and_monitors(self):
        m = _make_monitor()
        attach = {"sessionId": "s-new", "targetInfo": {"type": "page", "targetId": "t-new"}}
        sent = []

        async def _record_cdp(method, params=None, **kw):
            sent.append((method, kw.get("session_id")))

        m._cdp = _record_cdp
        asyncio.run(_feed(m, "Target.attachedToTarget", attach))
        assert any(method == "Network.enable" and sid == "s-new" for method, sid in sent)
        with m._state_lock:
            assert "s-new" in m._session_network_armed

    def test_worker_target_attached_network_enabled_and_violation_latched(self):
        """Defect-3 regression: dedicated-worker fetches are observed/blocked."""
        m = _make_monitor()
        attach = {"sessionId": "s-worker",
                  "targetInfo": {"type": "worker", "targetId": "t-worker"}}
        sent = []

        async def _record_cdp(method, params=None, **kw):
            sent.append((method, kw.get("session_id")))

        m._cdp = _record_cdp
        asyncio.run(_feed(m, "Target.attachedToTarget", attach))
        # The worker session is Network.enable'd like any page session.
        assert any(method == "Network.enable" and sid == "s-worker"
                   for method, sid in sent)
        with m._state_lock:
            assert "s-worker" in m._session_network_armed
        # A worker fetch to the IMDS address is observed → violation latched.
        asyncio.run(_feed(m, "Network.requestWillBeSent", {
            "requestId": "rw", "url": "http://169.254.169.254/latest/meta-data/",
            "type": "Fetch",
        }, "s-worker"))
        v = m.violation()
        assert v is not None and v["policy"] == "metadata"

    def test_oopif_iframe_target_attached_network_enabled_and_violation_latched(self):
        """Defect-3 regression: OOPIF iframe requests are observed/blocked."""
        m = _make_monitor()
        attach = {"sessionId": "s-iframe",
                  "targetInfo": {"type": "iframe", "targetId": "t-iframe"}}
        sent = []

        async def _record_cdp(method, params=None, **kw):
            sent.append((method, kw.get("session_id")))

        m._cdp = _record_cdp
        asyncio.run(_feed(m, "Target.attachedToTarget", attach))
        assert any(method == "Network.enable" and sid == "s-iframe"
                   for method, sid in sent)
        with m._state_lock:
            assert "s-iframe" in m._session_network_armed
        asyncio.run(_feed(m, "Network.requestWillBeSent", {
            "requestId": "ri", "url": "http://10.0.0.5/secret", "type": "Fetch",
        }, "s-iframe"))
        v = m.violation()
        assert v is not None and v["policy"] == "private"

    def test_initial_attach_arms_existing_worker_targets(self):
        """Defect-3 regression: pre-existing worker targets get Network.enable."""
        m = _make_monitor()
        sent = []

        async def _cdp(method, params=None, **kw):
            if method == "Target.getTargets":
                return {"result": {"targetInfos": [
                    {"type": "worker", "targetId": "w1", "url": "worker.js"},
                    {"type": "service_worker", "targetId": "sw1", "url": "sw.js"},
                    {"type": "page", "targetId": "p1", "url": "about:blank"},
                ]}}
            if method == "Target.attachToTarget":
                target_id = str(params["targetId"])
                return {"result": {"sessionId": f"s-{target_id}"}}
            sent.append((method, kw.get("session_id")))
            return {"result": {}}

        m._cdp = _cdp
        asyncio.run(m._attach_initial_pages())
        # Every armed target type is Network.enable'd on its own session.
        assert any(method == "Network.enable" and sid == "s-w1" for method, sid in sent)
        assert any(method == "Network.enable" and sid == "s-sw1" for method, sid in sent)
        assert any(method == "Network.enable" and sid == "s-p1" for method, sid in sent)

    def test_fencedframe_and_worklet_targets_armed(self):
        """Fix-2 regression: fencedframe + worklet targets get Network.enable.

        These network-capable target types were absent from
        ``_MONITOR_ARMED_TARGET_TYPES``, so their requests were never
        observed (and therefore never blocked). Every one of them must get a
        per-session ``Network.enable`` like any page/worker target.
        """
        m = _make_monitor()
        sent = []

        async def _cdp(method, params=None, **kw):
            if method == "Target.getTargets":
                return {"result": {"targetInfos": [
                    {"type": "fencedframe", "targetId": "ff1", "url": "about:blank"},
                    {"type": "auction_worklet", "targetId": "aw1", "url": "worklet.js"},
                    {"type": "interest_group_worklet", "targetId": "igw1", "url": "worklet.js"},
                    {"type": "shared_storage_worklet", "targetId": "ssw1", "url": "worklet.js"},
                ]}}
            if method == "Target.attachToTarget":
                target_id = str(params["targetId"])
                return {"result": {"sessionId": f"s-{target_id}"}}
            sent.append((method, kw.get("session_id")))
            return {"result": {}}

        m._cdp = _cdp
        asyncio.run(m._attach_initial_pages())
        for tid in ("ff1", "aw1", "igw1", "ssw1"):
            assert any(
                method == "Network.enable" and sid == f"s-{tid}" for method, sid in sent
            ), tid
        # A fenced-frame fetch to the IMDS address is observed → latched.
        asyncio.run(_feed(m, "Network.requestWillBeSent", {
            "requestId": "rff", "url": "http://169.254.169.254/latest/meta-data/",
            "type": "Fetch",
        }, "s-ff1"))
        v = m.violation()
        assert v is not None and v["policy"] == "metadata"

    def test_violation_latch_write_once_and_reset(self):
        m = _make_monitor()
        asyncio.run(_feed(m, "Network.requestWillBeSent", _rwbs({"url": "http://10.0.0.1/a"})))
        first = m.violation()
        asyncio.run(_feed(m, "Network.requestWillBeSent", _rwbs({"url": "http://10.0.0.2/b"})))
        second = m.violation()
        assert first == second  # write-once
        assert second["url"] == "http://10.0.0.1/a"
        m.reset()
        assert m.violation() is None
        assert m.request_log() == []

    def test_no_verdict_cache_across_requests(self):
        """Each request is validated fresh; a changed resolution is applied."""
        m = _make_monitor()
        # First request: public literal — passes with zero DNS.
        asyncio.run(_feed(m, "Network.requestWillBeSent",
                          _rwbs({"url": "http://93.184.216.34/a"})))
        assert m.violation() is None
        # Second request: same-looking host class resolves privately.
        with patch("socket.getaddrinfo", return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 0)),
        ]):
            asyncio.run(_feed(m, "Network.requestWillBeSent",
                              _rwbs({"url": "http://rebind.example/b"})))
        assert m.violation() is not None
        assert m.violation()["policy"] == "private"

    def test_request_log_bounded(self):
        m = _make_monitor()
        for i in range(5):
            asyncio.run(_feed(m, "Network.requestWillBeSent",
                              _rwbs({"requestId": f"r{i}", "url": "https://example.com/"})))
        log = m.request_log()
        assert len(log) == 5
        assert log[0]["request_id"] == "r0"


# ── Three-state semantics + armed path (fake CDP server) ───────────────────

class _FakeCdpServer:
    """Minimal CDP endpoint that arms page sessions and can push events."""

    def __init__(self, targets=("t1",), fail_attach=False):
        self.targets = list(targets)
        self.fail_attach = fail_attach
        self.ws_url = None
        self.connections = []
        self._thread = None
        self._server = None

    async def _handler(self, ws):
        self.connections.append(ws)
        async for raw in ws:
            msg = json.loads(raw)
            mid = msg["id"]
            method = msg["method"]
            if method == "Target.getTargets":
                infos = [{"type": "page", "targetId": t, "title": "", "url": "about:blank"}
                         for t in self.targets]
                await ws.send(json.dumps({"id": mid, "result": {"targetInfos": infos}}))
            elif method == "Target.attachToTarget":
                if self.fail_attach:
                    await ws.send(json.dumps({"id": mid, "error": {"code": -32000, "message": "boom"}}))
                else:
                    sid = "s-" + str(msg["params"]["targetId"])
                    await ws.send(json.dumps({"id": mid, "result": {"sessionId": sid}}))
            else:
                await ws.send(json.dumps({"id": mid, "result": {}}))

    def stop(self):
        pass


@pytest.fixture
def fake_cdp_server():
    """Start a real websockets CDP server on an ephemeral port; yield URL."""
    import socket as _socket

    sock = _socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = _FakeCdpServer()
    server.port = port

    async def _serve():
        async with websockets.serve(server._handler, "127.0.0.1", port, max_size=2**26):
            await asyncio.Future()

    server._thread = threading.Thread(target=lambda: asyncio.run(_serve()), daemon=True)
    server._thread.start()
    time.sleep(0.2)
    server.ws_url = f"ws://127.0.0.1:{port}/devtools/browser/x"
    yield server
    server.stop()


class TestThreeStateSemantics:
    def test_never_started_is_not_attach_failed(self):
        m = _make_monitor()
        assert m.attach_failed() is False
        assert m.armed() is False

    def test_attach_failure_on_dead_port(self):
        m = NetworkExecMonitor("ws://127.0.0.1:1/nonexistent", task_id="t1")
        m.start(timeout=1.0)
        assert m.attach_failed() is True
        assert m.armed() is False
        m.stop()

    def test_armed_then_activity(self, fake_cdp_server):
        m = NetworkExecMonitor(fake_cdp_server.ws_url, task_id="t1")
        m.start(timeout=5.0)
        assert m.attach_failed() is False
        assert m.armed() is True
        started = time.monotonic()
        assert m.saw_activity(started) is False  # connected+enabled, no events
        asyncio.run(_feed(m, "Network.requestWillBeSent",
                          _rwbs({"url": "https://example.com/"})))
        assert m.saw_activity(started) is True
        m.stop()

    def test_probe_success_arm(self, fake_cdp_server):
        m = NetworkExecMonitor(fake_cdp_server.ws_url, task_id="t1")
        m.start(timeout=5.0)
        started = time.monotonic()
        assert m.saw_activity(started) is False
        m.mark_probe_success()
        assert m.saw_activity(started) is True
        m.stop()


# ── browser_exec withhold tests (fake CLI + stubbed guard stack) ───────────

class _StubMonitor:
    def __init__(self, *, attach_failed=False, armed=True, saw_activity=True,
                 violation=None, last_known_url="", dropped=False):
        self._attach_failed = attach_failed
        self._armed = armed
        self._saw = saw_activity
        self._violation = violation
        self._last = last_known_url
        self._dropped = dropped
        self.stopped = False

    def attach_failed(self): return self._attach_failed
    def armed(self): return self._armed
    def saw_activity(self, exec_started): return self._saw
    def mark_probe_success(self): pass
    def violation(self): return dict(self._violation) if self._violation else None
    def reset(self): pass
    def last_known_url(self): return self._last
    def event_count(self): return 1 if self._saw else 0
    def dropped(self): return self._dropped
    def request_log(self, limit=200): return []
    def stop(self, timeout=5.0): self.stopped = True


class _FakeGuardProc(dict):
    """Dict-shaped fake of ``_spawn_ssrf_guard``'s return value."""

    def __init__(self):
        super().__init__(
            blocked=threading.Event(),
            arm_failed=threading.Event(),
            died=threading.Event(),
            markers=[],
            proc=None,
            report_sock=None,
        )


def _fake_cli(tmp_path, body):
    """Python-based fake CLI (executable via [sys.executable, path])."""
    script = tmp_path / "browser-use.py"
    script.write_text(body, encoding="utf-8")
    return str(script)


@pytest.fixture
def stub_browser_exec(monkeypatch):
    """Wire browser_exec's guard hooks to stub objects for one test."""
    import tools.browser_use_guard as bug

    state = {}

    def _install(monitor, *, landed=None, fake_guard=None):
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
            "ssrf_guard": fake_guard or _FakeGuardProc(),
            "error": None,
        }
        monkeypatch.setattr(
            bu_cli, "_ensure_exec_cdp_endpoint",
            lambda env, task_id, session: (
                env.setdefault("BU_CDP_WS", "ws://127.0.0.1:1/x"), None
            )[1],
        )
        monkeypatch.setattr(bug, "_prepare_guard", lambda *a, **k: ctx)
        monkeypatch.setattr(bug, "_guard_env", lambda env, ctx: env)
        monkeypatch.setattr(bug, "_guard_self_test", lambda ctx, env: None)
        monkeypatch.setattr(bu_cli, "_trusted_landed_url", lambda *a, **k: landed)
        return ctx

    state["install"] = _install
    return state


class TestBrowserExecWithhold:
    def test_attach_failure_fail_closed_withhold(self, tmp_path, monkeypatch, stub_browser_exec):
        cli = _fake_cli(tmp_path, "import sys\nsys.stdin.read()\nprint('SECRET')\n")
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [sys.executable, cli])
        monitor = _StubMonitor(attach_failed=True, armed=False, saw_activity=False)
        stub_browser_exec["install"](monitor, landed=None)
        result = json.loads(bu_cli.browser_exec("print('SECRET')"))
        assert "monitoring could not be verified" in result["error"]
        assert "SECRET" not in json.dumps(result)

    def test_violation_withholds_stdout_stderr_and_screenshot(self, tmp_path, monkeypatch, stub_browser_exec):
        cli = _fake_cli(
            tmp_path,
            "import sys\nsys.stdin.read()\nprint('SECRET_BODY')\n"
            "import pathlib\npathlib.Path(r'" + str(tmp_path).replace("\\", "\\\\") + "/nope.png').write_text('x')\n"
            "print('" + str(tmp_path).replace("\\", "\\\\") + "/nope.png')\n",
        )
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [sys.executable, cli])
        violation = {"url": "http://10.0.0.1/x", "policy": "private",
                     "event": "Network.requestWillBeSent", "request_id": "r1",
                     "ts": time.time(), "frame_id": "f1", "session_id": "s1"}
        monitor = _StubMonitor(violation=violation)
        stub_browser_exec["install"](monitor, landed=None)
        result = json.loads(bu_cli.browser_exec("print('x')"))
        assert "error" in result
        assert "10.0.0.1" in result["error"]
        assert "SECRET_BODY" not in json.dumps(result)
        assert "screenshot_path" not in result
        assert "stderr" not in result or "SECRET" not in result["stderr"]

    def test_session_path_endpoint_guaranteed(self, tmp_path, monkeypatch):
        """browser_exec(session=...) resolves an endpoint (C1 regression)."""
        calls = []

        def _record(env, task_id, session):
            calls.append((task_id, session))
            env["BU_CDP_WS"] = "ws://127.0.0.1:9222/x"
            return None

        monkeypatch.setattr(bu_cli, "_ensure_exec_cdp_endpoint", _record)
        # Guard disabled so no real CDP is needed.
        monkeypatch.setattr(bu_cli, "_read_browser_cfg", lambda: {"exec_network_monitor": "off"})
        cli = _fake_cli(tmp_path, "import sys\nsys.stdin.read()\nprint('bu:$BU_NAME')\n")
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [sys.executable, cli])
        monkeypatch.setattr(bu_cli, "_trusted_landed_url", lambda *a, **k: None)
        result = json.loads(bu_cli.browser_exec("print(1)", session="r7k2"))
        assert calls, "_ensure_exec_cdp_endpoint must be invoked for session= path"
        assert calls[0][1] == "r7k2"
        assert result["monitor"] == "disabled"

    def test_forged_full_marker_without_trusted_landing_withheld(self, tmp_path, monkeypatch, stub_browser_exec):
        """Defects 1+5: forged ARMED:full stdout cannot release output.

        The CLI prints a model-forged ``__HERMES_BROWSER_EXEC_ARMED__:full``
        marker and the browser is observed, but the trusted probe returns
        None (browser killed) — output must be withheld.
        """
        cli = _fake_cli(
            tmp_path,
            "import sys\nsys.stdin.read()\n"
            "print('__HERMES_BROWSER_EXEC_ARMED__:full', flush=True)\n"
            "print('SECRET_FORGED')\n",
        )
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [sys.executable, cli])
        monitor = _StubMonitor(saw_activity=True)  # browser observed
        stub_browser_exec["install"](monitor, landed=None)
        result = json.loads(bu_cli.browser_exec("print('x')"))
        assert "error" in result
        assert "trusted landing" in result["error"]
        assert "SECRET_FORGED" not in json.dumps(result)
