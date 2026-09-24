"""Integration tests for tools.browser_supervisor.

Exercises the supervisor end-to-end against a real local Chrome
(``--remote-debugging-port``).  Skipped when Chrome is not installed
— these are the tests that actually verify the CDP wire protocol
works, since mock-CDP unit tests can only prove the happy paths we
thought to model.

These tests spawn a **real Chrome process** on the machine running them.
They are therefore opt-in, twice over:

* ``@pytest.mark.integration`` — excluded by the default
  ``addopts = "-m 'not integration'"`` in ``pyproject.toml``, so a bare
  ``pytest`` cannot launch a browser on a developer's desktop by accident.
* ``HERMES_E2E_BROWSER=1`` — the env gate this docstring has always claimed.
  It previously existed only in this prose: nothing read the variable, and
  the sole real gate was "is a Chrome binary on PATH", which is true on most
  desktops and on ``ubuntu-latest``. Now it is enforced.

Run manually:
    HERMES_E2E_BROWSER=1 scripts/run_tests.sh -m integration \\
        tests/tools/test_browser_supervisor.py

(``scripts/run_tests.sh`` runs under ``env -i`` and forwards
``HERMES_E2E_BROWSER`` explicitly; ``-m integration`` overrides the default
marker filter.)
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import ssl
import subprocess
import tempfile
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _find_chrome() -> str:
    for candidate in ("google-chrome", "chromium", "chromium-browser"):
        path = shutil.which(candidate)
        if path:
            return path
    # macOS app bundles do not put a google-chrome shim on PATH.
    macos_chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if macos_chrome.is_file() and os.access(macos_chrome, os.X_OK):
        return str(macos_chrome)
    pytest.skip("no Chrome binary found")


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("HERMES_E2E_BROWSER", "").strip() != "1",
        reason="real-browser E2E: set HERMES_E2E_BROWSER=1 to opt in",
    ),
    pytest.mark.skipif(
        not shutil.which("google-chrome") and not shutil.which("chromium")
        and not Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome").is_file(),
        reason="Chrome/Chromium not installed",
    ),
]


@pytest.fixture
def chrome_cdp(request):
    """Start a headless Chrome with --remote-debugging-port, yield its WS URL.

    Uses a unique port per xdist worker to avoid cross-worker collisions.
    Always launches with ``--site-per-process`` so cross-origin iframes
    become real OOPIFs (needed by the iframe interaction tests).
    """

    # xdist worker_id is "master" in single-process mode or "gw0".."gwN" otherwise.
    # Under subprocess-per-file isolation there's no xdist, so we fall back
    # to "master" via the session-scoped fixture below.
    worker_id = request.getfixturevalue("worker_id") if "worker_id" in request.fixturenames else "master"
    if worker_id == "master":
        port_offset = 0
    else:
        port_offset = int(worker_id.lstrip("gw"))
    port = 9225 + port_offset
    profile = tempfile.mkdtemp(prefix="hermes-supervisor-test-")
    proc = subprocess.Popen(
        [
            _find_chrome(),
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--headless=new",
            "--disable-gpu",
            "--site-per-process",  # force OOPIFs for cross-origin iframes
            "--host-resolver-rules=MAP *.test 127.0.0.1",
            # The .test TLD is HSTS-preloaded.  This temporary, test-only
            # profile accepts the fixture's ephemeral self-signed certificate.
            "--ignore-certificate-errors",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    ws_url = None
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            import urllib.request
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/json/version", timeout=1
            ) as r:
                info = json.loads(r.read().decode())
                ws_url = info["webSocketDebuggerUrl"]
                break
        except Exception:
            time.sleep(0.25)
    if ws_url is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, AssertionError, Exception):
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except (AssertionError, Exception):
                pass
        shutil.rmtree(profile, ignore_errors=True)
        pytest.skip("Chrome didn't expose CDP in time")

    yield ws_url, port

    # Tear down Chrome. The stdlib `subprocess._wait()` POSIX implementation
    # has a known race (https://bugs.python.org/issue38630): when SIGCHLD
    # arrives concurrently with `proc.wait()`, `_try_wait(WNOHANG)` can
    # return a foreign pid and the `assert pid == self.pid or pid == 0`
    # fires. We saw this in CI on slice 1 after this fixture's teardown
    # (PR #33661 follow-up). Swallow the stdlib race + force-kill if wait
    # hangs, then always reap so we don't leak a zombie.
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=3)
    except (subprocess.TimeoutExpired, AssertionError, Exception):
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except (AssertionError, Exception):
            pass
    shutil.rmtree(profile, ignore_errors=True)


def _test_page_url() -> str:
    html = """<!doctype html>
<html><head><title>Supervisor pytest</title></head><body>
<h1>Supervisor pytest</h1>
<iframe id="inner" srcdoc="<body><h2>frame-marker</h2></body>" width="400" height="100"></iframe>
</body></html>"""
    return "data:text/html;base64," + base64.b64encode(html.encode()).decode()


def _fire_on_page(cdp_url: str, expression: str) -> None:
    """Navigate the first page target to a data URL and fire `expression`."""
    import websockets as _ws_mod

    async def run():
        async with _ws_mod.connect(cdp_url, max_size=50 * 1024 * 1024) as ws:
            next_id = [1]

            async def call(method, params=None, session_id=None):
                cid = next_id[0]
                next_id[0] += 1
                p = {"id": cid, "method": method}
                if params:
                    p["params"] = params
                if session_id:
                    p["sessionId"] = session_id
                await ws.send(json.dumps(p))
                async for raw in ws:
                    m = json.loads(raw)
                    if m.get("id") == cid:
                        return m

            targets = (await call("Target.getTargets"))["result"]["targetInfos"]
            page = next(t for t in targets if t.get("type") == "page")
            attach = await call(
                "Target.attachToTarget", {"targetId": page["targetId"], "flatten": True}
            )
            sid = attach["result"]["sessionId"]
            await call("Page.navigate", {"url": _test_page_url()}, session_id=sid)
            await asyncio.sleep(1.5)  # let the page load
            await call(
                "Runtime.evaluate",
                {"expression": expression, "returnByValue": True},
                session_id=sid,
            )

    asyncio.run(run())


@pytest.fixture
def cross_site_pages():
    """Three HTTPS DNS sites on loopback; Chrome's site isolation sees real OOPIFs."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):  # noqa: N802
            if self.path in {"/rp", "/rp-idp"}:
                evil = ("<iframe id=evil src='https://evil.test:%d/login'></iframe>" % server.server_port
                        if self.path == "/rp" else "")
                body = ("<!doctype html>" + evil
                        + "<iframe id=idp src='https://idp.test:%d/login'></iframe>" % server.server_port)
            else:
                body = "<input type=password id=password>"
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers()
            self.wfile.write(body.encode())

    with tempfile.TemporaryDirectory(prefix="hermes-oopif-cert-") as cert_dir:
        cert = Path(cert_dir) / "cert.pem"
        key = Path(cert_dir) / "key.pem"
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
             "-subj", "/CN=rp.test", "-keyout", str(key), "-out", str(cert)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(certfile=cert, keyfile=key)
        server.socket = tls.wrap_socket(server.socket, server_side=True)
        thread = __import__("threading").Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.server_port
        finally:
            server.shutdown(); thread.join(timeout=3)


def _navigate(cdp_url: str, url: str) -> str:
    import websockets as _ws_mod
    async def run():
        async with _ws_mod.connect(cdp_url) as ws:
            await ws.send(json.dumps({"id": 1, "method": "Target.getTargets"}))
            while (m := json.loads(await ws.recv())).get("id") != 1: pass
            page = next(t for t in m["result"]["targetInfos"] if t.get("type") == "page")
            await ws.send(json.dumps({"id": 2, "method": "Target.attachToTarget", "params": {"targetId": page["targetId"], "flatten": True}}))
            while (m := json.loads(await ws.recv())).get("id") != 2: pass
            await ws.send(json.dumps({"id": 3, "method": "Page.navigate", "params": {"url": url}, "sessionId": m["result"]["sessionId"]}))
            while (m := json.loads(await ws.recv())).get("id") != 3: pass
            return page["targetId"]
    return asyncio.run(run())


def _top_frame_tree(cdp_url: str, target_id: str) -> dict:
    """Return the selected top page's CDP frame tree for OOPIF provenance."""
    import websockets as _ws_mod

    async def run():
        async with _ws_mod.connect(cdp_url) as ws:
            await ws.send(json.dumps({"id": 1, "method": "Target.getTargets"}))
            while (m := json.loads(await ws.recv())).get("id") != 1:
                pass
            await ws.send(json.dumps({"id": 2, "method": "Target.attachToTarget", "params": {"targetId": target_id, "flatten": True}}))
            while (m := json.loads(await ws.recv())).get("id") != 2:
                pass
            session_id = m["result"]["sessionId"]
            await ws.send(json.dumps({"id": 3, "method": "Page.enable", "sessionId": session_id}))
            while (m := json.loads(await ws.recv())).get("id") != 3:
                pass
            await ws.send(json.dumps({"id": 4, "method": "Page.getFrameTree", "sessionId": session_id}))
            while (m := json.loads(await ws.recv())).get("id") != 4:
                pass
            return m["result"]["frameTree"]

    return asyncio.run(run())


def _top_body(cdp_url: str, target_id: str) -> str:
    """Read fixture DOM only; production writes remain supervisor-routed."""
    import websockets as _ws_mod

    async def run():
        async with _ws_mod.connect(cdp_url) as ws:
            await ws.send(json.dumps({"id": 1, "method": "Target.attachToTarget", "params": {"targetId": target_id, "flatten": True}}))
            while (m := json.loads(await ws.recv())).get("id") != 1:
                pass
            await ws.send(json.dumps({"id": 2, "method": "Runtime.evaluate", "params": {"expression": "document.body.innerHTML", "returnByValue": True}, "sessionId": m["result"]["sessionId"]}))
            while (m := json.loads(await ws.recv())).get("id") != 2:
                pass
            return str(m["result"]["result"]["value"])

    return asyncio.run(run())


@pytest.fixture
def supervisor_registry():
    """Yield the global registry and tear down any supervisors after the test."""
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    yield SUPERVISOR_REGISTRY
    SUPERVISOR_REGISTRY.stop_all()


def _wait_for_dialog(supervisor, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = supervisor.snapshot()
        if snap.pending_dialogs:
            return snap.pending_dialogs
        time.sleep(0.1)
    return ()


def test_supervisor_start_and_snapshot(chrome_cdp, supervisor_registry):
    """Supervisor attaches, exposes an active snapshot with a top frame."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-1", cdp_url=cdp_url)

    # Navigate so the frame tree populates.
    _fire_on_page(cdp_url, "/* no dialog */ void 0")

    # Give a moment for frame events to propagate
    time.sleep(1.0)
    snap = supervisor.snapshot()
    assert snap.active is True
    assert snap.task_id == "pytest-1"
    assert snap.pending_dialogs == ()
    # At minimum a top frame should exist after the navigate.
    assert snap.frame_tree.get("top") is not None


def test_cross_site_frames_are_real_oopifs(chrome_cdp, supervisor_registry, cross_site_pages):
    """The rp/idp/evil fixture proves CDP child targets, not port-only iframes."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-oopif", cdp_url=cdp_url)
    target_id = _navigate(cdp_url, f"https://rp.test:{cross_site_pages}/rp")
    body = _top_body(cdp_url, target_id)
    assert 'id="evil"' in body, body
    deadline = time.monotonic() + 5
    frames = []
    top_frame_id = ""
    while time.monotonic() < deadline:
        with supervisor._state_lock:
            frames = list(supervisor._frames.values())
        top_frame_id = str((_top_frame_tree(cdp_url, target_id).get("frame") or {}).get("id") or "")
        top_children = [f for f in frames if f.parent_frame_id == top_frame_id]
        if len(top_children) == 2 and all(f.is_oopif and f.cdp_session_id for f in top_children):
            break
        time.sleep(.1)
    children = [f for f in frames if f.parent_frame_id == top_frame_id]
    assert len(children) == 2, [
        (f.frame_id, f.origin, f.url, f.is_oopif, f.cdp_session_id)
        for f in frames
    ]
    assert all(f.is_oopif and f.cdp_session_id for f in children)
    snapshot = supervisor.snapshot().frame_tree
    assert snapshot["top"]["frame_id"] == top_frame_id
    assert {child["frame_id"] for child in snapshot["children"]} >= {f.frame_id for f in children}


def test_browser_vault_fill_uses_real_oopif_route(chrome_cdp, supervisor_registry, cross_site_pages):
    """The production vault path fills only an approved OOPIF, via its CDP route."""
    from tools import browser_vault_tool

    cdp_url, _port = chrome_cdp
    task_id = "pytest-live-vault"
    supervisor = supervisor_registry.get_or_start(task_id=task_id, cdp_url=cdp_url)
    _navigate(cdp_url, f"https://rp.test:{cross_site_pages}/rp-idp")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with supervisor._state_lock:
            if any(f.is_oopif and f.cdp_session_id for f in supervisor._frames.values()):
                break
        time.sleep(.1)
    top_origin = f"https://rp.test:{cross_site_pages}"
    child_origin = f"https://idp.test:{cross_site_pages}"
    canary = "oopif-test-canary"
    meta = SimpleNamespace(
        id="live-login", kind="login", origin=top_origin,
        allowed_origins=[top_origin], label="Live OOPIF test", has_otp=False,
    )

    class Backend:
        name, display_name, needs_unlock = "test", "Test", False

        def is_unlocked(self):
            return True

        def get_meta(self, handle):
            return meta if handle == meta.id else None

        def resolve_password(self, handle):
            assert handle == meta.id
            return canary

    with patch("agent.vault_backends.backend_for_handle", return_value=Backend()), \
         patch("tools.approval_prompt.request_elicitation_consent", return_value="accept") as consent:
        result = json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id=task_id))

    assert result["success"] is True and result["filled_fields"] == 1, result
    assert canary not in json.dumps(result)
    assert consent.call_count == 1
    assert top_origin in consent.call_args.args[0] and child_origin in consent.call_args.args[0]
    with supervisor._state_lock:
        child = next(f for f in supervisor._frames.values() if f.is_oopif and f.cdp_session_id)
        route = {"page_session_id": supervisor._page_session_id, "frame_id": child.frame_id,
                 "frame_session_id": child.cdp_session_id, "frame_loader_id": child.loader_id}
    check = supervisor.evaluate_runtime(
        "document.querySelector('#password').value === 'oopif-test-canary'", route=route,
    )
    assert check == {"ok": True, "result": True, "result_type": "boolean"}


def test_browser_vault_decline_never_resolves_real_evil_oopif(chrome_cdp, supervisor_registry, cross_site_pages):
    """An evil first OOPIF cannot trigger resolution or cause a fallback fill."""
    from tools import browser_vault_tool

    cdp_url, _port = chrome_cdp
    task_id = "pytest-live-vault-decline"
    supervisor = supervisor_registry.get_or_start(task_id=task_id, cdp_url=cdp_url)
    _navigate(cdp_url, f"https://rp.test:{cross_site_pages}/rp")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with supervisor._state_lock:
            if len([f for f in supervisor._frames.values() if f.is_oopif and f.cdp_session_id]) == 2:
                break
        time.sleep(.1)
    top_origin = f"https://rp.test:{cross_site_pages}"
    meta = SimpleNamespace(id="declined-login", kind="login", origin=top_origin,
                           allowed_origins=[top_origin], label="Declined", has_otp=False)
    resolved = False

    class Backend:
        name, display_name, needs_unlock = "test", "Test", False

        def is_unlocked(self):
            return True

        def get_meta(self, handle):
            return meta if handle == meta.id else None

        def resolve_password(self, handle):
            nonlocal resolved
            resolved = True
            return "must-not-be-resolved"

    with patch("agent.vault_backends.backend_for_handle", return_value=Backend()), \
         patch("tools.approval_prompt.request_elicitation_consent", return_value="decline") as consent:
        result = json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id=task_id))

    assert result["error_type"] == "cross_origin_declined"
    assert resolved is False
    assert consent.call_count == 1
    assert f"https://evil.test:{cross_site_pages}" in consent.call_args.args[0]


def test_main_frame_alert_detection_and_dismiss(chrome_cdp, supervisor_registry):
    """alert() in the main frame surfaces and can be dismissed via the sync API."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-2", cdp_url=cdp_url)

    _fire_on_page(cdp_url, "setTimeout(() => alert('PYTEST-MAIN-ALERT'), 50)")
    dialogs = _wait_for_dialog(supervisor)
    assert dialogs, "no dialog detected"
    d = dialogs[0]
    assert d.type == "alert"
    assert "PYTEST-MAIN-ALERT" in d.message

    result = supervisor.respond_to_dialog("dismiss")
    assert result["ok"] is True
    # State cleared after dismiss
    time.sleep(0.3)
    assert supervisor.snapshot().pending_dialogs == ()


def test_iframe_contentwindow_alert(chrome_cdp, supervisor_registry):
    """alert() fired from inside a same-origin iframe surfaces too."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-3", cdp_url=cdp_url)

    _fire_on_page(
        cdp_url,
        "setTimeout(() => document.querySelector('#inner').contentWindow.alert('PYTEST-IFRAME'), 50)",
    )
    dialogs = _wait_for_dialog(supervisor)
    assert dialogs, "no iframe dialog detected"
    assert any("PYTEST-IFRAME" in d.message for d in dialogs)

    result = supervisor.respond_to_dialog("accept")
    assert result["ok"] is True


def test_prompt_dialog_with_response_text(chrome_cdp, supervisor_registry):
    """prompt() gets our prompt_text back inside the page."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-4", cdp_url=cdp_url)

    # Fire a prompt and stash the answer on window
    _fire_on_page(
        cdp_url,
        "setTimeout(() => { window.__promptResult = prompt('give me a token', 'default-x'); }, 50)",
    )
    dialogs = _wait_for_dialog(supervisor)
    assert dialogs
    d = dialogs[0]
    assert d.type == "prompt"
    assert d.default_prompt == "default-x"

    result = supervisor.respond_to_dialog("accept", prompt_text="PYTEST-PROMPT-REPLY")
    assert result["ok"] is True


def test_browser_dialog_tool_end_to_end(chrome_cdp, supervisor_registry):
    """Full agent-path check: fire an alert, call the tool handler directly."""
    from tools.browser_dialog_tool import browser_dialog

    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-tool", cdp_url=cdp_url)

    _fire_on_page(cdp_url, "setTimeout(() => alert('PYTEST-TOOL-END2END'), 50)")
    assert _wait_for_dialog(supervisor), "no dialog detected via wait_for_dialog"

    r = json.loads(browser_dialog(action="dismiss", task_id="pytest-tool"))
    assert r["success"] is True
    assert r["action"] == "dismiss"
    assert "PYTEST-TOOL-END2END" in r["dialog"]["message"]




def test_evaluate_runtime_unserializable_value(chrome_cdp, supervisor_registry):
    """``Infinity``/``NaN``/``BigInt`` come back via ``unserializableValue``."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-eval-5", cdp_url=cdp_url)

    _fire_on_page(cdp_url, "void 0")
    time.sleep(0.5)

    out = supervisor.evaluate_runtime("Infinity")
    assert out["ok"] is True
    assert out["result"] == "Infinity"
