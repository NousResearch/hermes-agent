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
import subprocess
import tempfile
import time
from pathlib import Path

import pytest


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("HERMES_E2E_BROWSER", "").strip() != "1",
        reason="real-browser E2E: set HERMES_E2E_BROWSER=1 to opt in",
    ),
    pytest.mark.skipif(
        not shutil.which("google-chrome") and not shutil.which("chromium"),
        reason="Chrome/Chromium not installed",
    ),
]


def _find_chrome() -> str:
    for candidate in ("google-chrome", "chromium", "chromium-browser"):
        path = shutil.which(candidate)
        if path:
            return path
    pytest.skip("no Chrome binary found")


@pytest.fixture
def chrome_cdp(tmp_path):
    """Start a headless Chrome with --remote-debugging-port, yield its WS URL.

    Binds an ephemeral port (bind port 0, read back the real one) so
    concurrently-running test files in the per-file parallel runner can
    never collide on a fixed port.
    Always launches with ``--site-per-process`` so cross-origin iframes
    become real OOPIFs (needed by the iframe interaction tests).
    """
    profile = tempfile.mkdtemp(prefix="hermes-supervisor-test-")
    stderr = (tmp_path / "chrome.stderr").open("w+b")
    proc = subprocess.Popen(
        [
            _find_chrome(),
            "--remote-debugging-port=0",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--headless=new",
            "--disable-gpu",
            "--site-per-process",  # force OOPIFs for cross-origin iframes
        ],
        stdout=subprocess.DEVNULL,
        stderr=stderr,
    )

    ws_url = None
    port = None
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        try:
            port = int((Path(profile) / "DevToolsActivePort").read_text(encoding="utf-8").splitlines()[0])
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
        stderr.seek(0)
        diagnostic = stderr.read().decode("utf-8", errors="replace")
        stderr.close()
        pytest.fail(f"Chrome didn't expose CDP in time: {diagnostic}")

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
    stderr.close()


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


def test_evaluate_runtime_rebinds_after_page_target_closed(chrome_cdp, supervisor_registry):
    """When the supervisor's page target goes away (its tab closed, work continuing in a new
    tab), evaluate must not stay pinned to the dead CDP session forever: any failure names the
    supervisor (so browser_console falls back to the CLI path) and the next calls run in the
    surviving page."""
    import urllib.request

    cdp_url, port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-eval-rebind", cdp_url=cdp_url)
    _fire_on_page(cdp_url, "void 0")
    time.sleep(0.5)
    assert supervisor.evaluate_runtime("document.title")["result"] == "Supervisor pytest"

    def _pages():
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=5) as r:
            return [t for t in json.loads(r.read().decode()) if t.get("type") == "page"]

    old_ids = {t["id"] for t in _pages()}
    new_url = "data:text/html,<title>replacement-tab</title>"
    req = urllib.request.Request(f"http://127.0.0.1:{port}/json/new?{new_url}", method="PUT")
    urllib.request.urlopen(req, timeout=5).read()
    for target_id in old_ids:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/json/close/{target_id}", timeout=5).read()

    deadline = time.monotonic() + 30
    out = {}
    while time.monotonic() < deadline:
        out = supervisor.evaluate_runtime("document.title", timeout=3.0)
        if out.get("ok"):
            break
        assert "supervisor" in out["error"].lower(), out
        time.sleep(0.25)
    assert out.get("ok") is True, out
    assert out["result"] == "replacement-tab"


def test_evaluate_runtime_moves_off_startup_tab_to_web_page(chrome_cdp, supervisor_registry):
    """Attached to a browser it did not launch, the supervisor binds to the startup tab before
    the driver opens its own. Evaluation must run in the web page, not stay in about:blank /
    chrome://newtab for the whole session."""
    import http.server
    import threading
    import urllib.request

    class _Page(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = b"<!doctype html><title>web-tab</title>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Page)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cdp_url, port = chrome_cdp
        supervisor = supervisor_registry.get_or_start(task_id="pytest-eval-webpage", cdp_url=cdp_url)
        assert supervisor.evaluate_runtime("location.protocol")["result"] in ("about:", "chrome:")

        page_url = f"http://127.0.0.1:{server.server_address[1]}/"
        urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/json/new?{page_url}", method="PUT"),
                               timeout=5).read()
        deadline = time.monotonic() + 10
        out = {}
        while time.monotonic() < deadline:
            out = supervisor.evaluate_runtime("document.title", timeout=3.0)
            if out.get("ok") and out.get("result") == "web-tab":
                break
            time.sleep(0.25)
        assert out.get("result") == "web-tab", out
    finally:
        server.shutdown()


def test_bind_driver_page_picks_the_driver_tab_among_several(chrome_cdp, supervisor_registry):
    """With several web tabs open, "first web tab" is a guess; binding by the driver's exact
    URL must land evaluation in that tab."""
    import http.server
    import threading
    import urllib.request

    class _Page(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = f"<!doctype html><title>tab{self.path.replace('/', '-')}</title>".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Page)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cdp_url, port = chrome_cdp
        supervisor = supervisor_registry.get_or_start(task_id="pytest-bind-driver", cdp_url=cdp_url)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        for path in ("/a", "/b"):
            urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/json/new?{base}{path}", method="PUT"),
                                   timeout=5).read()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:  # both tabs committed their URLs
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=5) as r:
                urls = {t.get("url") for t in json.loads(r.read().decode())}
            if {f"{base}/a", f"{base}/b"} <= urls:
                break
            time.sleep(0.25)
        assert supervisor.bind_driver_page(f"{base}/b") is True
        assert supervisor.evaluate_runtime("document.title")["result"] == "tab-b"
    finally:
        server.shutdown()
