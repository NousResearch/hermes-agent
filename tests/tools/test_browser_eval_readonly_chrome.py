"""Opt-in real Chrome + supervisor regression (no fabricated CDP responses)."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tools import browser_cdp_tool as cdp
from tools import browser_supervisor as bs
from tools import browser_tool as bt
from tools import browser_tool_eval_policy as policy

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    os.environ.get("HERMES_E2E_BROWSER") != "1", reason="set HERMES_E2E_BROWSER=1")]


@pytest.fixture
def live_browser(tmp_path, monkeypatch):
    chrome = (shutil.which("google-chrome") or shutil.which("chromium") or
              "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if not Path(chrome).is_file():
        pytest.skip("Chrome/Chromium required")
    hits = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            iframe = (f'<iframe src="http://localhost:{self.server.server_port}/frame"></iframe>'
                      if self.path == "/" else "")
            self.wfile.write(("<title>inspection works</title>page works" + iframe).encode())
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    profile = tmp_path / "chrome-profile"
    proc = subprocess.Popen([chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
        "--disable-background-networking", "--site-per-process", "--no-first-run", "--remote-debugging-port=0",
        f"--user-data-dir={profile}", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sup = None
    try:
        deadline = time.monotonic() + 15
        while not (profile / "DevToolsActivePort").exists():
            assert proc.poll() is None, "Chrome exited"
            assert time.monotonic() < deadline, "Chrome did not become ready"
            time.sleep(.05)
        port, path = (profile / "DevToolsActivePort").read_text().splitlines()[:2]
        endpoint = f"ws://127.0.0.1:{port}{path}"
        def wire(method, params, target=None):
            return cdp._run_async(cdp._cdp_call(endpoint, method, params, target, 10))
        targets = wire("Target.getTargets", {})["targetInfos"]
        target = next(t["targetId"] for t in targets if t["type"] == "page")
        url = f"http://127.0.0.1:{server.server_port}/"
        wire("Page.navigate", {"url": url}, target)
        deadline = time.monotonic() + 10
        while wire("Runtime.evaluate", {"expression": "document.title", "returnByValue": True}, target)["result"].get("value") != "inspection works":
            assert time.monotonic() < deadline
            time.sleep(.05)
        wire("Runtime.evaluate", {"expression": "globalThis.savedApis=[fetch,XMLHttpRequest,WebSocket,EventSource,navigator.sendBeacon]"}, target)
        sup = bs.CDPSupervisor("readonly-test", endpoint)
        sup.start()
        # Create the OOPIF while auto-attach is live, not before the supervisor.
        wire("Page.reload", {}, target)
        deadline = time.monotonic() + 10
        while wire("Runtime.evaluate", {"expression": "document.title", "returnByValue": True}, target)["result"].get("value") != "inspection works":
            assert time.monotonic() < deadline
            time.sleep(.05)
        wire("Runtime.evaluate", {"expression": "globalThis.savedApis=[fetch,XMLHttpRequest,WebSocket,EventSource,navigator.sendBeacon]"}, target)
        monkeypatch.setattr(policy, "_eval_ssrf_guard_active", lambda task: True)
        monkeypatch.setattr(policy, "_allow_unsafe_browser_evaluate", lambda: False)
        monkeypatch.setattr(policy, "_current_page_private_url", lambda task: None)
        monkeypatch.setattr(policy, "_expression_targets_private_url", lambda expression: None)
        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(bt, "_last_session_key", lambda task: task)
        monkeypatch.setattr(bs.SUPERVISOR_REGISTRY, "get", lambda task: sup)
        monkeypatch.setattr(cdp, "_resolve_cdp_endpoint", lambda: endpoint)
        yield target, sup, hits, lambda m,p: wire(m,p,target)
    finally:
        if sup:
            sup.stop()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def test_real_console_and_cdp_network_and_page_recovery(live_browser):
    target, sup, hits, wire = live_browser
    console = json.loads(bt.browser_console(expression="document.title", task_id="readonly-test"))
    assert console["success"] and console["result"] == "inspection works"
    before = [h for h in hits if h not in {"/", "/frame", "/favicon.ico"}]
    attacks = [
        "fetch(location.origin+'/blocked')",
        "savedApis[0](location.origin+'/saved-fetch')",
        "(()=>{var x=new XMLHttpRequest();x.open('GET',location.origin+'/xhr');x.send()})()",
        "navigator.sendBeacon(location.origin+'/beacon','x')",
        "new WebSocket('ws://'+location.host+'/socket')",
        "new EventSource(location.origin+'/events')",
        "setTimeout(()=>fetch(location.origin+'/timer'),0)",
        "Promise.resolve().then(()=>fetch(location.origin+'/promise'))",
        "(async()=>{await Promise.resolve();fetch(location.origin+'/async')})()",
    ]
    for expression in attacks:
        response = json.loads(cdp.browser_cdp("Runtime.evaluate", {"expression": expression,
            "returnByValue": True}, target_id=target, task_id="readonly-test"))
        assert "Possible side-effect" in response["result"]["exceptionDetails"]["exception"]["description"]
    from tools.browser_supervisor import _schedule
    obj = _schedule(sup._cdp("Runtime.evaluate", {"expression": "globalThis"},
                            session_id=sup._page_session_id), sup._loop, timeout=10)
    params = cdp._guard_runtime_evaluate("readonly-test", "Runtime.callFunctionOn", {
        "functionDeclaration": "function(){return fetch(location.origin+'/function')}",
        "objectId": obj["result"]["result"]["objectId"], "returnByValue": True})
    call = _schedule(sup._cdp("Runtime.callFunctionOn", params,
                             session_id=sup._page_session_id), sup._loop, timeout=10)
    assert "Possible side-effect" in call["result"]["exceptionDetails"]["exception"]["description"]
    time.sleep(.1)  # allow any erroneously admitted detached work to reach the server
    assert [h for h in hits if h not in {"/", "/frame", "/favicon.ico"}] == before
    untouched = wire("Runtime.evaluate", {"expression": "savedApis.every((a,i)=>a===[fetch,XMLHttpRequest,WebSocket,EventSource,navigator.sendBeacon][i])", "returnByValue": True})
    assert untouched["result"]["value"] is True
    # Normal page-owned requests still work after success AND denied evaluations.
    response = wire("Runtime.evaluate", {"expression": "fetch('/after').then(r=>r.text())", "awaitPromise": True, "returnByValue": True})
    assert "page works" in response["result"]["value"]
    assert "/after" in hits
    xhr = wire("Runtime.evaluate", {"expression": "new Promise(resolve=>{var x=new XMLHttpRequest();x.open('GET','/after-xhr');x.onload=()=>resolve(x.responseText);x.send()})", "awaitPromise": True, "returnByValue": True})
    assert "page works" in xhr["result"]["value"] and "/after-xhr" in hits
    polling = wire("Runtime.evaluate", {"expression": "new Promise(resolve=>{var n=0;var id=setInterval(()=>{fetch('/poll').then(()=>{if(++n===2){clearInterval(id);resolve(n)}})},20)})", "awaitPromise": True, "returnByValue": True})
    assert polling["result"]["value"] == 2 and hits.count("/poll") == 2


def test_real_javascript_program_semantics_and_explicit_async_rejection(live_browser):
    target, sup, hits, wire = live_browser
    for expression, expected in [
        ("if(true){3}", 3),
        ("(()=>{var total=0;for(var i=0;i<3;i++)total+=i;return total})()", 3),
        ("(function(){function answer(){return 3}return answer()})()", 3),
        ("1; 2; 3", 3),
    ]:
        out = json.loads(bt.browser_console(expression=expression, task_id="readonly-test"))
        assert out["success"] and out["result"] == expected
    out = json.loads(cdp.browser_cdp("Runtime.evaluate", {"expression": "await 3", "replMode": True}, target_id=target))
    assert "asynchronous execution are unsupported" in out["error"]


def test_real_oopif_supervisor_dispatch_is_guarded(live_browser):
    target, sup, hits, wire = live_browser
    deadline = time.monotonic() + 10
    frame = None
    while frame is None:
        frame = next((f for f in sup.snapshot().frame_tree.get("children", [])
                      if f.get("session_id")), None)
        assert time.monotonic() < deadline, "cross-origin iframe did not attach"
        if frame is None:
            time.sleep(.05)
    frame_id = frame["frame_id"]
    good = json.loads(cdp.browser_cdp("Runtime.evaluate", {"expression": "document.title", "returnByValue": True},
                                    frame_id=frame_id, task_id="readonly-test"))
    assert good["success"] and good["result"]["result"]["value"] == "inspection works"
    before = [h for h in hits if h not in {"/", "/frame", "/favicon.ico"}]
    bad = json.loads(cdp.browser_cdp("Runtime.evaluate", {
        "expression": "Promise.resolve().then(()=>fetch(location.origin+'/oopif'))", "returnByValue": True},
        frame_id=frame_id, task_id="readonly-test"))
    assert "Possible side-effect" in bad["result"]["exceptionDetails"]["exception"]["description"]
    console = json.loads(bt.browser_console(expression="setTimeout(()=>fetch(location.origin+'/console'),0)",
                                           task_id="readonly-test"))
    assert not console["success"] and "Possible side-effect" in console["error"]
    time.sleep(.1)
    assert [h for h in hits if h not in {"/", "/frame", "/favicon.ico"}] == before
