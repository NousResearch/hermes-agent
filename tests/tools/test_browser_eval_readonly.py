"""Behavioral guard regressions: real V8 plus actual production dispatch seams."""
import json
import shutil
import subprocess
from unittest.mock import MagicMock

import pytest

from tools import browser_cdp_tool as cdp
from tools import browser_tool as bt
from tools import browser_tool_eval_policy as policy
from tools import browser_tool_session as session
from tools import browser_supervisor as bs


@pytest.fixture(autouse=True)
def guarded(monkeypatch):
    monkeypatch.setattr(policy, "_eval_ssrf_guard_active", lambda task: True)
    monkeypatch.setattr(policy, "_allow_unsafe_browser_evaluate", lambda: False)
    monkeypatch.setattr(policy, "_current_page_private_url", lambda task: None)
    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(bt, "_last_session_key", lambda task: task)
    monkeypatch.setattr(bs.SUPERVISOR_REGISTRY, "get", lambda task: None)


@pytest.mark.parametrize("frame", [None, "oopif"])
@pytest.mark.parametrize("method,key,text", [
    ("Runtime.evaluate", "expression", "if(true){3}"),
    ("Runtime.callFunctionOn", "functionDeclaration", "function(){return this.title}"),
    ("Debugger.evaluateOnCallFrame", "expression", "document.title"),
])
def test_both_dispatch_paths_enforce_engine_guard(monkeypatch, frame, method, key, text):
    seen = []
    async def call(endpoint, m, params, target, timeout):
        seen.append((m, params))
        return {}
    def supervisor(**kw):
        seen.append((kw["method"], kw["params"]))
        return json.dumps({"success": True})
    monkeypatch.setattr(cdp, "_resolve_cdp_endpoint", lambda: "ws://localhost:1234")
    monkeypatch.setattr(cdp, "_cdp_call", call)
    monkeypatch.setattr(cdp, "_browser_cdp_via_supervisor", supervisor)
    params = {key: text, "throwOnSideEffect": False, "returnByValue": True}
    out = json.loads(cdp.browser_cdp(method, params, frame_id=frame, task_id="test"))
    assert out["success"]
    expected = {**params, "throwOnSideEffect": True}
    if method != "Debugger.evaluateOnCallFrame":
        expected["awaitPromise"] = False
    assert seen == [(method, expected)]
    assert params["throwOnSideEffect"] is False


@pytest.mark.parametrize("frame", [None, "oopif"])
@pytest.mark.parametrize("method,params", [
    ("Page.addScriptToEvaluateOnNewDocument", {"source": "fetch('assembled')"}),
    ("Runtime.runScript", {"scriptId": "cached"}),
    ("Debugger.setScriptSource", {"scriptId": "existing", "scriptSource": "fetch('x')"}),
    ("Runtime.evaluate", {"expression": "await 3", "replMode": True}),
    ("Runtime.callFunctionOn", {"functionDeclaration": "async function(){}", "awaitPromise": True}),
])
def test_unsupported_entry_points_rejected_before_transport(monkeypatch, frame, method, params):
    monkeypatch.setattr(cdp, "_resolve_cdp_endpoint", lambda: pytest.fail("must not reach transport"))
    monkeypatch.setattr(cdp, "_browser_cdp_via_supervisor", lambda **kw: pytest.fail("must not dispatch"))
    out = json.loads(cdp.browser_cdp(method, params, frame_id=frame))
    assert "Blocked:" in out["error"]
    assert "unsupported" in out["error"]


@pytest.mark.parametrize("camofox", [False, True])
def test_console_without_capability_fails_closed(monkeypatch, camofox):
    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: camofox)
    monkeypatch.setattr(session, "_run_browser_command", lambda *a, **kw: pytest.fail("unsafe fallback"))
    monkeypatch.setattr(bt, "_camofox_eval", lambda *a, **kw: pytest.fail("unsafe REST fallback"))
    out = json.loads(bt.browser_console(expression="document.title", task_id="test"))
    assert not out["success"]
    assert "read-only" in out["error"]


def test_console_supervisor_receives_readonly_and_no_retry(monkeypatch):
    sup = MagicMock()
    sup.evaluate_runtime.return_value = {"ok": False, "error": "EvalError: Possible side-effect in debug-evaluate"}
    monkeypatch.setattr(bs.SUPERVISOR_REGISTRY, "get", lambda task: sup)
    monkeypatch.setattr(session, "_run_browser_command", lambda *a, **kw: pytest.fail("unsafe fallback"))
    out = json.loads(bt.browser_console(expression="setTimeout(()=>fetch('assembled'),0)", task_id="test"))
    assert not out["success"]
    sup.evaluate_runtime.assert_called_once_with("setTimeout(()=>fetch('assembled'),0)",
                                               throw_on_side_effect=True, await_promise=False)


def test_supervisor_wire_params_and_retry_keep_readonly():
    from tests.tools.test_browser_eval_supervisor_path import _make_supervisor_with_cdp_fn, _stop_supervisor
    calls = []
    async def transport(method, params=None, **kw):
        calls.append(params)
        if params["returnByValue"]:
            raise RuntimeError("Object reference chain is too long")
        return {"result": {"result": {"type": "object", "description": "body"}}}
    sup = _make_supervisor_with_cdp_fn(transport)
    try:
        assert sup.evaluate_runtime("document.body", throw_on_side_effect=True, await_promise=False)["ok"]
        assert len(calls) == 2
        assert all(p["throwOnSideEffect"] is True and p["awaitPromise"] is False for p in calls)
    finally:
        _stop_supervisor(sup)


@pytest.mark.parametrize("unsafe", [True, False])
def test_opt_out_and_local_params_are_unchanged(monkeypatch, unsafe):
    monkeypatch.setattr(policy, "_eval_ssrf_guard_active", lambda task: unsafe)
    monkeypatch.setattr(policy, "_allow_unsafe_browser_evaluate", lambda: unsafe)
    params = {"source": "fetch('x')"}
    assert cdp._guard_runtime_evaluate("test", "Page.addScriptToEvaluateOnNewDocument", params) is params


@pytest.mark.skipif(not shutil.which("node"), reason="Node required for real V8 regression")
def test_node_real_v8_rejects_deferred_work_and_preserves_globals():
    params = cdp._guard_runtime_evaluate("test", "Runtime.evaluate", {"returnByValue": True})
    script = r'''
const assert = require('node:assert/strict');
const inspector = require('node:inspector');
const s = new inspector.Session(); s.connect();
const post = (m,p) => new Promise((resolve,reject)=>s.post(m,p,(e,r)=>e?reject(e):resolve(r)));
(async()=>{
 const originalFetch = globalThis.fetch;
 const params = JSON.parse(process.argv[1]);
 const good = ['1+2', 'if(true){3}', '(()=>{var n=0;for(var i=0;i<3;i++)n+=i;return n})()',
               '(function(){function answer(){return 3}return answer()})()'];
 for(const expression of good){const r=await post('Runtime.evaluate',{...params,expression});
   assert.equal(r.exceptionDetails,undefined); assert.equal(r.result.value,3);}
 const bad = ["fetch(String.fromCharCode(104,116,116,112,58,47,47)+'169.254.169.254')",
  "setTimeout(()=>fetch('http://127.0.0.1'),0)",
  "queueMicrotask(()=>fetch('http://127.0.0.1'))",
  "Promise.resolve().then(()=>fetch('http://127.0.0.1'))",
  "(async()=>{await Promise.resolve();fetch('http://127.0.0.1')})()",
  "globalThis.fetch=()=>3", "var total=0;total", "function answer(){return 3};answer()"];
 for(const expression of bad){const r=await post('Runtime.evaluate',{...params,expression});
  assert.match(r.exceptionDetails.exception.description,/Possible side-effect/);}
 assert.equal(globalThis.fetch,originalFetch);
 const pageFetch=await fetch('data:text/plain,page-works'); assert.equal(await pageFetch.text(),'page-works');
 s.disconnect(); console.log(JSON.stringify({good:good.length,blocked:bad.length,pageFetch:'page-works'}));
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    result = subprocess.run([shutil.which("node"), "-e", script, json.dumps(params)],
                            capture_output=True, text=True, timeout=20, check=True)
    assert json.loads(result.stdout) == {"good": 4, "blocked": 8, "pageFetch": "page-works"}
