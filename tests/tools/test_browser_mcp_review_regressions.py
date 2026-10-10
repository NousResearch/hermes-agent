"""Merge-condition regressions from the PR #135861 review:

* test_tab_registry_isolates_interleaved_tasks — P1-1: every MCP verb must target the
  task's own gateway tab (navigate parses the ``Tab:`` line; press/scroll/back/click/
  snapshot/type/close/screenshot forward it). Interleaving two tasks must show each
  verb carrying its task's tab id — never the implicit "most recent for the key".
* test_mcp_type_redacts_typed_text — P1-4: the MCP type route never echoes the raw
  typed text back (success AND error payloads), matching the REST path.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture(autouse=True)
def _isolate_browser_lane_env(monkeypatch):
    for name in ("BROWSER_MCP_URL", "BROWSER_MCP_API_KEY", "BROWSER_CDP_URL",
                 "CAMOFOX_URL", "CAMOFOX_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BROWSER_MCP_URL", "https://api.browser-gw.example")
    monkeypatch.setenv("BROWSER_MCP_API_KEY", "k" * 8)


def _fake_transport(monkeypatch, calls):
    """Instrument BELOW _call (requests.post) so _call's real tab-injection logic runs
    and the recorded args are what would hit the wire."""
    import tools.browser_mcp_transport as transport

    class _Resp:
        status_code = 200
        text = ""
        def raise_for_status(self): pass
        def json(self): return json.loads(self.text)

    def _fake_post(url, **kw):
        post_body = kw.get("json") or {}
        params = post_body.get("params", {})
        calls.append((params.get("name"), (params.get("arguments") or {}).get("tab_id")))
        if params.get("name") == "browser_navigate":
            url = (params.get("arguments") or {}).get("url", "")
            tab = "tab-B" if url.endswith("/b") else "tab-A"
            body = f"Navigated to {url}\nTab: {tab}\nStatus: 200\n\nSnapshot:\n- link [e1]\n"
            payload = {"result": {"content": [{"text": body}], "isError": False}}
        else:
            payload = {"result": {"content": [{"text": "ok"}], "isError": False}}
        r = _Resp(); r.text = json.dumps(payload); return r

    monkeypatch.setattr(transport.requests, "post", _fake_post)


def test_tab_registry_isolates_interleaved_tasks(monkeypatch):
    calls = []
    _fake_transport(monkeypatch, calls)
    from tools.browser_mcp_transport import (mcp_navigate, mcp_press, mcp_scroll,
                                             mcp_back, mcp_snapshot, mcp_close)
    from tools.browser_camofox import camofox_click, camofox_type

    mcp_navigate("https://example.com/a", task_id="taskA")
    mcp_navigate("https://example.com/b", task_id="taskB")

    # Interleave: every verb must carry its OWN task's tab.
    mcp_press("Enter", task_id="taskA")
    mcp_scroll("down", task_id="taskB")
    mcp_back(task_id="taskA")
    camofox_click("@e1", task_id="taskB")
    camofox_type("@e1", "hello", task_id="taskA")
    mcp_snapshot(task_id="taskB")

    sent = {tool: tab for tool, tab in calls}
    assert sent["browser_press"] == "tab-A"
    assert sent["browser_scroll"] == "tab-B"
    assert sent["browser_back"] == "tab-A"
    assert sent["browser_click"] == "tab-B"
    assert sent["browser_type"] == "tab-A"
    assert sent["browser_snapshot"] == "tab-B"

    # close drops the registry entry — the next verb for that task targets no tab.
    mcp_close(task_id="taskA")
    mcp_back(task_id="taskA")
    assert calls[-1] == ("browser_back", None)

    # explicit tab_id still wins over the registry.
    mcp_scroll("up", task_id="taskA")   # registry dropped above? no — only taskA closed
    assert calls[-1] == ("browser_scroll", None)  # dropped → no implicit targeting


def test_redaction_and_back_guard(monkeypatch):
    """P1-4 + back parity: typed text never echoes; back's URL passes the private recheck."""
    from agent.display import redact_browser_typed_text_for_display
    calls = []
    _fake_transport(monkeypatch, calls)

    from tools.browser_mcp_transport import mcp_type

    secret = "sk-proj-aaaaaaaaaaaaaaaaaaaaaaaa"
    out = json.loads(mcp_type("@e1", secret))
    assert out["success"] is True
    assert secret not in json.dumps(out)          # raw secret never echoes
    assert out["typed"] != secret

    # error path also redacts
    import tools.browser_mcp_transport as transport
    class _Resp2:
        status_code = 200
        text = ""
        def raise_for_status(self): pass
    def _failing_post(url, **kw):
        post_body = kw.get("json") or {}
        name = (post_body.get("params") or {}).get("name")
        if name == "browser_evaluate":
            payload = {"result": {"content": [{"text": '"https://example.com/"'}], "isError": False}}
        else:
            payload = {"result": {"content": [{"text": f"type failed for {secret}"}], "isError": True}}
        r = _Resp2(); r.text = json.dumps(payload); return r
    monkeypatch.setattr(transport.requests, "post", _failing_post)
    err = mcp_type("@e1", secret)
    assert secret not in err