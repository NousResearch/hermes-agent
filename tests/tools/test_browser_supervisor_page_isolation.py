"""Unit tests: CDP supervisor attaches a dedicated page per task_id (#69727)."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


def _make_supervisor() -> Any:
    import threading
    from tools.browser_supervisor import CDPSupervisor

    sup = object.__new__(CDPSupervisor)
    sup.task_id = "session-A"
    sup.cdp_url = "ws://example.test/cdp"
    sup._state_lock = threading.Lock()
    sup._active = False
    sup._page_session_id = None
    sup._page_target_id = None
    sup._owns_page_target = False
    sup._child_sessions = {}
    sup._loop = None
    return sup


@pytest.mark.asyncio
async def test_attach_creates_dedicated_page_even_when_pages_exist():
    """Existing page targets must not be adopted by a second Hermes session."""
    sup = _make_supervisor()
    calls: List[Dict[str, Any]] = []

    async def fake_cdp(
        method: str,
        params: Optional[dict] = None,
        session_id: Optional[str] = None,
        timeout: float = 10.0,
    ) -> dict:
        calls.append({"method": method, "params": params, "session_id": session_id})
        if method == "Target.createTarget":
            return {"result": {"targetId": "NEW-TAB-A"}}
        if method == "Target.attachToTarget":
            return {"result": {"sessionId": "SID-A"}}
        if method in {
            "Page.enable",
            "Runtime.enable",
            "Target.setAutoAttach",
            "Page.addScriptToEvaluateOnNewDocument",
            "Fetch.enable",
        }:
            return {"result": {}}
        if method == "Target.getTargets":
            # Would previously steal EXISTING-SHARED — must not happen.
            return {
                "result": {
                    "targetInfos": [
                        {"targetId": "EXISTING-SHARED", "type": "page", "url": "https://baidu.com"},
                        {"targetId": "OTHER", "type": "page", "url": "https://sina.com"},
                    ]
                }
            }
        return {"result": {}}

    # Bypass dialog bridge install (Fetch/addScript) complexity for this unit.
    async def _noop_bridge(_session_id: str) -> None:
        return None

    sup._cdp = fake_cdp  # type: ignore[method-assign]
    sup._install_dialog_bridge = _noop_bridge  # type: ignore[method-assign]

    await sup._attach_initial_page()

    create_calls = [c for c in calls if c["method"] == "Target.createTarget"]
    assert len(create_calls) == 1
    assert create_calls[0]["params"] == {"url": "about:blank"}

    attach_calls = [c for c in calls if c["method"] == "Target.attachToTarget"]
    assert len(attach_calls) == 1
    assert attach_calls[0]["params"] == {"targetId": "NEW-TAB-A", "flatten": True}

    # Must not have preferred the existing shared page.
    assert all(
        c["params"].get("targetId") != "EXISTING-SHARED"
        for c in attach_calls
        if c.get("params")
    )
    assert sup._page_target_id == "NEW-TAB-A"
    assert sup._owns_page_target is True
    assert sup._page_session_id == "SID-A"


@pytest.mark.asyncio
async def test_two_supervisors_get_distinct_page_targets():
    """Simulates two Hermes sessions against one shared CDP browser."""
    created_ids = ["TAB-1", "TAB-2"]
    create_index = {"i": 0}

    async def make_attach(sup_name: str):
        sup = _make_supervisor()
        sup.task_id = sup_name
        attached: List[str] = []

        async def fake_cdp(
            method: str,
            params: Optional[dict] = None,
            session_id: Optional[str] = None,
            timeout: float = 10.0,
        ) -> dict:
            if method == "Target.createTarget":
                tid = created_ids[create_index["i"]]
                create_index["i"] += 1
                return {"result": {"targetId": tid}}
            if method == "Target.attachToTarget":
                attached.append(params["targetId"])
                return {"result": {"sessionId": f"SID-{params['targetId']}"}}
            if method == "Target.getTargets":
                return {
                    "result": {
                        "targetInfos": [
                            {"targetId": "SHARED", "type": "page", "url": "about:blank"},
                        ]
                    }
                }
            return {"result": {}}

        async def _noop_bridge(_session_id: str) -> None:
            return None

        sup._cdp = fake_cdp  # type: ignore[method-assign]
        sup._install_dialog_bridge = _noop_bridge  # type: ignore[method-assign]
        await sup._attach_initial_page()
        return sup, attached

    a, attached_a = await make_attach("session-A")
    b, attached_b = await make_attach("session-B")

    assert a._page_target_id == "TAB-1"
    assert b._page_target_id == "TAB-2"
    assert a._page_target_id != b._page_target_id
    assert attached_a == ["TAB-1"]
    assert attached_b == ["TAB-2"]
    assert "SHARED" not in attached_a + attached_b


@pytest.mark.asyncio
async def test_reconnect_reuses_owned_page_target():
    """After reconnect, re-attach the same dedicated tab when it still exists."""
    sup = _make_supervisor()
    sup._page_target_id = "OWNED-TAB"
    sup._owns_page_target = True
    methods: List[str] = []

    async def fake_cdp(
        method: str,
        params: Optional[dict] = None,
        session_id: Optional[str] = None,
        timeout: float = 10.0,
    ) -> dict:
        methods.append(method)
        if method == "Target.getTargets":
            return {
                "result": {
                    "targetInfos": [
                        {"targetId": "OWNED-TAB", "type": "page", "url": "about:blank"},
                        {"targetId": "OTHER", "type": "page", "url": "https://example.com"},
                    ]
                }
            }
        if method == "Target.createTarget":
            raise AssertionError("must not create a new tab when owned page still exists")
        if method == "Target.attachToTarget":
            assert params["targetId"] == "OWNED-TAB"
            return {"result": {"sessionId": "SID-REUSE"}}
        return {"result": {}}

    async def _noop_bridge(_session_id: str) -> None:
        return None

    sup._cdp = fake_cdp  # type: ignore[method-assign]
    sup._install_dialog_bridge = _noop_bridge  # type: ignore[method-assign]

    await sup._attach_initial_page()

    assert "Target.createTarget" not in methods
    assert methods.count("Target.getTargets") == 1
    assert sup._page_target_id == "OWNED-TAB"
    assert sup._page_session_id == "SID-REUSE"


def test_navigate_page_uses_owned_session_and_activates_target(monkeypatch):
    """Public navigation must hit the dedicated page session (#69727 review)."""
    sup = _make_supervisor()
    sup._active = True
    sup._page_session_id = "SID-NAV"
    sup._page_target_id = "TAB-NAV"
    sup._owns_page_target = True
    methods: List[tuple] = []

    async def fake_cdp(
        method: str,
        params: Optional[dict] = None,
        session_id: Optional[str] = None,
        timeout: float = 10.0,
    ) -> dict:
        methods.append((method, params, session_id))
        if method == "Page.navigate":
            return {"result": {"frameId": "frame-1", "loaderId": "load-1"}}
        return {"result": {}}

    class _Loop:
        def is_running(self) -> bool:
            return True

    class _Fut:
        def __init__(self, value):
            self._value = value

        def result(self, timeout=None):
            return self._value

    def schedule(coro, loop):
        # Run the coroutine on a private loop to completion.
        loop_local = asyncio.new_event_loop()
        try:
            return _Fut(loop_local.run_until_complete(coro))
        finally:
            loop_local.close()

    monkeypatch.setattr(
        "agent.async_utils.safe_schedule_threadsafe", schedule
    )
    sup._loop = _Loop()  # type: ignore[assignment]
    sup._cdp = fake_cdp  # type: ignore[method-assign]

    result = sup.navigate_page("https://example.com/search")

    assert result["ok"] is True
    assert result["target_id"] == "TAB-NAV"
    assert ("Target.activateTarget", {"targetId": "TAB-NAV"}, None) in methods
    assert any(
        m[0] == "Page.navigate"
        and m[1] == {"url": "https://example.com/search"}
        and m[2] == "SID-NAV"
        for m in methods
    )


def test_page_target_tab_ref_matches_agent_browser_target_order(monkeypatch):
    """The tab ref must identify the owned target, not merely activate it."""
    sup = _make_supervisor()
    sup._active = True
    sup._page_target_id = "TAB-OWNED"
    methods: List[str] = []

    async def fake_cdp(
        method: str,
        params: Optional[dict] = None,
        session_id: Optional[str] = None,
        timeout: float = 10.0,
    ) -> dict:
        methods.append(method)
        if method == "Target.getTargets":
            return {
                "result": {
                    "targetInfos": [
                        {"targetId": "DEVTOOLS", "type": "page", "url": "devtools://devtools"},
                        {"targetId": "TAB-OTHER", "type": "page", "url": "https://other.example"},
                        {"targetId": "TAB-OWNED", "type": "page", "url": "about:blank"},
                    ]
                }
            }
        return {"result": {}}

    class _Loop:
        def is_running(self) -> bool:
            return True

    class _Fut:
        def __init__(self, value):
            self._value = value

        def result(self, timeout=None):
            return self._value

    def schedule(coro, loop):
        loop_local = asyncio.new_event_loop()
        try:
            return _Fut(loop_local.run_until_complete(coro))
        finally:
            loop_local.close()

    monkeypatch.setattr("agent.async_utils.safe_schedule_threadsafe", schedule)
    sup._loop = _Loop()  # type: ignore[assignment]
    sup._cdp = fake_cdp  # type: ignore[method-assign]

    result = sup.page_target_tab_ref()

    assert result == {"ok": True, "target_id": "TAB-OWNED", "tab_ref": "t2"}
    assert methods == ["Target.getTargets"]


def test_cdp_follow_up_command_binds_target_inside_agent_browser_batch(monkeypatch, tmp_path):
    """Click/type-style operations must select the owned tab in the same daemon call."""
    import json
    from unittest.mock import MagicMock, mock_open

    import tools.browser_tool as browser_tool

    session = {
        "session_name": "cdp-task",
        "cdp_url": "wss://browser.example/cdp",
    }
    captured: List[List[str]] = []
    process = MagicMock()
    process.wait.return_value = 0
    process.returncode = 0
    stdout = json.dumps([
        {"command": ["tab", "t2"], "success": True, "result": []},
        {"command": ["click", "@e1"], "success": True, "result": {"clicked": "@e1"}},
    ])

    def capture_popen(command, **kwargs):
        captured.append(command)
        return process

    monkeypatch.setattr(browser_tool, "_get_session_info", lambda _task_id: session)
    monkeypatch.setattr(browser_tool, "_find_agent_browser", lambda: "/usr/bin/agent-browser")
    monkeypatch.setattr(browser_tool, "_ensure_cdp_supervisor", lambda _task_id: None)
    monkeypatch.setattr(browser_tool, "_bind_session_page_target", lambda _task_id, _session: None)
    monkeypatch.setattr(browser_tool, "_session_page_tab_ref", lambda _task_id, _session: "t2")
    monkeypatch.setattr(browser_tool, "_get_browser_engine", lambda: "auto")
    monkeypatch.setattr(browser_tool, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr(browser_tool, "_build_browser_env", lambda: {})
    monkeypatch.setattr(browser_tool, "_write_owner_pid", lambda *_args: None)
    monkeypatch.setattr(browser_tool, "_needs_chromium_sandbox_bypass", lambda: False)
    monkeypatch.setattr(browser_tool, "_safe_command_timeout", lambda: 10)
    monkeypatch.setattr(browser_tool.subprocess, "Popen", capture_popen)
    monkeypatch.setattr(browser_tool.os, "open", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(browser_tool.os, "close", lambda *_args: None)
    monkeypatch.setattr(browser_tool.os, "unlink", lambda *_args: None)
    monkeypatch.setattr(browser_tool.os, "makedirs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(browser_tool, "_read_command_output_files", lambda *_args: (stdout, ""))
    monkeypatch.setattr(browser_tool, "_unlink_command_output_files", lambda *_args: None)
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
    monkeypatch.setattr("builtins.open", mock_open(read_data=stdout))

    result = browser_tool._run_browser_command("task", "click", ["@e1"])

    assert result == {"success": True, "data": {"clicked": "@e1"}}
    assert captured == [[
        "/usr/bin/agent-browser",
        "--cdp",
        "wss://browser.example/cdp",
        "--json",
        "batch",
        "tab t2",
        "click @e1",
    ]]


def test_browser_navigate_cdp_uses_supervisor_page(monkeypatch):
    """browser_navigate on a CDP session must not fall through to unbound CLI."""
    import json
    import tools.browser_tool as bt
    import tools.browser_supervisor as bsup

    session = {
        "session_name": "cdp_test",
        "cdp_url": "ws://127.0.0.1:9222/devtools/browser/x",
        "page_target_id": "TASK-TAB",
        "_first_nav": True,
    }

    class _Sup:
        def page_target_id(self):
            return "TASK-TAB"

        def navigate_page(self, url, timeout=30.0):
            return {
                "ok": True,
                "target_id": "TASK-TAB",
                "frame_id": "f1",
                "loader_id": "l1",
            }

    class _Reg:
        def get(self, task_id):
            return _Sup()

    monkeypatch.setattr(bt, "_get_session_info", lambda key: session)
    monkeypatch.setattr(bt, "_ensure_cdp_supervisor", lambda task_id: None)
    monkeypatch.setattr(bt, "_bind_session_page_target", lambda tid, info: None)
    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(bt, "_is_local_backend", lambda: False)
    monkeypatch.setattr(bt, "_allow_private_urls", lambda: True)
    monkeypatch.setattr(bt, "_is_always_blocked_url", lambda url: False)
    monkeypatch.setattr(bt, "check_website_access", lambda url: None)
    monkeypatch.setattr(bt, "_get_cloud_provider", lambda: None)
    monkeypatch.setattr(bt, "_maybe_start_recording", lambda key: None)
    monkeypatch.setattr(bt, "_sensitive_query_param_name", lambda url: None)
    monkeypatch.setattr(bt, "_normalize_url_for_request", lambda url: url)
    monkeypatch.setattr(bsup, "SUPERVISOR_REGISTRY", _Reg())

    def _fail_cli(*a, **k):
        raise AssertionError("must not call agent-browser CLI for CDP navigate")

    monkeypatch.setattr(bt, "_run_browser_command", _fail_cli)

    out = json.loads(bt.browser_navigate("https://www.baidu.com", task_id="sess-A"))
    assert out["success"] is True
    assert out["page_target_id"] == "TASK-TAB"
    assert out["via"] == "cdp_supervisor"


def test_two_task_navigate_paths_keep_distinct_targets(monkeypatch):
    """Two task_ids must route navigate to different page targets."""
    import json
    import tools.browser_tool as bt
    import tools.browser_supervisor as bsup

    sessions = {
        "sess-A": {
            "session_name": "cdp_a",
            "cdp_url": "ws://127.0.0.1:9222/devtools/browser/x",
            "_first_nav": True,
        },
        "sess-B": {
            "session_name": "cdp_b",
            "cdp_url": "ws://127.0.0.1:9222/devtools/browser/x",
            "_first_nav": True,
        },
    }
    seen: Dict[str, str] = {}

    class _Sup:
        def __init__(self, tid: str, tab: str):
            self.tid = tid
            self.tab = tab

        def page_target_id(self):
            return self.tab

        def navigate_page(self, url, timeout=30.0):
            seen[self.tid] = self.tab
            return {"ok": True, "target_id": self.tab, "frame_id": "f"}

    class _Reg:
        def get(self, task_id):
            tab = "TAB-A" if task_id == "sess-A" else "TAB-B"
            return _Sup(task_id, tab)

    monkeypatch.setattr(bt, "_get_session_info", lambda key: sessions[key])
    monkeypatch.setattr(bt, "_ensure_cdp_supervisor", lambda task_id: None)
    monkeypatch.setattr(bt, "_bind_session_page_target", lambda tid, info: None)
    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(bt, "_is_local_backend", lambda: False)
    monkeypatch.setattr(bt, "_allow_private_urls", lambda: True)
    monkeypatch.setattr(bt, "_is_always_blocked_url", lambda url: False)
    monkeypatch.setattr(bt, "check_website_access", lambda url: None)
    monkeypatch.setattr(bt, "_get_cloud_provider", lambda: None)
    monkeypatch.setattr(bt, "_maybe_start_recording", lambda key: None)
    monkeypatch.setattr(bt, "_sensitive_query_param_name", lambda url: None)
    monkeypatch.setattr(bt, "_normalize_url_for_request", lambda url: url)
    monkeypatch.setattr(bsup, "SUPERVISOR_REGISTRY", _Reg())
    monkeypatch.setattr(
        bt,
        "_run_browser_command",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no CLI")),
    )

    a = json.loads(bt.browser_navigate("https://www.baidu.com", task_id="sess-A"))
    b = json.loads(bt.browser_navigate("https://www.sina.com.cn", task_id="sess-B"))
    assert a["page_target_id"] == "TAB-A"
    assert b["page_target_id"] == "TAB-B"
    assert seen == {"sess-A": "TAB-A", "sess-B": "TAB-B"}
