"""Unit tests: CDP supervisor attaches a dedicated page per task_id (#69727)."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import pytest


def _make_supervisor() -> Any:
    from tools.browser_supervisor import CDPSupervisor

    sup = object.__new__(CDPSupervisor)
    sup.task_id = "session-A"
    sup.cdp_url = "ws://example.test/cdp"
    sup._page_session_id = None
    sup._page_target_id = None
    sup._owns_page_target = False
    sup._child_sessions = {}
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
