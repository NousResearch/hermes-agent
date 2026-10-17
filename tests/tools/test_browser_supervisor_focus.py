"""Unit tests for ``CDPSupervisor.focus_page`` tab preference (mock CDP, no browser).

Covers the shared-CDP regression from #140196: ``focus_page`` must keep the page the
supervisor is already attached to when it is on the requested origin (and the accept
probe passes), instead of re-attaching to the FIRST same-origin target — which, with
several agents sharing one Chrome, can be another session's or a stale tab and would
receive the vault fill. Real-CDP coverage lives in ``tests/tools/test_browser_supervisor.py``.
"""
from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.browser_supervisor import CDPSupervisor

LOGIN_ORIGIN = "https://accounts.heb.com"
LOGIN_PROBE = "!!document.querySelector('input[type=password]')"


@pytest.fixture()
def running_loop():
    """A live asyncio loop on a daemon thread, so sync bridges via ``_schedule`` work."""
    loop = asyncio.new_event_loop()
    started = threading.Event()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        started.set()
        loop.run_forever()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    started.wait(timeout=5)
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


@pytest.fixture()
def supervisor(running_loop):
    sup = CDPSupervisor(task_id="test-focus", cdp_url="ws://127.0.0.1:0/devtools/browser/FAKE")
    sup._loop = running_loop
    sup._active = True
    # The walk path's attach extras: irrelevant to the preference under test.
    sup._enable_page_domains = AsyncMock()
    sup._install_dialog_bridge = AsyncMock()
    yield sup


def _eval_response(value):
    return {"result": {"result": {"type": "string", "value": value}}}


class TestFocusPagePrefersAttachedTab:
    def test_attached_same_origin_tab_wins_without_walk(self, supervisor, monkeypatch):
        """The already-attached page matches origin + probe: keep it, never walk targets."""
        get_targets = AsyncMock(return_value={"result": {"targetInfos": []}})
        evals = {
            "window.location.href": _eval_response(f"{LOGIN_ORIGIN}/login"),
            LOGIN_PROBE: _eval_response(True),
        }

        async def _cdp(method, params=None, *, session_id=None, timeout=10.0):
            if method == "Target.getTargets":
                return await get_targets(method, params)
            return evals[params["expression"]] if params["expression"] in evals else {
                "result": {"result": {"type": "undefined"}}
            }

        monkeypatch.setattr(supervisor, "_cdp", _cdp)
        supervisor._page_session_id = "sess-current"

        out = supervisor.focus_page(LOGIN_ORIGIN, accept=LOGIN_PROBE)

        assert out == {"ok": True, "url": f"{LOGIN_ORIGIN}/login"}
        # The whole point: no re-attach walk, the current session stays.
        get_targets.assert_not_called()
        assert supervisor._page_session_id == "sess-current"

    def test_attached_tab_failing_probe_falls_back_to_walk(self, supervisor, monkeypatch):
        """Attached page has no matching form (probe falsy): resolve via the target walk."""
        target_infos = [
            {"targetId": "target-stale", "type": "page", "url": f"{LOGIN_ORIGIN}/stale-login"},
            {"targetId": "target-agent", "type": "page", "url": f"{LOGIN_ORIGIN}/login"},
        ]

        async def _cdp(method, params=None, *, session_id=None, timeout=10.0):
            if method == "Target.getTargets":
                return {"result": {"targetInfos": target_infos}}
            if method == "Target.attachToTarget":
                return {"result": {"sessionId": f"attach-{params['targetId']}"}}
            if method == "Target.detachFromTarget":
                return {"result": {}}
            expr = params["expression"]
            if session_id == "sess-current":
                # Own tab already submitted its form: no password input left.
                return _eval_response(f"{LOGIN_ORIGIN}/done" if expr == "window.location.href" else False)
            return _eval_response(True)  # walk candidates all pass the probe

        monkeypatch.setattr(supervisor, "_cdp", _cdp)
        supervisor._page_session_id = "sess-current"

        out = supervisor.focus_page(LOGIN_ORIGIN, accept=LOGIN_PROBE)

        assert out == {"ok": True, "url": f"{LOGIN_ORIGIN}/stale-login"}
        assert supervisor._page_session_id == "attach-target-stale"

    def test_attached_tab_on_other_origin_falls_back_to_walk(self, supervisor, monkeypatch):
        """Own tab is elsewhere: the walk resolves the origin's tab as before."""

        async def _cdp(method, params=None, *, session_id=None, timeout=10.0):
            if method == "Target.getTargets":
                return {"result": {"targetInfos": [
                    {"targetId": "target-login", "type": "page", "url": f"{LOGIN_ORIGIN}/login"},
                ]}}
            if method == "Target.attachToTarget":
                return {"result": {"sessionId": "attach-target-login"}}
            expr = params["expression"]
            return _eval_response("https://www.example.com/other" if expr == "window.location.href" else True)

        monkeypatch.setattr(supervisor, "_cdp", _cdp)
        supervisor._page_session_id = "sess-current"

        out = supervisor.focus_page(LOGIN_ORIGIN, accept=LOGIN_PROBE)

        assert out == {"ok": True, "url": f"{LOGIN_ORIGIN}/login"}
        assert supervisor._page_session_id == "attach-target-login"

    def test_stale_attached_session_falls_back_to_walk(self, supervisor, monkeypatch):
        """Attached session is dead (evaluate raises): the walk still resolves a tab."""

        async def _cdp(method, params=None, *, session_id=None, timeout=10.0):
            if session_id == "sess-current":
                raise RuntimeError("Session with given id not found")
            if method == "Target.getTargets":
                return {"result": {"targetInfos": [
                    {"targetId": "target-login", "type": "page", "url": f"{LOGIN_ORIGIN}/login"},
                ]}}
            if method == "Target.attachToTarget":
                return {"result": {"sessionId": "attach-target-login"}}
            return _eval_response(True)

        monkeypatch.setattr(supervisor, "_cdp", _cdp)
        supervisor._page_session_id = "sess-current"

        out = supervisor.focus_page(LOGIN_ORIGIN, accept=LOGIN_PROBE)

        assert out == {"ok": True, "url": f"{LOGIN_ORIGIN}/login"}
        assert supervisor._page_session_id == "attach-target-login"

    def test_no_attached_session_walks_as_before(self, supervisor, monkeypatch):
        """Fresh supervisor (no page session yet): unchanged first-match behavior."""

        async def _cdp(method, params=None, *, session_id=None, timeout=10.0):
            assert session_id is None or session_id.startswith("attach-")
            if method == "Target.getTargets":
                return {"result": {"targetInfos": [
                    {"targetId": "target-login", "type": "page", "url": f"{LOGIN_ORIGIN}/login"},
                ]}}
            if method == "Target.attachToTarget":
                return {"result": {"sessionId": "attach-target-login"}}
            return _eval_response(True)

        monkeypatch.setattr(supervisor, "_cdp", _cdp)
        supervisor._page_session_id = None

        out = supervisor.focus_page(LOGIN_ORIGIN, accept=LOGIN_PROBE)

        assert out == {"ok": True, "url": f"{LOGIN_ORIGIN}/login"}
