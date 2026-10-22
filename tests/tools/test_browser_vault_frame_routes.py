"""Owned OOPIF selection and document lifecycle contracts for vault routes."""
import asyncio
import threading

import pytest


class TestVaultFrameRoutes:
    @pytest.mark.parametrize(("child_origin", "probe_failure"), [
        ("https://login.example", None), ("null", None), ("", None), ("", "world"), ("", "probe"),
    ])
    def test_supervisor_focuses_only_an_oopif_owned_by_the_selected_page(self, monkeypatch, child_origin, probe_failure):
        """The OOPIF probe must use the selected page's frame tree, not every
        attached iframe session in the browser."""
        from tools import browser_supervisor
        from tools.browser_supervisor_frames import FrameInfo

        supervisor = object.__new__(browser_supervisor.CDPSupervisor)
        supervisor._loop = type("Loop", (), {"is_running": lambda self: True})()
        supervisor._state_lock = threading.Lock()
        supervisor._page_session_id = None
        supervisor._frames = {
            "child-frame": FrameInfo("child-frame", "https://login.example/form",
                                      "https://login.example", "top-frame", True, "child-session"),
            # This session has a password too, but is not in the selected
            # page's frame tree and therefore must never be probed.
            "other-frame": FrameInfo("other-frame", "https://evil.example/form",
                                      "https://evil.example", "other-top", True, "other-session"),
            # Even a valid sibling in THIS page must not become fallback
            # authority after the first matching child reports an empty origin.
            "valid-sibling": FrameInfo("valid-sibling", "https://idp.test/form",
                                        "https://idp.test", "top-frame", True, "sibling-session"),
        }
        supervisor._frames["child-frame"].loader_id = "child-document"

        async def fake_cdp(method, params=None, *, session_id=None, timeout=10.0):
            if method == "Target.getTargets":
                return {"result": {"targetInfos": [{"type": "page", "targetId": "page", "url": "https://www.espn.com/login"}]}}
            if method == "Target.attachToTarget":
                return {"result": {"sessionId": "top-session"}}
            if method == "Page.getFrameTree":
                if session_id == "child-session":
                    return {"result": {"frameTree": {"frame": {
                        "id": "child-frame", "loaderId": "child-document", "securityOrigin": child_origin,
                    }}}}
                return {"result": {"frameTree": {"frame": {"id": "top-frame"}, "childFrames": [
                    {"frame": {"id": "child-frame"}},
                ]}}}
            if method == "Runtime.evaluate" and session_id == "top-session":
                return {"result": {"result": {"value": False}}}
            if method == "Runtime.evaluate" and session_id == "child-session":
                if params["expression"] == "self.origin":
                    assert params["contextId"] == 7
                    if probe_failure == "probe":
                        raise RuntimeError("origin context disappeared")
                    return {"result": {"result": {"value": child_origin}}}
                return {"result": {"result": {"value": True}}}
            if method == "Page.createIsolatedWorld" and session_id == "child-session":
                assert params["frameId"] == "child-frame"
                assert not params.get("grantUniveralAccess", False)
                if probe_failure == "world":
                    return {"error": {"message": "context unavailable"}}
                return {"result": {"executionContextId": 7}}
            raise AssertionError((method, params, session_id))

        supervisor._cdp = fake_cdp
        supervisor._enable_page_domains = lambda *_args, **_kwargs: asyncio.sleep(0)
        supervisor._install_dialog_bridge = lambda *_args, **_kwargs: asyncio.sleep(0)
        monkeypatch.setattr(browser_supervisor, "_schedule", lambda coro, _loop, timeout: asyncio.run(coro))

        result = supervisor.focus_page("https://www.espn.com", accept="hasPassword")
        assert result["ok"] is True
        assert result["frame_origin"] == child_origin
        assert result["route"] == {"page_session_id": "top-session", "frame_id": "child-frame",
                                   "frame_session_id": "child-session", "frame_loader_id": "child-document"}
        # The persistent page session remains the selected relying party.
        assert supervisor._page_session_id == "top-session"

    def test_supervisor_rejects_a_stale_cross_origin_frame_route(self):
        """A detached/replaced OOPIF cannot reuse a former consent route."""
        from tools import browser_supervisor
        from tools.browser_supervisor_frames import FrameInfo

        supervisor = object.__new__(browser_supervisor.CDPSupervisor)
        supervisor._loop = type("Loop", (), {"is_running": lambda self: True})()
        supervisor._state_lock = threading.Lock()
        supervisor._active = True
        supervisor._page_session_id = "top-session"
        supervisor._frames = {
            # Same frame id but a new child session means its old document is gone.
            "child-frame": FrameInfo("child-frame", "https://login.example/next",
                                      "https://login.example", "top-frame", True, "new-child-session"),
        }
        result = supervisor.evaluate_runtime(
            "window.location.origin",
            route={"page_session_id": "top-session", "frame_id": "child-frame",
                   "frame_session_id": "old-child-session"},
        )
        assert result["ok"] is False
        assert "no longer attached" in result["error"]

    def test_supervisor_rejects_same_session_after_child_document_replacement(self):
        """A reload can retain the OOPIF CDP session, but not its authority."""
        from tools import browser_supervisor
        from tools.browser_supervisor_frames import FrameInfo

        supervisor = object.__new__(browser_supervisor.CDPSupervisor)
        supervisor._loop = type("Loop", (), {"is_running": lambda self: True})()
        supervisor._state_lock = threading.Lock()
        supervisor._active = True
        supervisor._page_session_id = "top-session"
        child = FrameInfo("child-frame", "https://idp.test/login", "https://idp.test",
                          "top-frame", True, "child-session")
        child.loader_id = "new-document"
        supervisor._frames = {"child-frame": child}
        supervisor._cdp = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not evaluate"))
        result = supervisor.evaluate_runtime(
            "window.location.origin",
            route={"page_session_id": "top-session", "frame_id": "child-frame",
                   "frame_session_id": "child-session", "frame_loader_id": "old-document"},
        )
        assert result == {"ok": False, "error": "vault frame route is no longer attached"}
