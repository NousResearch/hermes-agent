"""Unit tests for tools.browser_supervisor_frames frame bookkeeping.

``Page.frameAttached`` / ``Page.frameNavigated`` events carry the EMITTING
session id (the process that contains the frame), never the frame's own
dedicated session. Only OOPIF targets get a routable session, recorded by
``_on_target_attached``. Stamping the emitting session on a same-origin
iframe made ``browser_cdp(frame_id=...)`` take the OOPIF path and run the
method in the TOP page's context while reporting the iframe's frame_id.
"""

import threading
from dataclasses import dataclass

from tools.browser_supervisor_frames import FrameTrackingMixin


class _Stub(FrameTrackingMixin):
    def __init__(self):
        self._frames = {}
        self._state_lock = threading.Lock()

    def snapshot(self):
        with self._state_lock:
            tree = self._build_frame_tree_locked()

        @dataclass(frozen=True)
        class _Snap:
            frame_tree: dict

        return _Snap(tree)


def _navigated(frame_id, url, origin, parent_id=None, name=""):
    frame = {"id": frame_id, "url": url, "securityOrigin": origin, "name": name}
    if parent_id:
        frame["parentId"] = parent_id
    return {"frame": frame}


def test_frame_attached_same_origin_has_no_session_id():
    sup = _Stub()
    sup._on_frame_navigated(_navigated("top", "https://a.test/", "https://a.test"), "page-sid")
    sup._on_frame_attached({"frameId": "f1", "parentFrameId": "top"}, "page-sid")
    assert sup._frames["f1"].cdp_session_id is None
    assert sup._frames["f1"].is_oopif is False


def test_frame_navigated_new_same_origin_has_no_session_id():
    sup = _Stub()
    sup._on_frame_navigated(_navigated("f2", "https://a.test/inner", "https://a.test", "top"), "page-sid")
    assert sup._frames["f2"].cdp_session_id is None


def test_oopif_session_id_survives_later_navigation():
    """A real OOPIF record (set by _on_target_attached) keeps its session id."""
    sup = _Stub()
    from tools.browser_supervisor_frames import FrameInfo
    sup._frames["oof"] = FrameInfo("oof", "https://b.test/", "https://b.test", "top", True, "child-sid")
    sup._on_frame_navigated(_navigated("oof", "https://b.test/2", "https://b.test", "top"), "page-sid")
    assert sup._frames["oof"].cdp_session_id == "child-sid"
    assert sup._frames["oof"].is_oopif is True


def test_browser_cdp_same_origin_frame_fails_closed(monkeypatch):
    """End-to-end: browser_cdp(frame_id=...) must not route a same-origin
    iframe call to the emitting (top-page) session."""
    import tools.browser_cdp_tool as cdp_tool
    import tools.browser_supervisor as supervisor_mod

    sup = _Stub()
    sup._on_frame_navigated(_navigated("top", "https://a.test/", "https://a.test"), "page-sid")
    sup._on_frame_attached({"frameId": "f1", "parentFrameId": "top"}, "page-sid")
    sup._on_frame_navigated(_navigated("f1", "https://a.test/inner", "https://a.test", "top"), "page-sid")

    monkeypatch.setattr(supervisor_mod.SUPERVISOR_REGISTRY, "get", lambda task_id: sup)

    out = cdp_tool._browser_cdp_via_supervisor(
        "default", "f1", "Runtime.evaluate", {"expression": "location.href"}, 5.0)
    assert "not an out-of-process iframe" in out
    assert "page-sid" not in out
