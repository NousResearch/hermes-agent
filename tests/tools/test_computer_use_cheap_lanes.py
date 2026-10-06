"""Capability-gated ``get_window_state`` delivery lanes (``_gws_args(mode)``).

Every capture used to fetch the FULL result — AX tree + screenshot — then throw half of it away:
vision mode discards the tree (the expensive part: Finder's AX walk measured 6.9 s, a busy Chrome
window's ~230 ms) and ax mode discards the screenshot (``has_image`` is false for ax in tool.py, so
the ScreenCaptureKit grab and its image tokens bought nothing). The driver exposes
``include_accessibility_tree:false`` / ``include_screenshot:false`` for exactly these lanes. Both
flags are sent ONLY when the live driver's tools/list schema advertises the property
(``supports_input_property`` fails closed before discovery and on old builds), so an unfitted driver
sees the pre-existing payload byte-for-byte. Sending BOTH false is a driver error, so the lanes are
mutually exclusive by mode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from tools.computer_use.cua_backend_capture import _CaptureMixin, _gws_is_empty


class _StubSession:
    """Session seam: capability map only — no transport, no driver."""

    def __init__(self, *, discovered: bool = True, props: tuple = ()) -> None:
        self._discovered = discovered
        self._props = set(props)

    @property
    def capabilities_discovered(self) -> bool:
        return self._discovered

    def supports_input_property(self, tool: str, property_name: str) -> bool:
        return tool == "get_window_state" and property_name in self._props

    def _has_tool(self, name: str) -> bool:
        return False  # modern drivers folded PNG capture into get_window_state


BOTH = ("include_accessibility_tree", "include_screenshot")


class _StubCapture(_CaptureMixin):
    """Capture-lane shell: enough state for ``_gws_args``/``capture``, no driver."""

    def __init__(self, session: Optional[_StubSession] = None) -> None:
        self._active_pid: Optional[int] = 607
        self._active_window_id: Optional[int] = 382
        self._session_id: Optional[str] = None
        self._last_app = ""
        self._session = session

    def _resolve_capture_windows(self, mode: str, app: Optional[str], pid: Optional[int],
                                 window_id: Optional[int]) -> List[Dict[str, Any]]:
        return [{"app_name": "Finder", "pid": 607, "window_id": 382, "title": "", "z_index": 1,
                 "off_screen": False}]

    def _set_active_target(self, target: Dict[str, Any]) -> None:
        self._active_pid, self._active_window_id = target["pid"], target["window_id"]


class TestGwsArgsLanes:

    def test_vision_lane_skips_the_ax_walk_when_advertised(self):
        stub = _StubCapture(_StubSession(props=BOTH))
        args = stub._gws_args("vision")
        assert args["include_accessibility_tree"] is False
        # The capture-only lane must still be a screenshot request — asking for neither is a driver error.
        assert "include_screenshot" not in args

    def test_ax_lane_skips_the_grab_when_advertised(self):
        stub = _StubCapture(_StubSession(props=BOTH))
        args = stub._gws_args("ax")
        assert args["include_screenshot"] is False
        assert "include_accessibility_tree" not in args

    def test_som_lane_always_wants_both_halves(self):
        stub = _StubCapture(_StubSession(props=BOTH))
        args = stub._gws_args("som")
        assert "include_screenshot" not in args and "include_accessibility_tree" not in args

    def test_flags_fail_closed_before_capability_discovery(self):
        """Pre-discovery the schema map is empty by construction; the payload must not change."""
        stub = _StubCapture(_StubSession(discovered=False, props=BOTH))
        for mode in ("vision", "ax", "som"):
            args = stub._gws_args(mode)
            assert "include_screenshot" not in args and "include_accessibility_tree" not in args

    def test_old_driver_without_the_properties_sees_the_old_payload(self):
        stub = _StubCapture(_StubSession(props=()))
        for mode in ("vision", "ax", "som"):
            args = stub._gws_args(mode)
            assert "include_screenshot" not in args and "include_accessibility_tree" not in args

    def test_sessionless_stub_is_unaffected(self):
        """The ax_walk_bound stubs run with ``_session = None`` — the lane gate must not raise there."""
        stub = _StubCapture(None)
        assert "include_accessibility_tree" not in stub._gws_args("vision")

    def test_partial_schema_only_sends_the_advertised_flag(self):
        """A driver that shipped include_screenshot but not the tree flag (or vice versa) gets one lane only."""
        stub = _StubCapture(_StubSession(props=("include_screenshot",)))
        assert stub._gws_args("ax")["include_screenshot"] is False
        assert "include_accessibility_tree" not in stub._gws_args("vision")

    def test_max_elements_bound_and_lane_flags_coexist(self):
        """The ax_max_elements bound (prefix of the same walk) and the vision lane are independent keys."""
        from tools.computer_use import cua_backend
        stub = _StubCapture(_StubSession(props=BOTH))
        # Whatever the configured bound is, the lane flag must ride along with it unchanged.
        bound = cua_backend._cua_configured_ax_max_elements()
        args = stub._gws_args("vision")
        assert args.get("max_elements", 0) == bound
        assert args["include_accessibility_tree"] is False


class TestCaptureEndToEnd:
    """The flags reach the actual tool call capture() makes, and the lanes keep their contract."""

    def _recorder(self, stub, out):
        calls: List[Dict[str, Any]] = []

        def fake_call(name, args):
            calls.append({"name": name, "args": args})
            return out
        stub._call_capture_tool = fake_call
        return calls

    def test_vision_capture_asks_for_the_capture_only_lane(self):
        stub = _StubCapture(_StubSession(props=BOTH))
        out = {"images": ["aW1n"], "image_mime_types": ["image/png"],
               "structuredContent": {"window_title": "Notes"}}
        calls = self._recorder(stub, out)
        cap = stub.capture("vision")
        gws = [c for c in calls if c["name"] == "get_window_state"]
        assert gws and gws[0]["args"]["include_accessibility_tree"] is False
        # The capture-only lane's whole point: title/metadata survive without the tree.
        assert cap.window_title == "Notes"
        assert cap.elements == []

    def test_ax_capture_asks_for_the_tree_only_lane(self):
        stub = _StubCapture(_StubSession(props=BOTH))
        out = {"structuredContent": {"elements": [{"index": 1, "role": "AXButton", "label": "OK"}],
                                     "window_title": "Notes"}}
        calls = self._recorder(stub, out)
        cap = stub.capture("ax")
        gws = [c for c in calls if c["name"] == "get_window_state"]
        assert gws and gws[0]["args"]["include_screenshot"] is False
        assert cap.png_b64 is None and cap.window_title == "Notes"

    def test_som_capture_sends_neither_flag(self):
        stub = _StubCapture(_StubSession(props=BOTH))
        out = {"images": ["aW1n"], "structuredContent": {"elements": [], "window_title": ""}}
        calls = self._recorder(stub, out)
        stub.capture("som")
        gws = [c for c in calls if c["name"] == "get_window_state"]
        assert gws and "include_screenshot" not in gws[0]["args"]
        assert "include_accessibility_tree" not in gws[0]["args"]


class TestTreeOnlyIsNotEmpty:
    """The ax lane's healthy response has NO screenshot half — the empty-result watchdog must
    key on the tree, or every tree-only capture would trigger a CLI re-fetch."""

    def test_tree_only_payload_is_not_empty(self):
        assert not _gws_is_empty({"structuredContent": {"elements": [{"index": 1, "role": "AXButton"}]}})
        assert not _gws_is_empty({"data": 'AXWindow "Notes"\n  - AXButton "OK"'})

    def test_truly_degenerate_payload_still_refetches(self):
        assert _gws_is_empty({"structuredContent": {"window_title": "x"}})
