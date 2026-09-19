"""Snapshot-scoped element actions — the second addressing mode (NousResearch/hermes-agent#47072, Surface 6).

cua-driver accepts an indexed action in two forms — *"pass element_token, or snapshot_id together with
element_index"* — and refuses a bare ``element_index`` (``snapshot_id_required``). #111313 restored the first
form. This file pins the second: a capture whose elements carry NO per-element token still yields a
``structuredContent.snapshot_id``, and the wrapper carries it back on the action, gated on the live input
schema exactly like the token.

Tests named ``test_control_*`` pass with and without the change; they are what make the others meaningful.
"""

from unittest.mock import MagicMock

from tools.computer_use.cua_backend import CuaDriverBackend

_OK = {"data": "ok", "images": [], "image_mime_types": [], "structuredContent": None, "isError": False}


def _backend(*, schema_props):
    """Backend with a mocked session whose live input schema advertises ``schema_props``: {tool: {prop, ...}}.
    No ``capabilities[]`` at all — the modern tools/list shape."""
    backend = CuaDriverBackend()
    backend._session = MagicMock()
    backend._session.call_tool.return_value = dict(_OK)
    backend._session.supports_capability = lambda cap, tool=None: False
    backend._session.supports_input_property = lambda tool, prop: prop in schema_props.get(tool, set())
    backend._active_pid = 111
    backend._active_window_id = 222
    return backend


def _wire_args(backend):
    _, args = backend._session.call_tool.call_args.args
    return args


class TestSnapshotIdAttachment:
    def test_snapshot_id_attached_when_capture_carried_no_tokens(self):
        """The case #111313 does not cover: elements without per-element tokens, snapshot handle present."""
        backend = _backend(schema_props={"click": {"element_token", "snapshot_id"}})
        backend._snapshot_tokens = {}
        backend._snapshot_id = "snap-7"
        backend.click(element=3, button="left")
        args = _wire_args(backend)
        assert args["element_index"] == 3
        assert args["snapshot_id"] == "snap-7"
        assert "element_token" not in args  # nothing minted — the capture carried none

    def test_index_without_token_in_a_tokenised_snapshot_falls_back_to_snapshot_id(self):
        """A snapshot may tokenise some elements and not others; an untokenised index still belongs to that
        snapshot, so the driver gets the handle it can resolve (or report stale) against."""
        backend = _backend(schema_props={"click": {"element_token", "snapshot_id"}})
        backend._snapshot_tokens = {1: "s00000009:1"}
        backend._snapshot_id = "s00000009"
        backend.click(element=2, button="left")
        args = _wire_args(backend)
        assert "element_token" not in args
        assert args["snapshot_id"] == "s00000009"

    def test_control_token_wins_over_snapshot_id(self):
        """Per-element token is the driver's preferred form; when it exists, the snapshot handle stays home."""
        backend = _backend(schema_props={"click": {"element_token", "snapshot_id"}})
        backend._snapshot_tokens = {5: "s00000001:5"}
        backend._snapshot_id = "s00000001"
        backend.click(element=5, button="left")
        args = _wire_args(backend)
        assert args["element_token"] == "s00000001:5"
        assert "snapshot_id" not in args

    def test_control_schema_without_snapshot_id_never_receives_it(self):
        """Drivers that do not advertise the property (``additionalProperties: false``) must never see it."""
        backend = _backend(schema_props={"click": {"element_token"}})
        backend._snapshot_tokens = {}
        backend._snapshot_id = "snap-7"
        backend.click(element=3, button="left")
        assert "snapshot_id" not in _wire_args(backend)

    def test_control_no_snapshot_context_sends_the_bare_index(self):
        """No token and no snapshot handle: today's behaviour, unchanged — the driver decides."""
        backend = _backend(schema_props={"click": {"element_token", "snapshot_id"}})
        backend._snapshot_tokens = {}
        backend._snapshot_id = None
        backend.click(element=3, button="left")
        args = _wire_args(backend)
        assert args["element_index"] == 3
        assert "snapshot_id" not in args and "element_token" not in args


def _capture_backend(gws_structured):
    """Backend whose ``list_windows`` yields one window and whose ``get_window_state`` returns *gws_structured*."""
    backend = CuaDriverBackend()
    backend._session = MagicMock()
    backend._session.supports_capability = lambda cap, tool=None: False
    backend._session.supports_input_property = lambda tool, prop: prop in {"element_token", "snapshot_id"}
    windows_payload = {"windows": [{"app_name": "Demo", "pid": 9, "window_id": 1,
                                    "is_on_screen": True, "title": "", "z_index": 0}]}

    def fake_call_tool(name, args, **_kw):
        if name == "list_windows":
            return {**_OK, "data": "", "structuredContent": windows_payload}
        if name == "get_window_state":
            return {**_OK, "data": "✅ Demo — 2 elements, turn 1\n", "structuredContent": gws_structured}
        return dict(_OK)

    backend._session.call_tool.side_effect = fake_call_tool
    return backend


class TestSnapshotIdCapture:
    def test_capture_records_snapshot_id_when_elements_carry_no_tokens(self):
        backend = _capture_backend({
            "snapshot_id": "s00000042",
            "elements": [{"element_index": 1, "role": "AXButton", "label": "OK"},
                         {"element_index": 2, "role": "AXButton", "label": "Cancel"}],
        })
        backend.capture(mode="ax")
        assert backend._snapshot_tokens == {}
        assert backend._snapshot_id == "s00000042"

    def test_fresh_capture_replaces_a_stale_snapshot_id(self):
        """Snapshot-cache invariant, same as for tokens: only the latest capture's handle is eligible."""
        backend = _capture_backend({
            "elements": [{"element_index": 1, "role": "AXButton", "label": "OK", "element_token": "snap2:1"}],
        })
        backend._snapshot_id = "stale"
        backend.capture(mode="ax")
        assert backend._snapshot_id is None
        assert backend._snapshot_tokens == {1: "snap2:1"}

    def test_new_target_disarms_the_snapshot_id_before_any_capture(self):
        backend = _backend(schema_props={})
        backend._snapshot_id = "snap-7"
        backend._set_active_target({"pid": 5, "window_id": 6})
        assert backend._snapshot_id is None

    def test_capture_then_click_carries_the_snapshot_id_on_the_wire(self):
        """Invariant through the real capture() -> click() path: an index from a token-less snapshot is
        dispatched with the handle that produced it, never bare."""
        backend = _capture_backend({
            "snapshot_id": "s00000042",
            "elements": [{"element_index": 1, "role": "AXButton", "label": "OK"}],
        })
        backend.capture(mode="ax")
        backend.click(element=1, button="left")
        name, args = backend._session.call_tool.call_args.args
        assert name == "click"
        assert args["element_index"] == 1
        assert args["snapshot_id"] == "s00000042"
        assert "element_token" not in args
