"""Element-token attachment for cua-driver element actions (moved out of test_computer_use.py).

Covers both driver generations: 0.21-era schemas that accept element_index beside element_token, and
0.3x+/0.5x strict schemas that accept only element_token (#134857, #135872)."""
from __future__ import annotations


class TestElementTokenAttachment:
    """Surface 6 (NousResearch/hermes-agent#47072): trycua/cua#1961 added
    an opaque `element_token` alongside `element_index` so the wrapper
    can carry per-snapshot handles instead of relying on raw indices that
    silently re-resolve when the snapshot is superseded.

    The contract the wrapper implements:
    1. capture() refreshes a per-snapshot {index -> token} map from
       structuredContent.elements.
    2. Whenever an action carrying element_index is about to hit cua-driver,
       look up the matching token and attach it — but ONLY for tools that
       advertise `accessibility.element_tokens` (Surface 4 gate). Older
       drivers reject unknown args via additionalProperties=false.
    3. cua-driver prefers token over index when both are supplied, so
       sending both is safe and stale-detection becomes explicit.
    """

    def _backend_with_session(self, capabilities):
        """Build a backend whose session reports the given capabilities map."""
        from unittest.mock import MagicMock
        from tools.computer_use.cua_backend import CuaDriverBackend

        backend = CuaDriverBackend()
        backend._session = MagicMock()
        backend._session.call_tool.return_value = {
            "data": "ok", "images": [], "image_mime_types": [],
            "structuredContent": None, "isError": False,
        }
        # `supports_capability(cap, tool=None)` honors the supplied map.
        def _supports(cap, tool=None):
            if tool is not None:
                return cap in capabilities.get(tool, set())
            return any(cap in caps for caps in capabilities.values())
        backend._session.supports_capability = _supports
        backend._active_pid = 111
        backend._active_window_id = 222
        return backend

    def test_token_attached_when_tool_advertises_capability(self):
        backend = self._backend_with_session({
            "click": {"input.pointer.click", "accessibility.element_tokens"},
        })
        backend._snapshot_tokens = {5: "s0001:5", 6: "s0001:6"}
        backend.click(element=5, button="left")
        name, args = backend._session.call_tool.call_args.args
        assert name == "click"
        assert args["element_index"] == 5
        # The matching token rode along — cua-driver will prefer it.
        assert args["element_token"] == "s0001:5"

    def test_token_attached_when_only_input_schema_advertises_it(self):
        """cua-driver >= 0.21 dropped the per-tool `capabilities[]` array from tools/list but advertises
        `element_token` in `click`'s inputSchema and REFUSES a bare element_index (`snapshot_id_required`).
        The token must ride along from the live schema alone, or every element click is refused."""
        backend = self._backend_with_session({})  # no capabilities[] at all — the modern shape
        backend._session.supports_input_property = lambda tool, prop: (tool, prop) == ("click", "element_token")
        backend._snapshot_tokens = {5: "s00000001:5"}
        backend.click(element=5, button="left")
        _, args = backend._session.call_tool.call_args.args
        assert args["element_token"] == "s00000001:5"

    def test_index_stripped_when_schema_is_token_only(self):
        """cua-driver 0.32+ strict schemas declare `element_token`, drop `element_index`, and set
        additionalProperties: false — sending the pair gets the action rejected before it runs
        (#132876). The wrapper must send the token alone when the live schema names the token
        but not the index."""
        backend = self._backend_with_session({})  # no capabilities[] — the modern shape
        backend._session.supports_input_property = lambda tool, prop: (tool, prop) == ("click", "element_token")
        backend._snapshot_tokens = {5: "s00000001:5"}
        backend.click(element=5, button="left")
        _, args = backend._session.call_tool.call_args.args
        assert args["element_token"] == "s00000001:5"
        assert "element_index" not in args

    def test_index_kept_when_schema_accepts_both(self):
        """0.21-era schemas keep `element_index` alongside `element_token` and accept both, so
        the strip must stay scoped to token-only schemas."""
        backend = self._backend_with_session({})
        backend._session.supports_input_property = lambda tool, prop: (
            (tool, prop) in {("click", "element_token"), ("click", "element_index")})
        backend._snapshot_tokens = {5: "s00000001:5"}
        backend.click(element=5, button="left")
        _, args = backend._session.call_tool.call_args.args
        assert args["element_token"] == "s00000001:5"
        assert args["element_index"] == 5

    def test_refused_when_token_only_schema_has_no_token_for_index(self):
        """A strict token-only schema never accepts the bare `element_index` either, so an index
        with no token from the latest capture (model reusing an older capture, or a capture that
        returned no tokens) must be refused locally with an actionable message instead of being
        sent to the driver for the same opaque `unknown argument element_index` rejection."""
        backend = self._backend_with_session({})
        backend._session.supports_input_property = lambda tool, prop: (tool, prop) == ("click", "element_token")
        backend._snapshot_tokens = {}  # index 5 not in the latest capture's tokens
        res = backend.click(element=5, button="left")
        assert res.ok is False
        assert res.code == "stale_element"
        assert "5" in res.message and "capture" in res.message
        backend._session.call_tool.assert_not_called()


    def test_capture_refreshes_snapshot_tokens(self):
        """A fresh capture should overwrite any stale tokens from a
        previous snapshot — token cache invariant: only the latest
        capture's tokens are eligible for attachment."""
        from unittest.mock import MagicMock
        from tools.computer_use.cua_backend import CuaDriverBackend

        backend = CuaDriverBackend()
        backend._session = MagicMock()
        backend._session.supports_capability = lambda cap, tool=None: True
        # Pretend an earlier capture left this stale state.
        backend._snapshot_tokens = {99: "stale:99"}

        windows_payload = {"windows": [{
            "app_name": "Demo", "pid": 9, "window_id": 1,
            "is_on_screen": True, "title": "", "z_index": 0,
        }]}

        def fake_call_tool(name, args):
            if name == "list_windows":
                return {"data": "", "images": [], "image_mime_types": [],
                        "structuredContent": windows_payload, "isError": False}
            if name == "get_window_state":
                return {
                    "data": '✅ Demo — 2 elements, turn 1\n',
                    "images": [], "image_mime_types": [],
                    "structuredContent": {"elements": [
                        {"element_index": 1, "role": "AXButton", "label": "OK",
                         "element_token": "snap2:1"},
                        {"element_index": 2, "role": "AXButton", "label": "X",
                         "element_token": "snap2:2"},
                    ]},
                    "isError": False,
                }
            return {"data": "", "images": [], "image_mime_types": [],
                    "structuredContent": None, "isError": False}

        backend._session.call_tool.side_effect = fake_call_tool
        backend.capture(mode="ax")

        # Stale 99 token is gone; only the two new tokens remain.
        assert backend._snapshot_tokens == {1: "snap2:1", 2: "snap2:2"}
