"""cua-driver 0.38 compatibility: markdown-only window reads and a restarted driver.

0.38.0 defaults ``get_window_state.tree_format`` to "markdown": the structured ``elements`` array (and its
per-element ``element_token``) is gone unless asked for, while every input tool accepts ONLY
``element_token``. The backend must (a) ask for ``tree_format: "both"`` when the live schema offers it, and
(b) still address rows of a markdown-only read through the documented ``<snapshot_id>:N`` token.
A driver restart surfaces as ``McpError("Connection closed")``, which must count as a recoverable
closed session.
"""

from __future__ import annotations

import copy
import json

from tests.tools.test_computer_use_element_token_contract import (
    FIXTURE, PID, WINDOW, StrictFakeDriver, _backend,
)

MARKDOWN = (
    "- [0] AXWindow \"Tasks\" [id=main actions=[raise]]\n"
    "    - [1] AXButton (Increment) [actions=[press]]\n"
    "    - [2] AXButton (Reset) [actions=[press]]\n"
)


class MarkdownOnlyDriver(StrictFakeDriver):
    """0.38-style read: markdown tree plus ``snapshot_id``, no ``elements`` unless tree_format asks for it."""

    def call_tool(self, name, args, timeout=30.0):
        if name != "get_window_state":
            return super().call_tool(name, args, timeout)
        self.calls.append((name, dict(args)))
        self.snapshot += 1
        sid = f"s{self.snapshot:08x}"
        sc = {"snapshot_id": sid, "pid": PID, "window_id": WINDOW, "window_title": "Tasks",
              "tree_markdown": MARKDOWN, "tree_format": args.get("tree_format", "markdown")}
        if args.get("tree_format") in ("elements", "both"):
            sc["elements"] = [{"element_index": i, "role": r, "label": l, "element_token": f"{sid}:{i}",
                               "frame": {"x": 10 * i, "y": 5, "w": 8, "h": 8}}
                              for i, (r, l) in enumerate([("AXWindow", "Tasks"), ("AXButton", "Increment"),
                                                          ("AXButton", "Reset")])]
        return self._ok(sc, data=MARKDOWN)


def _backend_038(*, with_tree_format: bool):
    backend, _ = _backend("cua_driver_0_32_0")
    schemas = copy.deepcopy(json.loads(FIXTURE.read_text())["cua_driver_0_32_0"])
    gws = schemas.setdefault("get_window_state", {"type": "object", "properties": {}})
    gws.setdefault("properties", {})
    if with_tree_format:
        gws["properties"]["tree_format"] = {"type": "string", "enum": ["markdown", "elements", "both"]}
    else:
        gws["properties"].pop("tree_format", None)
    driver = MarkdownOnlyDriver(schemas)
    backend._session._tool_schemas = schemas
    backend._session.call_tool = driver.call_tool
    return backend, driver


def test_capture_requests_both_tree_formats_when_offered():
    backend, driver = _backend_038(with_tree_format=True)
    cap = backend.capture(mode="ax", app="Main.py")
    gws_args = [a for n, a in driver.calls if n == "get_window_state"][-1]
    assert gws_args["tree_format"] == "both"
    assert any(e.bounds != (0, 0, 0, 0) for e in cap.elements)  # structured frames came back
    result = backend.click(element=2)
    assert result.ok, result.message
    name, args = driver.action_calls()[-1]
    assert (name, args["element_token"], "element_index" in args) == ("click", "s00000001:2", False)


def test_markdown_only_read_still_addresses_rows_by_snapshot_token():
    backend, driver = _backend_038(with_tree_format=False)
    cap = backend.capture(mode="ax", app="Main.py")
    gws_args = [a for n, a in driver.calls if n == "get_window_state"][-1]
    assert "tree_format" not in gws_args  # never sent to a driver that does not advertise it
    assert {1, 2} <= {e.index for e in cap.elements}
    result = backend.click(element=1)
    assert result.ok, result.message
    name, args = driver.action_calls()[-1]
    assert (name, args["element_token"], "element_index" in args) == ("click", "s00000001:1", False)


def test_mcp_connection_closed_counts_as_closed_session():
    from tools.computer_use.cua_backend_session import _CuaDriverSession

    class McpError(Exception):
        pass

    assert _CuaDriverSession._is_closed_session_error(McpError("Connection closed"))
    assert not _CuaDriverSession._is_closed_session_error(McpError("Invalid params"))
