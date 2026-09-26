"""Lossless drag options through the registered tool; no live driver or GUI input.

Regression for #68097, stacked on #106540's strict addressing contracts.
"""

import copy
import json
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("foreground", [False, True])
@pytest.mark.parametrize("addressing", ["pixels", "elements", "both"])
@pytest.mark.parametrize(
    "properties,options,pixel_code,element_code",
    [
        ([], {}, None, None),
        ([], {"button": "left", "modifiers": []}, None, None),
        (["button"], {"button": "RIGHT"}, None, "drag_options_unsupported"),
        (["modifier"], {"modifiers": [" SHIFT ", "CTRL"]}, None, "drag_options_unsupported"),
        (["button", "modifier"], {"button": "middle", "modifiers": ["alt"]}, None, "drag_options_unsupported"),
        ([], {"button": "right"}, "drag_button_unsupported", "drag_options_unsupported"),
        ([], {"modifiers": ["shift"]}, "drag_modifiers_unsupported", "drag_options_unsupported"),
        (["button"], {"button": "right", "modifiers": ["shift"]}, "drag_modifiers_unsupported", "drag_options_unsupported"),
        (["modifier"], {"button": "right", "modifiers": ["shift"]}, "drag_button_unsupported", "drag_options_unsupported"),
        (["button", "modifier"], {"button": "bogus"}, "bad_drag_button", "bad_drag_button"),
    ],
)
def test_drag_options_are_preserved_or_refused_before_input(
    monkeypatch, properties, options, pixel_code, element_code, addressing, foreground,
):
    import tools.computer_use_tool  # noqa: F401
    from tools.computer_use import tool
    from tools.computer_use.cua_backend import CuaDriverBackend
    from tools.computer_use.cua_backend_session import _CuaDriverSession
    from tools.registry import registry

    # Exercise the real raw-schema probe, not a truthy MagicMock capability.
    # Option properties may be pixel-only even when element fields coexist.
    session = _CuaDriverSession.__new__(_CuaDriverSession)
    fields = {"pid", "window_id", "session", "from_x", "from_y", "to_x", "to_y",
              "from_element", "to_element", "delivery_mode"} | set(properties)
    session._tool_schemas = {"drag": {"type": "object", "additionalProperties": False,
                                     "properties": {key: {} for key in fields}}}
    session._capabilities = {"drag": {"input.drag.button", "input.drag.modifier"}, "bring_to_front": set()}
    calls = []

    def transport(name, args, timeout=30.0):
        calls.append((name, dict(args)))
        if name == "drag":
            assert set(args) <= fields
        return {"isError": False, "data": {}, "structuredContent": {"effect": "confirmed"}}

    session.call_tool = transport
    backend = CuaDriverBackend.__new__(CuaDriverBackend)
    backend._session = session
    backend._session_id = "drag-contract"
    backend._snapshot_tokens = {}
    backend._active_pid = 42
    backend._active_window_id = 7
    monkeypatch.setattr(tool, "_get_backend", lambda **kwargs: backend)
    monkeypatch.setattr(tool, "_request_approval", lambda *args: None)
    args = {"action": "drag", **options}
    if addressing in {"elements", "both"}:
        args.update(from_element=5, to_element=7)
    if addressing in {"pixels", "both"}:
        args.update(from_coordinate=[10, 20], to_coordinate=[30, 40])
    if foreground:
        args.update(delivery_mode="foreground", bring_to_front=True)

    original = copy.deepcopy(args)
    result = json.loads(registry._tools["computer_use"].handler(args, session_id="drag-contract"))

    code = pixel_code if addressing == "pixels" else element_code
    if code:
        assert result["ok"] is False
        assert result["code"] == code
        assert calls == []  # Not even the separately requested focus may run.
    else:
        assert result["ok"] is True
        expected = {"pid": 42, "window_id": 7, "session": "drag-contract"}
        if addressing == "pixels":
            expected.update(from_x=10, from_y=20, to_x=30, to_y=40)
        else:
            expected.update(from_element=5, to_element=7)
        if options.get("button", "left").lower() != "left":
            expected["button"] = options["button"].lower()
        if options.get("modifiers"):
            expected["modifier"] = [value.strip().lower() for value in options["modifiers"]]
        if foreground:
            expected["delivery_mode"] = "foreground"
        assert calls == ([('bring_to_front', {"pid": 42, "window_id": 7})] if foreground else []) + [
            ("drag", expected),
        ]
    assert args == original  # Normalization must not mutate the caller's payload.


@pytest.mark.parametrize("action", ["click", "double_click", "right_click", "middle_click", "drag"])
@pytest.mark.parametrize("modifiers", ["shift,ctrl", ["shift", "q"], ["shift", 1], None, {"shift": True}])
def test_invalid_pointer_modifiers_are_rejected_before_approval(monkeypatch, action, modifiers):
    import tools.computer_use_tool  # noqa: F401
    from tools.computer_use import tool
    from tools.registry import registry

    approve = Mock(side_effect=AssertionError("invalid input must not prompt"))
    get_backend = Mock(side_effect=AssertionError("invalid input must not start a driver"))
    monkeypatch.setattr(tool, "_request_approval", approve)
    monkeypatch.setattr(tool, "_get_backend", get_backend)

    result = json.loads(registry._tools["computer_use"].handler({
        "action": action, "coordinate": [10, 20],
        "from_coordinate": [10, 20], "to_coordinate": [30, 40], "modifiers": modifiers,
    }))

    assert "modifier" in result["error"]
    approve.assert_not_called()
    get_backend.assert_not_called()
