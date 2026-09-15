"""Driver schemas remain authoritative when capability vocabulary omits element tokens."""

import json

import pytest

from tools import computer_use_tool  # noqa: F401 -- register the real handler
from tools.computer_use import tool
from tools.computer_use.cua_backend import CuaDriverBackend
from tools.registry import registry


@pytest.fixture(autouse=True)
def _auto_grant_approvals(grant_computer_use_approvals):
    """Pass through approval prompts for computer_use dispatch tests."""
    pass


@pytest.mark.parametrize("schema_token,legacy_capability", [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize("action", ["click", "double_click", "scroll", "set_value"])
def test_indexed_actions_send_tokens_only_when_the_driver_accepts_them(
    monkeypatch, schema_token, legacy_capability, action,
):
    backend = CuaDriverBackend()
    backend._set_active_target({"pid": 10, "window_id": 20})
    backend._snapshot_tokens = {1: "s00000001:1"}
    backend._session._tool_schemas = {action: {"properties": {"element_token": {}} if schema_token else {}}}
    backend._session._capabilities = {
        action: {"accessibility.element_tokens"} if legacy_capability else set(),
    }

    def native_action(name, args):
        assert name == action
        accepts_token = schema_token or legacy_capability
        accepted = args.get("element_token") == "s00000001:1" if accepts_token else "element_token" not in args
        return {"data": {"message": "accepted" if accepted else "invalid snapshot addressing"},
                "images": [], "structuredContent": {}, "isError": not accepted}

    monkeypatch.setattr(backend._session, "call_tool", native_action)
    monkeypatch.setattr(tool, "_get_backend", lambda session_id="": backend)
    args = {"action": action, "element": 1}
    if action == "scroll":
        args.update(direction="down", amount=1)
    if action == "set_value":
        args["value"] = "test"
    result = json.loads(registry.dispatch("computer_use", args))
    assert result["ok"] is True, result


@pytest.mark.parametrize("schema_coords,legacy_capability", [(True, False), (False, True), (False, False)])
def test_scroll_coordinates_sent_when_schema_or_capability_advertises(
    monkeypatch, schema_coords, legacy_capability,
):
    backend = CuaDriverBackend()
    backend._set_active_target({"pid": 10, "window_id": 20})
    backend._session._tool_schemas = {"scroll": {"properties": {"x": {}, "y": {}} if schema_coords else {}}}
    backend._session._capabilities = {
        "scroll": {"input.scroll.coordinates"} if legacy_capability else set(),
    }

    def native_action(name, args):
        assert name == "scroll"
        accepts_coords = schema_coords or legacy_capability
        if accepts_coords:
            assert args.get("x") == 100 and args.get("y") == 200
        else:
            assert "x" not in args and "y" not in args
        return {"data": {"message": "scrolled"}, "images": [], "structuredContent": {}, "isError": False}

    monkeypatch.setattr(backend._session, "call_tool", native_action)
    monkeypatch.setattr(tool, "_get_backend", lambda session_id="": backend)
    result = json.loads(registry.dispatch("computer_use", {
        "action": "scroll", "direction": "down", "amount": 2, "coordinate": [100, 200],
    }))
    assert result["ok"] is True, result

