"""Utility MCP handlers validate required arguments before any transport work.

``_make_utility_handler``'s docstring promises ``required`` validated before any
transport work. ``tools.mcp_tool_discovery._get_connected_server_for_call`` is the
single first-use connect point for lazy servers — acquiring it spawns the server —
so a call missing its required parameter must be rejected without reaching it: no
server is spawned/connected just to report a missing argument.
"""

import json

import pytest

from tools.mcp_tool_handlers import _make_get_prompt_handler, _make_read_resource_handler


@pytest.fixture
def acquisition(monkeypatch):
    """Transport-boundary double: records every connection attempt, never connects."""
    calls = []

    def fake_get(server_name):
        calls.append(server_name)
        return None  # "not connected": keeps the handler's post-acquisition path observable

    monkeypatch.setattr("tools.mcp_tool_discovery._get_connected_server_for_call", fake_get)
    return calls


class TestRequiredParameterBeforeTransport:
    @pytest.mark.parametrize("required,factory", [
        ("uri", _make_read_resource_handler),
        ("name", _make_get_prompt_handler),
    ], ids=["read_resource", "get_prompt"])
    def test_missing_required_parameter_never_connects(self, acquisition, required, factory):
        handler = factory("srv", 5.0)
        out = json.loads(handler({}))
        assert out["error"] == f"Missing required parameter '{required}'"
        assert acquisition == []

    def test_present_required_parameter_still_connects(self, acquisition):
        handler = _make_get_prompt_handler("srv", 5.0)
        out = json.loads(handler({"name": "greeting"}))
        assert "is not connected" in out["error"]
        assert acquisition == ["srv"]
