"""Regression for #121934: hidden tools must not execute when named directly."""

import json

import pytest

from model_tools import handle_function_call


@pytest.mark.parametrize("scope", [
    {"enabled_toolsets": []},
    {"enabled_toolsets": ["web"]},
    {"enabled_toolsets": ["file"], "disabled_toolsets": ["file"]},
    {"disabled_toolsets": ["file"]},
])
def test_dispatch_rejects_registered_tool_outside_session_grant(tmp_path, scope):
    target = tmp_path / "forbidden.txt"
    result = json.loads(handle_function_call(
        "write_file", {"path": str(target), "content": "forbidden"}, **scope,
    ))
    assert "not available in this session" in result["error"]
    assert not target.exists()


def test_dispatch_preserves_granted_tool_and_unrestricted_calls(tmp_path):
    target = tmp_path / "permitted.txt"
    args = {"path": str(target), "content": "permitted"}
    for scope in ({"enabled_toolsets": ["file"]}, {}):
        result = json.loads(handle_function_call("write_file", args, **scope))
        assert result["success"] is True
        assert target.read_text() == "permitted"
        target.unlink()
