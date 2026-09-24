"""Regression for #121934: hidden tools must not execute when named directly."""

import json

import pytest

from model_tools import get_tool_definitions, handle_function_call
from tools.registry import registry


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
        assert result["verified"] is True
        assert target.read_text() == "permitted"
        target.unlink()


def test_dispatch_respects_dynamic_browser_exec_grant(monkeypatch):
    entry = registry.get_entry("browser_exec")
    monkeypatch.setattr(entry, "check_fn", lambda: True)
    calls = []
    monkeypatch.setattr(registry, "dispatch", lambda name, args, **kwargs: calls.append(name) or '{"ok": true}')

    def names(scope):
        return {td["function"]["name"] for td in get_tool_definitions(
            enabled_toolsets=scope, quiet_mode=True, skip_tool_search_assembly=True,
        )}

    assert "browser_exec" not in names(["browser"])
    denied = json.loads(handle_function_call("browser_exec", {"code": "print(1)"}, enabled_toolsets=["browser"]))
    assert "not available in this session" in denied["error"]
    assert calls == []

    assert "browser_exec" in names(["browser", "terminal"])
    allowed = json.loads(handle_function_call(
        "browser_exec", {"code": "print(1)"}, enabled_toolsets=["browser", "terminal"],
    ))
    assert allowed == {"ok": True}
    assert calls == ["browser_exec"]
