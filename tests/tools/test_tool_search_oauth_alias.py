"""OAuth wire aliases resolve in tool_describe/tool_call arguments.

Regression for #120858: on the Anthropic OAuth wire ``session_search`` is
advertised as ``chat_history_lookup`` (and ``memory`` as ``context_notes``),
but ``tool_describe({"names": ["chat_history_lookup"]})`` answered
``not_found`` for a tool that exists — the reverse mapping covered only the
tool-call NAME, never names passed as ARGUMENTS. The bridge resolves a wire
alias to its registered tool as a last resort (a real tool registered under
the wire name always wins) and answers under the REQUESTED name so the model
sees a consistent name.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def _td(name: str, description: str = "", properties: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties or {}},
        },
    }


def _describe(names):
    from tools.tool_search import dispatch_tool_describe
    return json.loads(dispatch_tool_describe(
        {"names": names},
        current_tool_defs=[_td("session_search", "Recall past conversations.", {"query": {}})],
    ))


class TestToolDescribeOAuthWireAlias:
    def test_alias_returns_registered_schema_under_requested_name(self):
        result = _describe(["chat_history_lookup"])
        assert "not_found" not in result, result
        assert set(result["tools"]) == {"chat_history_lookup"}
        assert result["tools"]["chat_history_lookup"]["parameters"]["properties"] == {"query": {}}

    def test_memory_alias_resolves(self, monkeypatch):
        import tools.tool_search as tool_search
        from tools.tool_search import ToolSearchConfig, dispatch_tool_describe
        # memory is eager by default; a session that defers it gets the same fallback.
        cfg = ToolSearchConfig.from_raw({"defer": ["memory"]})
        monkeypatch.setattr(tool_search, "load_config_readonly", lambda: cfg)
        result = json.loads(dispatch_tool_describe(
            {"names": ["context_notes"]},
            current_tool_defs=[_td("memory", "Persist notes.", {"content": {}})],
        ))
        assert "not_found" not in result, result
        assert set(result["tools"]) == {"context_notes"}

    def test_canonical_name_still_resolves(self):
        result = _describe(["session_search"])
        assert set(result["tools"]) == {"session_search"}

    def test_registered_tool_wins_over_alias(self, monkeypatch):
        """A real tool actually named ``chat_history_lookup`` keeps precedence —
        the alias must not hijack it."""
        import tools.tool_search as tool_search
        from tools.tool_search import ToolSearchConfig, dispatch_tool_describe
        cfg = ToolSearchConfig.from_raw({"defer": ["session_search", "chat_history_lookup"]})
        monkeypatch.setattr(tool_search, "load_config_readonly", lambda: cfg)
        result = json.loads(dispatch_tool_describe(
            {"names": ["chat_history_lookup"]},
            current_tool_defs=[
                _td("session_search", "Recall past conversations."),
                _td("chat_history_lookup", "The real wire-named tool.", {"own": {}}),
            ],
        ))
        assert "not_found" not in result, result
        assert result["tools"]["chat_history_lookup"]["description"] == "The real wire-named tool."

    def test_direct_surface_wire_name_is_not_hijacked(self):
        """A native tool under the wire name that never enters the deferrable subset
        (core / direct surface) must block the alias too — precedence is native
        ownership, not deferability."""
        from tools.tool_search import dispatch_tool_describe
        result = json.loads(dispatch_tool_describe(
            {"names": ["chat_history_lookup"]},
            current_tool_defs=[
                _td("session_search", "Recall past conversations."),
                _td("chat_history_lookup", "The real wire-named tool.", {"own": {}}),
            ],
        ))
        assert "chat_history_lookup" not in result.get("tools", {}), result
        assert result.get("not_found") == ["chat_history_lookup"], result

    def test_unknown_name_still_not_found(self):
        result = _describe(["no_such_tool_xyz"])
        assert result.get("not_found") == ["no_such_tool_xyz"]
        assert result["tools"] == {}


class TestToolCallOAuthWireAlias:
    def test_batch_call_alias_dispatches_to_registered_tool(self):
        from tools.tool_search import resolve_underlying_call
        name, args, err = resolve_underlying_call({
            "calls": [{"name": "chat_history_lookup", "arguments": {"query": "past chats"}}],
        })
        assert err is None, err
        assert name == "session_search"
        assert args == {"query": "past chats"}

    def test_legacy_single_shape_alias_dispatches(self):
        from tools.tool_search import resolve_underlying_call
        name, args, err = resolve_underlying_call({
            "name": "chat_history_lookup", "arguments": {"query": "x"},
        })
        assert err is None, err
        assert name == "session_search"

    def test_deferrable_wire_name_is_not_hijacked(self, monkeypatch):
        """A session whose defer set really contains the wire name dispatches
        it natively — the alias fallback never fires."""
        import tools.tool_search as tool_search
        from tools.tool_search import ToolSearchConfig, resolve_underlying_call

        cfg = ToolSearchConfig.from_raw({"defer": ["chat_history_lookup"]})
        monkeypatch.setattr(tool_search, "load_config_readonly", lambda: cfg)
        name, _, err = resolve_underlying_call({
            "calls": [{"name": "chat_history_lookup", "arguments": {}}],
        })
        assert err is None, err
        assert name == "chat_history_lookup"

    def test_registered_direct_surface_wire_name_is_not_redirected(self, monkeypatch):
        """A registered tool under the wire name that is not deferrable is a wrong-door
        call, not an alias: never silently redirect it to ``session_search``."""
        import tools.tool_search as tool_search
        from tools.tool_search import resolve_underlying_call

        class _Entry:
            toolset = "desktop_ui"

        monkeypatch.setattr(tool_search, "_registry_entry",
                            lambda n: _Entry() if n == "chat_history_lookup" else None)
        name, _, err = resolve_underlying_call({
            "calls": [{"name": "chat_history_lookup", "arguments": {}}],
        })
        assert name is None and err is not None, (name, err)

    def test_unknown_call_still_rejected(self):
        from tools.tool_search import resolve_underlying_call
        name, _, err = resolve_underlying_call({
            "calls": [{"name": "no_such_tool_xyz", "arguments": {}}],
        })
        assert name is None and err is not None
