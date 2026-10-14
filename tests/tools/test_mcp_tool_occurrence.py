"""Persisted occurrence identity through native dispatch and the MCP event-loop thread."""

import json
import threading
from unittest.mock import MagicMock

from agent.tool_call_context import CURRENT_TOOL_CALL_UID, bind_occurrence, dispatch_occurrence, persisted_occurrence
from gateway.session_context import clear_session_vars, set_session_vars
from hermes_state import SessionDB


def test_persisted_occurrences_replay_across_restart_and_rotation(tmp_path, monkeypatch):
    import model_tools
    from mcp import types
    from tools import mcp_tool, mcp_tool_loop
    from tools.mcp_tool_handlers import _make_tool_handler

    received = []

    async def call_tool(name, *, arguments, **kwargs):
        received.append(kwargs)
        return types.CallToolResult(content=[types.TextContent(type="text", text="ok")])

    server = MagicMock()
    server._config = {"forward_session_context": True}
    server.session.call_tool = call_tool
    server._ready = threading.Event()
    server._ready.set()
    server._is_recycled_stdio.return_value = False
    server._stdio_children_dead.return_value = False
    monkeypatch.setitem(mcp_tool._servers, "occurrence-proof", server)
    mcp_tool_loop._ensure_mcp_loop()
    handler = _make_tool_handler("occurrence-proof", "inspect", 5)
    monkeypatch.setattr(model_tools.registry, "dispatch", lambda name, args, **kwargs: handler(args))
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    db.create_session("original", "api_server")
    messages = []
    occurrences = []
    # Identical intentional calls with even the same provider ID remain distinct.
    for _ in range(2):
        assistant = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "reused", "type": "function", "function": {"name": "inspect", "arguments": "{}"}}]}
        db.append_messages_batch("original", [assistant])
        messages.append(assistant)
        occurrences.append(persisted_occurrence(messages, "inspect", "reused"))
    assert occurrences[0].uid != occurrences[1].uid
    db.publish_compression_child(parent_session_id="original", child_session_id="rotated", source="api_server",
                                 messages=messages, require_compression_lease=False)
    db.close()
    db = SessionDB(db_path=path)
    try:
        restored = persisted_occurrence(db.get_messages_as_conversation("rotated"), "inspect", "reused")
        assert restored == occurrences[1]
    finally:
        db.close()
    tokens = set_session_vars(platform="api_server", session_key="run", session_id="rotated", request_session_id="original")
    try:
        for occurrence in (occurrences[0], occurrences[0], restored):
            with bind_occurrence(occurrence):
                result = model_tools._execute_tool("inspect", {}, {}, model_tools._CallIds(tool_call_id="reused"),
                                                  user_task=None, enabled_tools=None, skip_tool_execution_middleware=True)
                assert json.loads(result)["result"] == "ok"
            assert CURRENT_TOOL_CALL_UID.get() == ""
        assert [entry["meta"]["com.nousresearch.hermes/tool_call_uid"] for entry in received] == [
            occurrences[0].uid, occurrences[0].uid, occurrences[1].uid]
        server._config = {}
        with bind_occurrence(restored), dispatch_occurrence("inspect", "reused"):
            handler({})
        assert received[-1] == {}
    finally:
        clear_session_vars(tokens)


def test_missing_and_nested_calls_do_not_inherit_occurrence():
    messages = [{"role": "assistant", "tool_calls": [{"id": "same", "function": {"name": "inspect"}}],
                 "_tool_call_uids": {"same": "persisted"}}]
    occurrence = persisted_occurrence(messages, "inspect", "same")
    with bind_occurrence(occurrence), dispatch_occurrence("inspect", "same"):
        assert CURRENT_TOOL_CALL_UID.get() == "persisted"
        with dispatch_occurrence("inspect", "same"):
            assert CURRENT_TOOL_CALL_UID.get() == ""
        assert CURRENT_TOOL_CALL_UID.get() == "persisted"
    assert CURRENT_TOOL_CALL_UID.get() == ""
    with bind_occurrence(occurrence), dispatch_occurrence("other", "same"):
        assert CURRENT_TOOL_CALL_UID.get() == ""
    messages.append({"role": "assistant", "tool_calls": [{"id": "same", "function": {"name": "inspect"}}]})
    assert persisted_occurrence(messages, "inspect", "same") is None
