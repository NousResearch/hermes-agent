"""Origin anchors are bound by the session, never by model-supplied metadata."""
import json

from gateway.session_context import clear_session_vars, set_session_vars
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
from hermes_state import SessionDB
from tools import kanban_tools as kt


def test_create_captures_bound_origin_not_caller_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    state = SessionDB(db_path=tmp_path / "state.db")
    state.create_session("conversation", source="telegram")
    state.close()
    tokens = set_session_vars(session_id="conversation", profile="default", platform="telegram",
                              chat_id="group", thread_id="topic", message_id="native-message")
    try:
        result = json.loads(kt._handle_create({"title": "release", "assignee": "default"},
                                             tool_origin={"prompt": "untrusted secret metadata"}))
    finally:
        clear_session_vars(tokens)
    assert result["ok"], result
    with kbc.connect_closing() as conn:
        origin = kb.get_task(conn, result["task_id"]).origin
    assert origin["session_id"] == "conversation"
    assert origin["platform"] == "telegram"
    assert origin["chat_id"] == "group"
    assert origin["thread_id"] == "topic"
    assert origin["message_id"] == "native-message"
    assert "prompt_excerpt" not in origin
    assert "untrusted secret metadata" not in json.dumps(origin)
    shown = json.loads(kt._handle_show({"task_id": result["task_id"]}))
    assert shown["task"]["origin"] == origin
