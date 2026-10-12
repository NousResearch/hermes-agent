"""Real activity/steer/persistence contracts for the continuation of #106423."""

import json
import threading

from hermes_state import SessionDB
from model_tools import handle_function_call
from run_agent import AIAgent


def _recipient(db, session_id):
    agent = object.__new__(AIAgent)
    agent.session_id = session_id
    agent._session_db = db
    agent._session_db_created = True
    agent._pending_steer = None
    agent._pending_steer_lock = threading.Lock()
    agent.valid_tool_names = []
    return agent


def _send(target, body, sender):
    return json.loads(handle_function_call(
        "peer_send", {"target_session_id": target, "message": body},
        session_id=sender,
    ))


def test_peer_activity_delivers_user_row_with_provenance_and_persists(tmp_path, monkeypatch):
    """The real receive path works without recipient send tools and preserves history."""
    import tools.peer_messaging_tool as pm

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session("sender", source="cli")
        db.create_session("recipient", source="cli")
        discovered = json.loads(handle_function_call("peer_sessions", {}, session_id="sender"))
        assert [peer["session_id"] for peer in discovered["peers"]] == ["recipient"]
        assert _send("recipient", "The schema is ready", "sender")["success"]
        envelope = next(pm._inbox_dir("recipient").glob("*.json"))
        # Windows editors may add a BOM without changing the envelope's JSON.
        envelope.write_bytes(b"\xef\xbb\xbf" + envelope.read_bytes())
        agent = _recipient(db, "recipient")
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "probe", "type": "function",
             "function": {"name": "terminal", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "probe", "content": "done"},
        ]
        original_prefix = json.dumps(messages)
        agent._touch_activity("tool batch finished")
        agent._apply_pending_steer_to_tool_results(messages, 1)
        assert json.dumps(messages[:-1]) == original_prefix
        row = messages[-1]
        assert row["role"] == "user"
        assert "sender" in row["content"] and "The schema is ready" in row["content"]
        assert "peer" in row["content"].lower()
        assert agent._flush_messages_to_session_db(messages) is True
        stored = db.get_messages("recipient")
        assert stored[-1]["role"] == row["role"]
        assert stored[-1]["content"] == row["content"]
        assert not list(pm._inbox_dir("recipient").glob("*.json"))
        count = len(stored)
        agent._touch_activity("next activity")
        agent._apply_pending_steer_to_tool_results(messages, 1)
        assert agent._flush_messages_to_session_db(messages) is True
        assert len(db.get_messages("recipient")) == count
    finally:
        db.close()


def test_recipient_store_pins_mailbox_when_ambient_profile_changes(tmp_path, monkeypatch):
    """A callback must not drain another profile's same-ID inbox."""
    import tools.peer_messaging_tool as pm

    homes = [tmp_path / "first", tmp_path / "second"]
    databases = []
    try:
        for home, body in zip(homes, ("first profile payload", "second profile payload")):
            home.mkdir()
            monkeypatch.setenv("HERMES_HOME", str(home))
            db = SessionDB(db_path=home / "state.db")
            databases.append(db)
            db.create_session("recipient", source="cli")
            assert _send("recipient", body, "sender")["success"]
        agent = _recipient(databases[0], "recipient")
        assert pm.inject_peer_messages(agent) is True
        pending = agent._drain_pending_steer()
        assert "first profile payload" in pending
        assert "second profile payload" not in pending
        assert list(pm._inbox_dir("recipient").glob("*.json"))
    finally:
        for db in databases:
            db.close()
