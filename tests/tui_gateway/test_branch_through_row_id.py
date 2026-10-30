"""Tests for session.branch through_message_id / through_row_id row-level boundary slicing."""

import threading
import pytest

from tui_gateway import server


@pytest.fixture(autouse=True)
def _neuter_agent_prewarm_timer(monkeypatch):
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)


def test_session_branch_slices_through_message_id(monkeypatch, tmp_path):
    """session.branch with through_message_id cuts history at the specified durable row id."""
    persisted_rows = []

    class ProfileDB:
        def __init__(self, db_path=None):
            pass

        def get_session_title(self, _key):
            return "parent"

        def get_next_title_in_lineage(self, current):
            return f"{current} (branch)"

        def create_session(self, new_key, **kwargs):
            pass

        def append_messages_batch(self, session_id, messages, **kwargs):
            persisted_rows.extend(messages)
            return list(range(1, len(messages) + 1))

        def set_session_title(self, key, title):
            return True

        def set_auto_title(self, key, title, source=None):
            return True

        def get_session(self, key):
            return {"id": key, "cwd": str(tmp_path)}

        def update_session_cwd(self, *a, **k):
            return None

        def close(self):
            return None

    class FakeAgent:
        def __init__(self):
            self.model = "test-model"
            self.session_id = None

    # Simulate a Discord / multi-turn session with tool calls:
    # Row 1: user "Can you check discord?"
    # Row 2: assistant "Checking..." (has text, so in visible branch history)
    # Row 3: assistant "Here are the results"
    # Row 4: user "Thanks, now do step 2"
    # Row 5: assistant "Step 2 done."
    history = [
        {"role": "user", "content": "Can you check discord?", "_row_id": 1},
        {"role": "assistant", "content": "Checking...", "_row_id": 2},
        {"role": "assistant", "content": "Here are the results", "_row_id": 3},
        {"role": "user", "content": "Thanks, now do step 2", "_row_id": 4},
        {"role": "assistant", "content": "Step 2 done.", "_row_id": 5},
    ]

    parent = {
        "session_key": "parent-key",
        "history": history,
        "history_lock": threading.Lock(),
        "running": False,
        "cols": 80,
        "profile_home": None,
        "source": "tui",
        "agent": FakeAgent(),
        "created_at": 1.0,
        "last_active": 1.0,
        "cwd": str(tmp_path),
    }
    server._sessions["parent"] = parent
    monkeypatch.setattr(server, "_get_db", lambda: ProfileDB())
    monkeypatch.setattr("hermes_state_registry.acquire", ProfileDB)
    monkeypatch.setattr(server, "_claim_active_session_slot", lambda *a, **k: (None, None))
    monkeypatch.setattr(server, "_make_agent", lambda *a, **k: FakeAgent())
    monkeypatch.setattr(server, "_set_session_context", lambda *a, **k: {})
    monkeypatch.setattr(server, "_clear_session_context", lambda *a, **k: None)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda *a, **k: None)
    monkeypatch.setattr(server, "_attach_worker", lambda *a, **k: None)

    try:
        # Branch through row id 3.
        # Note: count=2 is passed as fallback (as Desktop sends folded UI message count=2),
        # but through_message_id=3 must take precedence and keep rows 1, 2, 3!
        persisted_rows.clear()
        resp = server.handle_request({
            "id": "b1",
            "method": "session.branch",
            "params": {
                "session_id": "parent",
                "count": 2,
                "through_message_id": 3,
            }
        })
        assert "result" in resp, resp
        assert resp["result"]["message_count"] == 3
        copied_contents = [r["content"] for r in persisted_rows]
        assert copied_contents == [
            "Can you check discord?",
            "Checking...",
            "Here are the results"
        ]

        # Branch through row id 4 (user turn).
        persisted_rows.clear()
        resp2 = server.handle_request({
            "id": "b2",
            "method": "session.branch",
            "params": {
                "session_id": "parent",
                "count": 2,
                "through_message_id": 4,
            }
        })
        assert "result" in resp2, resp2
        assert resp2["result"]["message_count"] == 4
        assert [r["content"] for r in persisted_rows] == [
            "Can you check discord?",
            "Checking...",
            "Here are the results",
            "Thanks, now do step 2"
        ]

        # through_row_id alias works identically
        persisted_rows.clear()
        resp3 = server.handle_request({
            "id": "b3",
            "method": "session.branch",
            "params": {
                "session_id": "parent",
                "count": 2,
                "through_row_id": 4,
            }
        })
        assert "result" in resp3, resp3
        assert resp3["result"]["message_count"] == 4

        # When through_message_id is absent, falls back to count
        persisted_rows.clear()
        resp4 = server.handle_request({
            "id": "b4",
            "method": "session.branch",
            "params": {
                "session_id": "parent",
                "count": 2,
            }
        })
        assert "result" in resp4, resp4
        assert resp4["result"]["message_count"] == 2
        assert [r["content"] for r in persisted_rows] == [
            "Can you check discord?",
            "Checking..."
        ]

        # When through_message_id matches no row (e.g. unpersisted history without row IDs), falls back to count
        parent["history"] = [
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},
        ]
        persisted_rows.clear()
        resp5 = server.handle_request({
            "id": "b5",
            "method": "session.branch",
            "params": {
                "session_id": "parent",
                "count": 2,
                "through_message_id": 999,
            }
        })
        assert "result" in resp5, resp5
        assert resp5["result"]["message_count"] == 2
        assert [r["content"] for r in persisted_rows] == ["msg 1", "msg 2"]

    finally:
        server._sessions.pop("parent", None)
