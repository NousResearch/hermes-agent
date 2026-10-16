"""``session.branch`` row-anchored boundaries (#139939).

A merged desktop bubble spans several persisted rows (tool interleavings, auto-continues), so the
bubble ``count`` it sent has no stable mapping onto the raw rows ``_branch_live`` slices and the
prefix landed inside the clicked answer. ``through_message_id`` names the terminal durable row of
the clicked bubble; the slice keeps everything through that row, and a row outside the transcript
refuses the branch instead of silently miscutting or widening it.
"""

import threading

import pytest

from tui_gateway import server


@pytest.fixture(autouse=True)
def _neuter_agent_prewarm_timer(monkeypatch):
    """Stub the deferred agent pre-warm timer (mirrors test_tui_gateway_server.py)."""
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    yield


class ProfileDB:
    """The display-projection read is absent on purpose: ``_branch_source_history`` falls back to
    the in-memory history, which carries the same ``_row_id`` keys the durable projection does."""

    def __init__(self, db_path=None):
        self.persisted: list[list[dict]] = []

    def get_session_title(self, _key):
        return "parent"

    def get_next_title_in_lineage(self, current):
        return f"{current} (branch)"

    def create_session(self, new_key, **kwargs):
        pass

    def append_messages_batch(self, session_id, messages, **kwargs):
        self.persisted.append(list(messages))
        return list(range(1, len(messages) + 1))

    def set_session_title(self, key, title):
        return True

    def set_auto_title(self, key, title, **kwargs):
        return True

    def get_session(self, key):
        return {"id": key, "cwd": "."}

    def update_session_cwd(self, *a, **k):
        return None

    def close(self):
        return None


class FakeAgent:
    def __init__(self):
        self.model = "test-model"
        self.session_id = None


def _tool_heavy_parent(tmp_path):
    """One clicked answer spanning two assistant rows (rows 12-13): a bubble count of 2 selects
    only through row 12's bubble, the boundary row 13 is what the child must keep."""
    return {
        "session_key": "parent-key",
        "history": [
            {"role": "user", "content": "question", "_row_id": 11},
            {"role": "assistant", "content": "part one of the clicked answer", "_row_id": 12},
            {"role": "assistant", "content": "part two of the clicked answer", "_row_id": 13},
            {"role": "user", "content": "later question", "_row_id": 14},
            {"role": "assistant", "content": "later answer", "_row_id": 15},
        ],
        "history_lock": threading.Lock(),
        "running": False,
        "cols": 80,
        "profile_home": None,
        "source": "desktop",
        "agent": FakeAgent(),
        "created_at": 1.0,
        "last_active": 1.0,
        "cwd": str(tmp_path),
    }


@pytest.fixture
def branched(monkeypatch, tmp_path):
    db = ProfileDB()

    def _setup(history_rows):
        server._sessions["parent"] = {**_tool_heavy_parent(tmp_path), "history": history_rows}
        monkeypatch.setattr(server, "_get_db", lambda: db)
        monkeypatch.setattr("hermes_state_registry.acquire", ProfileDB)
        monkeypatch.setattr(server, "_claim_active_session_slot", lambda *a, **k: (None, None))
        monkeypatch.setattr(server, "_make_agent", lambda *a, **k: FakeAgent())
        monkeypatch.setattr(server, "_set_session_context", lambda *a, **k: {})
        monkeypatch.setattr(server, "_clear_session_context", lambda *a, **k: None)
        monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
        monkeypatch.setattr(server, "_session_cwd", lambda s: str(tmp_path))
        monkeypatch.setattr(server, "_register_session_cwd", lambda *a, **k: None)
        monkeypatch.setattr(server, "_attach_worker", lambda *a, **k: None)
        return db

    yield _setup
    for key in list(server._sessions):
        server._sessions.pop(key, None)
    server._idempotency_keys.clear()


def test_branch_through_message_id_keeps_the_whole_clicked_answer(branched, tmp_path):
    db = branched([
        {"role": "user", "content": "question", "_row_id": 11},
        {"role": "assistant", "content": "part one of the clicked answer", "_row_id": 12},
        {"role": "assistant", "content": "part two of the clicked answer", "_row_id": 13},
        {"role": "user", "content": "later question", "_row_id": 14},
        {"role": "assistant", "content": "later answer", "_row_id": 15},
    ])

    # The legacy shape the desktop used to send: bubble count 2 slices inside the clicked
    # two-row answer, dropping its tail — the #139939 miscut.
    legacy = server.handle_request({"id": "b1", "method": "session.branch",
                                    "params": {"session_id": "parent", "count": 2, "name": "legacy"}})
    assert legacy["result"]["message_count"] == 2
    assert [row["content"] for row in db.persisted[-1]] == ["question", "part one of the clicked answer"]

    # The boundary names the clicked answer's terminal durable row: both of its rows stay.
    anchored = server.handle_request({"id": "b2", "method": "session.branch", "params": {
        "session_id": "parent", "count": 2, "through_message_id": 13, "name": "anchored"}})
    assert anchored["result"]["message_count"] == 3
    assert [row["content"] for row in db.persisted[-1]] == [
        "question", "part one of the clicked answer", "part two of the clicked answer"]


def test_branch_through_message_id_refuses_a_row_outside_the_transcript(branched):
    branched([
        {"role": "user", "content": "question", "_row_id": 11},
        {"role": "assistant", "content": "answer", "_row_id": 12},
    ])

    refused = server.handle_request({"id": "b1", "method": "session.branch", "params": {
        "session_id": "parent", "count": 2, "through_message_id": 999, "name": "foreign"}})
    assert refused["error"]["code"] == 4008
    assert "999" in refused["error"]["message"]
    # No child runtime or stored row was minted for the refused branch.
    assert list(server._sessions) == ["parent"]


def test_branch_without_through_message_id_keeps_the_count_contract(branched):
    branched([
        {"role": "user", "content": "question", "_row_id": 11},
        {"role": "assistant", "content": "answer", "_row_id": 12},
        {"role": "user", "content": "later", "_row_id": 13},
    ])

    legacy = server.handle_request({"id": "b1", "method": "session.branch",
                                    "params": {"session_id": "parent", "count": 2}})
    assert legacy["result"]["message_count"] == 2
