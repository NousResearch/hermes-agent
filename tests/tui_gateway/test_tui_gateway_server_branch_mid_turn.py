"""Mid-turn branch: source is cut at the turn-start boundary (contract 9)."""
import threading

import pytest

from tui_gateway import server


@pytest.fixture(autouse=True)
def _neuter_agent_prewarm_timer(request, monkeypatch):
    if request.node.get_closest_marker("real_agent_prewarm"):
        yield
        return
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    yield


def _make_parent(tmp_path, monkeypatch):
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
            return list(range(1, len(messages) + 1))

        def set_session_title(self, key, title, *a, **k):
            return True

        def set_auto_title(self, key, title, *a, **k):
            return True

        def discard_unadmitted_session(self, *a, **k):
            return None

        def get_session(self, key):
            return {"id": key, "cwd": str(tmp_path)}

        def update_session_cwd(self, *a, **k):
            return None

        def get_resume_conversations(self, key):
            # Display projection carries every row the model history collapses:
            # settled turns PLUS the in-flight prompt row the turn persisted at submit.
            display = [dict(m) for m in self._display]
            return [dict(m) for m in self._display], display

        def close(self):
            return None

    class FakeAgent:
        def __init__(self):
            self.model = "test-model"
            self.session_id = None

    def settled_rows():
        rows = [
            {"role": "user", "content": "q1", "_row_id": 1},
            {"role": "assistant", "content": "a1", "_row_id": 2},
            {"role": "user", "content": "q2", "_row_id": 3},
            {"role": "assistant", "content": "a2", "_row_id": 4},
        ]
        return rows

    # Model history: settled turns only. The in-flight prompt has NOT been committed.
    history = settled_rows()
    # ONE stable db instance (not a fresh one per _get_db()): the in-flight prompt row the
    # submit path persisted at submit must still be in the display projection the branch reads.
    db = ProfileDB()
    db._display = history + [{"role": "user", "content": "IN-FLIGHT q3", "_row_id": 5}]

    parent = {
        "session_key": "parent-key",
        "history": history,
        "display_history_prefix": [],
        "history_lock": threading.Lock(),
        "running": True,
        "cols": 80,
        "profile_home": None,
        "source": "desktop",
        "agent": FakeAgent(),
        "created_at": 1.0,
        "last_active": 1.0,
        "cwd": str(tmp_path),
        "inflight_turn": {"user": "IN-FLIGHT q3", "assistant": "", "streaming": True},
        # Recorded by _lock_in_submit_turn at turn start (len(prefix)+len(history) = 4).
        "branch_turn_boundary": 4,
    }
    server._sessions["parent"] = parent
    # Stable shared handle: profile_home is None so _workdir_owner_db -> _get_db().
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr("hermes_state_registry.acquire", lambda *a, **k: (db, True))
    monkeypatch.setattr(server, "_claim_active_session_slot", lambda *a, **k: (None, None))
    monkeypatch.setattr(server, "_make_agent", lambda *a, **k: FakeAgent())
    monkeypatch.setattr(server, "_set_session_context", lambda *a, **k: {})
    monkeypatch.setattr(server, "_clear_session_context", lambda *a, **k: None)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_session_cwd", lambda s: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda *a, **k: None)
    monkeypatch.setattr(server, "_attach_worker", lambda *a, **k: None)
    return parent


def test_mid_turn_branch_cuts_at_turn_start_boundary(tmp_path, monkeypatch):
    """A branch while the turn runs keeps the settled transcript and never the in-flight prompt."""
    _make_parent(tmp_path, monkeypatch)
    server._idempotency_keys.clear()
    try:
        res = server.handle_request({"id": "m1", "method": "session.branch",
                                     "params": {"session_id": "parent", "idempotency_key": "midturn-1"}})
        assert "result" in res, res
        result = res["result"]
        assert result["mid_turn"] is True
        assert result["message_count"] == 4, result["message_count"]
        texts = [m["text"] or m["content"] for m in result["messages"]]
        assert "IN-FLIGHT q3" not in texts
        assert texts == ["q1", "a1", "q2", "a2"]
        children = [sid for sid in server._sessions if sid != "parent"]
        assert len(children) == 1
    finally:
        for k in list(server._sessions):
            server._sessions.pop(k, None)
        server._idempotency_keys.clear()


def test_mid_turn_branch_count_bound_cut_also_respects_boundary(tmp_path, monkeypatch):
    """count + boundary: the boundary truncates the source first, then count bounds it."""
    _make_parent(tmp_path, monkeypatch)
    server._idempotency_keys.clear()
    try:
        res = server.handle_request({"id": "m2", "method": "session.branch",
                                     "params": {"session_id": "parent", "count": 2,
                                                "idempotency_key": "midturn-2"}})
        assert "result" in res, res
        assert res["result"]["mid_turn"] is True
        assert res["result"]["message_count"] == 2
        assert [m["text"] or m["content"] for m in res["result"]["messages"]] == ["q1", "a1"]
    finally:
        for k in list(server._sessions):
            server._sessions.pop(k, None)
        server._idempotency_keys.clear()


def test_settled_branch_keeps_full_transcript(tmp_path, monkeypatch):
    """No boundary (turn settled) -> whole projection, mid_turn False, q3 is committed content."""
    _make_parent(tmp_path, monkeypatch)
    server._idempotency_keys.clear()
    try:
        parent = server._sessions["parent"]
        # Turn settled: _clear_inflight_turn pops the boundary; q3 + its reply committed.
        parent.pop("branch_turn_boundary", None)
        parent["running"] = False
        parent["inflight_turn"] = None
        parent["history"].extend([
            {"role": "user", "content": "IN-FLIGHT q3", "_row_id": 5},
            {"role": "assistant", "content": "a3", "_row_id": 6},
        ])
        # The DB display (what the branch source reads) has the settled pair too.
        db = server._get_db()
        db._display = [dict(m) for m in parent["history"]]
        monkeypatch.setattr(server, "_get_db", lambda: db)
        monkeypatch.setattr("hermes_state_registry.acquire", lambda *a, **k: db)

        res = server.handle_request({"id": "s1", "method": "session.branch",
                                     "params": {"session_id": "parent", "idempotency_key": "settled-1"}})
        assert "result" in res, res
        result = res["result"]
        assert result["mid_turn"] is False
        assert result["message_count"] == 6
        assert [m["text"] or m["content"] for m in result["messages"]] == ["q1", "a1", "q2", "a2", "IN-FLIGHT q3", "a3"]
    finally:
        for k in list(server._sessions):
            server._sessions.pop(k, None)
        server._idempotency_keys.clear()


def test_idempotent_retry_mid_turn_returns_same_child(tmp_path, monkeypatch):
    """A retried mid-turn branch (same idempotency key) answers the SAME child."""
    _make_parent(tmp_path, monkeypatch)
    server._idempotency_keys.clear()
    try:
        params = {"session_id": "parent", "idempotency_key": "midturn-retry"}
        first = server.handle_request({"id": "r1", "method": "session.branch", "params": dict(params)})
        assert "result" in first, first
        second = server.handle_request({"id": "r2", "method": "session.branch", "params": dict(params)})
        assert "result" in second, second
        assert second["result"]["session_id"] == first["result"]["session_id"]
        assert second["result"]["stored_session_id"] == first["result"]["stored_session_id"]
        assert second["result"]["title"] == first["result"]["title"]
        assert second["result"]["parent"] == first["result"]["parent"]
        # The retry answered the existing child (idle now), so its transcript is the
        # 4 settled rows the fresh branch cut at the boundary — NOT a second copy.
        assert second["result"]["message_count"] == 4
        children = [sid for sid in server._sessions if sid != "parent"]
        assert len(children) == 1
    finally:
        for k in list(server._sessions):
            server._sessions.pop(k, None)
        server._idempotency_keys.clear()


def _make_rowid_parent(tmp_path, monkeypatch, display_rows):
    """A settled parent whose AUTHORITATIVE display has `display_rows` (row-id stamped). Used to
    prove the row boundary cuts where the clicked row is, independent of a stale visible count."""
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
            # Real DB mints row ids on insert; mirror that (start after any existing rows).
            start = getattr(self, "_next_id", 1)
            ids = []
            for m in messages:
                m["_row_id"] = start
                ids.append(start)
                start += 1
            self._next_id = start
            return ids

        def set_session_title(self, key, title, *a, **k):
            return True

        def set_auto_title(self, key, title, *a, **k):
            return True

        def discard_unadmitted_session(self, *a, **k):
            return None

        def get_session(self, key):
            return {"id": key, "cwd": str(tmp_path)}

        def update_session_cwd(self, *a, **k):
            return None

        def get_resume_conversations(self, key):
            rows = [dict(m) for m in display_rows]
            return rows, rows

        def close(self):
            return None

    class FakeAgent:
        def __init__(self):
            self.model = "test-model"
            self.session_id = None

    db = ProfileDB()
    parent = {
        "session_key": "parent-key",
        "history": [dict(m) for m in display_rows],
        "display_history_prefix": [],
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
    server._sessions["parent"] = parent
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr("hermes_state_registry.acquire", lambda *a, **k: (db, True))
    monkeypatch.setattr(server, "_claim_active_session_slot", lambda *a, **k: (None, None))
    monkeypatch.setattr(server, "_make_agent", lambda *a, **k: FakeAgent())
    monkeypatch.setattr(server, "_set_session_context", lambda *a, **k: {})
    monkeypatch.setattr(server, "_clear_session_context", lambda *a, **k: None)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_session_cwd", lambda s: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda *a, **k: None)
    monkeypatch.setattr(server, "_attach_worker", lambda *a, **k: None)
    return parent


def _rows(*texts):
    return [{"role": r, "content": t, "_row_id": i + 1} for i, (r, t) in enumerate(texts)]


def test_row_boundary_wins_over_stale_count(tmp_path, monkeypatch):
    """Defect 2: the client's LOCAL visible count is stale (a compacted projection). The
    authoritative display has 3 user/assistant pairs (6 rows). A `count` derived from the
    2-row local view cuts the child at row 2; the row boundary `through_message_id=4`
    (the clicked bubble's last durable row) must cut at row 4 instead — keeping pair 2."""
    display = _rows(
        ("user", "q1"), ("assistant", "a1"),
        ("user", "q2"), ("assistant", "a2"),
        ("user", "q3"), ("assistant", "a3"),
    )
    _make_rowid_parent(tmp_path, monkeypatch, display)
    server._idempotency_keys.clear()
    try:
        # The buggy path: count alone (== the stale local visible count, 2) -> wrong cut.
        count_only = server.handle_request({"id": "c1", "method": "session.branch",
                                            "params": {"session_id": "parent", "count": 2,
                                                       "idempotency_key": "rowid-count-only"}})
        assert count_only["result"]["message_count"] == 2
        assert [m["text"] for m in count_only["result"]["messages"]] == ["q1", "a1"]

        # The fixed path: count AND the row boundary -> boundary wins, cut lands at row 4.
        fixed = server.handle_request({"id": "c2", "method": "session.branch",
                                       "params": {"session_id": "parent", "count": 2,
                                                  "through_message_id": 4,
                                                  "idempotency_key": "rowid-boundary"}})
        assert fixed["result"]["message_count"] == 4
        assert [m["text"] for m in fixed["result"]["messages"]] == ["q1", "a1", "q2", "a2"]
    finally:
        for k in list(server._sessions):
            server._sessions.pop(k, None)
        server._idempotency_keys.clear()


def test_row_boundary_is_inclusive_of_the_row(tmp_path, monkeypatch):
    """The boundary keeps rows through `through_message_id` (inclusive); the next row stays out."""
    display = _rows(
        ("user", "q1"), ("assistant", "a1"),
        ("user", "q2"), ("assistant", "a2"),
    )
    _make_rowid_parent(tmp_path, monkeypatch, display)
    server._idempotency_keys.clear()
    try:
        res = server.handle_request({"id": "b1", "method": "session.branch",
                                     "params": {"session_id": "parent", "through_message_id": 1,
                                                "idempotency_key": "rowid-first"}})
        assert res["result"]["message_count"] == 1
        assert [m["text"] for m in res["result"]["messages"]] == ["q1"]
    finally:
        for k in list(server._sessions):
            server._sessions.pop(k, None)
        server._idempotency_keys.clear()


def test_row_boundary_lands_between_hidden_rows(tmp_path, monkeypatch):
    """Defect 2 regression: hidden rows (tool rows) sit BETWEEN the settled spine
    messages, and the renderer's local list is a compacted model projection shorter
    than the authoritative display. The authoritative display is

        1 u q1 | 2 t tool | 3 a a1 | 4 u q2 | 5 t tool | 6 a a2
        7 u q3 | 8 t tool | 9 a a3

    whose visible spine is [q1, a1, q2, a2, q3, a3] (6 messages). The user clicks
    "branch" at the bottom of the LAST message of the session they see. If their
    renderer holds the compacted projection (4 spine messages), the legacy
    count-based cut sends `count: 4` and lands the child two messages EARLY (at
    a2) — the "branch at the bottom of the last message ends at an earlier
    message" defect. The row boundary of the clicked bubble is `through_message_id:
    9` (its last durable row), which keeps the full spine regardless of how many
    hidden rows sit between the messages."""
    display = [
        {"role": "user", "content": "q1", "_row_id": 1},
        {"role": "tool", "content": "ran", "_row_id": 2},
        {"role": "assistant", "content": "a1", "_row_id": 3},
        {"role": "user", "content": "q2", "_row_id": 4},
        {"role": "tool", "content": "ran", "_row_id": 5},
        {"role": "assistant", "content": "a2", "_row_id": 6},
        {"role": "user", "content": "q3", "_row_id": 7},
        {"role": "tool", "content": "ran", "_row_id": 8},
        {"role": "assistant", "content": "a3", "_row_id": 9},
    ]
    _make_rowid_parent(tmp_path, monkeypatch, display)
    server._idempotency_keys.clear()
    try:
        # The buggy cut: the compacted local projection's spine count (4) — the
        # child ends at a2, two messages before the clicked last message.
        count_only = server.handle_request({"id": "h1", "method": "session.branch",
                                            "params": {"session_id": "parent", "count": 4,
                                                       "idempotency_key": "hidden-count-only"}})
        assert [m["text"] for m in count_only["result"]["messages"]] == ["q1", "a1", "q2", "a2"]

        # The fixed cut: the clicked bubble's last durable row (9) — the hidden
        # tool rows inside the span are filtered from the visible history, the
        # spine messages are all kept.
        fixed = server.handle_request({"id": "h2", "method": "session.branch",
                                       "params": {"session_id": "parent", "count": 4,
                                                  "through_message_id": 9,
                                                  "idempotency_key": "hidden-boundary"}})
        assert fixed["result"]["message_count"] == 6
        assert [m["text"] for m in fixed["result"]["messages"]] == ["q1", "a1", "q2", "a2", "q3", "a3"]
    finally:
        for k in list(server._sessions):
            server._sessions.pop(k, None)
        server._idempotency_keys.clear()
