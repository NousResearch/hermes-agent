"""A failed session.branch transcript copy must not leave a partial child row behind.

Regression companion to #107562: that merge rolled back partial *seeded* copies and states
the same compensation "applies to branch children", but the interactive session.branch call
site constructs the child without it. A copy failure after the first committed chunk (real
append_messages_batch transactions, chunk_rows=500) therefore leaves a durable child row
holding a truncated transcript even though the branch RPC returned an error.
Injects the failure at the real production seam;
no source reading, synthetic data, temp HERMES_HOME only.
"""

import threading

import pytest

from hermes_state import SessionDB


class _InjectedCopyError(RuntimeError):
    pass


class _FailingBatchDB(SessionDB):
    """Real SessionDB failing before copying or after one committed 500-row chunk.
    Subclassing (not wrapping) is required: chunk_rows recursion calls
    ``self.append_messages_batch``, so the override must sit ON the instance the
    production code holds."""

    def __init__(self, db_path=None, **kwargs):
        super().__init__(db_path=db_path, **kwargs)
        self.fail_after_rows = None
        self.rows_at_failure = None

    def append_messages_batch(self, session_id, messages, **kwargs):
        # Only inject at a leaf transaction, after parent seeding has finished.
        if self.fail_after_rows is not None and kwargs.get("chunk_rows") is None:
            committed_rows = self.message_count(session_id)
            if committed_rows == self.fail_after_rows:
                self.rows_at_failure = committed_rows
                raise _InjectedCopyError("injected branch copy failure")
        return super().append_messages_batch(session_id, messages, **kwargs)


@pytest.mark.parametrize("committed_rows", [0, 500])
def test_session_branch_cleans_up_partial_copy_on_failure(monkeypatch, tmp_path, committed_rows):
    pytest.importorskip("tui_gateway")
    monkeypatch.setattr("hermes_cli.banner.prefetch_update_check", lambda: None)
    from tui_gateway import server

    profile_home = tmp_path / "profiles" / "mlperf"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = _FailingBatchDB(tmp_path / "state.db")

    parent = "parent-key"
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"synthetic-msg-{i}"}
               for i in range(1200)]
    db.create_session(parent, source="tui")
    db.append_messages_batch(parent, history)
    assert db.message_count(parent) == 1200
    parent_messages = db.get_messages(parent)
    db.fail_after_rows = committed_rows

    class FakeAgent:
        def __init__(self):
            self.model = "test-model"
            self.session_id = None

    parent_record = {
        "session_key": parent,
        "history": [],
        "history_lock": threading.Lock(),
        "running": False,
        "cols": 80,
        "profile_home": str(profile_home),
        "source": "tui",
        "agent": FakeAgent(),
        "created_at": 1.0,
        "last_active": 1.0,
        "cwd": str(tmp_path),
    }
    monkeypatch.setattr(server, "_sessions", {"parent": parent_record})
    monkeypatch.setattr("hermes_state_registry.acquire", lambda *a, **k: db)
    monkeypatch.setattr(server, "_claim_active_session_slot", lambda *a, **k: (None, None))
    monkeypatch.setattr(server, "_make_agent", lambda *a, **k: FakeAgent())
    monkeypatch.setattr(server, "_set_session_context", lambda *a, **k: {})
    monkeypatch.setattr(server, "_clear_session_context", lambda *a, **k: None)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_session_cwd", lambda s: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda *a, **k: None)
    monkeypatch.setattr(server, "_attach_worker", lambda *a, **k: None)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda *a, **k: None)
    monkeypatch.setattr(server, "_project_info_for_cwd", lambda *a, **k: None)
    try:
        response = server.handle_request(
            {"id": "1", "method": "session.branch", "params": {"session_id": "parent", "name": "forked"}})
        assert db.rows_at_failure == committed_rows, "copy failure seam was not reached"
        assert db.get_messages(parent) == parent_messages
        # The copy failed, so the branch must fail — but must not leave a durable partial child.
        child_rows = [row for row in (db.list_sessions_rich() or [])
                      if row.get("parent_session_id") == parent]
        assert not child_rows, (
            f"partial branch child survived copy failure: "
            f"{[(r.get('id'), db.message_count(r.get('id'))) for r in child_rows]}")
        assert "error" in response or response.get("result", {}).get("ok") is False, response
    finally:
        for k in list(server._sessions):
            server._sessions.pop(k, None)
        db.close()
