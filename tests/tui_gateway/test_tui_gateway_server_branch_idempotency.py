"""TUI gateway ``session.branch`` idempotency: a retried branch returns the same child (#65410)."""

import threading
import time

import pytest

from tui_gateway import server


@pytest.fixture(autouse=True)
def _neuter_agent_prewarm_timer(request, monkeypatch):
    """Stub the deferred agent pre-warm timer (mirrors test_tui_gateway_server.py).

    ``session.create``/``session.branch`` paths may arm a background
    ``threading.Timer`` that calls whatever ``server._make_agent`` is patched in
    AT FIRE TIME; left live it lands in a later test's mock. Tests that exercise
    the deferred build itself opt back in with ``@pytest.mark.real_agent_prewarm``.
    """
    if request.node.get_closest_marker("real_agent_prewarm"):
        yield
        return
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    yield


@pytest.fixture(autouse=True)
def _reap_leaked_notification_pollers():
    """Stop and join notification pollers leaked by each test (mirrors
    test_tui_gateway_server.py): a leaked poller steals events off the
    process-global completion queue while a later test asserts on it."""
    yield
    pollers = [
        (stop, thread)
        for stop, thread in list(server._notification_pollers)
        if thread.is_alive()
    ]
    for stop, _thread in pollers:
        stop.set()
    deadline = time.time() + 3.0
    for _stop, thread in pollers:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        thread.join(timeout=remaining)
    server._notification_pollers[:] = [
        (stop, thread)
        for stop, thread in server._notification_pollers
        if thread.is_alive()
    ]


def test_session_branch_idempotency_key_dedupes_retry(monkeypatch, tmp_path):
    """A retried session.branch with the SAME idempotency_key returns the SAME
    child instead of a duplicate (#65410): a branch whose first response was
    lost must not leave two children behind. The hit answers the SAME result
    shape (title, parent, message_count) without re-copying the transcript."""

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

        def set_session_title(self, key, title):
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

    parent = {
        "session_key": "parent-key",
        "history": [{"role": "user", "content": "hi"}],
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
    monkeypatch.setattr(server, "_session_cwd", lambda s: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda *a, **k: None)
    monkeypatch.setattr(server, "_attach_worker", lambda *a, **k: None)
    server._idempotency_keys.clear()
    try:
        params = {"session_id": "parent", "name": "forked", "idempotency_key": "branch-live-retry-1"}
        first = server.handle_request({"id": "b1", "method": "session.branch", "params": dict(params)})
        assert "result" in first, first
        first_sid = first["result"]["session_id"]
        first_key = first["result"]["stored_session_id"]
        assert first["result"]["title"] == "forked"
        assert first["result"]["parent"] == "parent-key"

        # Client retries after a lost response: same key, same params.
        second = server.handle_request({"id": "b2", "method": "session.branch", "params": dict(params)})
        assert "result" in second, second
        assert second["result"]["session_id"] == first_sid
        assert second["result"]["stored_session_id"] == first_key
        assert second["result"]["title"] == "forked"
        assert second["result"]["parent"] == "parent-key"

        # Only ONE child runtime exists besides the parent.
        children = [sid for sid, s in server._sessions.items() if sid != "parent"]
        assert len(children) == 1
    finally:
        for k in list(server._sessions):
            server._sessions.pop(k, None)
        server._idempotency_keys.clear()
