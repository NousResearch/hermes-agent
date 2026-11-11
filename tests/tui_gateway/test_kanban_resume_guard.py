"""Gateway/desktop `session.resume` must refuse a Kanban worker transcript (#68779).

The refusal happens in `_resume_guard`, before ANY history read, live-reuse, lazy watch or
slash-worker build: a resumed Desktop/TUI session would be a write-capable writer with none
of the dispatcher's ownership env, invisible to the Kanban board.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_original_stdout = sys.stdout


@pytest.fixture(autouse=True)
def _restore_stdout():
    yield
    sys.stdout = _original_stdout


@pytest.fixture()
def server():
    # Same import-window pattern as tests/tui_gateway/test_protocol.py: the mocks only need
    # to cover the initial import of tui_gateway.server.
    import tui_gateway.server_requests
    import tui_gateway.transport
    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(get_hermes_home=MagicMock(return_value="/tmp/hermes_test")),
        "hermes_cli.env_loader": MagicMock(),
        "hermes_cli.banner": MagicMock(),
        "hermes_state": MagicMock(),
    }):
        import importlib
        mod = importlib.import_module("tui_gateway.server")

    methods = dict(mod._methods)
    real_stdout = mod._real_stdout
    yield mod
    mod._methods.clear()
    mod._methods.update(methods)
    mod._real_stdout = real_stdout
    for sid in list(mod._sessions):
        mod._close_session_by_id(sid, end_reason="test_cleanup")
    from tui_gateway import server_requests
    server_requests.reset_for_tests()
    mod._live_transports.clear()


class _KanbanDB:
    """Fake profile session store and the storage handle beside it: classifies `worker-1` as
    kanban-owned, `chat-1` not."""

    def __init__(self):
        self.reads = []
        self.sessions = SimpleNamespace(
            get=lambda session_id: SimpleNamespace(as_session_row=lambda: {"id": session_id}),
            get_by_title=lambda _title: None, is_kanban_owned=self._is_kanban_owned,
            retire_undrained_queue_rows=lambda _session_id: 0)
        self.lineage = SimpleNamespace(resolve_resume_target=lambda session_id: session_id,
                                       ancestor_display_prefix=lambda _session_id: ())
        self.messages = SimpleNamespace(assert_resume_safe=self._assert_resume_safe,
                                        read_resume_conversations=self._read_resume_conversations,
                                        read_conversation=self._read_conversation)

    def _is_kanban_owned(self, session_id):
        self.reads.append(("kanban-probe", session_id))
        return session_id == "worker-1"

    def _assert_resume_safe(self, session_id, **_kwargs):
        self.reads.append(("size-guard", session_id))
        return 0

    def _read_resume_conversations(self, session_id):
        self.reads.append(("lineage", session_id))
        return SimpleNamespace(model=(), display=())

    def _read_conversation(self, session_id, _request=None):
        self.reads.append(("tip", session_id))
        return ()


def _resume(server, monkeypatch, db, session_id):
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_persistence_beside", lambda store: store)
    monkeypatch.setattr(server, "_release_persistence", lambda _handle: None)
    return server.handle_request({
        "id": f"r-{session_id}",
        "method": "session.resume",
        "params": {"session_id": session_id, "omit_messages": True},
    })


def test_gateway_resume_refuses_kanban_worker_session(server, monkeypatch):
    """A kanban-owned transcript is refused before any history read or session build."""
    db = _KanbanDB()
    response = _resume(server, monkeypatch, db, "worker-1")

    err = response.get("error") or {}
    assert err.get("code") == 4132, response
    assert "Kanban" in err.get("message", "")
    # No history read ran: the refusal fired before anything touched the transcript.
    assert ("lineage", "worker-1") not in db.reads
    assert ("tip", "worker-1") not in db.reads


def test_gateway_resume_ordinary_session_unchanged(server, monkeypatch):
    """A non-kanban session flows past the guard exactly as before."""
    db = _KanbanDB()
    response = _resume(server, monkeypatch, db, "chat-1")

    assert "error" not in response, response
    # omit_messages reads the tip segment — proof the guard was survived.
    assert ("tip", "chat-1") in db.reads
