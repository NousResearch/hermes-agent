"""``rollback.for_prompt``: the read-only "checkpoints from this prompt on" RPC.

The happy path against real checkpoints and rows is in tests/agent/test_checkpoint_prompt_link_e2e.py;
this pins the refusals and the shape on a stand-in manager.
"""
import threading
import types

import pytest

from tui_gateway import server


class _Mgr:
    enabled = True
    seen: list = []

    def __init__(self, rows=None, container_reason=None):
        self._rows, self._reason = rows or [], container_reason

    def unsupported_backend_reason(self, task_id="default"):
        return self._reason

    def checkpoints_for_prompts(self, session_id, prompt_uids):
        _Mgr.seen.append((session_id, list(prompt_uids)))
        return self._rows


def _session(mgr):
    return {
        "agent": types.SimpleNamespace(_checkpoint_mgr=mgr, session_id="db-session"),
        "session_key": "db-session", "history": [], "history_lock": threading.Lock(), "history_version": 0,
        "running": False, "attached_images": [], "image_counter": 0, "cols": 80, "slash_worker": None,
        "show_reasoning": False, "tool_progress_mode": "all",
    }


def _call(mgr, params, monkeypatch, uids=("u1", "u2")):
    monkeypatch.setattr("tools.checkpoint_prompts.prompt_uids_from", lambda session_id, row_id, db_path=None: list(uids))
    server._sessions["sid"] = _session(mgr)
    try:
        return server.handle_request({"id": "1", "method": "rollback.for_prompt", "params": {"session_id": "sid", **params}})
    finally:
        server._sessions.pop("sid", None)


ROW = {"workdir": "/w", "hash": "h" * 40, "short_hash": "hhhhhhh", "timestamp": "2026-11-04T10:00:00+00:00",
       "reason": "before write_file", "prompt_uid": "u1", "files_changed_since": 2, "files_changed": 9, "session_id": "db-session"}


def test_returns_the_checkpoints_in_the_contract_shape(monkeypatch):
    _Mgr.seen = []
    resp = _call(_Mgr([dict(ROW)]), {"message_id": 7}, monkeypatch)

    assert resp["result"] == {"enabled": True, "checkpoints": [{
        "workdir": "/w", "hash": "h" * 40, "short_hash": "hhhhhhh", "timestamp": "2026-11-04T10:00:00+00:00",
        "message": "before write_file", "prompt_uid": "u1", "files_changed_since": 2}]}
    # Scoped to the session's own database id and to the picked prompt and the ones after it.
    assert _Mgr.seen == [("db-session", ["u1", "u2"])]


@pytest.mark.parametrize("bad", [None, "7", 7.5, True, [7]])
def test_message_id_must_be_an_integer(monkeypatch, bad):
    resp = _call(_Mgr(), {"message_id": bad}, monkeypatch)

    assert "result" not in resp and resp["error"]["code"] == 4014


def test_an_unknown_or_foreign_row_is_refused_not_guessed(monkeypatch):
    resp = _call(_Mgr([dict(ROW)]), {"message_id": 7}, monkeypatch, uids=())

    assert "result" not in resp and resp["error"]["code"] == 4014


def test_checkpoints_off_answers_disabled_and_empty(monkeypatch):
    mgr = _Mgr([dict(ROW)])
    mgr.enabled = False

    assert _call(mgr, {"message_id": 7}, monkeypatch)["result"] == {"enabled": False, "checkpoints": []}


def test_a_container_backed_session_is_refused_like_rollback_diff(monkeypatch):
    _Mgr.seen = []
    resp = _call(_Mgr([dict(ROW)], container_reason="Checkpoints are not taken for terminal.backend=docker"),
                 {"message_id": 7}, monkeypatch)

    assert "result" not in resp and resp["error"]["code"] == 5023
    assert _Mgr.seen == []  # the host tree is not this session's: nothing was compared
