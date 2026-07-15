from __future__ import annotations

import copy
from hermes_cli import kanban_db
import tui_gateway.server as srv


def _call(params): return srv._methods["kanban.activity"](1, params)["result"]
def _task(task_id, status, *, outcome=None, children=None):
    return {"task_id": task_id, "title": task_id, "status": status, "assignee": "worker", "block_reason": None, "parents": [], "children": children or [], "run": None if status not in {"running", "done"} else {"run_id": 1, "profile": "worker", "started_at": 10, "ended_at": 20 if status == "done" else None, "outcome": outcome, "last_heartbeat_at": 15, "max_runtime_seconds": 3600}}


def test_empty_implicit_response_omits_uninitialized_board(monkeypatch, tmp_path):
    missing = tmp_path / "missing.db"; monkeypatch.setattr(kanban_db, "activity_board_scope", lambda: None); monkeypatch.setattr(kanban_db, "list_boards", lambda include_archived=False: [{"slug": "default"}]); monkeypatch.setattr(kanban_db, "kanban_db_path", lambda board=None: missing)
    result = _call({}); assert result["boards"] == []; assert result["active_count"] == result["attention_count"] == 0; assert not missing.exists()


def test_implicit_pinned_scope_reads_only_pinned_board(monkeypatch, tmp_path):
    pinned = tmp_path / "alpha.db"; pinned.touch(); monkeypatch.setattr(kanban_db, "activity_board_scope", lambda: "alpha"); monkeypatch.setattr(kanban_db, "kanban_db_path", lambda board=None: pinned); seen = []
    monkeypatch.setattr(kanban_db, "get_activity_snapshot", lambda board: seen.append(board) or {"board": board, "checked_at": 10, "roots": []}); _call({}); assert seen == ["alpha"]


def test_multi_board_canonicalization_and_overflow(monkeypatch):
    snapshots = {"alpha": {"board": "alpha", "checked_at": 10, "roots": [_task("running", "running", children=[_task("blocked", "blocked")])]}, "zeta": {"board": "zeta", "checked_at": 10, "roots": [_task("failed", "done", outcome="stopped")]}}
    monkeypatch.setattr(kanban_db, "get_activity_snapshot", lambda board: copy.deepcopy(snapshots.get(board, {"board": board, "checked_at": 10, "roots": []})))
    result = _call({"boards": ["zeta", "ALPHA", "alpha"]}); assert [b["board"] for b in result["boards"]] == ["alpha", "zeta"]; assert result["active_count"] == 1 and result["attention_count"] == 2
    overflow = _call({"boards": [f"board-{i}" for i in range(srv._KANBAN_ACTIVITY_MAX_BOARDS + 3)]}); assert len(overflow["boards"]) == srv._KANBAN_ACTIVITY_MAX_BOARDS; assert overflow["diagnostics"] == [f"board-limit:{srv._KANBAN_ACTIVITY_MAX_BOARDS}"]


def test_private_backend_error_is_bounded(monkeypatch):
    def unavailable(*, board): raise RuntimeError("private path or database detail")
    monkeypatch.setattr(kanban_db, "get_activity_snapshot", unavailable); result = _call({"boards": ["broken"]}); assert result["boards"][0]["error"] == "board unavailable"; assert "private path" not in repr(result)


def test_rpc_does_not_touch_session_or_delivery_state(monkeypatch):
    sessions, pending = dict(srv._sessions), dict(srv._pending); monkeypatch.setattr(kanban_db, "get_activity_snapshot", lambda board: {"board": board, "checked_at": 10, "roots": [_task("running", "running")]})
    assert _call({"boards": ["default"]})["active_count"] == 1; assert srv._sessions == sessions and srv._pending == pending
