"""Usage counts this conversation's unfinished completion units, not global children."""
from types import SimpleNamespace

from hermes_state import SessionDB
from tools import async_delegation as ad
from tui_gateway.server import _get_usage, _session_usage_snapshot


def test_usage_counts_owned_completion_units(monkeypatch):
    records = {
        "group": {"status": "running", "parent_session_id": "a", "is_batch": True,
                  "goals": ["one", "two", "three"], "task_indexes": [0, 1, 2]},
        "other": {"status": "running", "parent_session_id": "b"},
        "other2": {"status": "finalizing", "parent_session_id": "b"},
    }
    monkeypatch.setattr(ad, "_records", records)
    a, b, empty = [SimpleNamespace(session_id=sid) for sid in ("a", "b", "empty")]
    assert [_get_usage(agent)["active_subagents"] for agent in (a, b, empty, None)] == [1, 2, 0, 0]
    assert [_session_usage_snapshot({"agent": agent})["active_subagents"]
            for agent in (a, b, empty)] == [1, 2, 0]
    assert ad.active_count() == 3  # Global completion/capacity consumers keep their contract.
    assert ad.active_task_count() == 5  # A group is not a count of individual live children.
    for status in ("running", "stalling", "finalizing"):
        records["group"]["status"] = status
        assert _get_usage(a)["active_subagents"] == 1
    for status in ("completed", "error", "interrupted", "stalled"):
        records["group"]["status"] = status
        assert _get_usage(a)["active_subagents"] == 0
        assert _get_usage(b)["active_subagents"] == 2


def test_usage_follows_compression_but_not_reused_tab(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "sessions.db")
    try:
        db.create_session(session_id="old", source="tui", model="test")
        db.create_session(session_id="tip", source="tui", model="test", parent_session_id="old")
        db.end_session("old", "compression")
        db.create_session(session_id="new", source="tui", model="test")
        records = {
            "before": {"status": "running", "parent_session_id": "old", "origin_ui_session_id": "tab"},
            "after": {"status": "running", "parent_session_id": "tip", "origin_ui_session_id": "tab"},
            "unowned": {"status": "running", "origin_ui_session_id": "tab"},
        }
        monkeypatch.setattr(ad, "_records", records)
        assert db.resolve_resume_session_id("old") == "tip"
        # Rotation and a rebuilt/resumed parent both use the durable compression lineage.
        for sid in ("old", "tip"):
            agent = SimpleNamespace(session_id=sid, _session_db=db)
            assert _get_usage(agent)["active_subagents"] == 2
        assert _get_usage(SimpleNamespace(session_id="new", _session_db=db))["active_subagents"] == 0
        # Missing DB retains exact-owner matching, never falls back to the shared tab.
        assert _get_usage(SimpleNamespace(session_id="tip"))["active_subagents"] == 1
        def unavailable(sid):
            raise RuntimeError("database unavailable")

        monkeypatch.setattr(db, "resolve_resume_session_id", unavailable)
        assert _get_usage(SimpleNamespace(session_id="tip", _session_db=db))["active_subagents"] == 1
    finally:
        db.close()
