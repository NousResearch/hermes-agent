"""delegate_task action='list' — read-only visibility of sibling-thread subagents.

Regression for the group-room double-dispatch (#135864): a fresh thread's session got
``count: 0`` plus a definitive "there is nothing to steer or stop" note for subagents
spawned by a sibling thread of the same bot (same profile, same process), concluded the
earlier delegation died, and re-dispatched the same long-running tasks into the same
worktree. The ownership scoping that (correctly) hides sibling children from steer/stop
must not make ``list`` deny that the work is already in flight.
"""

import json
from types import SimpleNamespace

from tools.delegate_tool import _handle_control_action, _register_subagent, _unregister_subagent


class _StubParentWithHome:
    """Caller session of one profile home: ``_session_db.db_path`` pins the home
    (the same parent-owned state _parent_live_home reads)."""

    def __init__(self, session_id: str, home):
        self.session_id = session_id
        self._session_db = SimpleNamespace(
            db_path=f"{home}/state.db",
            resolve_resume_session_id=lambda sid: sid,
        )


def _register_sibling(sid: str, home, *, goal: str = "trim the cursor", owner_sid: str = "sess-thread-a") -> None:
    _register_subagent({
        "subagent_id": sid,
        "parent_id": None,
        "depth": 0,
        "goal": goal,
        "model": "test-model",
        "started_at": 1000.0,
        "status": "running",
        "tool_count": 0,
        "agent": None,
        "owner_agent_session_id": owner_sid,
        "owner_home": str(home),
    })


def test_list_reports_sibling_thread_subagents_read_only(tmp_path):
    """A fresh thread in the same profile sees its bot's sibling-thread children as
    read-only summaries — not as a flat 'nothing is running'."""
    _register_sibling("sid-sib-list-1", tmp_path)
    try:
        caller = _StubParentWithHome("sess-thread-b", tmp_path)
        out = json.loads(_handle_control_action("list", None, None, caller))
        assert out["count"] == 0
        assert out["subagents"] == []
        siblings = out["subagents_in_other_sessions"]
        assert len(siblings) == 1
        entry = siblings[0]
        assert entry["goal"] == "trim the cursor"
        assert entry["owner_agent_session_id"] == "sess-thread-a"
        assert entry["status"] == "running"
        assert entry["controllable"] is False
        assert isinstance(entry["running_seconds"], float)
        # No id and no private fields: the model must not try to steer/stop it.
        assert "subagent_id" not in entry
        assert "agent" not in entry
        assert "other sessions of this profile" in out["note"]
        assert "do not re-dispatch" in out["note"]
    finally:
        _unregister_subagent("sid-sib-list-1")


def test_sibling_visibility_stays_within_profile_home(tmp_path):
    """A record from a DIFFERENT profile home must stay invisible — same-process
    multi-profile workers must not leak each other's goals."""
    _register_sibling("sid-sib-foreign-1", tmp_path / "other-home", goal="SECRET")
    try:
        caller = _StubParentWithHome("sess-thread-b", tmp_path)
        out = json.loads(_handle_control_action("list", None, None, caller))
        assert "subagents_in_other_sessions" not in out
        assert "SECRET" not in json.dumps(out)
        assert "spawn tree" in out["note"]
    finally:
        _unregister_subagent("sid-sib-foreign-1")


def test_owned_children_stay_out_of_sibling_field(tmp_path):
    """Records this conversation owns land only in the main list; the sibling field
    carries the OTHER sessions' work."""
    _register_sibling("sid-sib-owned-1", tmp_path, owner_sid="sess-thread-b")
    _register_sibling("sid-sib-owned-2", tmp_path)
    try:
        caller = _StubParentWithHome("sess-thread-b", tmp_path)
        out = json.loads(_handle_control_action("list", None, None, caller))
        assert out["count"] == 1
        assert out["subagents"][0]["subagent_id"] == "sid-sib-owned-1"
        assert [e["goal"] for e in out["subagents_in_other_sessions"]] == ["trim the cursor"]
        assert "note" not in out
    finally:
        _unregister_subagent("sid-sib-owned-1")
        _unregister_subagent("sid-sib-owned-2")


def test_caller_without_home_gets_no_siblings(tmp_path):
    """No usable SessionDB on the caller (or on old in-flight records) -> no sibling
    field: profile scoping fails closed rather than showing everything."""
    _register_sibling("sid-sib-nohome-1", tmp_path)
    try:
        caller = SimpleNamespace(session_id="sess-thread-b")  # no _session_db
        out = json.loads(_handle_control_action("list", None, None, caller))
        assert "subagents_in_other_sessions" not in out
    finally:
        _unregister_subagent("sid-sib-nohome-1")


def test_register_child_stamps_owner_home(tmp_path):
    """The spawn side records the owning conversation's profile home from
    parent-owned state (never ambient HERMES_HOME — see #91996)."""
    from tools.delegate_tool_child_run import _register_child
    from tools.delegate_tool_registry import _active_subagents

    child = SimpleNamespace(_subagent_id="sa-0-homestamp01", _delegate_depth=1)
    parent = _StubParentWithHome("sess-stamp-1", tmp_path)
    sid = _register_child(
        child, parent, "port the widget",
        owner_session_id="sess-stamp-1", owner_transport=object(), owner_session_record=object(),
    )
    assert sid == "sa-0-homestamp01"
    try:
        assert _active_subagents[sid]["owner_home"] == str(tmp_path)
    finally:
        _unregister_subagent(sid)
