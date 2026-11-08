"""A delegated child's stored session stays addressable by its subagent_id after the registry forgets it.

Through ``_build_child_agent`` with real AIAgents and a real SessionDB in a temp HERMES_HOME: the
child's row carries the id it was spawned with, and only the conversation it descends from (directly
or through an orchestrator child) resolves it.
"""


def test_spawned_children_resolve_by_subagent_id_only_under_their_conversation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tools import delegate_tool as dt
    import tools.delegate_tool_config as dtc
    from tools.delegate_tool_sessions import stored_child_session
    cfg = {"max_spawn_depth": 3}
    monkeypatch.setattr(dt, "_load_config", lambda: cfg)
    monkeypatch.setattr(dtc, "_load_config", lambda: cfg)
    db = SessionDB(db_path=tmp_path / "state.db")
    kw = dict(api_key="k", base_url="https://openrouter.ai/api/v1", provider="openrouter",
              api_mode="chat_completions", model="anthropic/claude-sonnet-4.6", platform="cli", quiet_mode=True,
              skip_context_files=True, skip_memory=True, save_trajectories=False, enabled_toolsets=["file"])
    parent = AIAgent(session_id="parent", session_db=db, **kw)
    agents = [parent]
    try:
        parent._ensure_db_session()
        mid = dt._build_child_agent(task_index=0, goal="mid", context=None, toolsets=["file"], model=None,
                                    max_iterations=4, task_count=1, parent_agent=parent)
        agents.append(mid)
        worker = dt._build_child_agent(task_index=0, goal="worker", context=None, toolsets=["file"], model=None,
                                       max_iterations=4, task_count=1, parent_agent=mid)
        agents.append(worker)
        for child in (mid, worker):
            child._ensure_db_session()
        assert stored_child_session(db, mid._subagent_id, ["parent"]) == mid.session_id
        assert stored_child_session(db, worker._subagent_id, ["parent"]) == worker.session_id
        db.create_session("elsewhere", source="cli")
        assert stored_child_session(db, mid._subagent_id, ["elsewhere"]) is None
        assert stored_child_session(db, "sa-0-00000000", ["parent"]) is None
    finally:
        for agent in reversed(agents):
            agent.close()
        db.close()


def test_a_child_rotated_past_any_hop_budget_resolves_to_its_newest_segment(tmp_path):
    """Compression rotations count as hops on the walk to the conversation: a long-lived child
    (compression.in_place: false) must still resolve to its newest segment, never an older one."""
    from hermes_state import SessionDB
    from tools.delegate_tool_sessions import stored_child_session
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("parent", source="cli")
        marker = {"_delegate_from": "parent", "_subagent_id": "sa-0-long"}
        db.create_session("seg-000", source="subagent", parent_session_id="parent", model_config=marker)
        for i in range(1, 80):
            db.end_session(f"seg-{i - 1:03d}", "compression")
            db.create_session(f"seg-{i:03d}", source="subagent", parent_session_id=f"seg-{i - 1:03d}",
                              model_config=marker)
        reads, get_session = [], db.get_session
        db.get_session = lambda sid: reads.append(sid) or get_session(sid)
        assert stored_child_session(db, "sa-0-long", ["parent"]) == "seg-079"
        # The 80 segments share their chain: one lookup reads each row once, not a walk per segment.
        assert len(reads) == len(set(reads)) == 81, len(reads)
    finally:
        db.close()


def test_a_finished_child_resolves_to_its_continuation_after_a_backward_clock_step(tmp_path):
    """The terminal segment follows verified compression links, not ``started_at``: a wall clock that
    stepped back between the rotation's two rows must not hand back the root."""
    from hermes_state import SessionDB
    from tools.delegate_tool_sessions import stored_child_session
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("parent", source="cli")
        marker = {"_delegate_from": "parent", "_subagent_id": "sa-0-clock"}
        db.create_session("clock-root", source="subagent", parent_session_id="parent", model_config=marker)
        db.end_session("clock-root", "compression")
        db.create_session("clock-tip", source="subagent", parent_session_id="clock-root", model_config=marker)
        db._execute_write(lambda conn: conn.executemany(
            "UPDATE sessions SET started_at = ? WHERE id = ?", [(1002.0, "clock-root"), (1001.0, "clock-tip")]))
        assert stored_child_session(db, "sa-0-clock", ["parent"]) == "clock-tip"
    finally:
        db.close()
