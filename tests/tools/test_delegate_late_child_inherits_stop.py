"""A stop that lands on an orchestrator subagent mid-fan-out reaches every grandchild it builds afterwards.

``AIAgent.interrupt()`` fans out to a snapshot of ``_active_children``; a child attached after that
snapshot (the orchestrator is still building siblings, or has not reached its next iteration check)
used to start with no signal and run to completion as an orphan. ``_attach_child`` now mirrors a
pending stop onto the newcomer.
"""
from types import SimpleNamespace

from tools.delegate_tool_child_run import _attach_child


class _Child:
    def __init__(self):
        self.stops = []

    def hard_interrupt(self, message=None, *, tool_reason=None):
        self.stops.append(message)


def test_child_attached_after_parent_stop_is_stopped_too():
    parent = SimpleNamespace(_active_children=[], _interrupt_requested=True, _interrupt_message="stop")
    late = _Child()
    _attach_child(parent, late)
    assert late in parent._active_children
    assert late.stops == ["stop"]


def test_child_attached_to_running_parent_is_left_alone():
    parent = SimpleNamespace(_active_children=[], _interrupt_requested=False)
    child = _Child()
    _attach_child(parent, child)
    assert child in parent._active_children and child.stops == []


def test_real_spawn_path_child_starts_interrupted(tmp_path, monkeypatch):
    """Through ``_build_child_agent`` with real AIAgents: the orchestrator is already stopped, so the
    grandchild it builds carries the interrupt before its conversation ever starts."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent
    from tools import delegate_tool as dt
    import tools.delegate_tool_config as dtc
    cfg = {"max_spawn_depth": 3}
    monkeypatch.setattr(dt, "_load_config", lambda: cfg)
    monkeypatch.setattr(dtc, "_load_config", lambda: cfg)
    kw = dict(api_key="k", base_url="https://openrouter.ai/api/v1", provider="openrouter",
              api_mode="chat_completions", model="anthropic/claude-sonnet-4.6", platform="cli", quiet_mode=True,
              skip_context_files=True, skip_memory=True, save_trajectories=False, enabled_toolsets=["file"])
    parent = AIAgent(session_id="p", **kw)
    mid = dt._build_child_agent(task_index=0, goal="mid", context=None, toolsets=["file"], model=None,
                                max_iterations=4, task_count=1, parent_agent=parent)
    try:
        mid.hard_interrupt("stop")
        grandchild = dt._build_child_agent(task_index=0, goal="gc", context=None, toolsets=["file"], model=None,
                                           max_iterations=4, task_count=1, parent_agent=mid)
        try:
            assert grandchild._interrupt_requested is True
        finally:
            grandchild.close()
    finally:
        mid.close()
        parent.close()
