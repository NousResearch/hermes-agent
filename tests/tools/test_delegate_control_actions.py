"""delegate_task(action=...) — model-facing live orchestration of subagents.

Covers the control plane added to delegate_task: action='list' /
'steer' / 'stop' resolve against the module-level _active_subagents
registry, scoped by the _delegate_parent_ref ownership chain so a
conversation can only control its own spawn tree. Also pins the two
integration contracts: control actions are synchronous (never
backgrounded) and never consume the per-turn subagent spawn cap.
"""

import json
import weakref

from tools.delegate_tool import (
    _handle_control_action,
    _is_descendant_of,
    _register_subagent,
    _unregister_subagent,
    delegate_task,
)


class _StubChild:
    """Weakref-able stand-in for a live child AIAgent."""

    def __init__(self, parent=None, accept_steer: bool = True):
        self.steered: list[str] = []
        self.accept_steer = accept_steer
        self._live_transcript_path = "/tmp/live/task-0.log"
        if parent is not None:
            self._delegate_parent_ref = weakref.ref(parent)

    def steer(self, text: str) -> bool:
        if not self.accept_steer:
            return False
        self.steered.append(text)
        return True


class _StubParent:
    pass


def _register(sid: str, child, **extra) -> None:
    record = {
        "subagent_id": sid,
        "parent_id": None,
        "depth": 0,
        "goal": "test goal",
        "model": "test-model",
        "started_at": 1000.0,
        "status": "running",
        "tool_count": 0,
        "agent": child,
    }
    record.update(extra)
    _register_subagent(record)


# ---------------------------------------------------------------------------
# Ownership chain
# ---------------------------------------------------------------------------


def test_direct_child_is_descendant():
    parent = _StubParent()
    child = _StubChild(parent)
    assert _is_descendant_of(child, parent) is True


def test_grandchild_is_descendant():
    parent = _StubParent()
    mid = _StubChild(parent)
    grandchild = _StubChild(mid)
    assert _is_descendant_of(grandchild, parent) is True


def test_foreign_agent_is_not_descendant():
    parent = _StubParent()
    other_parent = _StubParent()
    foreign = _StubChild(other_parent)
    assert _is_descendant_of(foreign, parent) is False


def test_missing_ref_is_not_descendant():
    parent = _StubParent()
    orphan = _StubChild()  # no parent ref
    assert _is_descendant_of(orphan, parent) is False
    assert _is_descendant_of(None, parent) is False


def test_dead_parent_ref_is_not_descendant():
    parent = _StubParent()
    child = _StubChild(parent)
    del parent
    import gc

    gc.collect()
    assert _is_descendant_of(child, _StubParent()) is False


# ---------------------------------------------------------------------------
# action='list'
# ---------------------------------------------------------------------------


def test_list_shows_only_own_children():
    parent = _StubParent()
    mine = _StubChild(parent)
    foreign = _StubChild(_StubParent())
    _register("sid-ctl-list-1", mine)
    _register("sid-ctl-list-2", foreign)
    try:
        out = json.loads(_handle_control_action("list", None, None, parent))
        assert out["count"] == 1
        entry = out["subagents"][0]
        assert entry["subagent_id"] == "sid-ctl-list-1"
        assert entry["goal"] == "test goal"
        assert entry["accepting_steer"] is True
        assert entry["live_transcript"] == "/tmp/live/task-0.log"
        # Internal fields must not leak
        assert "agent" not in entry
        assert "owner_transport" not in entry
    finally:
        _unregister_subagent("sid-ctl-list-1")
        _unregister_subagent("sid-ctl-list-2")


def test_list_empty_registry_has_note():
    out = json.loads(_handle_control_action("list", None, None, _StubParent()))
    assert out["count"] == 0
    assert "note" in out


# ---------------------------------------------------------------------------
# action='steer'
# ---------------------------------------------------------------------------


def test_steer_reaches_owned_child():
    parent = _StubParent()
    child = _StubChild(parent)
    _register("sid-ctl-steer-1", child)
    try:
        out = json.loads(
            _handle_control_action("steer", "sid-ctl-steer-1", "focus on X", parent)
        )
        assert out["status"] == "queued"
        assert child.steered == ["focus on X"]
    finally:
        _unregister_subagent("sid-ctl-steer-1")


def test_steer_foreign_child_is_refused():
    parent = _StubParent()
    foreign = _StubChild(_StubParent())
    _register("sid-ctl-steer-2", foreign)
    try:
        out = _handle_control_action("steer", "sid-ctl-steer-2", "hijack", parent)
        assert "No live subagent" in out
        assert foreign.steered == []
    finally:
        _unregister_subagent("sid-ctl-steer-2")


def test_steer_requires_message():
    parent = _StubParent()
    child = _StubChild(parent)
    _register("sid-ctl-steer-3", child)
    try:
        out = _handle_control_action("steer", "sid-ctl-steer-3", "   ", parent)
        assert "requires a non-empty 'message'" in out
    finally:
        _unregister_subagent("sid-ctl-steer-3")


def test_steer_requires_subagent_id():
    out = _handle_control_action("steer", "", "text", _StubParent())
    assert "requires subagent_id" in out


def test_steer_closed_acceptance_is_refused():
    parent = _StubParent()
    child = _StubChild(parent)
    _register("sid-ctl-steer-4", child, accepting_steer=False)
    try:
        out = _handle_control_action("steer", "sid-ctl-steer-4", "late", parent)
        assert "no longer accepting" in out
        assert child.steered == []
    finally:
        _unregister_subagent("sid-ctl-steer-4")


# ---------------------------------------------------------------------------
# action='stop'
# ---------------------------------------------------------------------------


def test_stop_interrupts_owned_child(monkeypatch):
    import tools.delegate_tool as dt

    parent = _StubParent()
    child = _StubChild(parent)
    _register("sid-ctl-stop-1", child)
    interrupted = []
    monkeypatch.setattr(
        dt, "request_hard_interrupt", lambda agent, reason: interrupted.append(agent) or True
    )
    try:
        out = json.loads(
            _handle_control_action("stop", "sid-ctl-stop-1", None, parent)
        )
        assert out["status"] == "interrupt_requested"
        assert interrupted == [child]
    finally:
        _unregister_subagent("sid-ctl-stop-1")


def test_stop_foreign_child_is_refused(monkeypatch):
    import tools.delegate_tool as dt

    parent = _StubParent()
    foreign = _StubChild(_StubParent())
    _register("sid-ctl-stop-2", foreign)
    interrupted = []
    monkeypatch.setattr(
        dt, "request_hard_interrupt", lambda agent, reason: interrupted.append(agent) or True
    )
    try:
        out = _handle_control_action("stop", "sid-ctl-stop-2", None, parent)
        assert "No live subagent" in out
        assert interrupted == []
    finally:
        _unregister_subagent("sid-ctl-stop-2")


def test_stop_unknown_id_mentions_completion_path():
    out = _handle_control_action("stop", "sid-gone", None, _StubParent())
    assert "No live subagent" in out
    assert "completion message" in out


# ---------------------------------------------------------------------------
# delegate_task() entrypoint routing
# ---------------------------------------------------------------------------


def test_delegate_task_routes_control_action_before_spawn_machinery():
    """action='list' must return synchronously without touching spawn paths
    (no goal/tasks required, no pause gate, no depth checks)."""
    parent = _StubParent()
    out = json.loads(delegate_task(action="list", parent_agent=parent))
    assert out["action"] == "list"


def test_delegate_task_control_action_bypasses_spawn_pause():
    from tools.delegate_tool import set_spawn_paused

    parent = _StubParent()
    set_spawn_paused(True)
    try:
        out = json.loads(delegate_task(action="list", parent_agent=parent))
        assert out["action"] == "list"
    finally:
        set_spawn_paused(False)


def test_delegate_task_unknown_action_is_an_error():
    out = delegate_task(action="pause", goal="g", parent_agent=_StubParent())
    assert "Unknown action" in out


def test_delegate_task_spawn_action_still_validates_goal():
    out = delegate_task(action="spawn", parent_agent=_StubParent())
    assert "Provide either 'goal'" in out


def test_delegate_task_requires_parent_agent_for_control():
    out = delegate_task(action="list", parent_agent=None)
    assert "requires a parent agent" in out


def test_empty_tasks_array_with_goal_is_single_task_not_batch_error():
    """Small models emit tasks=[] alongside goal; that must not trip the
    'Batch mode requires at least 2 tasks' gate (observed live with
    gpt-5.4-mini on Nous Portal)."""
    out = delegate_task(tasks=[], goal="", parent_agent=_StubParent())
    # Falls through to the single-goal validation, not the batch gate.
    assert "Provide either 'goal'" in out
    assert "at least 2 tasks" not in out


# ---------------------------------------------------------------------------
# Guardrail: control actions never consume the spawn cap
# ---------------------------------------------------------------------------


def test_spawn_count_zero_for_control_actions():
    from agent.tool_guardrails import _subagent_spawn_count

    assert _subagent_spawn_count({"action": "list"}) == 0
    assert _subagent_spawn_count({"action": "steer", "subagent_id": "x"}) == 0
    assert _subagent_spawn_count({"action": "stop", "subagent_id": "x"}) == 0
    # Spawn shapes unchanged
    assert _subagent_spawn_count({"goal": "g"}) == 1
    assert _subagent_spawn_count({"action": "spawn", "goal": "g"}) == 1
    assert _subagent_spawn_count({"tasks": [{"goal": "a"}, {"goal": "b"}]}) == 2


def test_control_action_not_blocked_at_spawn_cap():
    """Once the cap is hit, steer/stop must STILL work — that's when the
    user most needs to rein children in."""
    from agent.tool_guardrails import (
        LoopCapConfig,
        ToolCallGuardrailConfig,
        ToolCallGuardrailController,
    )

    cfg = ToolCallGuardrailConfig(loop_caps=LoopCapConfig(max_subagents=1))
    ctl = ToolCallGuardrailController(cfg)
    # Exhaust the cap with a spawn
    assert ctl.before_call("delegate_task", {"goal": "a"}).action == "allow"
    # A second spawn is blocked
    assert ctl.before_call("delegate_task", {"goal": "b"}).action == "block"
    # Control actions still pass on a fresh controller after cap exhaustion
    ctl2 = ToolCallGuardrailController(cfg)
    assert ctl2.before_call("delegate_task", {"goal": "a"}).action == "allow"
    assert (
        ctl2.before_call(
            "delegate_task", {"action": "stop", "subagent_id": "x"}
        ).action
        == "allow"
    )
    assert (
        ctl2.before_call("delegate_task", {"action": "list"}).action == "allow"
    )
    # And spawns remain blocked afterwards — the control call didn't reset it
    assert ctl2.before_call("delegate_task", {"goal": "c"}).action == "block"
