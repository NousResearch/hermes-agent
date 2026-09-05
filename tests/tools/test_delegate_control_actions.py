"""delegate_task(action=...) — model-facing live orchestration of subagents.

Covers the control plane added to delegate_task: action='list' /
'inspect' / 'steer' / 'stop' resolve against the module-level _active_subagents
registry, scoped by the _delegate_parent_ref ownership chain so a
conversation can only control its own spawn tree. Also pins the two
integration contracts: control actions are synchronous (never
backgrounded) and never consume the per-turn subagent spawn cap.
"""

import json
import time
import weakref

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _SUBAGENT_INSPECT_EVENT_LIMIT,
    _handle_control_action,
    _is_descendant_of,
    _owns_subagent_record,
    _register_subagent,
    _unregister_subagent,
    _wrap_subagent_inspect_callback,
    delegate_task,
    get_subagent_attribution,
)


class _StubChild:
    """Weakref-able stand-in for a live child AIAgent."""

    def __init__(self, parent=None, accept_steer: bool = True):
        self.steered: list[str] = []
        self.accept_steer = accept_steer
        self._live_transcript_path = "/tmp/live/task-0.log"
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.activity = {
            "current_tool": None,
            "api_call_count": 0,
            "max_iterations": 50,
            "last_activity_ts": time.time(),
        }
        if parent is not None:
            self._delegate_parent_ref = weakref.ref(parent)

    def get_activity_summary(self):
        return dict(self.activity)

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
# action='inspect'
# ---------------------------------------------------------------------------


def test_inspect_returns_live_activity_usage_and_sanitized_tool_evidence():
    parent = _StubParent()
    child = _StubChild(parent)
    child.activity.update(
        {"current_tool": "terminal", "api_call_count": 6, "last_activity_ts": time.time() - 4.0}
    )
    child.session_prompt_tokens = 1234
    child.session_completion_tokens = 321
    child.session_estimated_cost_usd = 0.0123456
    sid = "sid-ctl-inspect-1"
    _register(sid, child, tool_count=8, last_tool="read_file")
    cb = _wrap_subagent_inspect_callback(None, sid)
    try:
        cb("tool.started", "read_file", args=json.dumps({"path": "gateway/session.py", "content": "private"}))
        cb("tool.completed", "read_file", duration=0.25, is_error=False)
        out = json.loads(_handle_control_action("inspect", sid, None, parent))
        assert out["action"] == "inspect"
        assert out["activity"]["current_tool"] == "terminal"
        assert out["activity"]["tool_count"] == 9
        assert out["activity"]["last_tool"] == "read_file"
        assert out["activity"]["api_calls"] == 6
        assert out["usage"] == {
            "input_tokens": 1234,
            "output_tokens": 321,
            "estimated_cost_usd": 0.012346,
        }
        assert [e["type"] for e in out["recent_events"]] == ["tool_started", "tool_completed"]
        assert out["recent_events"][0]["tool_input"]["targets"]["path"] == "gateway/session.py"
        assert "private" not in json.dumps(out)
    finally:
        _unregister_subagent(sid)


def test_inspect_redacts_credentials_raw_result_and_thinking_text():
    parent = _StubParent()
    child = _StubChild(parent)
    sid = "sid-ctl-inspect-redaction"
    _register(sid, child)
    cb = _wrap_subagent_inspect_callback(None, sid)
    try:
        cb(
            "tool.started",
            "web_fetch",
            args=json.dumps(
                {
                    "url": "https://user:secret@example.com/path?token=abc#frag",
                    "body": "supersecret-body",
                }
            ),
        )
        cb("_thinking", "PRIVATE CHAIN OF THOUGHT")
        cb("tool.completed", "web_fetch", duration=1.5, is_error=True, result="RAW-SECRET-RESULT")
        out = json.loads(_handle_control_action("inspect", sid, None, parent))
        serialized = json.dumps(out)
        for secret in ("user:secret", "token=abc", "/path", "supersecret-body", "PRIVATE CHAIN", "RAW-SECRET"):
            assert secret not in serialized
        assert out["recent_events"][0]["tool_input"]["targets"]["url"] == "https://example.com"
        assert out["recent_events"][1]["status"] == "error"
    finally:
        _unregister_subagent(sid)


def test_inspect_fails_closed_on_slack_and_telegram_path_secrets():
    parent = _StubParent()
    child = _StubChild(parent)
    sid = "sid-ctl-inspect-path-secrets"
    _register(sid, child)
    cb = _wrap_subagent_inspect_callback(None, sid)
    urls = [
        "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX",
        "https://api.telegram.org/bot123456789:AAExampleSecret/sendMessage",
    ]
    try:
        for url in urls:
            cb("tool.started", "http_request", args=json.dumps({"endpoint": url}))
        out = json.loads(_handle_control_action("inspect", sid, None, parent))
        serialized = json.dumps(out)
        for secret in ("/services/", "XXXXXXXXXXXXXXXXXXXXXXXX", "bot123456789:AAExampleSecret", "sendMessage"):
            assert secret not in serialized
        assert out["recent_events"][0]["tool_input"]["targets"]["endpoint"] == "https://hooks.slack.com"
        assert out["recent_events"][1]["tool_input"]["targets"]["endpoint"] == "https://api.telegram.org"
    finally:
        _unregister_subagent(sid)


def test_inspect_url_path_allowlist_ships_empty():
    from tools import delegate_inspect

    assert delegate_inspect._TOOL_INPUT_PATH_PRESERVING_HOSTS == frozenset()


def test_inspect_event_ring_is_bounded_and_tool_count_is_total():
    parent = _StubParent()
    child = _StubChild(parent)
    sid = "sid-ctl-inspect-ring"
    _register(sid, child)
    cb = _wrap_subagent_inspect_callback(None, sid)
    try:
        total = _SUBAGENT_INSPECT_EVENT_LIMIT + 3
        for i in range(total):
            cb("tool.started", f"tool-{i}", args="{}")
        out = json.loads(_handle_control_action("inspect", sid, None, parent))
        assert len(out["recent_events"]) == _SUBAGENT_INSPECT_EVENT_LIMIT
        assert out["recent_events"][0]["tool"] == "tool-3"
        assert out["recent_events"][-1]["tool"] == f"tool-{total - 1}"
        assert out["activity"]["tool_count"] == total
    finally:
        _unregister_subagent(sid)


def test_inspect_capture_failure_degrades_without_breaking_inner_callback(monkeypatch):
    from tools import delegate_inspect

    parent = _StubParent()
    child = _StubChild(parent)
    sid = "sid-ctl-inspect-degraded"
    _register(sid, child)
    forwarded = []

    def _inner(*args, **kwargs):
        forwarded.append((args, kwargs))

    def _boom(_arguments):
        raise RuntimeError("secret-bearing exception")

    monkeypatch.setattr(delegate_inspect, "summarize_tool_arguments", _boom)
    cb = _wrap_subagent_inspect_callback(_inner, sid)
    try:
        cb("tool.started", "terminal", args='{"command":"secret"}')
        assert len(forwarded) == 1
        out = json.loads(_handle_control_action("inspect", sid, None, parent))
        assert out["activity"]["tool_count"] == 1
        assert out["telemetry"]["capture_degraded"] is True
        assert out["telemetry"]["capture_errors"] == 1
        assert "secret-bearing exception" not in json.dumps(out)
    finally:
        _unregister_subagent(sid)


def test_inspect_activity_failure_fails_closed():
    parent = _StubParent()
    child = _StubChild(parent)
    sid = "sid-ctl-inspect-activity-fail"

    def _boom():
        raise RuntimeError("activity source unavailable")

    child.get_activity_summary = _boom
    _register(sid, child)
    try:
        out = _handle_control_action("inspect", sid, None, parent)
        assert "activity telemetry is unavailable" in out
        assert '"action": "inspect"' not in out
    finally:
        _unregister_subagent(sid)


def test_inspect_samples_activity_outside_registry_lock():
    import tools.delegate_tool_registry as registry

    parent = _StubParent()
    child = _StubChild(parent)
    sid = "sid-ctl-inspect-lock"

    def _activity():
        acquired = registry._active_subagents_lock.acquire(blocking=False)
        assert acquired, "inspect sampled activity while holding registry lock"
        registry._active_subagents_lock.release()
        return dict(child.activity)

    child.get_activity_summary = _activity
    _register(sid, child)
    try:
        assert json.loads(_handle_control_action("inspect", sid, None, parent))["action"] == "inspect"
    finally:
        _unregister_subagent(sid)


def test_inspect_revalidates_target_after_sampling():
    parent = _StubParent()
    child = _StubChild(parent)
    sid = "sid-ctl-inspect-race"

    def _activity():
        _unregister_subagent(sid, agent=child)
        return dict(child.activity)

    child.get_activity_summary = _activity
    _register(sid, child)
    assert "No live subagent" in _handle_control_action("inspect", sid, None, parent)


def test_inspect_foreign_and_finished_children_fail_closed():
    parent = _StubParent()
    foreign = _StubChild(_StubParent())
    _register("sid-ctl-inspect-foreign", foreign)
    try:
        assert "No live subagent" in _handle_control_action("inspect", "sid-ctl-inspect-foreign", None, parent)
    finally:
        _unregister_subagent("sid-ctl-inspect-foreign")

    child = _StubChild(parent)
    _register("sid-ctl-inspect-finished", child)
    _unregister_subagent("sid-ctl-inspect-finished")
    assert "No live subagent" in _handle_control_action("inspect", "sid-ctl-inspect-finished", None, parent)


def test_list_snapshot_hides_private_inspect_state():
    from tools.delegate_tool import list_active_subagents

    parent = _StubParent()
    child = _StubChild(parent)
    sid = "sid-ctl-inspect-private"
    _register(sid, child, _inspect_events=[{"type": "tool_started", "tool": "x"}], _inspect_capture_errors=2)
    try:
        row = next(r for r in list_active_subagents() if r["subagent_id"] == sid)
        assert "_inspect_events" not in row
        assert "_inspect_capture_errors" not in row
    finally:
        _unregister_subagent(sid)


def test_inspect_normalizes_nonfinite_numeric_telemetry():
    parent = _StubParent()
    child = _StubChild(parent)
    child.activity.update(
        {"last_activity_ts": float("inf"), "api_call_count": float("inf"), "max_iterations": float("nan")}
    )
    child.session_estimated_cost_usd = float("inf")
    child.session_prompt_tokens = float("nan")
    sid = "sid-ctl-inspect-nonfinite"
    _register(sid, child, started_at=float("-inf"))
    cb = _wrap_subagent_inspect_callback(None, sid)
    try:
        cb("tool.completed", "terminal", duration=float("inf"), is_error=False)
        raw = _handle_control_action("inspect", sid, None, parent)
        assert "Infinity" not in raw and "NaN" not in raw
        out = json.loads(raw)
        assert out["running_seconds"] is None
        assert out["activity"]["seconds_since_activity"] is None
        assert out["activity"]["api_calls"] == 0
        assert out["usage"]["input_tokens"] == 0
        assert out["usage"]["estimated_cost_usd"] == 0.0
    finally:
        _unregister_subagent(sid)


def test_delegate_task_schema_exposes_inspect_action():
    actions = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["action"]["enum"]
    assert "inspect" in actions


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
    import tools.delegate_tool_registry as dt

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
    import tools.delegate_tool_registry as dt

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


def test_delegate_task_routes_inspect_before_spawn_machinery():
    parent = _StubParent()
    child = _StubChild(parent)
    sid = "sid-ctl-inspect-routing"
    _register(sid, child)
    try:
        out = json.loads(delegate_task(action="inspect", subagent_id=sid, parent_agent=parent))
        assert out["action"] == "inspect"
    finally:
        _unregister_subagent(sid)


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
    assert "No tasks provided" in out
    assert "one-entry" in out  # teaching error carries the canonical shape


def test_delegate_task_requires_parent_agent_for_control():
    out = delegate_task(action="list", parent_agent=None)
    assert "requires a parent agent" in out


def test_empty_tasks_array_with_goal_is_single_task_not_batch_error():
    """Small models emit tasks=[] alongside goal; that must not trip a
    batch-count gate (observed live with gpt-5.4-mini on Nous Portal) —
    it falls through to the no-tasks teaching error."""
    out = delegate_task(tasks=[], goal="", parent_agent=_StubParent())
    assert "No tasks provided" in out
    assert "at least 2 tasks" not in out


# ---------------------------------------------------------------------------
# Durable ownership: registry survives parent-agent object rebuilds
# (regression for deleg_88454b70 / sa-0-dc0100f4, 2026-08-17: CLI rebuilt its
# AIAgent mid-session; running child fell out of list/inspect/steer while completion
# delivery — which routes by durable session id — still worked)
# ---------------------------------------------------------------------------


class _StubParentWithSession:
    def __init__(self, session_id: str, session_db=None):
        self.session_id = session_id
        self._session_db = session_db


class _StubSessionDB:
    """resolve_resume_session_id lineage: old_id -> tip mapping."""

    def __init__(self, lineage=None):
        self._lineage = dict(lineage or {})

    def resolve_resume_session_id(self, session_id):
        return self._lineage.get(session_id, session_id)


def test_list_finds_child_after_parent_agent_rebuild():
    """A REBUILT parent object (weakref chain broken) must still see its
    running child via the durable owner_agent_session_id spine."""
    old_parent = _StubParentWithSession("sess-durable-1")
    child = _StubChild(old_parent)
    _register(
        "sid-durable-list-1", child, owner_agent_session_id="sess-durable-1"
    )
    # Simulate the CLI's `self.agent = None` + rebuild: a NEW object, same
    # durable conversation/session id. The weakref chain no longer reaches it.
    rebuilt_parent = _StubParentWithSession("sess-durable-1")
    assert _is_descendant_of(child, rebuilt_parent) is False  # identity broken
    try:
        out = json.loads(_handle_control_action("list", None, None, rebuilt_parent))
        assert out["count"] == 1
        assert out["subagents"][0]["subagent_id"] == "sid-durable-list-1"
    finally:
        _unregister_subagent("sid-durable-list-1")


def test_steer_resolves_after_parent_agent_rebuild():
    old_parent = _StubParentWithSession("sess-durable-2")
    child = _StubChild(old_parent)
    _register(
        "sid-durable-steer-1", child, owner_agent_session_id="sess-durable-2"
    )
    rebuilt_parent = _StubParentWithSession("sess-durable-2")
    try:
        out = json.loads(
            _handle_control_action(
                "steer", "sid-durable-steer-1", "keep going", rebuilt_parent
            )
        )
        assert out["status"] == "queued"
        assert child.steered == ["keep going"]
    finally:
        _unregister_subagent("sid-durable-steer-1")


def test_durable_ownership_does_not_leak_to_foreign_session():
    """A DIFFERENT conversation (different session id, no identity chain)
    must still be refused — the durable spine widens recovery, not access."""
    owner = _StubParentWithSession("sess-durable-3")
    child = _StubChild(owner)
    _register(
        "sid-durable-foreign-1", child, owner_agent_session_id="sess-durable-3"
    )
    intruder = _StubParentWithSession("sess-other-99")
    try:
        out = _handle_control_action(
            "steer", "sid-durable-foreign-1", "hijack", intruder
        )
        assert "No live subagent" in out
        assert child.steered == []
        out2 = json.loads(_handle_control_action("list", None, None, intruder))
        assert out2["count"] == 0
    finally:
        _unregister_subagent("sid-durable-foreign-1")


def test_durable_ownership_resolves_compression_lineage():
    """Delegation registered under a pre-compression session id must match
    the rotated parent whose SessionDB lineage maps old -> new."""
    db = _StubSessionDB({"sess-old-tip": "sess-new-tip"})
    child = _StubChild()  # no identity chain at all
    _register(
        "sid-durable-lineage-1", child, owner_agent_session_id="sess-old-tip"
    )
    rotated_parent = _StubParentWithSession("sess-new-tip", session_db=db)
    try:
        out = json.loads(
            _handle_control_action("list", None, None, rotated_parent)
        )
        assert out["count"] == 1
    finally:
        _unregister_subagent("sid-durable-lineage-1")


def test_owns_subagent_record_requires_some_spine():
    """No identity chain AND no durable owner id -> not owned (fail closed)."""
    child = _StubChild()
    record = {"subagent_id": "x", "agent": child}
    assert _owns_subagent_record(record, _StubParentWithSession("sess-a")) is False
    # And a record with an owner id but a parent with no session_id fails too.
    record2 = {"subagent_id": "y", "agent": child, "owner_agent_session_id": "s1"}
    assert _owns_subagent_record(record2, _StubParent()) is False


# ---------------------------------------------------------------------------
# Process-notification attribution: get_subagent_attribution
# ---------------------------------------------------------------------------


def test_attribution_resolves_live_child():
    parent = _StubParentWithSession("sess-attr-1")
    child = _StubChild(parent)
    _register(
        "sa-0-attr0001",
        child,
        delegation_id="deleg_attr_1",
        owner_agent_session_id="sess-attr-1",
    )
    try:
        info = get_subagent_attribution("sa-0-attr0001")
        assert info is not None
        assert info["subagent_id"] == "sa-0-attr0001"
        assert info["goal"] == "test goal"
        assert info["delegation_id"] == "deleg_attr_1"
    finally:
        _unregister_subagent("sa-0-attr0001")


def test_attribution_survives_child_completion():
    """After the child finishes (unregistered), attribution must still
    resolve — its background processes can outlive it."""
    parent = _StubParentWithSession("sess-attr-2")
    child = _StubChild(parent)
    _register(
        "sa-0-attr0002",
        child,
        delegation_id="deleg_attr_2",
        owner_agent_session_id="sess-attr-2",
    )
    _unregister_subagent("sa-0-attr0002")
    info = get_subagent_attribution("sa-0-attr0002")
    assert info is not None
    assert info["delegation_id"] == "deleg_attr_2"
    assert info["goal"] == "test goal"


def test_attribution_unknown_task_id_is_none():
    assert get_subagent_attribution("proc-not-a-subagent") is None
    assert get_subagent_attribution("") is None
    assert get_subagent_attribution(None) is None


def test_completion_notification_carries_delegation_attribution(monkeypatch):
    """format_process_notification on a child-started process completion must
    name the subagent + delegation instead of an anonymous output wall.

    Runs with delegation.surface_child_process_notifications=true (the
    non-default): this test pins the ATTRIBUTION path, which only renders
    when child process notifications are surfaced at all.
    """
    from tools.process_registry import ProcessRegistry

    monkeypatch.setattr(
        ProcessRegistry,
        "_surface_child_process_notifications",
        staticmethod(lambda: True),
    )
    parent = _StubParentWithSession("sess-attr-3")
    child = _StubChild(parent)
    _register(
        "sa-1-attr0003",
        child,
        delegation_id="deleg_attr_3",
        goal="run the npm ci for the desktop app",
    )
    try:
        reg = ProcessRegistry()
        reg.completion_queue.put(
            {
                "type": "completion",
                "session_id": "proc_deadbeef0001",
                "task_id": "sa-1-attr0003",
                "command": "npm ci",
                "exit_code": 0,
                "output": "added 1500 packages",
            }
        )
        results = reg.drain_notifications()
        assert len(results) == 1
        text = results[0][1]
        assert text is not None
        assert "Started by subagent sa-1-attr0003" in text
        assert "deleg_attr_3" in text
        assert "run the npm ci for the desktop app" in text
    finally:
        _unregister_subagent("sa-1-attr0003")


def _child_completion_evt(task_id="sa-9-supp0001", sid="proc_childnoise01"):
    return {
        "type": "completion",
        "session_id": sid,
        "task_id": task_id,
        "command": "npm ci",
        "exit_code": 0,
        "output": "added 1500 packages",
    }


def test_child_completion_notification_suppressed_by_default(monkeypatch):
    """With no user config, subagent-owned completion events are DROPPED from
    the parent drain (not delivered, not requeued)."""
    import hermes_cli.config as _cfg
    from tools.process_registry import ProcessRegistry

    monkeypatch.setattr(_cfg, "read_raw_config", lambda *a, **k: {})
    reg = ProcessRegistry()
    reg.completion_queue.put(_child_completion_evt())
    results = reg.drain_notifications()
    assert results == []
    # NOT requeued — children never drain notify events; requeueing would
    # pin the event in the queue forever.
    assert reg.completion_queue.qsize() == 0


def test_async_delegation_event_from_child_never_suppressed(monkeypatch):
    """The delegation result itself (type async_delegation) always flows to
    the parent even while the same child's process notifications are
    suppressed."""
    import hermes_cli.config as _cfg
    from tools.process_registry import ProcessRegistry

    monkeypatch.setattr(_cfg, "read_raw_config", lambda *a, **k: {})
    reg = ProcessRegistry()
    reg.completion_queue.put(_child_completion_evt(task_id="sa-9-supp0002"))
    reg.completion_queue.put(
        {
            "type": "async_delegation",
            "delegation_id": "deleg_supp_2",
            "task_id": "sa-9-supp0002",
            "goal": "port the widget",
            "status": "completed",
            "summary": "Widget ported successfully.",
        }
    )
    results = reg.drain_notifications()
    assert len(results) == 1
    evt, text = results[0]
    assert evt["type"] == "async_delegation"
    assert "ASYNC DELEGATION COMPLETE" in text
    assert reg.completion_queue.qsize() == 0


def test_parent_owned_completion_unaffected_by_suppression(monkeypatch):
    """Processes the parent itself started (non sa- task_id) still notify."""
    import hermes_cli.config as _cfg
    from tools.process_registry import ProcessRegistry

    monkeypatch.setattr(_cfg, "read_raw_config", lambda *a, **k: {})
    reg = ProcessRegistry()
    reg.completion_queue.put(
        {
            "type": "completion",
            "session_id": "proc_parent01",
            "task_id": "20260817_154314_30d98f",
            "command": "make build",
            "exit_code": 0,
            "output": "ok",
        }
    )
    results = reg.drain_notifications()
    assert len(results) == 1
    assert "Background process proc_parent01" in results[0][1]


def test_surface_flag_true_restores_child_notification_delivery(monkeypatch):
    """delegation.surface_child_process_notifications=true restores the legacy
    behavior: child completion delivered with attribution."""
    import hermes_cli.config as _cfg
    from tools.process_registry import ProcessRegistry

    monkeypatch.setattr(
        _cfg,
        "read_raw_config",
        lambda *a, **k: {
            "delegation": {"surface_child_process_notifications": True}
        },
    )
    reg = ProcessRegistry()
    reg.completion_queue.put(_child_completion_evt(task_id="sa-9-supp0003"))
    results = reg.drain_notifications()
    assert len(results) == 1
    text = results[0][1]
    assert "Background process proc_childnoise01" in text
    assert "Started by subagent sa-9-supp0003" in text


def test_child_watch_match_suppressed_by_default(monkeypatch):
    """watch_match events from sa- sessions follow the same suppression."""
    import hermes_cli.config as _cfg
    from tools.process_registry import ProcessRegistry

    monkeypatch.setattr(_cfg, "read_raw_config", lambda *a, **k: {})
    reg = ProcessRegistry()
    reg.completion_queue.put(
        {
            "type": "watch_match",
            "session_id": "proc_childwatch01",
            "task_id": "sa-9-supp0004",
            "command": "vitest --watch",
            "pattern": "FAIL",
            "output": "FAIL src/x.test.ts",
            "suppressed": 0,
        }
    )
    assert reg.drain_notifications() == []
    assert reg.completion_queue.qsize() == 0


def test_child_completion_with_collapsed_container_task_id_suppressed(monkeypatch):
    """Regression (child-notify leak, Aug 2026): terminal_tool stamps the
    COLLAPSED container key ("default"/session key) into the event's task_id
    — _resolve_container_task_id deliberately collapses subagent ids so
    children share the parent's container. The suppression gate must key on
    owner_task_id (the raw spawning id), or child events with
    task_id="default" walk straight past it into the parent chat."""
    import hermes_cli.config as _cfg
    from tools.process_registry import ProcessRegistry

    monkeypatch.setattr(_cfg, "read_raw_config", lambda *a, **k: {})
    reg = ProcessRegistry()
    evt = _child_completion_evt(task_id="default")
    evt["owner_task_id"] = "sa-9-supp0005"
    reg.completion_queue.put(evt)
    assert reg.drain_notifications() == []
    assert reg.completion_queue.qsize() == 0


def test_spawn_local_stamps_owner_task_id_and_event_carries_it(monkeypatch):
    """spawn_local(owner_task_id=...) survives to the completion event, so a
    real subagent-spawned process (collapsed task_id) is suppressed on
    drain. Exercises the actual spawn -> _move_to_finished -> drain path."""
    import time as _time

    import hermes_cli.config as _cfg
    from tools.process_registry import ProcessRegistry

    monkeypatch.setattr(_cfg, "read_raw_config", lambda *a, **k: {})
    reg = ProcessRegistry()
    session = reg.spawn_local(
        command="echo owner-stamp-e2e",
        task_id="default",
        owner_task_id="sa-9-supp0006",
    )
    session.notify_on_complete = True
    assert session.owner_task_id == "sa-9-supp0006"
    deadline = _time.time() + 15
    while not session.exited and _time.time() < deadline:
        _time.sleep(0.05)
    assert session.exited, "test process should exit promptly"
    _time.sleep(0.3)  # let the reader thread enqueue the completion event
    assert reg.drain_notifications() == []


def test_spawn_local_without_owner_defaults_to_task_id(monkeypatch):
    """Backward compat: callers that don't pass owner_task_id behave exactly
    as before (owner falls back to task_id; parent-owned still delivers)."""
    import time as _time

    import hermes_cli.config as _cfg
    from tools.process_registry import ProcessRegistry

    monkeypatch.setattr(_cfg, "read_raw_config", lambda *a, **k: {})
    reg = ProcessRegistry()
    session = reg.spawn_local(command="echo parent-e2e", task_id="default")
    session.notify_on_complete = True
    assert session.owner_task_id == "default"
    deadline = _time.time() + 15
    while not session.exited and _time.time() < deadline:
        _time.sleep(0.05)
    assert session.exited
    _time.sleep(0.3)
    results = reg.drain_notifications()
    assert len(results) == 1
    assert "completed normally" in results[0][1]


def test_attribution_line_uses_owner_task_id(monkeypatch):
    """format_process_notification resolves attribution from owner_task_id
    when task_id is a collapsed container key (surface flag on)."""
    import hermes_cli.config as _cfg
    from tools.process_registry import ProcessRegistry
    from tools.process_registry_notifications import format_process_notification

    monkeypatch.setattr(
        _cfg,
        "read_raw_config",
        lambda *a, **k: {
            "delegation": {"surface_child_process_notifications": True}
        },
    )
    parent = _StubParentWithSession("sess-attr-owner")
    child = _StubChild(parent)
    _register("sa-9-supp0007", child, delegation_id="deleg_attr_owner")
    try:
        reg = ProcessRegistry()
        evt = _child_completion_evt(task_id="default", sid="proc_ownerattr01")
        evt["owner_task_id"] = "sa-9-supp0007"
        reg.completion_queue.put(evt)
        results = reg.drain_notifications()
        assert len(results) == 1
        assert "Started by subagent sa-9-supp0007" in results[0][1]
        # And the standalone formatter agrees.
        text = format_process_notification(evt)
        assert "Started by subagent sa-9-supp0007" in text
    finally:
        _unregister_subagent("sa-9-supp0007")


def test_completion_notification_trims_subagent_output_wall():
    from tools.process_registry_notifications import format_process_notification

    parent = _StubParentWithSession("sess-attr-4")
    child = _StubChild(parent)
    _register("sa-2-attr0004", child, delegation_id="deleg_attr_4")
    try:
        big_output = "npm noise line\n" * 500
        text = format_process_notification(
            {
                "type": "completion",
                "session_id": "proc_deadbeef0002",
                "task_id": "sa-2-attr0004",
                "command": "npm ci",
                "exit_code": 0,
                "output": big_output,
            }
        )
        assert text is not None
        assert "output trimmed — subagent-owned process" in text
        assert len(text) < len(big_output)
    finally:
        _unregister_subagent("sa-2-attr0004")


def test_parent_owned_process_notification_unchanged():
    """Processes NOT started by a subagent keep the exact legacy shape."""
    from tools.process_registry_notifications import format_process_notification

    text = format_process_notification(
        {
            "type": "completion",
            "session_id": "proc_parentowned",
            "task_id": "20260817_154314_30d98f",  # CLI session task_id
            "command": "make build",
            "exit_code": 0,
            "output": "ok",
        }
    )
    assert text is not None
    assert "Started by subagent" not in text
    assert text.startswith("[IMPORTANT: Background process proc_parentowned")
    assert "Command: make build\nOutput:\nok]" in text


# ---------------------------------------------------------------------------
# Guardrail: control actions never consume the spawn cap
# ---------------------------------------------------------------------------
def test_spawn_count_zero_for_control_actions():
    from agent.tool_guardrails import _subagent_spawn_count

    assert _subagent_spawn_count({"action": "list"}) == 0
    assert _subagent_spawn_count({"action": "inspect", "subagent_id": "x"}) == 0
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
    assert (
        ctl2.before_call(
            "delegate_task", {"action": "inspect", "subagent_id": "x"}
        ).action
        == "allow"
    )
    # And spawns remain blocked afterwards — the control call didn't reset it
    assert ctl2.before_call("delegate_task", {"goal": "c"}).action == "block"
