"""Reasoning escalation: a child on a route with ``escalate_reasoning`` + ``escalate_ceiling`` that stops on budget
is retried on the same route at the next reasoning level until it finishes or the ceiling (or the provider's clamp)
stops the climb. The parent receives one result: the last attempt's entry, an ``attempts`` list and the cost summed
across attempts. Untouched routes behave exactly as today.

Two levels: ``_run_single_child`` driven directly with scripted children (the chain, triggers, clamp, handoff,
workspace and worktree rules), and the real ``delegate_task`` with a scripted ``AIAgent`` (route pinning, effort
inheritance, the result the parent model reads and the parent's cost rollup).
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.delegation_provider import DelegationProvider
from hermes_constants import parse_reasoning_effort
from tools import delegate_tool
from tools.delegate_tool_child_run import _effort_ran

GOAL = "Fix the flaky retry test in tests/test_retry.py"
TRANSCRIPT = "/tmp/hermes-live/deleg-1/task-0.log"


# ── Scripted children ────────────────────────────────────────────────────────

def M(summary="partial work: found the race in retry()", cost=0.01, calls=5):
    """Budget exhaustion (max_iterations)."""
    return {"kind": "max", "summary": summary, "cost": cost, "calls": calls}


def D(summary="fixed: retry() now awaits the lock", cost=0.02, calls=3):
    return {"kind": "done", "summary": summary, "cost": cost, "calls": calls}


def T(calls=2, cost=0.005):
    """No progress inside the inactivity window (after ``calls`` API calls)."""
    return {"kind": "timeout", "summary": None, "cost": cost, "calls": calls}


def X(kind, cost=0.0):
    return {"kind": kind, "summary": None, "cost": cost, "calls": 1}


class _FakeChild:
    """A child whose outcome is scripted per the reasoning level it actually runs at."""

    def __init__(self, script, log, kw):
        self.kwargs = kw
        self.script, self.log = script, log
        self.reasoning_config = kw.get("reasoning_config")
        self.provider, self.model, self.api_mode = kw.get("provider"), kw.get("model"), kw.get("api_mode")
        self.tool_progress_callback = kw.get("tool_progress_callback")
        self.session_id = f"child-{uuid.uuid4().hex[:6]}"
        self.session_estimated_cost_usd, self.session_cost_status = 0.0, "unknown"
        self.session_prompt_tokens = self.session_completion_tokens = 0
        self._calls, self._activity_ts = 0, time.time()
        self._stop = threading.Event()
        self._interrupt_requested = False
        self.messages, self.task_ids, self.start_cwds = [], [], []
        self.closed = False
        log.append(self)

    def run_conversation(self, user_message, task_id=None, stream_callback=None, **_):
        from tools.terminal_tool import get_session_cwd, record_session_cwd
        self.messages.append(user_message)
        self.task_ids.append(task_id)
        self.start_cwds.append(get_session_cwd(task_id))
        outcome = self.script[_effort_ran(self)]
        if callable(outcome):
            outcome = outcome(self, task_id, record_session_cwd)
        self.session_estimated_cost_usd = outcome["cost"]
        self.session_cost_status = "estimated"
        kind = outcome["kind"]
        self._calls = outcome["calls"] if kind != "timeout0" else 0
        if kind in ("timeout", "timeout0"):
            self._stop.wait(10)
            return {"final_response": "", "interrupted": True, "api_calls": self._calls, "messages": []}
        if kind == "raise":
            raise RuntimeError("transport exploded")
        result = {"final_response": outcome["summary"] or "", "api_calls": self._calls, "messages": []}
        if kind == "done":
            result["completed"] = True
        elif kind == "max":
            result["completed"] = False
        elif kind == "error":
            result.update(final_response="provider rejected the request", failed=True, error="HTTP 400")
        elif kind == "interrupted":
            result.update(final_response="Operation interrupted.", interrupted=True)
        elif kind == "completed_empty":
            result.update(final_response="", completed=True)
        return result

    def get_activity_summary(self):
        return {"api_call_count": self._calls, "current_tool": None, "last_activity_ts": self._activity_ts,
                "max_iterations": 50}

    def hard_interrupt(self, *_a, **_k):
        self._stop.set()

    def interrupt(self, *_a, **_k):
        self._stop.set()

    def steer(self, _text):
        return True

    def close(self):
        self.closed = True


def _parent(**overrides):
    parent = SimpleNamespace(
        session_id="parent", _current_task_id="parent-task", _active_children=[],
        _active_children_lock=threading.Lock(), _touch_activity=lambda _d: None, _interrupt_requested=False,
    )
    for key, value in overrides.items():
        setattr(parent, key, value)
    return parent


def _run_direct(monkeypatch, script, *, effort="low", ceiling="high", enabled=True, provider="openrouter",
                model="anthropic/claude-sonnet-4", api_mode="chat_completions", timeout=None, parent=None,
                builder=True, progress=None, schema=None, first_attrs=None):
    """``(entry, [children in attempt order], parent)`` for a first child at ``effort`` run by ``_run_single_child``.
    Children carry what ``_build_children`` pins; the builder mirrors it (same route, next level, handoff goal)."""
    log: list[_FakeChild] = []
    events = progress if progress is not None else []

    def make(kw):
        child = _FakeChild(script, log, kw)
        child.tool_progress_callback = lambda event, *a, **k: events.append((child, event, k))
        if enabled is not None:
            child._delegate_escalate_reasoning = enabled
        if ceiling is not None:
            child._delegate_escalate_ceiling = ceiling
        if builder and enabled and ceiling:
            child._delegate_build_retry = lambda e, g: make({**kw, "reasoning_config": parse_reasoning_effort(e),
                                                             "goal": g})
        if schema is not None:
            child._delegate_output_schema = schema
        child._live_transcript_path = TRANSCRIPT
        return child

    first = make({"reasoning_config": parse_reasoning_effort(effort) if effort else None, "provider": provider,
                  "model": model, "api_mode": api_mode, "goal": GOAL})
    for key, value in (first_attrs or {}).items():
        setattr(first, key, value)
    parent = parent or _parent()
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: timeout)
    monkeypatch.setattr(delegate_tool, "_get_worktree_isolation", lambda: False)
    entry = delegate_tool._run_single_child(0, GOAL, child=first, parent_agent=parent)
    return entry, log, parent


def _efforts(log):
    return [_effort_ran(c) for c in log]


# ── The climb ────────────────────────────────────────────────────────────────

def test_steps_up_one_level_per_budget_stop_until_the_child_completes(monkeypatch):
    script = {"low": M("low got halfway", cost=0.01), "medium": M("medium got further", cost=0.02),
              "high": D("high finished it", cost=0.04)}
    entry, log, _ = _run_direct(monkeypatch, script, effort="low", ceiling="xhigh")
    assert _efforts(log) == ["low", "medium", "high"]
    assert entry["status"] == "completed" and entry["exit_reason"] == "completed"
    assert entry["summary"] == "high finished it"
    assert entry["attempts"] == [{"effort": "low", "exit_reason": "max_iterations"},
                                 {"effort": "medium", "exit_reason": "max_iterations"},
                                 {"effort": "high", "exit_reason": "completed"}]
    assert entry["cost_usd"] == pytest.approx(0.07) and entry["_child_cost_usd"] == pytest.approx(0.07)
    assert all(c.closed for c in log)


def test_stops_as_soon_as_an_attempt_completes_below_the_ceiling(monkeypatch):
    entry, log, _ = _run_direct(monkeypatch, {"low": M(), "medium": D("done at medium")}, effort="low",
                                ceiling="max")
    assert _efforts(log) == ["low", "medium"]
    assert entry["summary"] == "done at medium"
    assert [a["effort"] for a in entry["attempts"]] == ["low", "medium"]


def test_a_budget_stop_with_no_summary_is_failed_but_still_retries(monkeypatch):
    entry, log, _ = _run_direct(monkeypatch, {"low": M(summary=""), "medium": D()}, effort="low", ceiling="medium")
    assert _efforts(log) == ["low", "medium"]
    assert entry["attempts"][0] == {"effort": "low", "exit_reason": "max_iterations"}
    assert entry["status"] == "completed"
    # The handoff points at the transcript because there was no summary.
    handoff = log[1].messages[0]
    assert "left no summary" in handoff and TRANSCRIPT in handoff


def test_a_lone_budget_stop_without_a_summary_is_reported_failed_unchanged(monkeypatch):
    """No escalation configured: the same empty budget stop stays exactly today's failed entry."""
    entry, log, _ = _run_direct(monkeypatch, {"low": M(summary="")}, effort="low", ceiling=None, enabled=None)
    assert len(log) == 1
    assert (entry["status"], entry["exit_reason"]) == ("failed", "max_iterations")
    assert "attempts" not in entry


def test_a_timeout_after_api_calls_retries(monkeypatch):
    entry, log, _ = _run_direct(monkeypatch, {"low": T(calls=3, cost=0.01), "medium": D(cost=0.02)},
                                effort="low", ceiling="medium", timeout=0.3)
    assert _efforts(log) == ["low", "medium"]
    assert entry["attempts"] == [{"effort": "low", "exit_reason": "timeout"},
                                 {"effort": "medium", "exit_reason": "completed"}]
    # A timed-out attempt's spend still counts.
    assert entry["cost_usd"] == pytest.approx(0.03)
    handoff = log[1].messages[0]
    assert "timed out" in handoff and "left no summary" in handoff and TRANSCRIPT in handoff


def test_a_timeout_with_zero_api_calls_never_reached_the_model_and_does_not_retry(monkeypatch):
    entry, log, _ = _run_direct(monkeypatch, {"low": X("timeout0")}, effort="low", ceiling="high", timeout=0.3)
    assert len(log) == 1
    assert entry["status"] == "timeout" and entry["api_calls"] == 0
    assert "attempts" not in entry and "cost_usd" not in entry


@pytest.mark.parametrize("kind, status, exit_reason", [
    ("done", "completed", "completed"),
    ("completed_empty", "failed", "completed"),  # completed is success for escalation, whatever the status
    ("interrupted", "interrupted", "interrupted"),
    ("error", "failed", "error"),
    ("raise", "error", "error"),  # the worker raised
])
def test_never_retries_on_completion_interrupt_or_error(monkeypatch, kind, status, exit_reason):
    outcome = D() if kind == "done" else X(kind)
    entry, log, _ = _run_direct(monkeypatch, {"low": outcome, "medium": D()}, effort="low", ceiling="xhigh")
    assert len(log) == 1
    assert entry["status"] == status and entry.get("exit_reason") == exit_reason
    assert "attempts" not in entry


def test_never_retries_once_the_parent_was_interrupted(monkeypatch):
    entry, log, _ = _run_direct(monkeypatch, {"low": M(), "medium": D()}, effort="low", ceiling="high",
                                parent=_parent(_interrupt_requested=True))
    assert len(log) == 1 and entry["exit_reason"] == "max_iterations" and "attempts" not in entry


# ── Where the climb starts and stops ─────────────────────────────────────────

@pytest.mark.parametrize("first_attrs, expected", [
    ({"reasoning_config": None}, ["medium", "high"]),  # Hermes' default level
    ({"reasoning_config": {"enabled": True}}, ["medium", "high"]),
    ({"reasoning_config": {"enabled": False}}, ["none", "minimal"]),
    ({"reasoning_config": "not-a-dict", "_wire_reasoning_config": {"effort": "LOW"}}, ["low", "medium"]),
])
def test_steps_from_the_level_the_child_actually_ran(monkeypatch, first_attrs, expected):
    entry, _log, _ = _run_direct(monkeypatch, {lvl: (M() if lvl == expected[0] else D()) for lvl in expected},
                                effort=None, ceiling="xhigh", first_attrs=first_attrs)
    assert [a["effort"] for a in entry["attempts"]] == expected


def test_stops_at_the_ceiling(monkeypatch):
    entry, log, _ = _run_direct(monkeypatch, {"low": M(), "medium": M("medium ran out too")}, effort="low",
                                ceiling="medium")
    assert _efforts(log) == ["low", "medium"]
    assert entry["exit_reason"] == "max_iterations" and entry["summary"] == "medium ran out too"
    assert entry["attempts"][-1] == {"effort": "medium", "exit_reason": "max_iterations"}


@pytest.mark.parametrize("effort, ceiling", [("high", "high"), ("xhigh", "high"), ("max", "MAX ")])
def test_no_retry_when_the_child_already_runs_at_or_above_the_ceiling(monkeypatch, effort, ceiling):
    entry, log, _ = _run_direct(monkeypatch, {effort: M()}, effort=effort, ceiling=ceiling)
    assert len(log) == 1 and "attempts" not in entry


@pytest.mark.parametrize("enabled, ceiling", [(True, None), (False, "xhigh"), (None, "xhigh"), (True, "huge")])
def test_escalation_is_off_without_both_the_switch_and_a_valid_ceiling(monkeypatch, enabled, ceiling):
    entry, log, _ = _run_direct(monkeypatch, {"low": M(), "medium": D()}, effort="low", ceiling=ceiling,
                                enabled=enabled, builder=False)
    assert len(log) == 1 and "attempts" not in entry


@pytest.mark.parametrize("effort, provider, model, api_mode, retries", [
    ("max", "openrouter", "openai/gpt-5.5", "chat_completions", False),  # ultra clamps back to max
    ("xhigh", "openai-codex", "gpt-5.5", "codex_responses", False),  # legacy Codex: max clamps to xhigh
    ("high", "openai-codex", "gpt-5.5", "codex_responses", True),
    ("max", "openai-codex", "gpt-5.6", "codex_app_server", True),  # the app server sends ultra verbatim
    ("max", "openai-codex", "gpt-6-astra", "codex_app_server", True),
    ("max", "openai-codex", "gpt-5.6", "codex_responses", False),
])
def test_stops_when_the_provider_clamp_would_not_send_a_stronger_level(
        monkeypatch, effort, provider, model, api_mode, retries):
    from agent.reasoning_effort import EFFORT_LADDER
    nxt = EFFORT_LADDER[EFFORT_LADDER.index(effort) + 1]
    entry, log, _ = _run_direct(monkeypatch, {effort: M(), nxt: D()}, effort=effort, ceiling="ultra",
                                provider=provider, model=model, api_mode=api_mode)
    assert (len(log) == 2) is retries
    assert ("attempts" in entry) is retries


@pytest.mark.parametrize("target", ["agent.reasoning_effort.clamp_effort",
                                    "tools.delegate_tool_child_run.clamp_effort"])
def test_the_clamp_is_read_where_tests_and_providers_patch_it(monkeypatch, target):
    with patch(target, side_effect=lambda effort, supported, overrides=None: "low"):
        entry, log, _ = _run_direct(monkeypatch, {"low": M(), "medium": D()}, effort="low", ceiling="high")
    assert len(log) == 1 and "attempts" not in entry


# ── Handoff and workspace ────────────────────────────────────────────────────

def test_handoff_carries_goal_summary_transcript_and_working_dir(monkeypatch, tmp_path):
    from tools.terminal_tool import record_session_cwd
    parent_dir, worked_in = tmp_path / "repo", tmp_path / "repo" / "pkg"
    worked_in.mkdir(parents=True)
    record_session_cwd("parent-task", str(parent_dir))

    def low(child, task_id, record):
        record(task_id, str(worked_in))  # the attempt cd'd into a subdirectory
        return M("narrowed it to pkg/retry.py:42")

    _entry, log, _ = _run_direct(monkeypatch, {"low": low, "medium": D()}, effort="low", ceiling="medium")
    first, retry = log
    assert first.start_cwds == [str(parent_dir)]
    handoff = retry.messages[0]
    assert handoff.startswith(GOAL)
    assert handoff.count(GOAL) == 1
    assert "narrowed it to pkg/retry.py:42" in handoff
    assert TRANSCRIPT in handoff
    assert str(worked_in) in handoff
    assert "'low'" in handoff and "'medium'" in handoff
    # The retry starts where the previous attempt was working, in its own terminal session.
    assert retry.start_cwds == [str(worked_in)]
    assert retry.task_ids[0] != first.task_ids[0]


def test_a_retry_with_no_working_dir_of_its_own_falls_back_to_the_parents(monkeypatch, tmp_path):
    from tools.terminal_tool import record_session_cwd
    record_session_cwd("parent-task", str(tmp_path))
    _, (_first, retry), _ = _run_direct(monkeypatch, {"low": M(), "medium": D()}, effort="low", ceiling="medium")
    assert retry.start_cwds == [str(tmp_path)] and str(tmp_path) in retry.messages[0]


def test_only_the_final_attempt_emits_a_completion(monkeypatch):
    events = []
    _entry, log, _ = _run_direct(monkeypatch, {"low": M(), "medium": T(), "high": D()}, effort="low",
                                ceiling="high", timeout=0.3, progress=events)
    completes = [(child, kw) for child, event, kw in events if event == "subagent.complete"]
    assert len(completes) == 1 and completes[0][0] is log[-1]
    assert completes[0][1]["status"] == "completed"
    assert [event for _, event, _ in events].count("subagent.start") == 3


def test_an_escalating_attempt_gets_no_schema_retry_turn(monkeypatch):
    schema = {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}
    entry, (low, _medium), _ = _run_direct(
        monkeypatch, {"low": M("not json"), "medium": D('{"ok": true}')}, effort="low", ceiling="medium",
        schema=schema)
    assert len(low.messages) == 1  # no correction turn on the exhausted child
    assert entry["schema_valid"] is True and entry["summary"] == '{"ok": true}'


def test_a_failed_retry_build_ends_the_chain_with_a_normal_result(monkeypatch):
    events = []

    def boom(_effort, _goal):
        raise ValueError("owner refused")

    entry, log, _ = _run_direct(monkeypatch, {"low": M("ran out")}, effort="low", ceiling="high", builder=False,
                                progress=events, first_attrs={"_delegate_build_retry": boom})
    assert len(log) == 1 and "attempts" not in entry
    assert entry["exit_reason"] == "max_iterations" and entry["summary"] == "ran out"
    assert [e for _, e, _ in events].count("subagent.complete") == 1


def test_a_child_without_a_retry_builder_is_rerun_at_the_next_level(monkeypatch):
    """Direct ``_run_single_child`` callers (no ``_build_children``): the same child is rerun one level up."""
    script = {"low": M("low ran out"), "medium": D("medium done")}
    log: list = []
    child = _FakeChild(script, log, {"reasoning_config": {"enabled": True, "effort": "low"}})
    child._delegate_escalate_reasoning, child._delegate_escalate_ceiling = True, "medium"
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: None)
    monkeypatch.setattr(delegate_tool, "_get_worktree_isolation", lambda: False)
    entry = delegate_tool._run_single_child(0, GOAL, child=child, parent_agent=_parent())
    assert len(log) == 1 and len(child.messages) == 2
    assert child.reasoning_config == {"enabled": True, "effort": "medium"}
    assert "low ran out" in child.messages[1]
    assert entry["attempts"] == [{"effort": "low", "exit_reason": "max_iterations"},
                                 {"effort": "medium", "exit_reason": "completed"}]


@pytest.mark.parametrize("cancel_during_build", [False, True, "cleanup"])
def test_background_retry_tracking_and_cancellation(monkeypatch, cancel_during_build):
    from tools import delegate_tool_dispatch as dispatch
    from tools.delegate_tool_child_run import _attach_child

    callbacks = {}
    monkeypatch.setattr("tools.async_delegation.dispatch_async_delegation_batch",
                        lambda **kwargs: callbacks.update(kwargs) or {})
    parent = _parent()
    observations = []

    def low(first, *_):
        unit = dispatch._Batch(
            children=[(0, {"goal": GOAL}, first)], task_list=[{"goal": GOAL}], parent_agent=parent,
            context=None, top_role="", creds={"model": None}, live_writers=[], max_children=1,
            live_deleg_id=None, live_paths=[], origin_wake_sid="", origin_ui_session_id="",
            origin_owner_transport=None, origin_owner_session_record=None,
            origin_session_history_delivery=False, overall_start=time.monotonic(),
        )
        dispatch._dispatch_unit(unit, "unit", None, {})
        builder = first._delegate_build_retry

        def build(effort, goal):
            retry = builder(effort, goal)
            _attach_child(parent, retry)
            if cancel_during_build is True:
                callbacks["interrupt_fn"]()
            return retry

        first._delegate_build_retry = build
        if cancel_during_build == "cleanup":
            close = first.close
            first.close = lambda: (callbacks["interrupt_fn"](), close())
        observations.append(callbacks["progress_fn"]())
        return M()

    def medium(retry, *_):
        observations.append(retry in parent._active_children)
        retry._calls, retry._activity_ts = 7, 123
        observations.append(callbacks["progress_fn"]())
        callbacks["interrupt_fn"]()
        observations.append(retry._stop.is_set())
        return M()  # even budget exhaustion cannot restart a cancelled unit

    _entry, log, _ = _run_direct(monkeypatch, {"low": low, "medium": medium, "high": D()}, parent=parent)
    if cancel_during_build:
        assert len(log) == 2 and not log[1].messages
        assert log[1]._stop.is_set() and log[1].closed
    else:
        assert observations[1:] == [False, (((1, 7, None, 123),), False), True]
        assert len(log) == 2
    assert not parent._active_children


def test_reused_child_stays_open_and_accumulates_cost_once(monkeypatch):
    states = []
    original = _FakeChild.run_conversation

    def cumulative(self, *args, **kwargs):
        states.append(self.closed)
        previous = self.session_estimated_cost_usd
        result = original(self, *args, **kwargs)
        self.session_estimated_cost_usd += previous
        return result

    monkeypatch.setattr(_FakeChild, "run_conversation", cumulative)
    entry, (child,), _ = _run_direct(monkeypatch, {"low": M(cost=0.01), "medium": D(cost=0.02)}, builder=False)
    assert states == [False, False]
    assert child.closed
    assert entry["cost_usd"] == pytest.approx(0.03)
    assert child.session_estimated_cost_usd == pytest.approx(0.03)


def test_worktree_is_created_once_and_finalized_only_when_the_chain_ends(monkeypatch, tmp_path):
    from tools import subagent_worktree
    order = []
    tree = {"path": str(tmp_path / "wt"), "branch": "hermes/sa-0"}
    (tmp_path / "wt").mkdir()

    def create(*_a, **_k):
        order.append("create")
        return dict(tree)

    def finalize(info):
        order.append("finalize")
        return {"path": info["path"], "state": "archived"}

    monkeypatch.setattr(subagent_worktree, "local_backend_active", lambda: True)
    monkeypatch.setattr(subagent_worktree, "create_subagent_worktree", create)
    monkeypatch.setattr(subagent_worktree, "finalize_subagent_worktree", finalize)

    def attempt(kind):
        def run(child, task_id, record):
            order.append(("run", _effort_ran(child)))
            return {"max": M(), "done": D()}[kind] if kind != "timeout" else T()
        return run

    log: list = []
    monkeypatch.setattr(delegate_tool, "_get_worktree_isolation", lambda: True)
    # _run_direct forces isolation off; drive the same children by hand with it on.
    script = {"low": attempt("max"), "medium": attempt("timeout"), "high": attempt("done")}

    def make(kw):
        child = _FakeChild(script, log, kw)
        child._delegate_escalate_reasoning, child._delegate_escalate_ceiling = True, "high"
        child._delegate_build_retry = lambda e, g: make({**kw, "reasoning_config": parse_reasoning_effort(e)})
        child._live_transcript_path = TRANSCRIPT
        return child

    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: 0.3)
    entry = delegate_tool._run_single_child(0, GOAL, child=make({"reasoning_config": parse_reasoning_effort("low")}),
                                            parent_agent=_parent())
    assert order == ["create", ("run", "low"), ("run", "medium"), ("run", "high"), "finalize"]
    assert entry["worktree"] == {"path": tree["path"], "state": "archived"}
    for child in log:
        assert child.start_cwds == [tree["path"]]  # every attempt works in the live tree
        assert child.messages[0].count("[WORKTREE ISOLATION]") == 1
    assert tree["path"] in log[1].messages[0]


# ── Through delegate_task ────────────────────────────────────────────────────

LOCAL = {"base_url": "http://127.0.0.1:8080/v1", "api_key": "k", "model": "qwen3-local"}


class _Backend(DelegationProvider):
    def __init__(self, routes):
        self.routes = routes

    @property
    def name(self):
        return "escalation"

    def route_task(self, task, config):
        route = self.routes[task["index"]]
        return dict(config) if route is None else dict(route)


def _mock_parent(reasoning_config=None):
    parent = MagicMock()
    parent.base_url, parent.api_key, parent.provider = "https://openrouter.ai/api/v1", "parent-key", "openrouter"
    parent.api_mode, parent.model, parent.platform = "chat_completions", "anthropic/claude-sonnet-4", "cli"
    parent.request_overrides = {}
    parent.reasoning_config = reasoning_config
    parent.providers_allowed = parent.providers_ignored = parent.providers_order = parent.provider_sort = None
    parent._session_db, parent._delegate_depth, parent._active_children = None, 0, []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = parent.tool_progress_callback = parent.thinking_callback = None
    parent._interrupt_requested = False
    parent._memory_manager = None
    parent.session_estimated_cost_usd, parent.session_cost_source, parent.session_cost_status = 0.0, "none", "unknown"
    return parent


def _delegate(script, routes, *, cfg=None, parent=None):
    """``(result json, [children in build order], [TaskRoute], parent)`` for one delegate_task call."""
    from tools import delegate_tool_routes
    log: list[_FakeChild] = []
    resolved = []
    real_resolve = delegate_tool_routes.resolve_task_routes

    def resolve(*a, **k):
        routes_, err = real_resolve(*a, **k)
        resolved.extend(routes_)
        return routes_, err

    parent = parent or _mock_parent()
    with (
        patch("tools.delegate_tool._load_config", return_value=cfg or {}),
        patch("tools.delegate_tool_routes._active_provider", return_value=_Backend(routes)),
        patch("tools.delegate_tool_routes.resolve_task_routes", side_effect=resolve),
        patch("run_agent.AIAgent", side_effect=lambda **kw: _FakeChild(script, log, kw)),
        patch("tools.delegate_tool._get_worktree_isolation", return_value=False),
        patch("tools.delegate_tool._get_child_timeout", return_value=None),
    ):
        result = json.loads(delegate_tool.delegate_task(
            tasks=[{"goal": f"task number {i}"} for i in range(len(routes))], parent_agent=parent))
    return result, log, resolved, parent


ESC = {**LOCAL, "escalate_reasoning": True, "escalate_ceiling": "high", "label": "esc"}


def test_delegate_task_returns_one_result_with_attempts_and_summed_cost():
    result, log, (route,), parent = _delegate(
        {"low": M(cost=0.01), "medium": M(cost=0.02), "high": D("all green", cost=0.04)},
        [{**ESC, "reasoning_effort": "low", "max_iterations": 9, "child_timeout_seconds": 300}])
    (entry,) = result["results"]
    assert entry["summary"] == "all green" and entry["status"] == "completed"
    assert entry["attempts"] == [{"effort": "low", "exit_reason": "max_iterations"},
                                 {"effort": "medium", "exit_reason": "max_iterations"},
                                 {"effort": "high", "exit_reason": "completed"}]
    assert entry["cost_usd"] == pytest.approx(0.07)
    assert "_child_cost_usd" not in entry
    assert parent.session_estimated_cost_usd == pytest.approx(0.07)
    assert entry["route"]["reasoning_effort"] == "high"
    # Same route and limits on every attempt; the route itself never learns the bumped level.
    assert [c.kwargs["max_iterations"] for c in log] == [9, 9, 9]
    assert all(c._delegate_child_timeout == 300 for c in log)
    assert all(c._live_transcript_path == log[0]._live_transcript_path for c in log)
    assert all(getattr(c, "_delegation_id", None) == log[0]._delegation_id for c in log)
    assert route.reasoning_effort == "low" and route.routing_cfg.get("reasoning_effort") == "low"
    assert log[0]._delegate_last_attempt is log[-1]
    assert parent._active_children == []


def test_delegate_task_without_a_retry_keeps_todays_entry():
    result, log, _, parent = _delegate({"low": D(cost=0.01)}, [{**ESC, "reasoning_effort": "low"}])
    (entry,) = result["results"]
    assert len(log) == 1 and "attempts" not in entry
    assert entry["cost_usd"] == pytest.approx(0.01) and parent.session_estimated_cost_usd == pytest.approx(0.01)


@pytest.mark.parametrize("cfg, parent_reasoning, expected", [
    ({"reasoning_effort": "low"}, None, ["low", "medium"]),  # delegation.reasoning_effort
    ({}, {"enabled": True, "effort": "medium"}, ["medium", "high"]),  # the parent's level
])
def test_a_blank_route_effort_steps_from_the_inherited_level(cfg, parent_reasoning, expected):
    result, _log, (route,), _ = _delegate({expected[0]: M(), expected[1]: D()}, [{**ESC, "escalate_ceiling": "xhigh"}],
                                         cfg=cfg, parent=_mock_parent(parent_reasoning))
    (entry,) = result["results"]
    assert [a["effort"] for a in entry["attempts"]] == expected
    assert route.reasoning_effort is None and "reasoning_effort" not in route.routing_cfg


def test_a_refused_sibling_and_an_untouched_route_never_escalate():
    result, log, _, _ = _delegate({"low": M(), "medium": D(), "high": M()},
                                  [{**ESC, "reasoning_effort": "low"}, {**LOCAL, "reasoning_effort": "high"},
                                   {**ESC, "escalate_ceiling": "huge"}])
    by_index = {e["task_index"]: e for e in result["results"]}
    assert [a["effort"] for a in by_index[0]["attempts"]] == ["low", "medium"]
    assert "attempts" not in by_index[1] and by_index[1]["exit_reason"] == "max_iterations"
    (refused,) = result["refused"] if "refused" in result else [e for e in result["results"] if e["task_index"] == 2]
    assert "attempts" not in refused and "escalate_ceiling" in refused["error"]
    assert len(log) == 3  # two attempts for task 0, one for task 1, none for the refused task


# ── Provider wire vocabularies and overrides ────────────────────────────────

@pytest.mark.parametrize("effort, provider, model, retries", [
    ("high", "tencent-tokenhub", "hunyuan-t2", False),  # high and xhigh both clamp to high
    ("high", "nebius-token-factory", "meta-llama/x", False),
    ("medium", "kimi-coding", "k3", False),  # K3 sends medium as high already
    ("high", "kimi-coding", "k3", True),  # xhigh overrides to max
    ("high", "deepseek", "deepseek-v4", True),  # xhigh overrides to max
    ("medium", "zai", "glm-5.2", False),  # GLM-5.2's floor is high
    ("high", "zai", "glm-5.2", True),
    ("medium", "opencode-zen", "x-preview-f-free", True),  # medium→low, high→high
])
def test_the_clamp_guard_uses_the_providers_wire_vocabulary(monkeypatch, effort, provider, model, retries):
    from agent.reasoning_effort import EFFORT_LADDER
    nxt = EFFORT_LADDER[EFFORT_LADDER.index(effort) + 1]
    entry, log, _ = _run_direct(monkeypatch, {effort: M(), nxt: D()}, effort=effort, ceiling="max",
                                provider=provider, model=model)
    assert (len(log) == 2) is retries
    assert ("attempts" in entry) is retries


# ── Timed-out predecessor must exit before the workspace handoff ────────────

def test_a_timed_out_worker_is_drained_and_awaited_before_the_retry_starts(monkeypatch):
    """The stop is cooperative: a worker that only unwinds once its transport is drained must have exited before
    the retry runs in the same workspace."""
    order, drains = [], []

    def low(child, *_):
        released = threading.Event()
        child._drain_transports_after_abandonment = lambda reason: (drains.append(reason), released.set())
        child._stop = threading.Event()  # interrupt alone does NOT wake this worker

        def run(**_k):
            released.wait(5)
            time.sleep(0.05)
            order.append("low exited")
            return {"final_response": "", "interrupted": True, "api_calls": 3, "messages": []}
        child.run_conversation = run
        return T(calls=3)

    def medium(child, *_):
        order.append("medium started")
        return D()

    original = _FakeChild.run_conversation

    def run_conversation(self, user_message, task_id=None, stream_callback=None, **kw):
        if _effort_ran(self) == "low" and not self.messages:
            self.messages.append(user_message)
            self.task_ids.append(task_id)
            self._calls = 3
            low(self)
            return self.run_conversation()
        return original(self, user_message, task_id=task_id, stream_callback=stream_callback, **kw)

    monkeypatch.setattr(_FakeChild, "run_conversation", run_conversation)
    entry, log, _ = _run_direct(monkeypatch, {"low": T(calls=3), "medium": medium}, effort="low", ceiling="medium",
                                timeout=0.3)
    assert "delegate_timeout_escalate" in drains
    assert order == ["low exited", "medium started"]
    assert _efforts(log) == ["low", "medium"] and entry["status"] == "completed"
    assert log[0].closed


def test_a_wedged_timed_out_worker_ends_the_chain_without_sharing_the_workspace(monkeypatch, caplog):
    import logging
    from tools import delegate_tool_child_run
    monkeypatch.setattr(delegate_tool_child_run, "_ESCALATION_TIMEOUT_SETTLE_SECONDS", 0.2)
    release = threading.Event()
    original = _FakeChild.run_conversation

    def run_conversation(self, user_message, task_id=None, stream_callback=None, **kw):
        if _effort_ran(self) == "low":
            self.messages.append(user_message)
            self._calls = 3
            release.wait(10)  # ignores the interrupt
            return {"final_response": "", "interrupted": True, "api_calls": 3, "messages": []}
        return original(self, user_message, task_id=task_id, stream_callback=stream_callback, **kw)

    monkeypatch.setattr(_FakeChild, "run_conversation", run_conversation)
    try:
        with caplog.at_level(logging.WARNING, logger="tools.delegate_tool_child_run"):
            entry, log, _ = _run_direct(monkeypatch, {"low": T(calls=3), "medium": D()}, effort="low",
                                        ceiling="medium", timeout=0.3)
        assert len(log) == 1 and "attempts" not in entry
        assert entry["status"] == "timeout"
        assert entry["model"] == "anthropic/claude-sonnet-4"  # timed-out entries carry route/model observability
        assert "skipping reasoning escalation" in caplog.text
        assert not log[0].closed  # close deferred to the worker's exit
    finally:
        release.set()
    deadline = time.monotonic() + 5
    while not log[0].closed and time.monotonic() < deadline:
        time.sleep(0.02)
    assert log[0].closed


# ── Stable subagent_id across attempts ───────────────────────────────────────

def test_retries_keep_the_dispatched_subagent_id_for_steer_and_stop():
    from tools.delegate_tool_registry import _active_subagents, get_subagent_attribution
    seen = {}

    def medium(child, task_id, _record):
        record = _active_subagents.get(child._subagent_id)
        seen["registered_agent"] = record and record.get("agent")
        seen["attribution"] = get_subagent_attribution(task_id)
        return D()

    _result, log, _, _ = _delegate({"low": M(), "medium": medium},
                                  [{**ESC, "reasoning_effort": "low", "escalate_ceiling": "medium"}])
    first, retry = log
    assert retry._subagent_id == first._subagent_id
    assert seen["registered_agent"] is retry  # steer/stop by the dispatched id reach the live retry
    # Each attempt still runs its own terminal session; attribution resolves back to the subagent.
    assert retry.task_ids[0] == f"{first._subagent_id}-attempt-2" != first.task_ids[0]
    assert seen["attribution"]["subagent_id"] == first._subagent_id


def test_a_detached_units_retry_is_never_attached_to_the_parent():
    from tools.delegate_tool_child_run import _detach_child
    attached_at_build = []
    parent = _mock_parent()

    def low(child, *_):
        _detach_child(parent, child)  # a background unit is detached before it runs
        builder = child._delegate_build_retry

        def build(effort, goal):
            retry = builder(effort, goal)
            attached_at_build.append(retry in parent._active_children)
            return retry
        child._delegate_build_retry = build
        return M()

    _delegate({"low": low, "medium": D()}, [{**ESC, "reasoning_effort": "low", "escalate_ceiling": "medium"}],
              parent=parent)
    assert attached_at_build == [False]
