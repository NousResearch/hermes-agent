"""A helper waiting on its own sub-helpers can be steered mid-wait.

Below the top level a helper joins its sub-helpers inside its delegate_task call. A steer that reaches the helper
(directly, or relayed from an ancestor) now ends that join at once: finished results come back, the still-running
sub-helpers keep running (not stopped, still registered/steerable), and ``delegate_task(action='wait')`` rejoins them.
The top level, and a join no steer reaches, behave exactly as before.
"""

import json
import threading
import time
import weakref
from types import SimpleNamespace

import pytest

from agent.interrupt_control import InterruptControlMixin
from tools import delegate_tool_dispatch as dispatch
from tools.delegate_tool import _register_subagent, _unregister_subagent, delegate_task, steer_subagent
from tools.delegate_tool_dispatch import _Batch, _execute_and_aggregate, surface_uncollected, wait_for_detached
from tests.waiting import wait_until


class _Agent(InterruptControlMixin):
    """The real steer/drain surface without the rest of AIAgent."""

    _executing_tools = True
    _interrupt_requested = False
    _pending_steer = None
    _pending_redirect = None
    api_mode = "chat_completions"

    def __init__(self, depth, parent=None, session_id=None):
        self._delegate_depth = depth
        self._pending_steer_lock = threading.Lock()
        self._pending_redirect_lock = threading.Lock()
        self._tool_worker_threads = set()
        self._tool_worker_threads_lock = threading.Lock()
        self._execution_thread_id = None
        self._delegate_spinner = None
        self.quiet_mode = True
        self.session_id = session_id
        self._delegate_parent_ref = weakref.ref(parent) if parent is not None else None


_REAL_FINALIZE = dispatch._finalize_child_results


@pytest.fixture(autouse=True)
def _no_finalize(monkeypatch):
    # Host finalization (hooks, memory, cost rollup) is covered elsewhere; these tests pin the join.
    monkeypatch.setattr(dispatch, "_finalize_child_results", lambda *a, **k: None)


@pytest.fixture
def joined(monkeypatch):
    """Task indexes the join has collected (it prints one completion line per collected child)."""
    seen = []
    monkeypatch.setattr(dispatch, "_report_child_done", lambda _p, _s, entry, *a: seen.append(entry["task_index"]))
    return seen


def _settled(agent, idx):
    """True once the detached sub-helper ``idx`` has finished (its result is ready for ``wait``)."""
    return any(f.done() for d in vars(agent).get("_detached_joins") or () for f, i in d.pending.items() if i == idx)


class _Kids:
    """Children whose run blocks until released (or stopped); ``fast`` ones return at once."""

    def __init__(self, n, fast=(), groups=None, prefix="sa"):
        self.release = {i: threading.Event() for i in range(n)}
        self.started = {i: threading.Event() for i in range(n)}
        self.alive = set()
        self.fast = set(fast)
        self.children = [SimpleNamespace(_delegate_role="leaf", _subagent_id=f"{prefix}-{i}", _interrupt_requested=False)
                         for i in range(n)]
        self.tasks = [{"goal": f"task {i}", **({"group": groups[i]} if groups and groups[i] else {})} for i in range(n)]

    def run(self, i, task, child):
        self.started[i].set()
        self.alive.add(i)
        try:
            while i not in self.fast and not self.release[i].wait(0.02):
                if child._interrupt_requested:
                    return {"task_index": i, "status": "interrupted", "summary": None, "error": "stopped",
                            "api_calls": 1, "duration_seconds": 0}
            return {"task_index": i, "status": "completed", "summary": f"s{i}", "error": None, "api_calls": 1,
                    "duration_seconds": 0}
        finally:
            self.alive.discard(i)

    def batch(self, parent):
        b = _Batch(
            task_list=self.tasks, children=[(i, t, c) for i, (t, c) in enumerate(zip(self.tasks, self.children))],
            parent_agent=parent, creds={}, context=None, top_role="leaf", max_children=len(self.tasks),
            live_deleg_id=None, live_writers=[], live_paths=[], origin_wake_sid="", origin_ui_session_id="",
            origin_owner_transport=None, origin_owner_session_record=None, origin_session_history_delivery=False,
            overall_start=time.monotonic(),
        )
        b.run_child = self.run
        return b


def _in_thread(fn, *args):
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("out", fn(*args)), daemon=True)
    t.start()
    return t, box


def _wait_json(agent):
    return json.loads(wait_for_detached(agent))


def _steer_and_time(agent, thread, text="new direction"):
    t0 = time.monotonic()
    assert agent.steer(text)
    thread.join(3)
    assert not thread.is_alive(), "the steer did not end the wait"
    return time.monotonic() - t0


def test_steer_ends_the_join_at_once_and_wait_collects_the_rest(joined):
    helper, kids = _Agent(1), _Kids(2, fast={0})
    t, box = _in_thread(_execute_and_aggregate, kids.batch(helper))
    assert kids.started[1].wait(3)
    wait_until(lambda: joined == [0], timeout=3)
    assert _steer_and_time(helper, t) < 1.5
    out = box["out"]
    assert [r["task_index"] for r in out["results"]] == [0]
    assert [r["subagent_id"] for r in out["still_running"]] == ["sa-1"]
    assert "delegate_task(action=\"wait\")" in out["note"]
    assert kids.alive == {1}, "the running sub-helper must not be stopped"
    assert helper._drain_pending_steer() == "new direction"  # the helper reads it at its next boundary
    kids.release[1].set()
    got = _wait_json(helper)
    assert got["status"] == "completed"
    assert [r["task_index"] for r in got["batches"][0]["results"]] == [1]
    assert "still_running" not in got["batches"][0]
    assert "Nothing to wait for" in wait_for_detached(helper)


def test_a_second_steer_interrupts_wait_and_wait_can_resume():
    helper, kids = _Agent(1), _Kids(2)
    t, box = _in_thread(_execute_and_aggregate, kids.batch(helper))
    assert kids.started[0].wait(3) and kids.started[1].wait(3)
    _steer_and_time(helper, t, "first")
    assert box["out"]["results"] == [] and len(box["out"]["still_running"]) == 2
    helper._drain_pending_steer()
    kids.release[0].set()
    wait_until(lambda: _settled(helper, 0), timeout=3)
    t2, box2 = _in_thread(_wait_json, helper)
    assert _steer_and_time(helper, t2, "second") < 1.5
    got = box2["out"]
    assert got["status"] == "steered" and "wait" in got["note"]
    assert [r["task_index"] for r in got["batches"][0]["results"]] == [0]
    assert [r["task_index"] for r in got["batches"][0]["still_running"]] == [1]
    helper._drain_pending_steer()
    kids.release[1].set()
    assert [r["task_index"] for r in _wait_json(helper)["batches"][0]["results"]] == [1]


def test_interrupt_during_wait_reports_running_children_interrupted():
    helper, kids = _Agent(1), _Kids(2)
    t, _ = _in_thread(_execute_and_aggregate, kids.batch(helper))
    assert kids.started[1].wait(3)
    _steer_and_time(helper, t)
    helper._drain_pending_steer()
    t2, box2 = _in_thread(_wait_json, helper)
    helper._interrupt_requested = True
    t2.join(3)
    assert not t2.is_alive()
    got = box2["out"]
    assert got["status"] == "interrupted"
    assert {r["status"] for r in got["batches"][0]["results"]} == {"interrupted"}
    for e in kids.release.values():
        e.set()


def test_interrupt_during_a_join_is_unchanged():
    helper, kids = _Agent(1), _Kids(2)
    t, box = _in_thread(_execute_and_aggregate, kids.batch(helper))
    assert kids.started[1].wait(3)
    helper._interrupt_requested = True
    t.join(3)
    assert {r["status"] for r in box["out"]["results"]} == {"interrupted"}
    assert "still_running" not in box["out"] and not vars(helper).get("_detached_joins")
    for e in kids.release.values():
        e.set()


def test_uncollected_results_are_surfaced_on_the_helpers_entry(joined):
    helper, kids = _Agent(1), _Kids(3, fast={0})
    t, _ = _in_thread(_execute_and_aggregate, kids.batch(helper))
    assert kids.started[1].wait(3) and kids.started[2].wait(3)
    wait_until(lambda: joined == [0], timeout=3)
    _steer_and_time(helper, t)
    kids.release[1].set()  # finishes after the early return, never collected
    wait_until(lambda: _settled(helper, 1), timeout=3)
    entry = surface_uncollected(helper, {"task_index": 0, "status": "completed", "summary": "helper done"})
    by = {r["task_index"]: r for r in entry["uncollected_sub_results"]}
    assert set(by) == {1, 2}
    assert by[1]["status"] == "completed"
    assert by[2]["status"] == "interrupted" and "finished before collecting" in by[2]["error"]
    assert kids.children[2]._interrupt_requested is True  # the still-running one is stopped, not orphaned
    assert "Nothing to wait for" in wait_for_detached(helper)
    assert surface_uncollected(helper, {"x": 1}) == {"x": 1}


def test_helper_entry_carries_uncollected_results_through_run_single_child(monkeypatch):
    import tools.delegate_tool as dt
    seen = {}
    monkeypatch.setattr(dispatch, "surface_uncollected", lambda child, entry: seen.setdefault("child", child) and entry)
    child = SimpleNamespace(run_conversation=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    dt._run_single_child(0, "goal", child=child, parent_agent=SimpleNamespace(_active_children=[]))
    assert seen["child"] is child


@pytest.mark.parametrize("depth", [0, 1])
def test_no_steer_means_the_same_join(depth):
    helper, kids = _Agent(depth), _Kids(3, fast={0, 1, 2})
    out = _execute_and_aggregate(kids.batch(helper))
    assert [r["task_index"] for r in out["results"]] == [0, 1, 2]
    assert set(out) == {"results", "total_duration_seconds"}


def test_top_level_join_ignores_steers():
    """depth 0 (one-shot / gateway inline): a steer is delivered when the call returns, as before."""
    top, kids = _Agent(0), _Kids(2)
    t, box = _in_thread(_execute_and_aggregate, kids.batch(top))
    assert kids.started[1].wait(3)
    top.steer("later")
    t.join(1.0)
    assert t.is_alive(), "a top-level join must not be cut short by a steer"
    for e in kids.release.values():
        e.set()
    t.join(3)
    assert len(box["out"]["results"]) == 2 and "still_running" not in box["out"]


def test_top_level_single_child_still_runs_in_the_callers_thread():
    top = _Agent(0)
    caller = threading.current_thread()
    kids = _Kids(1, fast={0})
    b = kids.batch(top)
    ran_in = []
    b.run_child = lambda i, t, c: ran_in.append(threading.current_thread()) or kids.run(i, t, c)
    _execute_and_aggregate(b)
    assert ran_in == [caller]


def test_a_steer_already_queued_when_the_join_starts_does_not_end_it():
    helper, kids = _Agent(1), _Kids(1)
    helper.steer("old")
    t, box = _in_thread(_execute_and_aggregate, kids.batch(helper))
    assert kids.started[0].wait(3)
    t.join(0.8)
    assert t.is_alive()
    assert _steer_and_time(helper, t, "new") < 1.5
    assert box["out"]["still_running"][0]["task_index"] == 0
    kids.release[0].set()
    helper._drain_pending_steer()
    assert _wait_json(helper)["batches"][0]["results"][0]["status"] == "completed"


def test_wait_action_is_routed_and_advertised():
    from tools.delegate_tool import DELEGATE_TASK_SCHEMA
    assert "wait" in DELEGATE_TASK_SCHEMA["parameters"]["properties"]["action"]["enum"]
    assert "Nothing to wait for" in delegate_task(action="wait", parent_agent=_Agent(1))


# ── deeper trees: every joined level is steerable; a relayed steer counts for that level ──

def _register(sid, agent, *, parent_id=None, depth=0, owner_sid="root-session"):
    _register_subagent({
        "subagent_id": sid, "parent_id": parent_id, "depth": depth, "goal": f"goal of {sid}", "model": "m",
        "started_at": time.time(), "status": "running", "tool_count": 0, "agent": agent,
        "owner_agent_session_id": owner_sid,
    })


def test_depth_three_tree_every_level_breaks_out_and_rejoins():
    root = type("Root", (), {"session_id": "root-session"})()
    d1 = _Agent(1, root, session_id="d1-session")
    d2 = _Agent(2, d1, session_id="d2-session")
    leaves = _Kids(2, prefix="d3")
    d2_rejoined = {}

    def d1_runs_d2(i, task, child):  # d1's only sub-helper is d2, itself joined on two depth-3 leaves
        first = _execute_and_aggregate(leaves.batch(d2))
        d2_rejoined["early"] = first
        d2_rejoined["read"] = d2._drain_pending_steer()
        got = _wait_json(d2)
        d2_rejoined["wait"] = got
        return {"task_index": i, "status": "completed", "summary": "d2 done", "error": None, "api_calls": 1,
                "duration_seconds": 0}

    d1_kids = _Kids(1)
    d1_kids.children[0]._subagent_id = "d2"
    b = d1_kids.batch(d1)
    b.run_child = d1_runs_d2
    _register("d1", d1)
    _register("d2", d2, parent_id="d1", depth=1, owner_sid="d1-session")
    try:
        t, box = _in_thread(_execute_and_aggregate, b)
        assert leaves.started[0].wait(3) and leaves.started[1].wait(3)
        t0 = time.monotonic()
        relayed = []
        assert steer_subagent("d1", "pivot", relayed_to=relayed) and relayed == ["d2"]
        t.join(3)
        assert not t.is_alive() and time.monotonic() - t0 < 1.5, "depth-1 join did not end on its steer"
        assert box["out"]["still_running"][0]["subagent_id"] == "d2"
        deadline = time.monotonic() + 3
        while "early" not in d2_rejoined and time.monotonic() < deadline:
            time.sleep(0.02)
        assert len(d2_rejoined["early"]["still_running"]) == 2, "depth-2 join did not end on the relayed steer"
        assert "pivot" in d2_rejoined["read"]
        assert leaves.alive == {0, 1}
        for e in leaves.release.values():
            e.set()
        assert d1._drain_pending_steer() == "pivot"
        got = _wait_json(d1)
        assert got["status"] == "completed" and got["batches"][0]["results"][0]["summary"] == "d2 done"
        assert [r["task_index"] for r in d2_rejoined["wait"]["batches"][0]["results"]] == [0, 1]
    finally:
        for e in leaves.release.values():
            e.set()
        _unregister_subagent("d2")
        _unregister_subagent("d1")


# ── grouped batches, independent_completions on and off, several batches in flight ──

@pytest.mark.parametrize("independent", [True, False])
def test_grouped_batches_list_running_children_per_group_and_wait_collects_all(monkeypatch, joined, independent):
    import tools.delegate_tool_config as cfg
    monkeypatch.setattr(cfg, "_get_independent_completions", lambda: independent)
    helper = _Agent(1)
    first = _Kids(3, fast={0}, groups=["a", "a", "b"], prefix="x")
    second = _Kids(2, groups=["c", None], prefix="y")
    t1, box1 = _in_thread(_execute_and_aggregate, first.batch(helper))
    assert first.started[2].wait(3)
    wait_until(lambda: joined == [0], timeout=3)
    _steer_and_time(helper, t1, "one")
    out = box1["out"]
    assert out["still_running_by_group"] == {"a": ["x-1"], "b": ["x-2"]}
    helper._drain_pending_steer()
    t2, box2 = _in_thread(_execute_and_aggregate, second.batch(helper))  # a second batch, also left running
    assert second.started[1].wait(3)
    _steer_and_time(helper, t2, "two")
    assert box2["out"]["still_running_by_group"] == {"c": ["y-0"], "": ["y-1"]}
    helper._drain_pending_steer()
    for kids in (first, second):
        for e in kids.release.values():
            e.set()
    got = _wait_json(helper)
    assert got["status"] == "completed"
    assert [[r["task_index"] for r in b["results"]] for b in got["batches"]] == [[1, 2], [0, 1]]


def test_uncollected_grandchild_cost_reaches_the_root_through_real_finalization(monkeypatch):
    """helper $1 + a finished, never-collected grandchild $2: the root's session cost must rise by $3."""
    from tools.delegate_tool_results import _run_child_lifecycle
    monkeypatch.setattr(dispatch, "_finalize_child_results", _REAL_FINALIZE)
    root = SimpleNamespace(session_id="root-session", _active_children=[], session_estimated_cost_usd=0.0)
    helper = _Agent(1, session_id="helper-session")
    grand = _Kids(2)

    def costly(i, task, child):
        return {**grand.run(i, task, child), "_child_cost_usd": 2.0 if i == 0 else 0.0}

    def helper_runs(**_kw):
        b = grand.batch(helper)
        b.run_child = costly
        t, _ = _in_thread(_execute_and_aggregate, b)
        assert grand.started[0].wait(3) and grand.started[1].wait(3)
        _steer_and_time(helper, t)
        helper._drain_pending_steer()
        grand.release[0].set()  # finishes, but the helper ends without action='wait'
        wait_until(lambda: _settled(helper, 0), timeout=3)
        helper.session_estimated_cost_usd = 1.0  # the helper's own spend
        return {"final_response": "helper done", "completed": True, "messages": [], "api_calls": 1}

    helper.run_conversation = helper_runs
    try:
        entry = _run_child_lifecycle(0, "goal", child=helper, parent_agent=root)
    finally:
        for e in grand.release.values():
            e.set()
    assert {r["task_index"] for r in entry["uncollected_sub_results"]} == {0, 1}
    assert entry["cost_usd"] == pytest.approx(3.0)
    assert root.session_estimated_cost_usd == pytest.approx(3.0)


def test_live_manifest_stays_open_until_the_detached_batch_settles(tmp_path, joined):
    from tools.delegation_live_log import _manifest_path, _write_manifest
    helper, kids = _Agent(1), _Kids(2, fast={0})
    b = kids.batch(helper)
    b.live_deleg_id, b.live_home = "deleg-x", tmp_path
    _manifest_path("deleg-x", tmp_path).parent.mkdir(parents=True)
    _write_manifest("deleg-x", kids.tasks, ["a.log", "b.log"], home=tmp_path)
    manifest = lambda: json.loads(_manifest_path("deleg-x", tmp_path).read_text())
    t, _ = _in_thread(_execute_and_aggregate, b)
    assert kids.started[1].wait(3)
    wait_until(lambda: joined == [0], timeout=3)
    _steer_and_time(helper, t)
    m = manifest()
    assert "completed" not in m, "children still run: the delegation is not complete"
    assert [x["status"] for x in m["tasks"]] == ["completed", "running"]
    helper._drain_pending_steer()
    kids.release[1].set()
    assert _wait_json(helper)["status"] == "completed"
    m = manifest()
    assert "completed" in m and [x["status"] for x in m["tasks"]] == ["completed", "completed"]
