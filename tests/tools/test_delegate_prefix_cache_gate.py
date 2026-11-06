"""Prompt-cache warm-up gate for batch subagents (delegation.prefix_stagger_seconds)."""

import threading
import time
from types import SimpleNamespace

import pytest

from hermes_constants import get_hermes_home
from tests.waiting import wait_until
from tools import delegate_tool_dispatch
from tools.delegate_tool_child_run import _attach_child
from tools.delegate_tool_prefix_gate import PrefixCacheGate


def _child(**kw):
    base = dict(provider="anthropic", base_url="https://api.anthropic.com", model="claude", api_mode="anthropic_messages",
                _use_prompt_caching=True, session_api_calls=0, _last_activity_desc=None, _interrupt_requested=False)
    return SimpleNamespace(**{**base, **kw})


def test_cache_sharing_siblings_start_only_after_the_leaders_first_response_begins():
    leader, sibling = _child(), _child()
    other_route, uncached = _child(model="other"), _child(_use_prompt_caching=False)
    children = [(i, {}, c) for i, c in enumerate((leader, sibling, other_route, uncached))]
    gate = PrefixCacheGate.for_children(children, max_wait_s=30)
    assert gate is not None
    started, first_chunk, release = set(), threading.Event(), threading.Event()

    def leader_run():
        started.add("leader")
        first_chunk.wait(5)
        leader._last_activity_desc = "receiving stream response"  # first chunk of the leader's first response
        release.wait(5)

    threads = [threading.Thread(target=gate.run, args=(leader, leader_run))]
    threads += [threading.Thread(target=gate.run, args=(c, lambda n=n: started.add(n)))
                for c, n in ((sibling, "sibling"), (other_route, "other_route"), (uncached, "uncached"))]
    for t in threads:
        t.start()
    wait_until(lambda: {"leader", "other_route", "uncached"} <= started, timeout=5)  # nothing to share: no wait
    assert "sibling" not in started  # held while the leader's request is still in flight
    first_chunk.set()
    wait_until(lambda: "sibling" in started, timeout=5)  # released by the first chunk, long before the 30 s cap
    for t in threads[1:]:
        t.join(5)
    assert threads[0].is_alive()  # the leader is still streaming; the siblings did not wait for it to finish
    release.set()
    threads[0].join(5)


def test_gate_is_off_by_default_bounded_by_the_cap_and_a_stopped_sibling_never_starts():
    leader, sibling, stopped = _child(), _child(), _child(_interrupt_requested=True)
    children = [(0, {}, leader), (1, {}, sibling), (2, {}, stopped)]
    assert PrefixCacheGate.for_children(children, max_wait_s=0) is None
    assert PrefixCacheGate.for_children([(0, {}, _child(_use_prompt_caching=False)), (1, {}, sibling)], 30) is None

    gate = PrefixCacheGate.for_children(children, max_wait_s=0.2)
    assert gate is not None
    t0 = time.monotonic()
    assert gate.run(sibling, lambda: "ran") == "ran"  # the leader never answered: the cap releases the sibling
    assert 0.15 <= time.monotonic() - t0 < 2
    assert gate.run(stopped, lambda: "ran") is None


def _gated_batch(parent, kids, run_child):
    tasks = [{"goal": f"g{i}"} for i in range(len(kids))]
    return SimpleNamespace(parent_agent=parent, task_list=tasks, children=[(i, tasks[i], c) for i, c in enumerate(kids)],
                           live_deleg_id=None, max_children=len(kids), unit_id=None, live_paths=[], run_child=run_child)


def test_background_siblings_ignore_a_parent_stop_and_a_sibling_stopped_while_waiting_is_released():
    (get_hermes_home() / "config.yaml").write_text("delegation:\n  prefix_stagger_seconds: 30\n", encoding="utf-8")
    parent = SimpleNamespace(_interrupt_requested=False, _delegate_spinner=None, _delegate_depth=0,
                             _active_children=[], _active_children_lock=threading.Lock())
    closed, ran = [], []
    kids = [_child(name=f"c{i}", _delegate_role="leaf", close=lambda n=f"c{i}": closed.append(n)) for i in range(3)]
    for c in kids:
        _attach_child(parent, c)

    def run_child(i, task, c):
        ran.append(i)
        if i == 0:
            wait_until(lambda: closed, timeout=5)  # c2 was stopped and released while the gate still held it
            c.session_api_calls = 1  # the leader's first response landed: the gate opens
        return {"task_index": i, "status": "completed", "summary": "x", "api_calls": 1, "duration_seconds": 0}

    def stop_soon():
        wait_until(lambda: 0 in ran, timeout=5)  # the leader's request is in flight; the siblings are gated
        parent._interrupt_requested = True  # the user stops the parent's foreground turn
        kids[2]._interrupt_requested = True  # and this one sibling individually

    threading.Thread(target=stop_soon).start()
    results = []
    delegate_tool_dispatch._run_children_parallel(_gated_batch(parent, kids, run_child), results,
                                                  honor_parent_interrupt=False)
    assert [r["status"] for r in results] == ["completed", "completed", "interrupted"]
    assert sorted(ran) == [0, 1]  # a detached unit outlives the parent's turn; only the stopped sibling is skipped
    assert closed == ["c2"] and [c.name for c in parent._active_children] == ["c0", "c1"]  # c0/c1 own their cleanup


def test_a_non_streamed_first_response_or_a_failed_leader_releases_siblings_before_the_cap():
    leader, sibling = _child(), _child()
    gate = PrefixCacheGate(leader, max_wait_s=30)
    sibling_ran = threading.Event()

    def leader_run():
        leader.session_api_calls = 1  # a non-streamed first response landed; the leader keeps working
        sibling_ran.wait(20)

    threading.Thread(target=gate.run, args=(leader, leader_run)).start()
    t0 = time.monotonic()
    assert gate.run(sibling, lambda: sibling_ran.set() or "ran") == "ran"
    assert time.monotonic() - t0 < 10  # released by the API-call count, not by the leader exiting

    leader, sibling = _child(), _child()
    gate = PrefixCacheGate(leader, max_wait_s=30)

    def leader_fails():
        raise RuntimeError("provider 500 before any chunk")

    with pytest.raises(RuntimeError):
        gate.run(leader, leader_fails)
    t0 = time.monotonic()
    assert gate.run(sibling, lambda: "ran") == "ran"
    assert time.monotonic() - t0 < 10  # nothing will ever be cached: the leader's exit opens the gate


def test_queued_children_cancelled_by_a_parent_stop_are_detached_and_closed():
    # Default config (no gate): this teardown runs with the option off.
    parent = SimpleNamespace(_interrupt_requested=False, _delegate_spinner=None, _delegate_depth=0,
                             _active_children=[], _active_children_lock=threading.Lock())
    closed = []
    kids = [_child(name=f"c{i}", _delegate_role="leaf", close=lambda n=f"c{i}": closed.append(n)) for i in range(3)]
    for c in kids:
        _attach_child(parent, c)
    started, unblock = threading.Event(), threading.Event()

    def run_child(i, task, c):
        started.set()
        unblock.wait(5)  # holds the only worker, so c1 and c2 stay queued
        return {"task_index": i, "status": "completed", "summary": "x", "api_calls": 1, "duration_seconds": 0}

    def stop_soon():
        started.wait(5)
        parent._interrupt_requested = True

    batch = _gated_batch(parent, kids, run_child)
    batch.max_children = 1
    threading.Thread(target=stop_soon).start()
    results = []
    delegate_tool_dispatch._run_children_parallel(batch, results, honor_parent_interrupt=True)
    unblock.set()
    assert [r["status"] for r in results] == ["interrupted"] * 3
    assert sorted(closed) == ["c1", "c2"] and [c.name for c in parent._active_children] == ["c0"]
