"""Prompt-cache warm-up gate for batch subagents (delegation.prefix_stagger_seconds)."""

import threading
import time
from types import SimpleNamespace

from tests.waiting import wait_until
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
