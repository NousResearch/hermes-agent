"""Prompt-cache warm-up gate for batch subagents (delegation.prefix_stagger_seconds)."""

import threading
import time
from types import SimpleNamespace

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
    started, release = {}, threading.Event()

    def leader_run():
        started["leader"] = time.monotonic()
        time.sleep(0.3)
        leader._last_activity_desc = "receiving stream response"  # first chunk of the leader's first response
        release.wait(5)
        return "leader"

    def mark(name):
        started[name] = time.monotonic()
        return name

    threads = [threading.Thread(target=gate.run, args=(leader, leader_run))]
    threads += [threading.Thread(target=gate.run, args=(c, lambda n=n: mark(n)))
                for c, n in ((sibling, "sibling"), (other_route, "other_route"), (uncached, "uncached"))]
    for t in threads:
        t.start()
    time.sleep(0.15)
    assert "sibling" not in started  # held while the leader's request is still in flight
    assert {"other_route", "uncached"} <= started.keys()  # nothing to share: no wait
    for t in threads[1:]:
        t.join(5)
    release.set()
    threads[0].join(5)
    assert started["sibling"] - started["leader"] >= 0.25  # released by the first chunk, long before the 30 s cap


def test_gate_is_off_by_default_bounded_by_the_cap_and_a_stopped_sibling_never_starts():
    leader, sibling, stopped = _child(), _child(), _child(_interrupt_requested=True)
    children = [(0, {}, leader), (1, {}, sibling), (2, {}, stopped)]
    assert PrefixCacheGate.for_children(children, max_wait_s=0) is None
    assert PrefixCacheGate.for_children([(0, {}, _child(_use_prompt_caching=False)), (1, {}, sibling)], 30) is None

    gate = PrefixCacheGate.for_children(children, max_wait_s=0.2)
    t0 = time.monotonic()
    assert gate.run(sibling, lambda: "ran") == "ran"  # the leader never answered: the cap releases the sibling
    assert 0.15 <= time.monotonic() - t0 < 2
    assert gate.run(stopped, lambda: "ran") is None
